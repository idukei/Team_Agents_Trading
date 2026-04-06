from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone

import aiohttp
import feedparser
import structlog

from agents.monitor.eps_scorer import compute_eps
from core.config import settings
from core.db.redis import exists, publish_event, set_cache
from core.models.events import EventSource, EventTrigger

log = structlog.get_logger(__name__)

# Monitored leaders with their RSS sources and EventSource mapping
MONITORED_LEADERS: list[dict] = [
    {
        "name": "Trump Truth Social",
        "source": EventSource.TRUMP_TRUTH_SOCIAL,
        "url": f"{settings.rsshub_url}/truthsocial/user/realDonaldTrump",
        "poll_interval_s": 15,
    },
    {
        "name": "Trump X/Twitter",
        "source": EventSource.TRUMP_X,
        "url": f"{settings.rsshub_url}/twitter/user/realDonaldTrump",
        "poll_interval_s": 15,
    },
    {
        "name": "Federal Reserve",
        "source": EventSource.FED_ANNOUNCEMENT,
        "url": "https://www.federalreserve.gov/feeds/press_all.xml",
        "poll_interval_s": 30,
    },
    {
        "name": "ECB",
        "source": EventSource.ECB_ANNOUNCEMENT,
        "url": "https://www.ecb.europa.eu/rss/press.html",
        "poll_interval_s": 30,
    },
    {
        "name": "US Treasury",
        "source": EventSource.TREASURY,
        "url": "https://home.treasury.gov/news/press-releases/feed",
        "poll_interval_s": 60,
    },
    {
        "name": "OPEC",
        "source": EventSource.OPEC,
        "url": "https://www.opec.org/opec_web/en/press_room/rss.htm",
        "poll_interval_s": 60,
    },
]

# Keywords→asset mapping for EPS affected_assets_guess
KEYWORD_ASSET_MAP: dict[str, list[str]] = {
    "tariff": ["SPY", "QQQ", "USDCNH"],
    "semiconductor": ["NVDA", "SMH", "AMD"],
    "china": ["SPY", "USDCNH", "FXI"],
    "oil": ["USO", "XOM", "CVX"],
    "bitcoin": ["BTC/USD"],
    "crypto": ["BTC/USD", "ETH/USD"],
    "rate": ["TLT", "SPY", "GLD"],
    "inflation": ["TLT", "GLD", "SPY"],
    "bank": ["JPM", "BAC", "XLF"],
    "tech": ["QQQ", "NVDA", "AAPL"],
}


def _guess_affected_assets(text: str) -> list[str]:
    lower = text.lower()
    assets: set[str] = set()
    for kw, asset_list in KEYWORD_ASSET_MAP.items():
        if kw in lower:
            assets.update(asset_list)
    return list(assets)[:6]


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class LeadersMonitor:
    """
    Polls RSS feeds of political and institutional leaders.
    For each new post: applies keyword filter → computes EPS → publishes to Redis.
    """

    def __init__(self, on_event: asyncio.Queue) -> None:
        self._queue = on_event
        self._seen: set[str] = set()            # content hashes already processed
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={"User-Agent": "TeamTrade/1.0 RSS Monitor"},
        )
        tasks = [
            asyncio.create_task(self._poll_leader(leader))
            for leader in MONITORED_LEADERS
        ]
        log.info("leaders_monitor_started", sources=len(MONITORED_LEADERS))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _poll_leader(self, leader: dict) -> None:
        while True:
            try:
                await self._fetch_and_process(leader)
            except Exception as e:
                log.warning("leaders_monitor_poll_error", name=leader["name"], error=str(e))
            await asyncio.sleep(leader["poll_interval_s"])

    async def _fetch_and_process(self, leader: dict) -> None:
        assert self._session is not None
        try:
            async with self._session.get(leader["url"]) as resp:
                if resp.status != 200:
                    return
                content = await resp.text()
        except Exception:
            return

        feed = feedparser.parse(content)
        for entry in feed.entries[:5]:      # check last 5 entries
            text = entry.get("summary") or entry.get("title") or ""
            if not text:
                continue

            content_hash = _content_hash(text)
            if content_hash in self._seen:
                continue

            # Check Redis dedup (survives restarts)
            cache_key = f"leader_seen:{content_hash}"
            if await exists(cache_key):
                self._seen.add(content_hash)
                continue

            self._seen.add(content_hash)
            await set_cache(cache_key, "1", ttl_s=86400)  # 24h dedup window

            # Quick keyword filter (no LLM, < 1ms)
            from agents.monitor.eps_scorer import count_keyword_hits
            hits, matched = count_keyword_hits(text)

            # Level-1 sources (Trump/Fed/ECB) always get EPS computed
            # Level-2 sources only if they hit at least 1 keyword
            is_level1 = leader["source"] in {
                EventSource.TRUMP_TRUTH_SOCIAL, EventSource.TRUMP_X,
                EventSource.FED_ANNOUNCEMENT, EventSource.ECB_ANNOUNCEMENT,
                EventSource.FOMC,
            }
            if not is_level1 and hits == 0:
                log.debug("leaders_no_keywords", source=leader["name"], text=text[:80])
                continue

            eps, urgency, _ = compute_eps(
                source=leader["source"],
                text=text,
                is_novel=True,        # we just deduped, it's novel
                vix_norm=0.5,         # TODO: get real VIX from Market Data Agent
            )

            trigger = EventTrigger(
                event_source=leader["source"],
                eps_score=eps,
                urgency=urgency,
                raw_content=text,
                source_account=leader["name"],
                affected_assets_guess=_guess_affected_assets(text),
            )

            log.info(
                "leader_statement_detected",
                source=leader["name"],
                eps=eps,
                urgency=urgency,
                keywords=matched,
                text_preview=text[:120],
            )

            await self._queue.put(trigger)
