from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone

import aiohttp
import feedparser
import structlog

from agents.monitor.eps_scorer import compute_eps
from core.config import settings
from core.db.redis import exists, set_cache
from core.models.events import EventSource, EventTrigger

log = structlog.get_logger(__name__)

RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.marketwatch.com/marketwatch/topstories",
    "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
]

MARKETAUX_URL = "https://api.marketaux.com/v1/news/all"
NEWSDATA_URL = "https://newsdata.io/api/1/news"


def _hash_title(title: str) -> str:
    return hashlib.md5(title.lower().strip().encode()).hexdigest()[:12]


class NewsMonitor:
    """
    Polls financial news from:
    - Marketaux API (100 req/day free)
    - NewsData.io API (200 credits/day free)
    - RSS feeds (unlimited)

    Deduplicates by title hash.
    Fires EventTrigger only for news with EPS >= eps_light_pipeline.
    """

    def __init__(self, on_event: asyncio.Queue) -> None:
        self._queue = on_event
        self._seen: set[str] = set()

    async def start(self) -> None:
        log.info("news_monitor_started")
        tasks = [
            asyncio.create_task(self._rss_loop()),
            asyncio.create_task(self._marketaux_loop()),
        ]
        if settings.newsdata_api_key:
            tasks.append(asyncio.create_task(self._newsdata_loop()))
        await asyncio.gather(*tasks, return_exceptions=True)

    # ── RSS ───────────────────────────────────────────────────────────────────
    async def _rss_loop(self) -> None:
        while True:
            for url in RSS_FEEDS:
                try:
                    await self._process_rss(url)
                except Exception as e:
                    log.debug("rss_error", url=url, error=str(e))
            await asyncio.sleep(10)     # RSS: every 10s

    async def _process_rss(self, url: str) -> None:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                content = await resp.text()
        feed = feedparser.parse(content)
        for entry in feed.entries[:10]:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            text = f"{title}. {summary}"
            await self._evaluate_and_fire(text, title)

    # ── Marketaux ─────────────────────────────────────────────────────────────
    async def _marketaux_loop(self) -> None:
        if not settings.marketaux_api_key:
            return
        while True:
            try:
                await self._fetch_marketaux()
            except Exception as e:
                log.warning("marketaux_error", error=str(e))
            await asyncio.sleep(900)    # 15 min → stays within 100 req/day

    async def _fetch_marketaux(self) -> None:
        params = {
            "api_token": settings.marketaux_api_key.get_secret_value(),
            "filter_entities": "true",
            "language": "en",
            "limit": 10,
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(MARKETAUX_URL, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
        for article in data.get("data", []):
            title = article.get("title", "")
            description = article.get("description", "")
            text = f"{title}. {description}"
            tickers = [e["symbol"] for e in article.get("entities", []) if e.get("symbol")]
            await self._evaluate_and_fire(text, title, affected_assets=tickers)

    # ── NewsData ──────────────────────────────────────────────────────────────
    async def _newsdata_loop(self) -> None:
        while True:
            try:
                await self._fetch_newsdata()
            except Exception as e:
                log.warning("newsdata_error", error=str(e))
            await asyncio.sleep(1800)   # 30 min → stays within 200 credits/day

    async def _fetch_newsdata(self) -> None:
        params = {
            "apikey": settings.newsdata_api_key.get_secret_value(),
            "category": "business,finance",
            "language": "en",
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(NEWSDATA_URL, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
        for article in data.get("results", []):
            title = article.get("title", "")
            description = article.get("description", "")
            text = f"{title}. {description}"
            await self._evaluate_and_fire(text, title)

    # ── Shared evaluation ──────────────────────────────────────────────────────
    async def _evaluate_and_fire(
        self,
        text: str,
        title: str,
        affected_assets: list[str] | None = None,
    ) -> None:
        h = _hash_title(title)
        if h in self._seen:
            return
        cache_key = f"news_seen:{h}"
        if await exists(cache_key):
            self._seen.add(h)
            return

        self._seen.add(h)
        await set_cache(cache_key, "1", ttl_s=43200)    # 12h dedup

        eps, urgency, matched = compute_eps(
            source=EventSource.NEWS,
            text=text,
            is_novel=True,
            vix_norm=0.3,
        )

        if eps < settings.eps_light_pipeline:
            return

        trigger = EventTrigger(
            event_source=EventSource.NEWS,
            eps_score=eps,
            urgency=urgency,
            raw_content=title,
            affected_assets_guess=affected_assets or [],
        )
        log.info("news_event_fired", title=title[:80], eps=eps, keywords=matched)
        await self._queue.put(trigger)
