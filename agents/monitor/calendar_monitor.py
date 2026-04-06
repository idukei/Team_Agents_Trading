from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from typing import Any

import aiohttp
import structlog

from agents.monitor.eps_scorer import compute_eps
from core.config import settings
from core.models.events import EventSource, EventTrigger, EventUrgency

log = structlog.get_logger(__name__)

HORIZONFX_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Map calendar event names to EventSource
CALENDAR_SOURCE_MAP: dict[str, EventSource] = {
    "Non-Farm Payroll": EventSource.NFP,
    "Non-Farm Employment": EventSource.NFP,
    "CPI": EventSource.CPI,
    "Consumer Price Index": EventSource.CPI,
    "PCE": EventSource.PCE,
    "Personal Consumption": EventSource.PCE,
    "FOMC": EventSource.FOMC,
    "Federal Funds Rate": EventSource.FOMC,
    "GDP": EventSource.GDP,
    "PPI": EventSource.PPI,
    "Producer Price Index": EventSource.PPI,
    "PMI": EventSource.PMI,
    "Retail Sales": EventSource.RETAIL_SALES,
    "Unemployment Claims": EventSource.UNEMPLOYMENT_CLAIMS,
    "Initial Jobless": EventSource.UNEMPLOYMENT_CLAIMS,
    "ECB": EventSource.ECB_ANNOUNCEMENT,
    "ECB Rate": EventSource.ECB_ANNOUNCEMENT,
}


def _parse_event_source(title: str) -> EventSource:
    for keyword, source in CALENDAR_SOURCE_MAP.items():
        if keyword.lower() in title.lower():
            return source
    return EventSource.NEWS


class CalendarMonitor:
    """
    Polls HorizonFX Economic Calendar every 5 minutes.
    Publishes PRE_EVENT alerts at T-60s and DATA_RELEASE events when actuals arrive.
    """

    def __init__(self, on_event: asyncio.Queue) -> None:
        self._queue = on_event
        self._today_events: list[dict] = []
        self._pre_event_fired: set[str] = set()
        self._release_fired: set[str] = set()

    async def start(self) -> None:
        log.info("calendar_monitor_started")
        await asyncio.gather(
            self._refresh_loop(),
            self._pre_event_watchdog(),
        )

    async def _refresh_loop(self) -> None:
        while True:
            try:
                await self._fetch_events()
            except Exception as e:
                log.warning("calendar_fetch_error", error=str(e))
            await asyncio.sleep(300)    # every 5 minutes

    async def _fetch_events(self) -> None:
        async with aiohttp.ClientSession() as session:
            async with session.get(HORIZONFX_URL, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return
                events: list[dict] = await resp.json(content_type=None)

        today = date.today().isoformat()
        high_impact = [
            e for e in events
            if e.get("impact", "").lower() in ("high", "medium")
            and e.get("date", "").startswith(today)
        ]
        self._today_events = high_impact
        log.debug("calendar_refreshed", high_impact_count=len(high_impact))

        # Check for released actuals (actual value appeared)
        for event in high_impact:
            event_id = event.get("id") or f"{event['title']}_{event['date']}"
            actual = event.get("actual")
            consensus = event.get("forecast") or event.get("previous")

            if actual and event_id not in self._release_fired:
                self._release_fired.add(event_id)
                await self._fire_release_event(event, actual, consensus)

    async def _pre_event_watchdog(self) -> None:
        """Check every 10s whether any event is within 60s."""
        while True:
            now = datetime.now(timezone.utc)
            for event in self._today_events:
                event_id = event.get("id") or f"{event['title']}_{event['date']}"
                if event_id in self._pre_event_fired:
                    continue
                try:
                    event_time = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
                    seconds_until = (event_time - now).total_seconds()
                    if 0 < seconds_until <= 60:
                        self._pre_event_fired.add(event_id)
                        await self._fire_pre_event(event, seconds_until)
                except (ValueError, KeyError):
                    pass
            await asyncio.sleep(10)

    async def _fire_pre_event(self, event: dict, seconds_until: float) -> None:
        title = event.get("title", "Unknown")
        source = _parse_event_source(title)

        trigger = EventTrigger(
            event_source=source,
            eps_score=65.0,             # pre-event warning, medium priority
            urgency=EventUrgency.MEDIUM,
            raw_content=f"PRE-EVENT in {int(seconds_until)}s: {title}",
            event_name=title,
            consensus=_safe_float(event.get("forecast")),
            previous=_safe_float(event.get("previous")),
        )
        log.info("calendar_pre_event", title=title, seconds_until=int(seconds_until))
        await self._queue.put(trigger)

    async def _fire_release_event(self, event: dict, actual: Any, consensus: Any) -> None:
        title = event.get("title", "Unknown")
        source = _parse_event_source(title)

        actual_f = _safe_float(actual)
        consensus_f = _safe_float(consensus)
        surprise = None
        if actual_f is not None and consensus_f is not None:
            surprise = actual_f - consensus_f

        # Only fire if there's a meaningful surprise (non-zero)
        if surprise is not None and abs(surprise) < 0.001:
            return

        eps, urgency, _ = compute_eps(
            source=source,
            text=f"{title} actual={actual} consensus={consensus}",
            is_novel=True,
            vix_norm=0.5,
        )

        trigger = EventTrigger(
            event_source=source,
            eps_score=eps,
            urgency=urgency,
            raw_content=f"ECONOMIC RELEASE: {title} — Actual: {actual} | Consensus: {consensus}",
            event_name=title,
            actual=actual_f,
            consensus=consensus_f,
            previous=_safe_float(event.get("previous")),
            surprise=surprise,
        )
        log.info("calendar_release", title=title, actual=actual, consensus=consensus, surprise=surprise, eps=eps)
        await self._queue.put(trigger)


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        # Remove % signs and other common suffixes
        cleaned = str(val).replace("%", "").replace("K", "").replace("M", "").strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return None
