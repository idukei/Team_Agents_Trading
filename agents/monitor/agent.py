from __future__ import annotations

import asyncio
import time

import structlog

from agents.monitor.calendar_monitor import CalendarMonitor
from agents.monitor.eps_scorer import should_activate_pipeline
from agents.monitor.leaders_monitor import LeadersMonitor
from agents.monitor.news_monitor import NewsMonitor
from agents.monitor.price_monitor import PriceMonitor
from core.config import settings
from core.db.redis import publish_event
from core.models.events import EventTrigger

log = structlog.get_logger(__name__)


class MonitorAgent:
    """
    Always-on agent running 4 parallel sub-monitors as asyncio tasks.
    Receives EventTrigger objects from sub-monitors via internal queue.
    Filters by EPS threshold and publishes to Redis Stream.

    No LLM consumption — pure data collection, filtering, and routing.
    """

    def __init__(self) -> None:
        self._event_queue: asyncio.Queue[EventTrigger] = asyncio.Queue(maxsize=1000)
        self._last_event_per_source: dict[str, float] = {}   # source → timestamp for dedup
        self.price_monitor: PriceMonitor | None = None

    async def start(self) -> None:
        log.info("monitor_agent_starting")

        self.price_monitor = PriceMonitor(self._event_queue)

        sub_monitors = [
            LeadersMonitor(self._event_queue),
            CalendarMonitor(self._event_queue),
            NewsMonitor(self._event_queue),
            self.price_monitor,
        ]

        tasks = [
            asyncio.create_task(monitor.start(), name=type(monitor).__name__)
            for monitor in sub_monitors
        ]
        tasks.append(asyncio.create_task(self._publish_loop(), name="MonitorPublishLoop"))

        log.info("monitor_agent_started", sub_monitors=len(sub_monitors))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _publish_loop(self) -> None:
        """
        Drain the internal queue, apply global dedup, and publish to Redis Stream.
        Rate-limit same source: ignore duplicate events within 10s window.
        """
        while True:
            try:
                trigger: EventTrigger = await asyncio.wait_for(
                    self._event_queue.get(), timeout=5.0
                )
            except asyncio.TimeoutError:
                continue

            # Global rate-limit: same source every 10s max (prevents spam)
            source_key = f"{trigger.event_source}:{trigger.affected_assets_guess[:1]}"
            last_time = self._last_event_per_source.get(source_key, 0)
            now = time.monotonic()
            if (now - last_time) < 10.0 and trigger.eps_score < 90:
                log.debug("monitor_rate_limited", source=trigger.event_source, eps=trigger.eps_score)
                continue
            self._last_event_per_source[source_key] = now

            pipeline_mode = should_activate_pipeline(trigger)

            log.info(
                "monitor_event_published",
                source=trigger.event_source,
                eps=trigger.eps_score,
                urgency=trigger.urgency,
                pipeline=pipeline_mode,
                assets=trigger.affected_assets_guess,
            )

            # Publish to Redis Stream (triggers LangGraph pipeline)
            await publish_event(
                settings.redis_stream_events,
                trigger.to_redis_dict(),
            )
