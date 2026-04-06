from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone

import structlog
from alpaca.data.live import StockDataStream, CryptoDataStream
from alpaca.data.models import Quote, Trade

from agents.monitor.eps_scorer import compute_eps
from core.config import settings
from core.models.events import EventSource, EventTrigger, EventUrgency

log = structlog.get_logger(__name__)


class PriceMonitor:
    """
    Connects to Alpaca WebSocket streams for real-time price/volume data.
    Detects price spikes (>= threshold % in window) and volume spikes.
    Publishes EventTrigger to queue.

    Also maintains a shared price_buffer dict for Market Data Agent.
    """

    def __init__(self, on_event: asyncio.Queue) -> None:
        self._queue = on_event
        # price_buffers[asset] = deque of (timestamp_ms, price, volume) tuples
        self.price_buffers: dict[str, deque] = {
            asset: deque(maxlen=settings.price_buffer_size)
            for asset in settings.all_assets
        }
        self._last_price: dict[str, float] = {}
        self._volume_ma: dict[str, float] = {}       # rolling 20-period volume MA

    async def start(self) -> None:
        log.info("price_monitor_started", assets=settings.all_assets)
        tasks = []
        if settings.monitored_equities:
            tasks.append(asyncio.create_task(self._stream_stocks()))
        if settings.monitored_crypto:
            tasks.append(asyncio.create_task(self._stream_crypto()))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _stream_stocks(self) -> None:
        stream = StockDataStream(
            api_key=settings.alpaca_api_key.get_secret_value(),
            secret_key=settings.alpaca_secret_key.get_secret_value(),
        )

        async def handle_quote(q: Quote) -> None:
            mid = (float(q.ask_price) + float(q.bid_price)) / 2
            await self._on_price_update(str(q.symbol), mid, 0, int(q.timestamp.timestamp() * 1000))

        async def handle_trade(t: Trade) -> None:
            await self._on_price_update(
                str(t.symbol), float(t.price), float(t.size),
                int(t.timestamp.timestamp() * 1000)
            )

        stream.subscribe_quotes(handle_quote, *settings.monitored_equities)
        stream.subscribe_trades(handle_trade, *settings.monitored_equities)

        while True:
            try:
                await stream.run()
            except Exception as e:
                log.warning("stock_stream_error", error=str(e))
                await asyncio.sleep(5)

    async def _stream_crypto(self) -> None:
        stream = CryptoDataStream(
            api_key=settings.alpaca_api_key.get_secret_value(),
            secret_key=settings.alpaca_secret_key.get_secret_value(),
        )

        async def handle_trade(t: Trade) -> None:
            await self._on_price_update(
                str(t.symbol), float(t.price), float(t.size),
                int(t.timestamp.timestamp() * 1000)
            )

        stream.subscribe_trades(handle_trade, *settings.monitored_crypto)

        while True:
            try:
                await stream.run()
            except Exception as e:
                log.warning("crypto_stream_error", error=str(e))
                await asyncio.sleep(5)

    async def _on_price_update(self, asset: str, price: float, volume: float, ts_ms: int) -> None:
        buf = self.price_buffers.get(asset)
        if buf is None:
            return

        buf.append((ts_ms, price, volume))
        self._last_price[asset] = price

        # Update rolling volume MA
        if volume > 0:
            prev_ma = self._volume_ma.get(asset, volume)
            self._volume_ma[asset] = prev_ma * 0.95 + volume * 0.05

        # Check spike conditions
        await self._check_price_spike(asset, price, ts_ms, buf)
        if volume > 0:
            await self._check_volume_spike(asset, volume)

    async def _check_price_spike(
        self, asset: str, current_price: float, ts_ms: int, buf: deque
    ) -> None:
        window_ms = settings.price_spike_window_s * 1000
        cutoff_ms = ts_ms - window_ms

        prices_in_window = [p for (t, p, _) in buf if t >= cutoff_ms]
        if len(prices_in_window) < 2:
            return

        base_price = prices_in_window[0]
        pct_change = abs((current_price - base_price) / base_price * 100)

        if pct_change >= settings.price_spike_threshold_pct:
            direction = "up" if current_price > base_price else "down"
            eps, urgency, _ = compute_eps(
                source=EventSource.PRICE_SPIKE,
                text=f"Price spike {asset} {direction} {pct_change:.2f}%",
                is_novel=True,
                vix_norm=min(pct_change / 3.0, 1.0),
            )
            trigger = EventTrigger(
                event_source=EventSource.PRICE_SPIKE,
                eps_score=eps,
                urgency=urgency,
                raw_content=f"PRICE SPIKE: {asset} moved {direction} {pct_change:.2f}% in {settings.price_spike_window_s}s",
                affected_assets_guess=[asset],
                expected_direction_hint=f"BEARISH_{asset}" if direction == "down" else f"BULLISH_{asset}",
            )
            log.info("price_spike_detected", asset=asset, pct=pct_change, direction=direction)
            await self._queue.put(trigger)

    async def _check_volume_spike(self, asset: str, volume: float) -> None:
        ma = self._volume_ma.get(asset, 0)
        if ma <= 0:
            return
        ratio = volume / ma
        if ratio >= settings.volume_spike_multiplier:
            trigger = EventTrigger(
                event_source=EventSource.VOLUME_SPIKE,
                eps_score=55.0,
                urgency=EventUrgency.MEDIUM,
                raw_content=f"VOLUME SPIKE: {asset} volume is {ratio:.1f}x normal",
                affected_assets_guess=[asset],
            )
            log.info("volume_spike_detected", asset=asset, ratio=ratio)
            await self._queue.put(trigger)

    def get_latest_price(self, asset: str) -> float | None:
        return self._last_price.get(asset)
