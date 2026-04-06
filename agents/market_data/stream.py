from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from core.config import settings


@dataclass
class Tick:
    timestamp_ms: int
    price: float
    volume: float
    bid: float = 0.0
    ask: float = 0.0

    @property
    def spread(self) -> float:
        if self.ask > 0 and self.bid > 0:
            return self.ask - self.bid
        return 0.0

    @property
    def spread_bps(self) -> float:
        mid = (self.bid + self.ask) / 2
        if mid > 0 and self.spread > 0:
            return (self.spread / mid) * 10_000
        return 0.0

    @property
    def obi(self) -> float:
        """Order book imbalance: positive = more bids (bullish pressure)."""
        total = self.bid + self.ask
        if total == 0:
            return 0.0
        return (self.bid - self.ask) / total


class PriceBuffer:
    """
    Circular buffer of Tick objects for a single asset.
    Provides numpy arrays for indicator calculation.
    """

    def __init__(self, maxlen: int = 500) -> None:
        self._buf: deque[Tick] = deque(maxlen=maxlen)

    def push(self, tick: Tick) -> None:
        self._buf.append(tick)

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def prices(self):
        import numpy as np
        return np.array([t.price for t in self._buf], dtype=float)

    @property
    def volumes(self):
        import numpy as np
        return np.array([t.volume for t in self._buf], dtype=float)

    @property
    def timestamps_ms(self):
        import numpy as np
        return np.array([t.timestamp_ms for t in self._buf], dtype=np.int64)

    @property
    def bids(self):
        import numpy as np
        return np.array([t.bid for t in self._buf], dtype=float)

    @property
    def asks(self):
        import numpy as np
        return np.array([t.ask for t in self._buf], dtype=float)

    @property
    def latest(self) -> Tick | None:
        return self._buf[-1] if self._buf else None

    @property
    def latest_price(self) -> float | None:
        return self._buf[-1].price if self._buf else None

    @property
    def latest_obi(self) -> float:
        return self._buf[-1].obi if self._buf else 0.0

    @property
    def latest_spread_bps(self) -> float:
        return self._buf[-1].spread_bps if self._buf else 0.0

    def prices_in_window(self, window_ms: int) -> list[float]:
        if not self._buf:
            return []
        cutoff = self._buf[-1].timestamp_ms - window_ms
        return [t.price for t in self._buf if t.timestamp_ms >= cutoff]

    def volume_ma(self, periods: int = 20) -> float:
        """Rolling volume moving average over last N periods."""
        import numpy as np
        vols = self.volumes
        if len(vols) < 2:
            return float(vols[-1]) if len(vols) > 0 else 0.0
        window = min(periods, len(vols))
        return float(np.mean(vols[-window:]))


# Global registry of buffers — shared between PriceMonitor and MarketDataAgent
class BufferRegistry:
    def __init__(self) -> None:
        self._buffers: dict[str, PriceBuffer] = {}

    def get_or_create(self, asset: str) -> PriceBuffer:
        if asset not in self._buffers:
            self._buffers[asset] = PriceBuffer(maxlen=settings.price_buffer_size)
        return self._buffers[asset]

    def get(self, asset: str) -> PriceBuffer | None:
        return self._buffers.get(asset)

    def all_assets(self) -> list[str]:
        return list(self._buffers.keys())


# Singleton registry used by both PriceMonitor and MarketDataAgent
buffer_registry = BufferRegistry()
