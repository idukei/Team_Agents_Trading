from __future__ import annotations

import numpy as np


class AdaptiveZScore:
    """
    Adaptive Z-score with ATR-based dynamic threshold.
    Higher ATR → more volatile market → require higher Z-score to trigger.
    """

    def __init__(
        self,
        windows: tuple[int, int, int] = (20, 50, 200),
        base_threshold: float = 3.0,
    ) -> None:
        self._windows = windows
        self._base_threshold = base_threshold

    def compute(self, prices: np.ndarray, atr: float, price: float) -> tuple[float, bool, float]:
        """
        Compute Z-score for the latest price point.

        Returns (z_score, is_anomaly, dynamic_threshold).
        """
        if len(prices) < self._windows[0]:
            return 0.0, False, self._base_threshold

        # Use the shortest window for responsiveness
        window = min(self._windows[0], len(prices))
        series = prices[-window:]

        mean = float(np.mean(series))
        std = float(np.std(series, ddof=1))

        if std == 0:
            return 0.0, False, self._base_threshold

        z_score = (prices[-1] - mean) / std

        # Dynamic threshold: ATR-adjusted
        # High volatility → require stronger signal (4σ vs 3σ)
        atr_pct = (atr / price * 100) if price > 0 else 0
        if atr_pct > 2.0:
            dynamic_threshold = self._base_threshold + 1.0  # 4σ in high vol
        elif atr_pct > 1.0:
            dynamic_threshold = self._base_threshold + 0.5  # 3.5σ
        else:
            dynamic_threshold = self._base_threshold

        is_anomaly = abs(z_score) >= dynamic_threshold

        return round(float(z_score), 4), is_anomaly, round(dynamic_threshold, 2)

    def check_volume_anomaly(
        self, volumes: np.ndarray, window: int = 20
    ) -> tuple[float, bool]:
        """Separate Z-score for volume spikes."""
        if len(volumes) < window:
            return 0.0, False
        series = volumes[-window:]
        mean = float(np.mean(series))
        std = float(np.std(series, ddof=1))
        if std == 0 or mean == 0:
            return 0.0, False
        z = (volumes[-1] - mean) / std
        return round(float(z), 4), z > 2.5


adaptive_zscore = AdaptiveZScore()
