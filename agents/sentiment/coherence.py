from __future__ import annotations

import time

import structlog

from core.models.signals import SentimentSignal
from core.models.trade import Direction

log = structlog.get_logger(__name__)

# In-memory recent signal cache: asset → (timestamp, direction, confidence)
_recent_signals: dict[str, tuple[float, Direction, float]] = {}
_WINDOW_S = 1800  # 30-minute conflict window


def validate_coherence(signal: SentimentSignal) -> SentimentSignal:
    """
    Check signal coherence against recent history.
    - Same asset + same direction in last 30min → increase confidence
    - Same asset + opposite direction → mark as CONFLICTED
    Returns (potentially modified) signal.
    """
    now = time.monotonic()
    asset = signal.primary_asset
    recent = _recent_signals.get(asset)

    if recent:
        prev_time, prev_direction, prev_confidence = recent
        age_s = now - prev_time

        if age_s < _WINDOW_S:
            if prev_direction == signal.direction:
                # Same direction reinforces confidence
                boosted = min(1.0, signal.confidence * 1.1)
                log.debug("coherence_reinforced", asset=asset, old_conf=signal.confidence, new_conf=boosted)
                signal = signal.model_copy(update={"confidence": round(boosted, 3)})
            else:
                # Opposing direction — mark conflicted
                log.info(
                    "coherence_conflict",
                    asset=asset,
                    current=signal.direction,
                    previous=prev_direction,
                    age_s=int(age_s),
                )
                signal = signal.model_copy(update={"conflicted": True, "confidence": signal.confidence * 0.7})

    # Update cache
    _recent_signals[asset] = (now, signal.direction, signal.confidence)
    return signal


def clear_signal_cache(asset: str | None = None) -> None:
    if asset:
        _recent_signals.pop(asset, None)
    else:
        _recent_signals.clear()
