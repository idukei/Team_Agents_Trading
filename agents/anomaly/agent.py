from __future__ import annotations

import time

import numpy as np
import structlog

from agents.anomaly.ensemble import classify_anomaly_type, detect_anomaly, severity_from_confidence
from agents.anomaly.llm_validator import validate_anomaly_llm
from agents.market_data.stream import buffer_registry
from core.models.signals import AnomalyAlert, AnomalyClassification
from graph.state import TradingState

log = structlog.get_logger(__name__)


async def anomaly_node(state: TradingState) -> dict:
    """
    LangGraph node: runs 3-model ensemble anomaly detection.
    Activates in two modes:
    - Reactive: triggered by EventTrigger (fast path)
    - Continuous: background scan every 5s (handled by background task)
    """
    t0 = time.monotonic()
    trigger = state.get("event_trigger")

    # Determine which asset to scan
    assets = []
    if trigger and trigger.affected_assets_guess:
        assets = trigger.affected_assets_guess[:2]   # check top-2 assets
    else:
        # No specific trigger: scan all monitored assets
        from core.config import settings
        assets = settings.monitored_equities[:3]

    best_alert: AnomalyAlert | None = None
    best_confidence = 0.0

    for asset in assets:
        buf = buffer_registry.get(asset)
        if buf is None or len(buf) < 30:
            continue

        prices = buf.prices
        volumes = buf.volumes
        latest = buf.latest
        if latest is None:
            continue

        atr = float(np.std(np.diff(prices[-50:]))) if len(prices) >= 50 else 0.01

        is_anomaly, confidence, details = detect_anomaly(
            asset=asset,
            prices=prices,
            volumes=volumes,
            spread_bps=latest.spread_bps,
            obi=latest.obi,
            atr=atr,
        )

        if not is_anomaly:
            continue

        # Classify anomaly type
        prices_valid = prices[-2:] if len(prices) >= 2 else prices
        price_return = (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 else 0.0
        vol_ma = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(volumes[-1])
        vol_ratio = volumes[-1] / vol_ma if vol_ma > 0 else 1.0
        z_score = details.get("z_score_value", 0.0)

        anomaly_type = classify_anomaly_type(price_return, vol_ratio, latest.obi, z_score)
        severity = severity_from_confidence(confidence, details.get("votes", 0))

        # LLM validation (fast llama-8b)
        recent_event = trigger.raw_content[:100] if trigger else "No event context"
        try:
            classification, reasoning = await validate_anomaly_llm(
                asset=asset,
                anomaly_type=anomaly_type,
                z_score=z_score,
                volume_ratio=vol_ratio,
                obi=latest.obi,
                recent_event=recent_event,
            )
        except Exception as e:
            log.warning("anomaly_llm_validation_failed", error=str(e))
            classification = AnomalyClassification.UNKNOWN
            reasoning = ""

        # Filter out data glitches
        if classification == AnomalyClassification.DATA_GLITCH:
            log.debug("anomaly_classified_as_glitch", asset=asset)
            continue

        alert = AnomalyAlert(
            anomaly_type=anomaly_type,
            asset=asset,
            severity=severity,
            ml_confidence=confidence,
            llm_classification=classification,
            expected_move={
                "direction": "UP" if price_return > 0 else "DOWN",
                "magnitude_pct": round(abs(price_return) * 100, 3),
                "window_seconds": 60,
            },
            false_positive_risk="LOW" if details.get("votes", 0) == 3 else "MEDIUM",
            models_agreed=details.get("votes", 0),
            z_score=z_score,
            isolation_score=details.get("isolation_forest", {}).get("confidence"),
            lstm_reconstruction_error=details.get("lstm", {}).get("error"),
        )

        if confidence > best_confidence:
            best_confidence = confidence
            best_alert = alert

    latency_ms = int((time.monotonic() - t0) * 1000)

    if best_alert:
        log.info(
            "anomaly_detected",
            asset=best_alert.asset,
            type=best_alert.anomaly_type,
            severity=best_alert.severity,
            confidence=best_alert.ml_confidence,
            classification=best_alert.llm_classification,
            latency_ms=latency_ms,
        )

    metadata = state.get("session_metadata")
    if metadata:
        metadata.record_agent_latency("anomaly", latency_ms)

    return {"anomaly_alert": best_alert}
