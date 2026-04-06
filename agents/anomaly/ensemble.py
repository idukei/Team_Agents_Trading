from __future__ import annotations

import numpy as np
import structlog

from agents.anomaly.isolation_forest import isolation_forest_detector
from agents.anomaly.lstm_autoencoder import lstm_detector, SEQ_LEN, INPUT_DIM
from agents.anomaly.zscore import adaptive_zscore
from core.models.signals import AnomalyAlert, AnomalyType

log = structlog.get_logger(__name__)

VOTES_REQUIRED = 2      # 2 out of 3 models must agree


def _build_features(
    price_return: float,
    volume_ratio: float,
    spread_bps: float,
    obi: float,
) -> np.ndarray:
    return np.array([price_return, volume_ratio, spread_bps, obi], dtype=float)


def detect_anomaly(
    asset: str,
    prices: np.ndarray,
    volumes: np.ndarray,
    spread_bps: float,
    obi: float,
    atr: float,
) -> tuple[bool, float, dict]:
    """
    Runs 3-model ensemble. Returns (is_anomaly, confidence, details).
    Requires VOTES_REQUIRED (2/3) models to agree.
    """
    if len(prices) < 2 or len(volumes) < 2:
        return False, 0.0, {}

    current_price = prices[-1]
    prev_price = prices[-2]
    price_return = (current_price - prev_price) / prev_price if prev_price > 0 else 0.0
    volume_ma = float(np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes))
    volume_ratio = volumes[-1] / volume_ma if volume_ma > 0 else 1.0

    votes = 0
    confidences = []
    details = {}

    # Model 1: Isolation Forest
    if isolation_forest_detector.is_loaded():
        features = _build_features(price_return, volume_ratio, spread_bps, obi)
        if_anomaly, if_conf = isolation_forest_detector.predict(features)
        if if_anomaly:
            votes += 1
        confidences.append(if_conf)
        details["isolation_forest"] = {"anomaly": if_anomaly, "confidence": if_conf}
    else:
        details["isolation_forest"] = {"anomaly": False, "confidence": 0.0, "not_loaded": True}

    # Model 2: Adaptive Z-score
    z_score, z_anomaly, z_threshold = adaptive_zscore.compute(prices, atr, current_price)
    vol_z, vol_z_anomaly = adaptive_zscore.check_volume_anomaly(volumes)
    z_combined = z_anomaly or vol_z_anomaly
    z_conf = min(1.0, abs(z_score) / (z_threshold + 0.001))
    if z_combined:
        votes += 1
    confidences.append(z_conf)
    details["z_score"] = {"anomaly": z_combined, "z_score": z_score, "threshold": z_threshold}

    # Model 3: LSTM Autoencoder
    if lstm_detector.is_loaded() and len(prices) >= SEQ_LEN:
        seq_len_prices = prices[-SEQ_LEN:]
        seq_len_vols = volumes[-SEQ_LEN:] if len(volumes) >= SEQ_LEN else np.ones(SEQ_LEN)

        # Normalize features for LSTM input
        returns = np.diff(seq_len_prices, prepend=seq_len_prices[0]) / (seq_len_prices + 1e-10)
        vol_ratios = seq_len_vols / (np.mean(seq_len_vols) + 1e-10)
        spreads = np.full(SEQ_LEN, spread_bps / 100.0)
        obis = np.full(SEQ_LEN, obi)

        sequence = np.stack([returns, vol_ratios, spreads, obis], axis=1)
        lstm_anomaly, lstm_error = lstm_detector.predict(sequence)
        lstm_conf = min(1.0, lstm_error / 0.1)
        if lstm_anomaly:
            votes += 1
        confidences.append(lstm_conf)
        details["lstm"] = {"anomaly": lstm_anomaly, "error": lstm_error}
    else:
        details["lstm"] = {"anomaly": False, "confidence": 0.0, "not_loaded": True}

    is_anomaly = votes >= VOTES_REQUIRED
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0
    details["votes"] = votes
    details["z_score_value"] = z_score

    return is_anomaly, round(avg_confidence, 4), details


def classify_anomaly_type(
    price_return: float,
    volume_ratio: float,
    obi: float,
    z_score: float,
) -> AnomalyType:
    """Heuristic classification before LLM validation."""
    abs_return = abs(price_return)

    if volume_ratio > 2.0 and abs_return < 0.003:
        return AnomalyType.VOLUME_ACCUMULATION_SILENT
    elif abs_return > 0.01:
        return AnomalyType.PRICE_SPIKE_SUDDEN
    elif abs(obi) > 0.7:
        return AnomalyType.OBI_IMBALANCE_EXTREME
    elif abs(z_score) > 3.5:
        return AnomalyType.MULTIVARIATE_OUTLIER
    return AnomalyType.MULTIVARIATE_OUTLIER


def severity_from_confidence(confidence: float, votes: int) -> str:
    if confidence > 0.85 or votes == 3:
        return "CRITICAL"
    elif confidence > 0.70 or votes == 2:
        return "HIGH"
    elif confidence > 0.50:
        return "MEDIUM"
    return "LOW"
