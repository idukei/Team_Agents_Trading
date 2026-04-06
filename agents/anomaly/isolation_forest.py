from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import structlog
from sklearn.ensemble import IsolationForest

log = structlog.get_logger(__name__)

MODEL_PATH = Path("models/isolation_forest.pkl")


class IsolationForestDetector:
    """
    Multivariate anomaly detection using IsolationForest.
    Trained on (price_return, volume_ratio, spread_bps, obi) features.
    """

    def __init__(self) -> None:
        self._model: IsolationForest | None = None
        self._load()

    def _load(self) -> None:
        if MODEL_PATH.exists():
            try:
                self._model = joblib.load(MODEL_PATH)
                log.info("isolation_forest_loaded", path=str(MODEL_PATH))
            except Exception as e:
                log.warning("isolation_forest_load_failed", error=str(e))

    def fit(self, X: np.ndarray, contamination: float = 0.05) -> None:
        """Train on normal market data. X shape: (n_samples, 4)."""
        self._model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X)
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(self._model, MODEL_PATH)
        log.info("isolation_forest_trained", samples=len(X))

    def predict(self, features: np.ndarray) -> tuple[bool, float]:
        """
        Returns (is_anomaly, anomaly_score).
        anomaly_score: negative = anomaly, 0 = boundary, positive = normal
        score is normalized to [0, 1] for confidence.
        """
        if self._model is None:
            return False, 0.0

        features_2d = features.reshape(1, -1)
        raw_score = self._model.decision_function(features_2d)[0]
        prediction = self._model.predict(features_2d)[0]   # -1 = anomaly, 1 = normal

        # Normalize to confidence [0, 1]: more negative = more anomalous
        confidence = float(max(0.0, min(1.0, -raw_score / 0.5 + 0.5)))
        is_anomaly = prediction == -1

        return is_anomaly, confidence

    def is_loaded(self) -> bool:
        return self._model is not None


# Singleton
isolation_forest_detector = IsolationForestDetector()
