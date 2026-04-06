from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog

log = structlog.get_logger(__name__)

MODEL_PATH = Path("models/lstm_autoencoder.pt")
THRESHOLD_PATH = Path("models/lstm_threshold.npy")

SEQ_LEN = 30          # sequence length
INPUT_DIM = 4         # features: price_return, volume_ratio, spread_bps, obi


class LSTMAutoencoder:
    """
    LSTM Autoencoder for temporal anomaly detection.
    Detects anomalies based on reconstruction error exceeding learned threshold.
    Especially powerful for detecting silent institutional accumulation.
    """

    def __init__(self) -> None:
        self._model = None
        self._threshold: float = 0.05
        self._loaded = False
        self._load()

    def _load(self) -> None:
        if not MODEL_PATH.exists():
            log.debug("lstm_model_not_found", path=str(MODEL_PATH))
            return
        try:
            import torch
            self._model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            self._model.eval()
            if THRESHOLD_PATH.exists():
                self._threshold = float(np.load(THRESHOLD_PATH))
            self._loaded = True
            log.info("lstm_autoencoder_loaded", threshold=self._threshold)
        except Exception as e:
            log.warning("lstm_load_failed", error=str(e))

    def predict(self, sequence: np.ndarray) -> tuple[bool, float]:
        """
        sequence shape: (SEQ_LEN, INPUT_DIM)
        Returns (is_anomaly, reconstruction_error).
        """
        if not self._loaded or self._model is None:
            return False, 0.0

        if sequence.shape != (SEQ_LEN, INPUT_DIM):
            return False, 0.0

        try:
            import torch
            with torch.no_grad():
                x = torch.FloatTensor(sequence).unsqueeze(0)  # (1, SEQ_LEN, INPUT_DIM)
                reconstructed = self._model(x)
                error = float(torch.mean((x - reconstructed) ** 2).item())
            return error > self._threshold, round(error, 8)
        except Exception as e:
            log.warning("lstm_predict_error", error=str(e))
            return False, 0.0

    def is_loaded(self) -> bool:
        return self._loaded


# Singleton
lstm_detector = LSTMAutoencoder()


# ── PyTorch model definition (used by train_anomaly_models.py) ─────────────────

def build_lstm_autoencoder_model(input_dim: int = INPUT_DIM, hidden_dim: int = 64):
    """Build LSTM Autoencoder model. Used during training."""
    try:
        import torch
        import torch.nn as nn

        class LSTMAEModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

            def forward(self, x):
                # x: (batch, seq_len, input_dim)
                _, (h, c) = self.encoder(x)
                # Repeat hidden state for each decoder step
                h_repeat = h.squeeze(0).unsqueeze(1).repeat(1, x.shape[1], 1)
                out, _ = self.decoder(h_repeat)
                return out

        return LSTMAEModel()
    except ImportError:
        raise ImportError("PyTorch required for LSTM autoencoder training")
