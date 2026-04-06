"""
Train IsolationForest and LSTM Autoencoder on historical market data.
Run: uv run python scripts/train_anomaly_models.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import structlog

from core.config import settings
from core.db.postgres import close_pool
from core.db.timescale import fetch_ohlcv
from core.logging import configure_logging

log = structlog.get_logger(__name__)

Path("models").mkdir(exist_ok=True)


def build_features(df: pd.DataFrame) -> np.ndarray:
    """Extract (price_return, volume_ratio, spread_bps_approx, obi_approx) features."""
    df = df.copy().dropna()
    price_return = df["close"].pct_change().fillna(0).values
    volume_ma = df["volume"].rolling(20, min_periods=1).mean().values
    volume_ratio = (df["volume"].values / (volume_ma + 1e-10))
    spread_approx = ((df["high"] - df["low"]) / df["close"] * 10000).fillna(0).values
    obi_approx = ((df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)).fillna(0).values

    X = np.column_stack([price_return, volume_ratio, spread_approx, obi_approx])
    return X.astype(float)


async def train_isolation_forest() -> None:
    log.info("training_isolation_forest")
    all_features = []

    for asset in settings.monitored_equities[:5]:      # use first 5 equities
        df = await fetch_ohlcv(asset, timeframe="1d", limit=500)
        if df.empty:
            log.warning("no_data_for_asset", asset=asset)
            continue
        X = build_features(df)
        all_features.append(X)
        log.debug("asset_features", asset=asset, rows=len(X))

    if not all_features:
        log.error("no_training_data")
        return

    X_combined = np.vstack(all_features)
    X_combined = X_combined[~np.any(np.isnan(X_combined), axis=1)]

    from agents.anomaly.isolation_forest import IsolationForestDetector
    detector = IsolationForestDetector.__new__(IsolationForestDetector)
    detector._model = None
    detector.fit(X_combined, contamination=0.03)
    log.info("isolation_forest_trained", samples=len(X_combined))


async def train_lstm_autoencoder() -> None:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        log.warning("pytorch_not_available_skipping_lstm")
        return

    from agents.anomaly.lstm_autoencoder import SEQ_LEN, INPUT_DIM, build_lstm_autoencoder_model

    log.info("training_lstm_autoencoder")
    all_features = []

    for asset in settings.monitored_equities[:5]:
        df = await fetch_ohlcv(asset, timeframe="1h", limit=2000)
        if df.empty:
            continue
        X = build_features(df)
        all_features.append(X)

    if not all_features:
        log.error("no_training_data_lstm")
        return

    X = np.vstack(all_features)
    X = X[~np.any(np.isnan(X), axis=1)]

    # Normalize each feature to [0, 1]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    np.save("models/lstm_scaler_min.npy", scaler.data_min_)
    np.save("models/lstm_scaler_scale.npy", scaler.scale_)

    # Build sequences
    sequences = []
    for i in range(len(X_scaled) - SEQ_LEN):
        sequences.append(X_scaled[i:i + SEQ_LEN])
    sequences = np.array(sequences, dtype=np.float32)

    dataset = TensorDataset(torch.tensor(sequences))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = build_lstm_autoencoder_model(INPUT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(20):
        total_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            log.info("lstm_epoch", epoch=epoch, loss=total_loss / len(loader))

    # Compute threshold as 95th percentile of reconstruction errors on training data
    model.eval()
    errors = []
    with torch.no_grad():
        for (batch,) in loader:
            output = model(batch)
            batch_errors = torch.mean((batch - output) ** 2, dim=(1, 2))
            errors.extend(batch_errors.numpy().tolist())
    threshold = float(np.percentile(errors, 95))

    torch.save(model, "models/lstm_autoencoder.pt")
    np.save("models/lstm_threshold.npy", threshold)
    log.info("lstm_autoencoder_trained", threshold=threshold, sequences=len(sequences))


async def main() -> None:
    configure_logging()
    from core.db.postgres import get_pool
    await get_pool()

    await asyncio.gather(
        train_isolation_forest(),
        train_lstm_autoencoder(),
    )
    await close_pool()
    log.info("anomaly_models_training_complete")


if __name__ == "__main__":
    asyncio.run(main())
