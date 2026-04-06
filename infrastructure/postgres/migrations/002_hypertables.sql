-- Migration 002: TimescaleDB hypertables for time-series market data

-- OHLCV (Open, High, Low, Close, Volume)
CREATE TABLE IF NOT EXISTS ohlcv (
    time        TIMESTAMPTZ NOT NULL,
    asset       TEXT NOT NULL,
    timeframe   TEXT NOT NULL,          -- '1m', '5m', '1h', '1d'
    open        NUMERIC(18,6) NOT NULL,
    high        NUMERIC(18,6) NOT NULL,
    low         NUMERIC(18,6) NOT NULL,
    close       NUMERIC(18,6) NOT NULL,
    volume      NUMERIC(20,2) NOT NULL,
    vwap        NUMERIC(18,6),
    trade_count INTEGER,
    PRIMARY KEY (time, asset, timeframe)
);

SELECT create_hypertable('ohlcv', by_range('time'), if_not_exists => TRUE);

-- Add compression policy (compress chunks older than 7 days)
SELECT add_compression_policy('ohlcv', INTERVAL '7 days', if_not_exists => TRUE);

-- Real-time quotes tick data
CREATE TABLE IF NOT EXISTS quotes (
    time        TIMESTAMPTZ NOT NULL,
    asset       TEXT NOT NULL,
    bid_price   NUMERIC(18,6) NOT NULL,
    ask_price   NUMERIC(18,6) NOT NULL,
    bid_size    NUMERIC(20,2) NOT NULL,
    ask_size    NUMERIC(20,2) NOT NULL,
    spread_bps  NUMERIC(8,4)
);

SELECT create_hypertable('quotes', by_range('time'), if_not_exists => TRUE);

-- Anomaly detection log
CREATE TABLE IF NOT EXISTS anomaly_log (
    time            TIMESTAMPTZ NOT NULL,
    asset           TEXT NOT NULL,
    anomaly_type    TEXT NOT NULL,
    severity        TEXT NOT NULL,
    ml_confidence   NUMERIC(4,3) NOT NULL,
    llm_classification TEXT,
    z_score         NUMERIC(8,4),
    isolation_score NUMERIC(8,6),
    lstm_error      NUMERIC(10,8),
    triggered_trade BOOLEAN NOT NULL DEFAULT FALSE
);

SELECT create_hypertable('anomaly_log', by_range('time'), if_not_exists => TRUE);

-- Indicators cache (pre-computed, refreshed by Market Data Agent)
CREATE TABLE IF NOT EXISTS indicator_snapshots (
    time        TIMESTAMPTZ NOT NULL,
    asset       TEXT NOT NULL,
    rsi14       NUMERIC(6,3),
    vwap        NUMERIC(18,6),
    atr14       NUMERIC(18,6),
    obi         NUMERIC(6,4),
    bb_upper    NUMERIC(18,6),
    bb_middle   NUMERIC(18,6),
    bb_lower    NUMERIC(18,6),
    trend       TEXT,
    volatility_regime TEXT,
    PRIMARY KEY (time, asset)
);

SELECT create_hypertable('indicator_snapshots', by_range('time'), if_not_exists => TRUE);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_ohlcv_asset_time ON ohlcv(asset, time DESC);
CREATE INDEX IF NOT EXISTS idx_quotes_asset_time ON quotes(asset, time DESC);
CREATE INDEX IF NOT EXISTS idx_anomaly_asset ON anomaly_log(asset, time DESC);
