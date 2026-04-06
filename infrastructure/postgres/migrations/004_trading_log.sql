-- Migration 004: Trading session log and performance metrics

-- Daily performance summary
CREATE TABLE IF NOT EXISTS daily_performance (
    date            DATE PRIMARY KEY,
    starting_capital NUMERIC(18,2) NOT NULL,
    ending_capital  NUMERIC(18,2) NOT NULL,
    pnl_usd         NUMERIC(18,4) NOT NULL,
    pnl_pct         NUMERIC(8,6) NOT NULL,
    total_trades    INTEGER NOT NULL DEFAULT 0,
    winning_trades  INTEGER NOT NULL DEFAULT 0,
    losing_trades   INTEGER NOT NULL DEFAULT 0,
    max_drawdown_pct NUMERIC(8,6),
    sharpe_daily    NUMERIC(8,6),
    fireworks_cost_usd NUMERIC(10,6),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Agent performance log (latency + accuracy tracking)
CREATE TABLE IF NOT EXISTS agent_performance (
    time            TIMESTAMPTZ NOT NULL,
    agent_name      TEXT NOT NULL,
    event_id        TEXT,
    latency_ms      INTEGER NOT NULL,
    success         BOOLEAN NOT NULL,
    error_msg       TEXT,
    tokens_used     INTEGER,
    model_used      TEXT
);

SELECT create_hypertable('agent_performance', by_range('time'), if_not_exists => TRUE);

-- CTS weight history (updated by post_trade_analyst daily)
CREATE TABLE IF NOT EXISTS cts_weights (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    effective_date  DATE NOT NULL,
    sentiment_weight NUMERIC(4,3) NOT NULL DEFAULT 0.32,
    anomaly_weight  NUMERIC(4,3) NOT NULL DEFAULT 0.28,
    market_weight   NUMERIC(4,3) NOT NULL DEFAULT 0.20,
    precedent_weight NUMERIC(4,3) NOT NULL DEFAULT 0.12,
    obi_weight      NUMERIC(4,3) NOT NULL DEFAULT 0.08,
    reason          TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Insert default weights
INSERT INTO cts_weights (effective_date, reason)
VALUES (CURRENT_DATE, 'Initial default weights')
ON CONFLICT DO NOTHING;

-- Fireworks.ai budget tracking
CREATE TABLE IF NOT EXISTS llm_usage (
    time            TIMESTAMPTZ NOT NULL,
    agent_name      TEXT NOT NULL,
    model           TEXT NOT NULL,
    prompt_tokens   INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens    INTEGER NOT NULL,
    estimated_cost_usd NUMERIC(10,8)
);

SELECT create_hypertable('llm_usage', by_range('time'), if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_agent_perf_name ON agent_performance(agent_name, time DESC);
CREATE INDEX IF NOT EXISTS idx_llm_usage_agent ON llm_usage(agent_name, time DESC);
