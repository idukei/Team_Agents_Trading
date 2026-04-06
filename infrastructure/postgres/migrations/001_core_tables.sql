-- Migration 001: Core tables

-- Event triggers log (append-only)
CREATE TABLE IF NOT EXISTS event_triggers (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id    TEXT NOT NULL UNIQUE,
    event_source TEXT NOT NULL,
    eps_score   NUMERIC(5,2) NOT NULL,
    raw_content TEXT NOT NULL,
    affected_assets JSONB NOT NULL DEFAULT '[]',
    expected_direction_hint TEXT,
    urgency     TEXT NOT NULL DEFAULT 'NORMAL',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Sentiment signals
CREATE TABLE IF NOT EXISTS sentiment_signals (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id        TEXT NOT NULL REFERENCES event_triggers(event_id),
    direction       TEXT NOT NULL,
    primary_asset   TEXT NOT NULL,
    secondary_assets JSONB NOT NULL DEFAULT '[]',
    magnitude       NUMERIC(4,3) NOT NULL,
    confidence      NUMERIC(4,3) NOT NULL,
    time_horizon    TEXT NOT NULL,
    llm_reasoning   TEXT,
    fireworks_model TEXT,
    latency_ms      INTEGER,
    conflicted      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Trade proposals
CREATE TABLE IF NOT EXISTS trade_proposals (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trade_id        TEXT NOT NULL UNIQUE,
    event_id        TEXT REFERENCES event_triggers(event_id),
    asset           TEXT NOT NULL,
    direction       TEXT NOT NULL,
    entry_type      TEXT NOT NULL,
    entry_price     NUMERIC(18,6) NOT NULL,
    position_size_usd NUMERIC(18,2) NOT NULL,
    stop_loss       NUMERIC(18,6) NOT NULL,
    take_profit     JSONB NOT NULL,
    max_hold_seconds INTEGER NOT NULL,
    cts_score       NUMERIC(4,3) NOT NULL,
    strategy_type   TEXT NOT NULL,
    reasoning_summary TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Risk decisions
CREATE TABLE IF NOT EXISTS risk_decisions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trade_id        TEXT NOT NULL REFERENCES trade_proposals(trade_id),
    status          TEXT NOT NULL,
    approved_size_usd NUMERIC(18,2),
    rejection_reason TEXT,
    adjustments     JSONB NOT NULL DEFAULT '{}',
    var_95          NUMERIC(10,6),
    portfolio_correlation NUMERIC(4,3),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Execution results
CREATE TABLE IF NOT EXISTS execution_results (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trade_id        TEXT NOT NULL REFERENCES trade_proposals(trade_id),
    order_id        TEXT NOT NULL,
    filled_price    NUMERIC(18,6) NOT NULL,
    filled_qty      NUMERIC(18,6) NOT NULL,
    fill_timestamp_ms BIGINT NOT NULL,
    slippage_bps    NUMERIC(8,4),
    tp1_hit         BOOLEAN NOT NULL DEFAULT FALSE,
    tp2_hit         BOOLEAN NOT NULL DEFAULT FALSE,
    sl_hit          BOOLEAN NOT NULL DEFAULT FALSE,
    timeout_exit    BOOLEAN NOT NULL DEFAULT FALSE,
    pnl_usd         NUMERIC(18,4),
    exit_price      NUMERIC(18,6),
    exit_timestamp_ms BIGINT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_event_triggers_source ON event_triggers(event_source);
CREATE INDEX IF NOT EXISTS idx_event_triggers_created ON event_triggers(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trade_proposals_asset ON trade_proposals(asset);
CREATE INDEX IF NOT EXISTS idx_execution_results_trade ON execution_results(trade_id);
