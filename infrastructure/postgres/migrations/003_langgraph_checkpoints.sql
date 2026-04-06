-- Migration 003: LangGraph PostgresSaver checkpoint tables
-- These tables are created automatically by langgraph-checkpoint-postgres
-- but we define them here for documentation and explicit control.
-- If LangGraph creates them first, these will be no-ops.

CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id   TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type        TEXT,
    checkpoint   JSONB NOT NULL,
    metadata     JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id   TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel     TEXT NOT NULL,
    version     TEXT NOT NULL,
    type        TEXT NOT NULL,
    blob        BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id   TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id     TEXT NOT NULL,
    idx         INTEGER NOT NULL,
    channel     TEXT NOT NULL,
    type        TEXT,
    blob        BYTEA NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

-- Index for fast thread lookups
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread ON checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_thread ON checkpoint_writes(thread_id, checkpoint_id);
