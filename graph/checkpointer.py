from __future__ import annotations

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from core.config import settings

log = structlog.get_logger(__name__)

_checkpointer: AsyncPostgresSaver | None = None


async def get_checkpointer() -> AsyncPostgresSaver:
    global _checkpointer
    if _checkpointer is None:
        dsn = settings.postgres_dsn.get_secret_value()
        _checkpointer = AsyncPostgresSaver.from_conn_string(dsn)
        await _checkpointer.setup()       # creates checkpoint tables if not exist
        log.info("postgres_checkpointer_initialized")
    return _checkpointer


def make_thread_id(event_id: str) -> str:
    """Stable thread_id for a pipeline run. Enables time-travel debugging."""
    return f"trade_{event_id[:16]}"
