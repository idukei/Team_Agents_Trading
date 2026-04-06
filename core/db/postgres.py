from __future__ import annotations

import asyncpg
import structlog
from asyncpg import Pool

from core.config import settings

log = structlog.get_logger(__name__)

_pool: Pool | None = None


async def get_pool() -> Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=settings.postgres_dsn.get_secret_value(),
            min_size=settings.postgres_pool_min,
            max_size=settings.postgres_pool_max,
            command_timeout=30,
            statement_cache_size=100,
        )
        log.info("postgres_pool_created", min=settings.postgres_pool_min, max=settings.postgres_pool_max)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        log.info("postgres_pool_closed")


async def ping_postgres() -> bool:
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        log.error("postgres_ping_failed", error=str(e))
        return False
