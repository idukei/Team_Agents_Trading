from __future__ import annotations

from typing import AsyncIterator

import redis.asyncio as aioredis
import structlog

from core.config import settings

log = structlog.get_logger(__name__)

_client: aioredis.Redis | None = None


def get_client() -> aioredis.Redis:
    global _client
    if _client is None:
        _client = aioredis.from_url(
            settings.redis_url,
            decode_responses=False,   # raw bytes for stream entries
            socket_keepalive=True,
            retry_on_timeout=True,
        )
    return _client


async def close_client() -> None:
    global _client
    if _client:
        await _client.aclose()
        _client = None


async def ping_redis() -> bool:
    try:
        return await get_client().ping()
    except Exception as e:
        log.error("redis_ping_failed", error=str(e))
        return False


async def publish_event(stream: str, data: dict[str, str]) -> str:
    """Publish a message to a Redis Stream. Returns entry ID."""
    client = get_client()
    entry_id = await client.xadd(
        stream,
        data,
        maxlen=settings.redis_stream_maxlen,
        approximate=True,
    )
    return entry_id.decode() if isinstance(entry_id, bytes) else entry_id


async def consume_events(
    stream: str,
    last_id: str = "$",
    block_ms: int = 5000,
    count: int = 10,
) -> AsyncIterator[tuple[str, dict[bytes, bytes]]]:
    """Async generator that yields (entry_id, fields) from a Redis Stream.

    Blocks up to block_ms waiting for new entries.
    Use last_id='$' to only receive new messages (not historical).
    Use last_id='0' to replay from beginning.
    """
    client = get_client()
    current_id = last_id
    while True:
        try:
            results = await client.xread(
                {stream: current_id},
                count=count,
                block=block_ms,
            )
            if results:
                for _stream, entries in results:
                    for entry_id, fields in entries:
                        eid = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                        current_id = eid
                        yield eid, fields
        except Exception as e:
            log.error("redis_stream_read_error", stream=stream, error=str(e))
            await aioredis.asyncio.sleep(1)


async def set_cache(key: str, value: str, ttl_s: int = 3600) -> None:
    await get_client().setex(key, ttl_s, value)


async def get_cache(key: str) -> str | None:
    val = await get_client().get(key)
    return val.decode() if val else None


async def exists(key: str) -> bool:
    return bool(await get_client().exists(key))
