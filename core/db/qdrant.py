from __future__ import annotations

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from core.config import settings

log = structlog.get_logger(__name__)

EMBEDDING_DIM = 384          # sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=settings.qdrant_url)
    return _client


async def close_client() -> None:
    global _client
    if _client:
        await _client.close()
        _client = None


async def ensure_collections() -> None:
    """Create Qdrant collections if they don't exist."""
    client = get_client()

    for collection_name in [
        settings.qdrant_collection_precedents,
        settings.qdrant_collection_news,
    ]:
        existing = await client.collection_exists(collection_name)
        if not existing:
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            log.info("qdrant_collection_created", name=collection_name)
        else:
            log.debug("qdrant_collection_exists", name=collection_name)


async def upsert_precedent(
    point_id: str,
    embedding: list[float],
    payload: dict,
) -> None:
    """Store a trading precedent (historical event + outcome) in Qdrant."""
    client = get_client()
    await client.upsert(
        collection_name=settings.qdrant_collection_precedents,
        points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
    )


async def search_precedents(
    query_embedding: list[float],
    limit: int = 5,
    score_threshold: float = 0.60,
) -> list[dict]:
    """Search for similar historical precedents by semantic similarity."""
    client = get_client()
    results = await client.search(
        collection_name=settings.qdrant_collection_precedents,
        query_vector=query_embedding,
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
    )
    return [
        {"score": hit.score, **hit.payload}
        for hit in results
    ]


async def ping_qdrant() -> bool:
    try:
        await get_client().get_collections()
        return True
    except Exception as e:
        log.error("qdrant_ping_failed", error=str(e))
        return False
