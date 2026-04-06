from __future__ import annotations

import structlog
from sentence_transformers import SentenceTransformer

from core.db.qdrant import search_precedents, upsert_precedent

log = structlog.get_logger(__name__)

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        log.info("embedding_model_loaded")
    return _model


def embed(text: str) -> list[float]:
    """Generate 384-dim embedding for a text string."""
    model = get_embedding_model()
    return model.encode(text, normalize_embeddings=True).tolist()


async def query_precedents(event_text: str, limit: int = 5) -> list[dict]:
    """
    Search Qdrant for the most similar historical trading precedents.
    Returns list of dicts with: score, date, event_desc, asset, direction,
    return_15min, return_60min, outcome_summary.
    """
    try:
        embedding = embed(event_text)
        results = await search_precedents(embedding, limit=limit, score_threshold=0.55)
        log.debug("rag_query_results", count=len(results), query_preview=event_text[:80])
        return results
    except Exception as e:
        log.warning("rag_query_failed", error=str(e))
        return []


async def store_precedent(
    event_id: str,
    event_text: str,
    asset: str,
    direction: str,
    return_15min: float,
    return_60min: float,
    outcome_summary: str,
) -> None:
    """Store a completed trade outcome as a precedent for future RAG queries."""
    embedding = embed(event_text)
    payload = {
        "event_id": event_id,
        "event_text": event_text,
        "asset": asset,
        "direction": direction,
        "return_15min": return_15min,
        "return_60min": return_60min,
        "outcome_summary": outcome_summary,
    }
    # Use a stable ID based on event_id to allow updates
    point_id = event_id[:32].replace("-", "")
    await upsert_precedent(point_id, embedding, payload)
    log.info("precedent_stored", event_id=event_id, asset=asset, return_15min=return_15min)


def format_precedents_for_prompt(precedents: list[dict]) -> str:
    """Format precedents list as readable context for LLM prompt."""
    if not precedents:
        return "No se encontraron precedentes históricos similares."
    lines = []
    for i, p in enumerate(precedents, 1):
        score = p.get("score", 0)
        lines.append(
            f"{i}. [Similitud: {score:.2f}] {p.get('event_text', '')[:120]}\n"
            f"   → {p.get('asset')} | {p.get('direction')} | "
            f"15min: {p.get('return_15min', 0):+.2f}% | 60min: {p.get('return_60min', 0):+.2f}%\n"
            f"   {p.get('outcome_summary', '')}"
        )
    return "\n".join(lines)
