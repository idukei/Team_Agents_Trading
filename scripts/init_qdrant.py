"""
Initialize Qdrant collections and optionally seed with historical precedents.
Run: uv run python scripts/init_qdrant.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from core.db.qdrant import close_client, ensure_collections, ping_qdrant
from core.logging import configure_logging

log = structlog.get_logger(__name__)

# Sample historical precedents to seed RAG system
SAMPLE_PRECEDENTS = [
    {
        "event_id": "sample_trump_tariff_2025_01",
        "event_text": "Trump announces 35% tariffs on Chinese semiconductors effective immediately",
        "asset": "NVDA",
        "direction": "SHORT",
        "return_15min": -1.8,
        "return_60min": -3.2,
        "outcome_summary": "NVDA dropped 3.2% in 60min following semiconductor tariff announcement",
    },
    {
        "event_id": "sample_fomc_2024_11",
        "event_text": "FOMC holds rates steady, signals no cuts in 2025, hawkish tone",
        "asset": "TLT",
        "direction": "SHORT",
        "return_15min": -0.8,
        "return_60min": -1.5,
        "outcome_summary": "TLT dropped 1.5% on hawkish Fed hold decision",
    },
    {
        "event_id": "sample_nfp_2024_10",
        "event_text": "NFP beats expectations: 350K jobs vs 200K consensus, unemployment 3.6%",
        "asset": "SPY",
        "direction": "LONG",
        "return_15min": 0.6,
        "return_60min": 0.9,
        "outcome_summary": "SPY rallied 0.9% on strong jobs report above consensus",
    },
    {
        "event_id": "sample_trump_crypto_2025_03",
        "event_text": "Trump says he wants to make America the crypto capital of the world, will sign pro-crypto executive order",
        "asset": "BTC/USD",
        "direction": "LONG",
        "return_15min": 3.5,
        "return_60min": 5.1,
        "outcome_summary": "BTC surged 5.1% in 60min on Trump pro-crypto statement",
    },
    {
        "event_id": "sample_cpi_hot_2024_09",
        "event_text": "CPI comes in hot: 3.8% vs 3.4% consensus, core CPI 3.6% vs 3.2% expected",
        "asset": "GLD",
        "direction": "LONG",
        "return_15min": 0.4,
        "return_60min": 0.7,
        "outcome_summary": "Gold rose 0.7% on hot CPI as inflation hedge demand increased",
    },
]


async def main() -> None:
    configure_logging()

    if not await ping_qdrant():
        log.error("qdrant_not_available")
        return

    await ensure_collections()

    # Seed sample precedents
    from agents.sentiment.rag import store_precedent
    for p in SAMPLE_PRECEDENTS:
        try:
            await store_precedent(**p)
        except Exception as e:
            log.error("seed_precedent_failed", event_id=p["event_id"], error=str(e))

    log.info("qdrant_initialized", precedents_seeded=len(SAMPLE_PRECEDENTS))
    await close_client()


if __name__ == "__main__":
    asyncio.run(main())
