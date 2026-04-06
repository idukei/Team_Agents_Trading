from __future__ import annotations

from core.config import settings
from core.models.events import EventSource, EventTrigger, EventUrgency, SOURCE_WEIGHTS

# Keywords that indicate high market impact — grouped by category
HIGH_IMPACT_KEYWORDS: dict[str, list[str]] = {
    "forex_indices": [
        "tariff", "trade war", "sanction", "deal", "agreement", "nato",
        "rate hike", "rate cut", "inflation", "recession", "gdp", "default",
        "debt ceiling", "fiscal", "stimulus", "bailout",
    ],
    "oil_commodities": [
        "opec", "oil", "energy", "pipeline", "embargo", "strategic reserve",
        "crude", "natural gas", "lng", "refinery",
    ],
    "crypto": [
        "bitcoin", "btc", "crypto", "sec", "ban", "regulate", "digital currency",
        "cbdc", "blockchain", "ethereum", "stablecoin",
    ],
    "equities": [
        "earnings", "acquisition", "merger", "bankruptcy", "investigation",
        "fine", "antitrust", "ceo fired", "ceo resign", "fraud", "hack",
        "data breach", "recall", "layoff", "strike",
    ],
    "macro_extreme": [
        "war", "ceasefire", "invasion", "nuclear", "terror", "pandemic",
        "lockdown", "election", "coup", "collapse",
    ],
}

# Flatten all keywords for fast lookup
_ALL_KEYWORDS = {
    kw: category
    for category, keywords in HIGH_IMPACT_KEYWORDS.items()
    for kw in keywords
}


def count_keyword_hits(text: str) -> tuple[int, list[str]]:
    """Count how many high-impact keywords appear in text (case-insensitive)."""
    lower = text.lower()
    hits = [kw for kw in _ALL_KEYWORDS if kw in lower]
    return len(hits), hits


def compute_eps(
    source: EventSource,
    text: str,
    is_novel: bool,
    vix_norm: float,
) -> tuple[float, EventUrgency, list[str]]:
    """
    Compute Event Priority Score (EPS) using the formula:

    EPS = (source_weight × 0.35) + (keyword_intensity × 0.30)
        + (novelty × 0.20) + (volatility × 0.15)

    Returns (eps_score, urgency, matched_keywords).
    """
    source_weight = SOURCE_WEIGHTS.get(source, 0.4)
    keyword_hits, matched = count_keyword_hits(text)

    # Keyword intensity: 1 hit = 0.4, 2+ = 0.7, strong phrase combo = 1.0
    if keyword_hits == 0:
        keyword_intensity = 0.0
    elif keyword_hits == 1:
        keyword_intensity = 0.4
    elif keyword_hits == 2:
        keyword_intensity = 0.7
    else:
        keyword_intensity = 1.0

    novelty = 1.0 if is_novel else 0.2
    vix_component = max(0.0, min(1.0, vix_norm))

    raw = (
        source_weight * 0.35
        + keyword_intensity * 0.30
        + novelty * 0.20
        + vix_component * 0.15
    )
    eps = round(raw * 100, 2)

    if eps >= 90:
        urgency = EventUrgency.IMMEDIATE
    elif eps >= settings.eps_full_pipeline:
        urgency = EventUrgency.HIGH
    elif eps >= settings.eps_light_pipeline:
        urgency = EventUrgency.MEDIUM
    else:
        urgency = EventUrgency.LOW

    return eps, urgency, matched


def should_activate_pipeline(trigger: EventTrigger) -> str:
    """Return 'FULL', 'LIGHT', or 'SKIP' based on EPS score."""
    if trigger.eps_score >= settings.eps_full_pipeline:
        return "FULL"
    if trigger.eps_score >= settings.eps_light_pipeline:
        return "LIGHT"
    return "SKIP"
