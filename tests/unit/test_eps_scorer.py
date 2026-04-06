from agents.monitor.eps_scorer import compute_eps, count_keyword_hits, should_activate_pipeline
from core.models.events import EventSource, EventTrigger, EventUrgency


def test_keyword_hit_tariff():
    hits, matched = count_keyword_hits("We are imposing tariffs on semiconductors")
    assert hits >= 2
    assert "tariff" in matched or "semiconductor" in matched


def test_keyword_miss():
    hits, _ = count_keyword_hits("Good morning, have a nice day")
    assert hits == 0


def test_eps_trump_tariff():
    eps, urgency, matched = compute_eps(
        source=EventSource.TRUMP_TRUTH_SOCIAL,
        text="Imposing 35% tariffs on Chinese semiconductors",
        is_novel=True,
        vix_norm=0.7,
    )
    assert eps >= 75
    assert urgency in (EventUrgency.IMMEDIATE, EventUrgency.HIGH)


def test_eps_low_source():
    eps, urgency, _ = compute_eps(
        source=EventSource.NEWS,
        text="Local weather forecast for New York",
        is_novel=True,
        vix_norm=0.2,
    )
    assert eps < 50
    assert urgency == EventUrgency.LOW


def test_eps_non_novel_reduces_score():
    eps_novel, _, _ = compute_eps(EventSource.FED_ANNOUNCEMENT, "rate hike", True, 0.5)
    eps_old, _, _ = compute_eps(EventSource.FED_ANNOUNCEMENT, "rate hike", False, 0.5)
    assert eps_novel > eps_old


def test_pipeline_routing():
    trigger_high = EventTrigger(
        event_source=EventSource.TRUMP_TRUTH_SOCIAL,
        eps_score=88.0,
        raw_content="tariff announcement",
    )
    assert should_activate_pipeline(trigger_high) == "FULL"

    trigger_mid = EventTrigger(
        event_source=EventSource.NEWS,
        eps_score=62.0,
        raw_content="some news",
    )
    assert should_activate_pipeline(trigger_mid) == "LIGHT"

    trigger_low = EventTrigger(
        event_source=EventSource.NEWS,
        eps_score=30.0,
        raw_content="irrelevant",
    )
    assert should_activate_pipeline(trigger_low) == "SKIP"
