from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EventSource(StrEnum):
    TRUMP_TRUTH_SOCIAL = "TRUMP_TRUTH_SOCIAL"
    TRUMP_X = "TRUMP_X"
    FED_POWELL = "FED_POWELL"
    FED_ANNOUNCEMENT = "FED_ANNOUNCEMENT"
    ECB_LAGARDE = "ECB_LAGARDE"
    ECB_ANNOUNCEMENT = "ECB_ANNOUNCEMENT"
    TREASURY = "TREASURY"
    OPEC = "OPEC"
    CEO_SP500 = "CEO_SP500"
    FOMC = "FOMC"
    CPI = "CPI"
    NFP = "NFP"
    PCE = "PCE"
    GDP = "GDP"
    PPI = "PPI"
    PMI = "PMI"
    RETAIL_SALES = "RETAIL_SALES"
    UNEMPLOYMENT_CLAIMS = "UNEMPLOYMENT_CLAIMS"
    PRICE_SPIKE = "PRICE_SPIKE"
    VOLUME_SPIKE = "VOLUME_SPIKE"
    NEWS = "NEWS"
    ANOMALY_CONTINUOUS = "ANOMALY_CONTINUOUS"


class EventUrgency(StrEnum):
    IMMEDIATE = "IMMEDIATE"   # EPS > 90, market-moving in seconds
    HIGH = "HIGH"             # EPS 75-90
    MEDIUM = "MEDIUM"         # EPS 50-75
    LOW = "LOW"               # EPS < 50, log only


# Source weight mapping for EPS calculation
SOURCE_WEIGHTS: dict[EventSource, float] = {
    EventSource.TRUMP_TRUTH_SOCIAL: 1.0,
    EventSource.TRUMP_X: 1.0,
    EventSource.FED_POWELL: 1.0,
    EventSource.FED_ANNOUNCEMENT: 1.0,
    EventSource.ECB_LAGARDE: 1.0,
    EventSource.ECB_ANNOUNCEMENT: 1.0,
    EventSource.FOMC: 1.0,
    EventSource.CPI: 0.95,
    EventSource.NFP: 0.95,
    EventSource.PCE: 0.90,
    EventSource.GDP: 0.90,
    EventSource.TREASURY: 0.85,
    EventSource.OPEC: 0.85,
    EventSource.PPI: 0.80,
    EventSource.PMI: 0.75,
    EventSource.RETAIL_SALES: 0.70,
    EventSource.UNEMPLOYMENT_CLAIMS: 0.65,
    EventSource.CEO_SP500: 0.70,
    EventSource.NEWS: 0.40,
    EventSource.PRICE_SPIKE: 0.60,
    EventSource.VOLUME_SPIKE: 0.55,
    EventSource.ANOMALY_CONTINUOUS: 0.50,
}


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


class EventTrigger(BaseModel):
    """Published to Redis Stream when a monitored event passes EPS threshold."""

    model_config = {"frozen": True}

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_source: EventSource
    eps_score: float = Field(ge=0.0, le=100.0)
    urgency: EventUrgency = EventUrgency.MEDIUM
    raw_content: str
    source_account: str | None = None          # e.g. "@realDonaldTrump"
    affected_assets_guess: list[str] = Field(default_factory=list)
    expected_direction_hint: str | None = None  # e.g. "BEARISH_TECH_BULLISH_USD"
    timestamp_ms: int = Field(default_factory=_now_ms)
    # Economic calendar fields (populated for calendar events)
    event_name: str | None = None
    consensus: float | None = None
    actual: float | None = None
    previous: float | None = None
    surprise: float | None = None              # actual - consensus

    @field_validator("eps_score")
    @classmethod
    def round_eps(cls, v: float) -> float:
        return round(v, 2)

    @field_validator("affected_assets_guess", mode="before")
    @classmethod
    def upper_assets(cls, v: Any) -> list[str]:
        if isinstance(v, list):
            return [a.upper() for a in v]
        return v

    @property
    def age_seconds(self) -> float:
        return (_now_ms() - self.timestamp_ms) / 1000

    def to_redis_dict(self) -> dict[str, str]:
        return {"data": self.model_dump_json()}

    @classmethod
    def from_redis_dict(cls, d: dict[bytes, bytes]) -> "EventTrigger":
        return cls.model_validate_json(d[b"data"])
