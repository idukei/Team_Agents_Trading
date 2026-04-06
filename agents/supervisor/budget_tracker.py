from __future__ import annotations

from datetime import date

import structlog

from core.config import settings

log = structlog.get_logger(__name__)


class BudgetTracker:
    def __init__(self) -> None:
        self._date = date.today()
        self._daily_cost: float = 0.0
        self._agent_costs: dict[str, float] = {}

    def _check_reset(self) -> None:
        today = date.today()
        if today != self._date:
            log.info("budget_daily_reset", prev_total=self._daily_cost)
            self._date = today
            self._daily_cost = 0.0
            self._agent_costs.clear()

    def record_usage(self, agent: str, tokens: int, cost_usd: float) -> None:
        self._check_reset()
        self._daily_cost += cost_usd
        self._agent_costs[agent] = self._agent_costs.get(agent, 0.0) + cost_usd

        if self._daily_cost >= settings.fireworks_budget_daily_usd * 0.8:
            log.warning(
                "budget_80pct_reached",
                total=self._daily_cost,
                limit=settings.fireworks_budget_daily_usd,
            )

    @property
    def is_budget_exceeded(self) -> bool:
        self._check_reset()
        return self._daily_cost >= settings.fireworks_budget_daily_usd

    @property
    def remaining_usd(self) -> float:
        self._check_reset()
        return max(0.0, settings.fireworks_budget_daily_usd - self._daily_cost)

    def summary(self) -> dict:
        return {
            "daily_cost_usd": round(self._daily_cost, 4),
            "daily_limit_usd": settings.fireworks_budget_daily_usd,
            "remaining_usd": round(self.remaining_usd, 4),
            "by_agent": {k: round(v, 4) for k, v in self._agent_costs.items()},
        }


budget_tracker = BudgetTracker()
