from __future__ import annotations

import time

import structlog

log = structlog.get_logger(__name__)

MAX_LATENCY_MS = {
    "sentiment": 2000,
    "market_data": 500,
    "anomaly": 3000,
    "strategy": 5000,
    "risk": 100,
    "execution": 60000,
}


class HealthMonitor:
    def __init__(self) -> None:
        self._agent_errors: dict[str, int] = {}
        self._agent_last_success: dict[str, float] = {}

    def record_success(self, agent: str) -> None:
        self._agent_errors[agent] = 0
        self._agent_last_success[agent] = time.monotonic()

    def record_failure(self, agent: str) -> None:
        self._agent_errors[agent] = self._agent_errors.get(agent, 0) + 1
        count = self._agent_errors[agent]
        if count >= 3:
            log.warning("agent_repeated_failures", agent=agent, failures=count)

    def get_degraded_agents(self, latencies_ms: dict[str, int]) -> list[str]:
        degraded = []
        for agent, latency in latencies_ms.items():
            max_lat = MAX_LATENCY_MS.get(agent, 5000)
            if latency > max_lat:
                degraded.append(agent)
                log.warning("agent_latency_exceeded", agent=agent, latency_ms=latency, max_ms=max_lat)
        return degraded

    def get_pipeline_mode(self, degraded: list[str]) -> str:
        if "market_data" in degraded:
            return "PAUSED"         # cannot trade without market data
        if "sentiment" in degraded and "anomaly" in degraded:
            return "TECHNICAL_ONLY"
        if "sentiment" in degraded:
            return "LIGHT"
        return "FULL"
