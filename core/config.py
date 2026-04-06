from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Fireworks.ai ──────────────────────────────────────────────────────────
    fireworks_api_key: SecretStr
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"
    fireworks_budget_daily_usd: float = 5.0

    model_llm_large: str = "accounts/fireworks/models/llama-v3p3-70b-instruct"
    model_llm_fast: str = "accounts/fireworks/models/llama-v3p1-8b-instruct"
    model_llm_cheap: str = "accounts/fireworks/models/gemma2-9b-it"
    model_llm_cot: str = "accounts/fireworks/models/deepseek-r1"

    # ── OpenRouter (chat permanente dashboard) ────────────────────────────────
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    model_chat: str = "qwen/qwen3.6-plus:free"

    # ── Dashboard web ─────────────────────────────────────────────────────────
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8000

    # ── Alpaca Markets ────────────────────────────────────────────────────────
    alpaca_api_key: SecretStr
    alpaca_secret_key: SecretStr
    alpaca_paper: bool = True
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_data_ws_url: str = "wss://stream.data.alpaca.markets/v2/iex"
    alpaca_trade_ws_url: str = "wss://paper-api.alpaca.markets/stream"

    # ── Free news APIs ────────────────────────────────────────────────────────
    marketaux_api_key: SecretStr | None = None
    alpha_vantage_api_key: SecretStr | None = None
    newsdata_api_key: SecretStr | None = None

    # ── PostgreSQL / TimescaleDB ──────────────────────────────────────────────
    postgres_dsn: SecretStr = SecretStr(
        "postgresql://tt:teamtrade_pass@localhost:5432/teamtrade"
    )
    postgres_pool_min: int = 2
    postgres_pool_max: int = 10

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    redis_stream_events: str = "events.triggers"
    redis_stream_maxlen: int = 10_000

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_precedents: str = "trading_precedents"
    qdrant_collection_news: str = "news_embeddings"

    # ── Telegram Bot ──────────────────────────────────────────────────────────
    telegram_bot_token: SecretStr
    telegram_chat_id: str
    telegram_approval_timeout_s: int = 30

    # ── RSSHub ────────────────────────────────────────────────────────────────
    rsshub_url: str = "http://localhost:1200"

    # ── EPS thresholds ────────────────────────────────────────────────────────
    eps_full_pipeline: float = 75.0
    eps_light_pipeline: float = 50.0

    # ── CTS thresholds ────────────────────────────────────────────────────────
    cts_execute: float = 0.68
    cts_half_size: float = 0.52

    # ── Risk limits ───────────────────────────────────────────────────────────
    risk_max_trade_pct: float = 0.01        # max 1% capital per trade
    risk_min_rr: float = 1.5                # min 1.5:1 risk/reward
    risk_max_daily_loss_pct: float = 0.025  # circuit breaker at 2.5%
    risk_max_open_trades: int = 3
    risk_circuit_breaker_pct: float = 0.025
    risk_flash_crash_pct: float = 0.02      # 2% move triggers flash crash detection
    risk_flash_crash_window_s: int = 60
    risk_pre_event_window_s: int = 180      # ±3 min around high-impact events
    risk_max_portfolio_correlation: float = 0.75
    risk_slippage_max_pct: float = 0.15     # max slippage as % of TP

    # ── Monitored assets ──────────────────────────────────────────────────────
    monitored_equities: list[str] = [
        "SPY", "QQQ", "NVDA", "AAPL", "TSLA", "JPM", "SMH", "GLD", "TLT"
    ]
    monitored_crypto: list[str] = ["BTC/USD", "ETH/USD"]
    price_spike_threshold_pct: float = 0.5
    price_spike_window_s: int = 30
    volume_spike_multiplier: float = 3.0

    # ── LangSmith ─────────────────────────────────────────────────────────────
    langsmith_api_key: SecretStr | None = None
    langchain_tracing_v2: bool = False
    langchain_project: str = "teamtrade"

    # ── System ────────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    capital_usd: float = 10_000.0

    # ── Market data buffer ────────────────────────────────────────────────────
    price_buffer_size: int = 500    # circular buffer per asset
    indicator_rsi_period: int = 14
    indicator_atr_period: int = 14
    indicator_bb_period: int = 20
    indicator_bb_std: float = 2.0

    @field_validator("monitored_equities", "monitored_crypto", mode="before")
    @classmethod
    def parse_comma_list(cls, v: str | list) -> list[str]:
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    @property
    def all_assets(self) -> list[str]:
        return self.monitored_equities + self.monitored_crypto

    @property
    def alpaca_base_url_live(self) -> str:
        return "https://api.alpaca.markets"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
