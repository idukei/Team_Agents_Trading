# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

RUN pip install uv

WORKDIR /app
COPY pyproject.toml .
RUN uv sync --no-dev

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

# System deps for PyTorch and ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 teamtrade && chown -R teamtrade:teamtrade /app
USER teamtrade

# Models directory for persisted ML models
RUN mkdir -p /app/models

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import asyncio; asyncio.run(__import__('core.db.redis', fromlist=['ping_redis']).ping_redis())" || exit 1

ENTRYPOINT ["python", "main.py"]
