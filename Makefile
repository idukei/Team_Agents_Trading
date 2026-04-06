.PHONY: up down logs migrate seed train init-qdrant test lint format typecheck clean

# ── Docker infrastructure ─────────────────────────────────────────────────────
up:
	docker compose -f infrastructure/docker-compose.yml up -d
	@echo "Waiting for services to be ready..."
	@sleep 5
	@docker compose -f infrastructure/docker-compose.yml ps

down:
	docker compose -f infrastructure/docker-compose.yml down

down-volumes:
	docker compose -f infrastructure/docker-compose.yml down -v

logs:
	docker compose -f infrastructure/docker-compose.yml logs -f

# ── Database ──────────────────────────────────────────────────────────────────
migrate:
	@echo "Running PostgreSQL migrations..."
	@for f in infrastructure/postgres/migrations/*.sql; do \
		echo "  Applying $$f..."; \
		docker compose -f infrastructure/docker-compose.yml exec -T postgres \
			psql -U tt -d teamtrade -f /dev/stdin < $$f; \
	done
	@echo "Migrations complete."

# ── Data initialization ───────────────────────────────────────────────────────
seed:
	@echo "Seeding 2 years of OHLCV data..."
	uv run python scripts/seed_historical.py

init-qdrant:
	@echo "Initializing Qdrant collections..."
	uv run python scripts/init_qdrant.py

backfill-rag:
	@echo "Backfilling RAG embeddings..."
	uv run python scripts/backfill_rag.py

# ── ML models ─────────────────────────────────────────────────────────────────
train:
	@echo "Training anomaly detection models..."
	uv run python scripts/train_anomaly_models.py

# ── Run system ────────────────────────────────────────────────────────────────
run:
	uv run python main.py

run-monitor-only:
	uv run python main.py --monitor-only

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

test-coverage:
	uv run pytest tests/ --cov=. --cov-report=html -v

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy . --ignore-missing-imports

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	uv sync --all-extras

install-prod:
	uv sync

# ── Full initialization (new environment) ─────────────────────────────────────
bootstrap: up
	@sleep 8
	@$(MAKE) migrate
	@$(MAKE) seed
	@$(MAKE) init-qdrant
	@$(MAKE) train
	@echo ""
	@echo "TeamTrade bootstrap complete! Run 'make run' to start the system."

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
