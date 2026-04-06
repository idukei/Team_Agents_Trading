# TeamTrade — Sistema Multi-Agente IA para Trading Event-Driven

> **Versión:** 1.0.0 | **Python:** 3.13+ | **Stack:** LangGraph · Redis Streams · TimescaleDB · Fireworks.ai · Alpaca Markets

Sistema autónomo de trading algorítmico que reacciona a eventos de mercado en **< 60 segundos**: declaraciones de líderes políticos (Trump, Fed, BCE), publicaciones de calendarios económicos (CPI, NFP, FOMC), anomalías de precio/volumen y noticias financieras. La única API de pago es Fireworks.ai para inferencia LLM (~$2-8/día).

---

## Índice

1. [Filosofía del Sistema](#filosofía-del-sistema)
2. [Arquitectura General](#arquitectura-general)
3. [Stack Tecnológico](#stack-tecnológico)
4. [Estructura del Proyecto](#estructura-del-proyecto)
5. [Los 9 Agentes](#los-9-agentes)
   - [Monitor Agent](#1-monitor-agent)
   - [Sentiment Agent](#2-sentiment-agent)
   - [Market Data Agent](#3-market-data-agent)
   - [Anomaly Detection Agent](#4-anomaly-detection-agent)
   - [Strategy Decision Agent](#5-strategy-decision-agent)
   - [Risk Management Agent](#6-risk-management-agent)
   - [Execution Agent](#7-execution-agent)
   - [Notification Agent](#8-notification-agent)
   - [Supervisor Agent](#9-supervisor-agent)
6. [LangGraph — Orquestación del Pipeline](#langgraph--orquestación-del-pipeline)
7. [Modelos de Datos](#modelos-de-datos)
8. [Infraestructura](#infraestructura)
9. [Flujo End-to-End](#flujo-end-to-end)
10. [Instalación y Configuración](#instalación-y-configuración)
11. [Estrategias de Trading](#estrategias-de-trading)
12. [Métricas de Éxito](#métricas-de-éxito)
13. [Seguridad y Gestión de Riesgos](#seguridad-y-gestión-de-riesgos)
14. [Fases de Implementación](#fases-de-implementación)

---

## Filosofía del Sistema

TeamTrade opera bajo tres principios fundamentales:

**1. Coste operativo mínimo**
Toda la infraestructura de datos es gratuita. Alpaca Markets (WebSocket), HorizonFX (calendario económico), RSS feeds (noticias), RSSHub (Truth Social/X). La única variable de coste es el consumo LLM via Fireworks.ai, la plataforma de inferencia de menor latencia disponible (~4x más rápida que alternativas).

**2. Event-driven, no polling**
El sistema no consume recursos monitorizando constantemente sin razón. Cada componente se activa únicamente cuando existen señales relevantes. El bus de mensajes (Redis Streams) es el sistema nervioso que coordina la activación selectiva. No hay "agentes siempre activos" con LLM corriendo en bucle.

**3. Convergencia multi-señal antes de ejecutar**
Nunca se ejecuta un trade basándose en una sola señal. Se requiere convergencia de al menos dos de los tres agentes de análisis (Sentiment, Market Data, Anomaly). El **Composite Trade Score (CTS)** cuantifica esa convergencia y determina el tamaño de posición proporcionalmente.

---

## Arquitectura General

```
┌─────────────────────────────────────────────────────────────────┐
│                  FUENTES DE DATOS (100% GRATUITAS)              │
├──────────┬───────────────┬──────────────────┬───────────────────┤
│ Alpaca   │ HorizonFX     │ Marketaux/RSS    │ RSSHub            │
│ WebSocket│ Calendar API  │ NewsData.io      │ Truth Social/X    │
└────┬─────┴──────┬────────┴────────┬─────────┴──────┬────────────┘
     │            │                 │                 │
     └────────────┴─────────────────┴─────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │    MONITOR AGENT    │  ← asyncio, sin LLM
                    │  EPS Scoring (0-100)│
                    └──────────┬──────────┘
                               │ Redis Stream: events.triggers
                    ┌──────────▼──────────┐
                    │   SUPERVISOR NODE   │  ← LangGraph entry
                    └──────────┬──────────┘
                               │ Send API scatter (paralelo)
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐   ┌──────────────┐  ┌──────────┐
        │SENTIMENT │   │ MARKET DATA  │  │ ANOMALY  │
        │  AGENT   │   │    AGENT     │  │  AGENT   │
        │RAG+LLM   │   │ numpy+pandas │  │ IF+LSTM  │
        └────┬─────┘   └──────┬───────┘  └────┬─────┘
             │                │               │
             └────────────────┴───────────────┘
                              │ convergencia
                    ┌─────────▼──────────┐
                    │  STRATEGY AGENT    │  ← CTS + Fireworks.ai
                    └─────────┬──────────┘
                              │ TradeProposal
                    ┌─────────▼──────────┐
                    │   RISK AGENT       │  ← 3 capas, < 5ms
                    └─────────┬──────────┘
                              │
               ┌──────────────┴──────────────────┐
               ▼                                 ▼
     ┌─────────────────┐             ┌───────────────────┐
     │ NOTIFY PROPOSAL │             │  NOTIFY RESULT    │
     │  (Telegram)     │             │  (trade cerrado)  │
     └────────┬────────┘             └───────────────────┘
              │ [interrupt] human-in-the-loop
              ▼
     ┌─────────────────┐
     │ EXECUTION AGENT │  ← Alpaca REST
     └─────────────────┘
```

---

## Stack Tecnológico

| Componente | Tecnología | Coste |
|---|---|---|
| Lenguaje | Python 3.13+ | Gratuito |
| Orquestación | LangGraph 0.2+ | Gratuito |
| Bus de mensajes | Redis Streams 7.4 | Gratuito (self-hosted) |
| Series temporales | TimescaleDB 2.16 / PostgreSQL 16 | Gratuito (self-hosted) |
| Vector DB (RAG) | Qdrant 1.9 | Gratuito (self-hosted) |
| Checkpointing | LangGraph PostgresSaver | Gratuito |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Gratuito (local) |
| ML anomalías | scikit-learn + PyTorch | Gratuito |
| LLM inferencia | Fireworks.ai | **~$2-8/día** |
| Datos mercado | Alpaca Markets | Gratuito (paper) |
| Noticias | Marketaux + NewsData.io + RSS | Gratuito |
| RSS proxy | RSSHub (self-hosted Docker) | Gratuito |
| Notificaciones | Telegram Bot API | Gratuito |
| Observabilidad | LangSmith (5K traces/mes gratis) | Gratuito |

---

## Estructura del Proyecto

```
teamtrade/
├── main.py                          # Entrypoint — arranca todo el sistema
├── pyproject.toml                   # Dependencias (uv como package manager)
├── Makefile                         # Comandos: up, migrate, seed, train, test
├── .env.example                     # Template de variables de entorno
│
├── core/                            # Módulos compartidos por todos los agentes
│   ├── config.py                    # pydantic-settings: todos los parámetros
│   ├── logging.py                   # structlog con JSON renderer
│   ├── exceptions.py                # Jerarquía de excepciones
│   ├── models/
│   │   ├── events.py                # EventTrigger, EventSource, EPS enums
│   │   ├── signals.py               # SentimentSignal, MarketContext, AnomalyAlert
│   │   ├── trade.py                 # TradeProposal, RiskDecision, ExecutionResult
│   │   └── portfolio.py             # PortfolioState, Position, SessionMetadata
│   └── db/
│       ├── postgres.py              # asyncpg pool factory
│       ├── timescale.py             # Helpers para hypertables OHLCV
│       ├── redis.py                 # aioredis: publish/consume streams
│       └── qdrant.py                # Qdrant async client, collections
│
├── agents/
│   ├── monitor/
│   │   ├── agent.py                 # Orquestador: lanza 4 sub-monitores
│   │   ├── price_monitor.py         # Alpaca WebSocket: spikes precio/volumen
│   │   ├── calendar_monitor.py      # HorizonFX: PRE_EVENT + data release
│   │   ├── news_monitor.py          # Marketaux + RSS: dedup por hash
│   │   ├── leaders_monitor.py       # RSSHub: Trump/Fed/BCE/CEOs
│   │   └── eps_scorer.py            # Fórmula EPS + routing por umbral
│   │
│   ├── sentiment/
│   │   ├── agent.py                 # Nodo LangGraph: RAG → LLM → coherence
│   │   ├── rag.py                   # Qdrant search + sentence-transformers
│   │   ├── llm_analyzer.py          # Fireworks.ai client + retry (tenacity)
│   │   └── coherence.py             # Validación determinística 30min window
│   │
│   ├── market_data/
│   │   ├── agent.py                 # Nodo LangGraph: calcula MarketContext
│   │   ├── stream.py                # PriceBuffer circular deque(500) + registry
│   │   ├── indicators.py            # RSI, VWAP, ATR, OBI, Bollinger (numpy)
│   │   └── levels.py                # Williams fractal pivots + liquidity pools
│   │
│   ├── anomaly/
│   │   ├── agent.py                 # Nodo LangGraph: ensemble + LLM validation
│   │   ├── isolation_forest.py      # IsolationForest wrapper + persistencia
│   │   ├── zscore.py                # Adaptive Z-score con threshold ATR-based
│   │   ├── lstm_autoencoder.py      # PyTorch LSTM AE + threshold percentil 95
│   │   ├── ensemble.py              # Votación 2/3 + classify_anomaly_type
│   │   └── llm_validator.py         # llama-8b clasificación rápida (< 80ms)
│   │
│   ├── strategy/
│   │   ├── agent.py                 # Nodo LangGraph: filtros → CTS → LLM
│   │   ├── regime_filters.py        # Hard rules: circuit breaker, spread, etc.
│   │   ├── cts_scorer.py            # Fórmula CTS + size multiplier
│   │   └── llm_optimizer.py         # Chain-of-thought params + validator
│   │
│   ├── risk/
│   │   ├── agent.py                 # Nodo LangGraph: 3 capas en < 5ms
│   │   ├── trade_layer.py           # Capa 1: R/R, max risk 1%, slippage
│   │   ├── portfolio_layer.py       # Capa 2: VaR, correlación, exposición
│   │   └── systemic_layer.py        # Capa 3: circuit breaker, flash crash
│   │
│   ├── execution/
│   │   ├── agent.py                 # Nodo LangGraph: submit + background mgmt
│   │   ├── alpaca_client.py         # REST client (paper/live toggle)
│   │   ├── order_manager.py         # LIMIT + TTF 30s → MARKET fallback
│   │   └── position_manager.py      # TP1→50%+BE, TP2→100%, timeout exit
│   │
│   ├── notification/
│   │   ├── agent.py                 # Nodo LangGraph + métodos de notificación
│   │   ├── telegram_bot.py          # python-telegram-bot + command handlers
│   │   ├── message_templates.py     # Templates Markdown por tipo de evento
│   │   └── approval_handler.py      # Inline buttons + asyncio.Event(30s)
│   │
│   └── supervisor/
│       ├── agent.py                 # Nodo entry: conflict resolution + health
│       ├── health_monitor.py        # Latencias por agente + degraded mode
│       ├── budget_tracker.py        # Tracking coste Fireworks.ai diario
│       └── post_trade_analyst.py    # Análisis 22:00 UTC + actualiza RAG
│
├── graph/
│   ├── state.py                     # TradingState TypedDict + Annotated reducers
│   ├── builder.py                   # StateGraph: nodos + edges + compile
│   ├── nodes.py                     # Thin wrappers + agent singletons
│   ├── edges.py                     # route_after_supervisor (Send API scatter)
│   └── checkpointer.py              # AsyncPostgresSaver factory
│
├── infrastructure/
│   ├── docker-compose.yml           # 4 servicios: postgres, redis, qdrant, rsshub
│   ├── Dockerfile                   # Multi-stage: builder + runtime Python 3.13
│   ├── postgres/
│   │   ├── init.sql                 # Extensiones: timescaledb, uuid-ossp
│   │   └── migrations/
│   │       ├── 001_core_tables.sql  # event_triggers, trade_proposals, etc.
│   │       ├── 002_hypertables.sql  # ohlcv, quotes, anomaly_log (TimescaleDB)
│   │       ├── 003_langgraph_checkpoints.sql
│   │       └── 004_trading_log.sql  # daily_performance, llm_usage, cts_weights
│   └── redis/
│       └── redis.conf               # maxmemory 512mb + AOF persistence
│
├── scripts/
│   ├── seed_historical.py           # 2 años OHLCV via yfinance → TimescaleDB
│   ├── train_anomaly_models.py      # Entrena IsolationForest + LSTM AE
│   └── init_qdrant.py               # Crea colecciones + seed precedentes iniciales
│
└── tests/
    ├── conftest.py                  # Env vars de test, sys.path setup
    └── unit/
        ├── test_eps_scorer.py
        ├── test_cts_scorer.py
        ├── test_indicators.py
        └── test_risk_layers.py
```

---

## Los 9 Agentes

### 1. Monitor Agent

**Archivo:** `agents/monitor/agent.py`

El único agente que corre permanentemente sin consumir LLM. Orquesta 4 sub-monitores como tareas asyncio independientes, recibe sus eventos via cola interna, aplica rate-limiting global y publica a Redis Stream.

#### 1.1 Political Leaders Monitor (`leaders_monitor.py`)

Monitoriza declaraciones de líderes con impacto probado en mercados:

| Figura | Fuente | Latencia | Weight EPS |
|---|---|---|---|
| Trump | Truth Social (RSSHub) | < 15s | 1.0 |
| Trump | X/Twitter (RSSHub) | < 15s | 1.0 |
| Fed (Powell) | fed.gov RSS | < 5s | 1.0 |
| BCE (Lagarde) | ecb.europa.eu RSS | < 5s | 1.0 |
| US Treasury | treasury.gov RSS | < 10s | 0.85 |
| OPEC | opec.org RSS | < 30s | 0.85 |

Flujo de detección:
1. **Polling RSS** cada 15-30s via RSSHub (self-hosted Docker)
2. **Deduplicación** por hash SHA-256 del contenido (Redis cache 24h)
3. **Keyword filter** sin LLM (< 1ms): `tariff`, `sanction`, `rate`, `ban`, `oil`, `crypto`...
4. **EPS scoring**: si pasa filtro → calcula score → publica EventTrigger

#### 1.2 Economic Calendar Monitor (`calendar_monitor.py`)

Integra HorizonFX API (gratuita, actualización cada 5 minutos):
- Carga eventos del día al arrancar
- Publica `PRE_EVENT` alert a T-60s de cada evento high/medium impact
- Al detectar `actual != null` calcula **surprise = actual - consensus**
- Solo activa pipeline si hay sorpresa significativa (≠ 0)

Eventos clave monitorizados: CPI, PCE, NFP, FOMC, ECB Rate Decision, GDP, PPI, PMI, Retail Sales, Unemployment Claims.

#### 1.3 News Monitor (`news_monitor.py`)

Tres fuentes con rotación para respetar límites gratuitos:
- **RSS feeds** (Reuters, MarketWatch, CNBC, WSJ): polling cada 10s, ilimitado
- **Marketaux API**: cada 15min (100 req/día gratuitas)
- **NewsData.io**: cada 30min (200 créditos/día gratuitos)

Deduplicación por MD5 del título (Redis cache 12h).

#### 1.4 Price Monitor (`price_monitor.py`)

Alpaca WebSocket en tiempo real:
- Detecta **price spikes**: cambio ≥ 0.5% en ventana de 30s
- Detecta **volume spikes**: volumen actual ≥ 3x media móvil de 20 períodos
- Mantiene `PriceBuffer` (deque de 500 ticks) por activo — compartido con Market Data Agent via `buffer_registry` singleton

#### 1.5 EPS Scoring (`eps_scorer.py`)

```
EPS = (source_weight × 0.35) + (keyword_intensity × 0.30)
    + (novelty × 0.20) + (vix_normalized × 0.15)

EPS > 75  → Full Pipeline (Sentiment + Market Data + Anomaly en paralelo)
EPS 50-75 → Light Pipeline (Market Data + Anomaly)
EPS < 50  → Log, sin acción
```

---

### 2. Sentiment Agent

**Archivo:** `agents/sentiment/agent.py`

Pipeline en 3 fases ejecutado en paralelo con los otros agentes de análisis:

#### Fase 1 — RAG Query (< 100ms)

`rag.py` busca en Qdrant los 5 precedentes históricos más similares al texto del evento usando embeddings locales (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensiones). Umbral de similitud coseno: 0.55.

```python
# Ejemplo de precedente recuperado:
{
    "score": 0.87,
    "event_text": "Trump announces 35% tariffs on Chinese semiconductors...",
    "asset": "NVDA",
    "direction": "SHORT",
    "return_15min": -1.8,
    "return_60min": -3.2,
    "outcome_summary": "NVDA dropped 3.2% in 60min..."
}
```

#### Fase 2 — LLM Analysis (< 400ms)

`llm_analyzer.py` llama a Fireworks.ai con prompt especializado. Selección de modelo por complejidad del texto:
- Textos > 200 chars: `llama-v3p3-70b-instruct` ($0.90/1M tokens, TTFT ~400ms)
- Textos cortos: `gemma2-9b-it` ($0.10/1M tokens, TTFT ~60ms)

El prompt incluye: precedentes RAG + condiciones técnicas actuales + texto del evento. La respuesta es JSON estructurado con `direction`, `magnitude`, `confidence`, `time_horizon`, `reasoning`.

**retry con tenacity**: 3 intentos con backoff exponencial (1s → 8s). Si falla, el pipeline continúa sin señal de sentimiento (degraded mode).

#### Fase 3 — Coherence Validation (< 5ms)

`coherence.py` compara con señales previas del mismo activo (ventana 30min):
- Misma dirección → aumenta confianza ×1.1
- Dirección opuesta → marca `conflicted=True`, reduce confianza ×0.7

---

### 3. Market Data Agent

**Archivo:** `agents/market_data/agent.py`

Lee del `buffer_registry` singleton (llenado en tiempo real por PriceMonitor) y calcula indicadores técnicos con numpy. Sin llamadas a APIs externas en el momento del análisis.

#### Indicadores calculados (`indicators.py`)

| Indicador | Implementación | Uso |
|---|---|---|
| **RSI(14)** | Wilder's smoothing | Momentum, evita entradas overbought |
| **VWAP** | ∑(precio × volumen) / ∑volumen | Precio justo intradía |
| **ATR(14)** | Wilder's TR smoothing | Sizing dinámico del stop-loss |
| **OBI** | (bid_size - ask_size) / total | Presión direccional a corto plazo |
| **Bollinger(20,2)** | Media ± 2σ | Detección de sobre-extensión |

#### Niveles clave (`levels.py`)

Algoritmo **Williams Fractal**: identifica máximos/mínimos locales con ventana de 2 períodos en datos OHLCV históricos de TimescaleDB. **Liquidity pools**: histograma de volumen por precio (20 bins) → top-3 concentraciones.

---

### 4. Anomaly Detection Agent

**Archivo:** `agents/anomaly/agent.py`

Ensemble de 3 modelos estadísticos/ML. Requiere acuerdo de **mínimo 2 de 3** para activar alerta.

#### Modelo 1 — Isolation Forest (`isolation_forest.py`)

- Entrenado sobre features: `(price_return, volume_ratio, spread_bps, obi)`
- 200 árboles, contamination=0.03, seed=42
- Persistido en `models/isolation_forest.pkl`
- Score negativo → anomalía, normalizado a confianza [0, 1]

#### Modelo 2 — Adaptive Z-score (`zscore.py`)

```python
z = (precio_actual - rolling_mean) / rolling_std

# Threshold dinámico basado en ATR:
# ATR > 2%  → threshold = 4σ  (mercado volátil, requiere señal más fuerte)
# ATR 1-2%  → threshold = 3.5σ
# ATR < 1%  → threshold = 3σ  (mercado tranquilo)
```

También detecta volume anomalies con Z-score > 2.5.

#### Modelo 3 — LSTM Autoencoder (`lstm_autoencoder.py`)

- Arquitectura: Encoder LSTM(4→64) + Decoder LSTM(64→4)
- Input: secuencias de 30 ticks con 4 features normalizadas
- Threshold: percentil 95 del error de reconstrucción en datos de entrenamiento
- Especialmente potente para detectar **acumulación institucional silenciosa** (volumen sube, precio no se mueve → suele preceder breakouts)

#### LLM Validation

Tras detección estadística, `llm_validator.py` usa `llama-v3p1-8b-instruct` (rápido y barato) para clasificar la anomalía:

`INSTITUTIONAL_ACCUMULATION` | `NEWS_OVERREACTION` | `BULL_TRAP` | `BEAR_TRAP` | `DATA_GLITCH` | `GENUINE_BREAKOUT`

Las clasificadas como `DATA_GLITCH` son descartadas.

---

### 5. Strategy Decision Agent

**Archivo:** `agents/strategy/agent.py`

El cerebro del sistema. Tres niveles de decisión:

#### Nivel 1 — Regime Filters (`regime_filters.py`)

Reglas duras que cancelan el trade instantáneamente (sin LLM):

- Circuit breaker activo
- Pérdida diaria ≥ 2.5% del capital
- Máximo de trades abiertos (3) alcanzado
- Trade opuesto ya abierto para el mismo activo
- Spread > 3× normal (~15bps)
- Evento de alto impacto en < 3 minutos

#### Nivel 2 — CTS Scoring (`cts_scorer.py`)

```
CTS = (sentiment.magnitude × sentiment.confidence × 0.32)   # señal LLM
    + (anomaly.ml_confidence × direction_align × 0.28)       # señal ML
    + (market.trend_score × 0.20)                            # técnico
    + (precedent.return_normalized × 0.12)                   # histórico
    + (obi_alignment × 0.08)                                 # libro órdenes

CTS ≥ 0.68 → Ejecutar con tamaño completo
CTS 0.52-0.68 → Ejecutar con 50% de tamaño
CTS < 0.52 → No operar
```

Penalizaciones automáticas:
- Señal de sentimiento marcada `conflicted` → componente = 0
- RSI > 75 en trade LONG o RSI < 25 en trade SHORT → penalización -0.2
- Alto false_positive_risk en anomalía → confianza ×0.5

#### Nivel 3 — LLM Parameter Optimization (`llm_optimizer.py`)

Para CTS ≥ 0.68, invoca `llama-v3p3-70b-instruct` con chain-of-thought estructurado. El LLM recibe: señales convergentes, contexto técnico, precedentes, régimen de volatilidad. Produce parámetros exactos de trade en JSON.

**Validador determinístico** post-LLM (nunca puede saltarse):
- Calcula posición máxima basada en 1% de riesgo del capital
- Verifica R/R ≥ 1.5:1 — si no se cumple, auto-ajusta TP
- Limita `max_hold_seconds` a [60, 1800]

---

### 6. Risk Management Agent

**Archivo:** `agents/risk/agent.py`

Tres capas determinísticas ejecutadas en < 5ms. Ningún LLM puede influir en estas decisiones.

#### Capa 1 — Por Trade (`trade_layer.py`)

- **Riesgo máximo**: 1% del capital por trade. Si `risk_usd > max_risk_usd`, reduce el tamaño proporcionalmente.
- **R/R mínimo**: 1.5:1. Si no se cumple después del ajuste de tamaño → REJECTED.
- **Estimación de slippage**: `spread × size`. Si el slippage estimado supera el 15% del TP → REJECTED.

#### Capa 2 — Portfolio (`portfolio_layer.py`)

- **Riesgo total**: suma del riesgo de todas las posiciones abiertas + nueva posición ≤ 5% del capital.
- **Correlación**: si el nuevo activo es del mismo sector que posiciones existentes (correlación > 0.75), reduce tamaño al 50%.
- **VaR paramétrico (95%, 1 día)**: `size × daily_vol × 1.645`. Reportado como métrica.

#### Capa 3 — Sistémica (`systemic_layer.py`)

- Circuit breaker: pérdida diaria ≥ 2.5% → REJECTED.
- Flash crash: `volatility_regime == EXTREME` → REJECTED.
- Pre-event: evento en < 3min → reduce tamaño al 40%.
- Spread alto (> 10bps) → reduce tamaño al 60%.

El estado final puede ser `APPROVED`, `APPROVED_REDUCED` o `REJECTED`.

---

### 7. Execution Agent

**Archivo:** `agents/execution/agent.py`

Ejecuta las órdenes aprobadas via Alpaca Markets REST API. El toggle `ALPACA_PAPER=true` permite paper trading sin cambios de código.

#### Estrategia de ejecución (`order_manager.py`)

1. **LIMIT order** en zona de liquidez (nivel identificado por Market Data Agent): menor slippage, mejor precio de entrada.
2. Si no se llena en **30 segundos** (TTF configurable): cancela el LIMIT y convierte a MARKET.
3. **MARKET directo** solo cuando urgencia = IMMEDIATE y EPS > 92.

#### Gestión de posición (`position_manager.py`)

Loop de monitoreo que corre como tarea asyncio en background (no bloquea el grafo):

```
TP1 alcanzado:
  → Cierra 50% de la posición
  → Mueve SL a breakeven (precio de entrada)

TP2 alcanzado:
  → Cierra el 100% restante

SL alcanzado:
  → Cierra posición completa

max_hold_seconds agotado:
  → Market close del remanente
```

---

### 8. Notification Agent

**Archivo:** `agents/notification/agent.py`

Telegram Bot como interfaz humana en tiempo real. Sin coste, push instantáneo en cualquier dispositivo.

#### Tipos de notificación

| Evento | Contenido | Cuándo |
|---|---|---|
| Alerta política | EPS, texto, activos afectados | Inmediato tras detectar declaración |
| Evento económico | Actual, consenso, sorpresa | Al publicarse el dato |
| Trade propuesto | Dirección, precio, SL, TP, CTS | Antes de ejecutar |
| Orden ejecutada | Precio fill, slippage | Tras confirmar fill |
| TP1/TP2/SL | P&L parcial/total, duración | En cada evento |
| Anomalía crítica | Tipo, confianza ML, clasificación LLM | Solo severity HIGH/CRITICAL |
| Circuit breaker | Motivo, P&L del día | Inmediato |

#### Human-in-the-Loop (`approval_handler.py`)

```
┌──────────────────────────────────────────┐
│  📈 TRADE PROPUESTO — NVDA SHORT        │
│  Entrada: LIMIT @ $876.50               │
│  SL: $884.00 | TP: $868 / $860         │
│  CTS: 0.76 | Tamaño: $1,500            │
│                                          │
│  Respondiendo automáticamente en 30s... │
│                                          │
│  [✅ APROBAR]    [❌ CANCELAR]          │
└──────────────────────────────────────────┘
```

El approval handler usa `asyncio.Event` para esperar la respuesta del botón. Si no hay respuesta en 30s, el comportamiento depende del modo configurado (automático en paper trading).

#### Comandos disponibles

```
/status       → Estado del sistema, posiciones abiertas, P&L
/portfolio    → Resumen de portfolio
/pause        → Pausa entradas (modo close-only)
/resume       → Reanuda operación normal
/close_all    → KILL SWITCH: cierra todas las posiciones
/help         → Lista de comandos
```

---

### 9. Supervisor Agent

**Archivo:** `agents/supervisor/agent.py`

Nodo de entrada en el grafo LangGraph. Responsabilidades:

**Resolución de conflictos**: cuando Sentiment dice BULLISH y Anomaly dice SHORT, el supervisor aplica política de desempate basada en confianza relativa. Si `anomaly.ml_confidence > sentiment.confidence + 0.15`, marca la señal de sentimiento como `conflicted`. En caso contrario, gana el sentimiento.

**Budget tracking** (`budget_tracker.py`): monitoriza el consumo diario de Fireworks.ai. Al superar el 80% del presupuesto, activa modo `TECHNICAL_ONLY` (desactiva agentes LLM, opera solo con Market Data y Anomaly).

**Health monitoring** (`health_monitor.py`): rastrea latencias de cada agente. Latencias excesivas activan modos de degradación:

```
market_data falla → PAUSED (sin datos técnicos = sin trades)
sentiment falla   → LIGHT (CTS penalizado -0.15)
anomaly falla     → continúa (CTS penalizado -0.10)
```

**Post-trade analyst** (`post_trade_analyst.py`): scheduler que corre a las 22:00 UTC. Consulta los trades del día en PostgreSQL, calcula métricas de performance, y almacena los trades exitosos como nuevos precedentes en Qdrant (el sistema aprende de sus propias operaciones).

---

## LangGraph — Orquestación del Pipeline

### StateGraph y TradingState

El estado global fluye y muta a través del grafo. Cada nodo lee el estado completo y devuelve solo los campos que modifica.

```python
class TradingState(TypedDict):
    event_trigger: EventTrigger | None
    sentiment_signal: SentimentSignal | None
    market_context: MarketContext | None
    anomaly_alert: AnomalyAlert | None
    trade_proposal: TradeProposal | None
    risk_decision: RiskDecision | None
    execution_result: ExecutionResult | None
    portfolio_state: PortfolioState       # persiste entre runs
    session_metadata: SessionMetadata
    awaiting_human_approval: bool
    error_log: Annotated[list[str], lambda a, b: a + b]  # reducer acumulador
```

El campo `error_log` usa un **Annotated reducer** que acumula errores sin sobreescribir. Esto es crítico porque sentiment, market_data y anomaly escriben en paralelo — sin reducer habría condición de carrera.

### Scatter Pattern con Send API

```python
def route_after_supervisor(state: TradingState) -> list[Send]:
    eps = state["event_trigger"].eps_score

    if eps >= 75:   # Full pipeline
        return [
            Send("sentiment", state),
            Send("market_data", state),
            Send("anomaly", state),
        ]
    elif eps >= 50:  # Light pipeline
        return [Send("market_data", state), Send("anomaly", state)]
    else:
        return "end_node"
```

Los tres nodos de análisis reciben el mismo estado y corren **verdaderamente en paralelo** (no secuencial). Todos convergen en `strategy_node` que espera los outputs de los tres.

### Checkpointing con PostgresSaver

```python
graph = builder.compile(
    checkpointer=AsyncPostgresSaver.from_conn_string(dsn),
    interrupt_before=["execution"],
)
```

Cada superstep del grafo queda persistido en PostgreSQL. Esto permite:
- **Recuperación ante reinicios**: el sistema retoma exactamente donde estaba
- **Time-travel debugging**: `graph.get_state_history(config)` devuelve todos los snapshots
- **Human-in-the-loop nativo**: `interrupt_before=["execution"]` pausa el grafo antes de ejecutar, la aprobación Telegram reanuda con `graph.invoke(None, config)`

---

## Modelos de Datos

### EventTrigger

Contrato de salida del Monitor Agent, entrada del pipeline:

```python
EventTrigger(
    event_id="uuid-v4",
    event_source=EventSource.TRUMP_TRUTH_SOCIAL,
    eps_score=94.0,
    urgency=EventUrgency.IMMEDIATE,
    raw_content="Imposing 35% tariffs on Chinese semiconductors...",
    affected_assets_guess=["NVDA", "SMH", "QQQ"],
    expected_direction_hint="BEARISH_TECH",
    timestamp_ms=1712345678901,
)
```

### TradeProposal

Contrato de salida del Strategy Agent, con validación automática SL/TP:

```python
TradeProposal(
    asset="NVDA",
    direction=Direction.SHORT,
    entry_type=OrderType.LIMIT,
    entry_price=876.50,
    position_size_usd=1500.0,
    stop_loss=884.00,        # validado: SHORT → SL > entry ✓
    take_profit=[868.00, 860.00],  # validado: SHORT → TP < entry ✓
    max_hold_seconds=600,
    cts_score=0.76,
    strategy_type=StrategyType.POLITICAL_SCALP,
)
```

Si SL o TP violan la dirección, el constructor Pydantic lanza `ValueError` antes de llegar al Risk Agent.

---

## Infraestructura

### Docker Compose

```bash
make up    # Levanta 4 servicios
```

| Servicio | Imagen | Puerto | Propósito |
|---|---|---|---|
| `postgres` | timescale/timescaledb:2.16-pg16 | 5432 | OHLCV, trades, checkpoints |
| `redis` | redis:7.4-alpine | 6379 | Event bus + cache |
| `qdrant` | qdrant/qdrant:v1.9.0 | 6333/6334 | Vector DB para RAG |
| `rsshub` | diygod/rsshub:latest | 1200 | RSS proxy para Truth Social/X |

### Migraciones PostgreSQL

```
001_core_tables.sql   → event_triggers, trade_proposals, risk_decisions, execution_results
002_hypertables.sql   → ohlcv (TimescaleDB), quotes, anomaly_log, indicator_snapshots
003_langgraph_checkpoints.sql → checkpoints, checkpoint_blobs, checkpoint_writes
004_trading_log.sql   → daily_performance, agent_performance, llm_usage, cts_weights
```

Las tablas de series temporales (`ohlcv`, `quotes`, etc.) son **hypertables de TimescaleDB** con compresión automática de chunks > 7 días.

---

## Flujo End-to-End

Escenario real: Trump publica en Truth Social sobre aranceles a semiconductores.

```
T+0s    Trump publica: "Imposing 35% tariffs on Chinese semiconductors"

T+12s   RSSHub detecta nuevo post (polling 15s)
        LeadersMonitor lee texto

T+12s   Keyword filter (< 1ms):
        "tariff" + "semiconductor" + fuente=Trump → HIT

T+12s   EPS calculado: 94 → FULL PIPELINE
        EventTrigger publicado a Redis Stream
        Telegram: "🚨 ALERTA CRÍTICA — TRUMP STATEMENT"

T+12s   LangGraph consumer recibe evento
        supervisor_node → Send API scatter

T+12s   [PARALELO]:
        sentiment_node → RAG query (Qdrant) + Fireworks.ai llama-70b
        market_data_node → lee buffer NVDA, calcula indicadores
        anomaly_node → ensemble 3 modelos NVDA

T+45s   sentiment_node completa:
        direction=SHORT, primary=NVDA, magnitude=0.82, confidence=0.79

T+43s   market_data_node completa:
        price=$875.40, RSI=68.3, OBI=-0.42, trend=BULLISH_WEAKENING

T+44s   anomaly_node completa:
        VOLUME_ACCUMULATION_SILENT, HIGH, ml_confidence=0.71

T+45s   strategy_node:
        Regime filters → OK
        CTS = (0.82×0.79×0.32) + (0.71×1.0×0.28) + (0.6×0.20) + ... = 0.74
        CTS ≥ 0.68 → LLM optimizer (deepseek-r1 chain-of-thought)
        TradeProposal: SHORT NVDA LIMIT@$876.50, SL=$884, TP=[$868,$860]

T+50s   risk_node (< 5ms):
        Layer1: risk=$75, max=$100 ✓ | R/R=2.1 ✓
        Layer2: VaR=$49 ✓ | correlation=0.2 ✓
        Layer3: no circuit breaker ✓ | no flash crash ✓
        APPROVED $1,500

T+51s   interrupt_before=["execution"] → grafo pausado
        notify_proposal → Telegram con botones

T+53s   Modo automático: await asyncio.Event (30s timeout)
        → APPROVED automático

T+55s   execution_node:
        LIMIT SHORT NVDA $876.50 → Alpaca REST
        Alpaca confirma fill $876.45 (slippage -0.06bps)
        Telegram: "✅ ORDEN EJECUTADA"

T+8min  position_manager (background task):
        NVDA = $868.20 → TP1 hit
        Cierra 50% → PnL parcial +$61.88
        SL movido a breakeven $876.45

T+14min NVDA = $861.50 → TP2 hit
        Cierra restante → PnL total +$113.40
        Telegram: "💰 TRADE CERRADO | +$113.40 | R/R: 2.2:1"

T+22:00 post_trade_analyst:
        Registra en daily_performance
        Almacena como precedente en Qdrant
        "NVDA SHORT 2025-XX-XX: -3.2% in 60min after Trump tariff tweet"
```

**Tiempo total evento → orden: 55 segundos.**

---

## Instalación y Configuración

### Requisitos

- Docker + Docker Compose
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes)

### 1. Clonar y configurar

```bash
git clone <repo>
cd teamtrade

# Copiar y editar variables de entorno
cp .env.example .env
```

Variables **obligatorias** en `.env`:

```env
FIREWORKS_API_KEY=fw_...        # Obtener en fireworks.ai (gratuito para registrarse)
ALPACA_API_KEY=PK...            # Alpaca Markets paper trading (gratuito)
ALPACA_SECRET_KEY=...
TELEGRAM_BOT_TOKEN=...          # BotFather en Telegram → /newbot
TELEGRAM_CHAT_ID=...            # Tu chat ID (envía /start al bot y consulta la API)
```

Variables opcionales (mejoran el sistema):

```env
MARKETAUX_API_KEY=...           # 100 req/día gratis
NEWSDATA_API_KEY=...            # 200 créditos/día gratis
LANGSMITH_API_KEY=...           # Observabilidad LangSmith (5K traces/mes)
LANGCHAIN_TRACING_V2=true
```

### 2. Instalar dependencias

```bash
# Instalar uv si no está disponible
curl -LsSf https://astral.sh/uv/install.sh | sh

# Instalar dependencias del proyecto
make install
```

### 3. Inicializar infraestructura

```bash
# Levantar todos los servicios Docker
make up

# Aplicar migraciones de base de datos
make migrate

# Cargar 2 años de datos históricos OHLCV
make seed            # ~10-15 minutos

# Inicializar colecciones Qdrant con precedentes de ejemplo
make init-qdrant

# Entrenar modelos de detección de anomalías
make train           # ~5-10 minutos
```

### 4. Ejecutar el sistema

```bash
# Fase 1: solo monitoreo + notificaciones Telegram (sin trading)
python main.py --monitor-only

# Sistema completo con paper trading
python main.py

# Con Docker
docker build -f infrastructure/Dockerfile -t teamtrade .
docker run --env-file .env teamtrade
```

### 5. Verificar funcionamiento

```bash
# Tests unitarios (no requieren servicios externos)
make test-unit

# Simular un evento manualmente (publicar a Redis Stream)
python -c "
import asyncio, json
import redis.asyncio as aioredis

async def main():
    r = aioredis.from_url('redis://localhost:6379')
    trigger = {
        'event_id': 'test-001',
        'event_source': 'TRUMP_TRUTH_SOCIAL',
        'eps_score': 92.0,
        'urgency': 'IMMEDIATE',
        'raw_content': 'Imposing 50% tariffs on all Chinese goods immediately',
        'affected_assets_guess': ['SPY', 'QQQ', 'NVDA'],
        'timestamp_ms': 1712345678901
    }
    await r.xadd('events.triggers', {'data': json.dumps(trigger)})
    print('Event published!')
    await r.aclose()

asyncio.run(main())
"
```

El bot de Telegram debería recibir la alerta en < 2s.

---

## Estrategias de Trading

### 1. Political Statement Scalp (estrategia primaria)

**Trigger**: `LeaderStatementEvent` con EPS > 80  
**Assets**: directamente mencionados o afectados via keyword→assets map  
**Timeframe**: 2-20 minutos  
**Edge**: el mercado tarda 30-120s en reaccionar completamente; el sistema reacciona en < 60s  
**Ejemplo**: Trump anuncia aranceles → SHORT semiconductores → beneficio antes del impacto completo

### 2. Economic Event Scalp

**Trigger**: CPI/NFP/FOMC con `surprise ≠ 0` (actual ≠ consenso)  
**Assets**: EURUSD, SPY, GLD, TLT según tipo de dato  
**Timeframe**: 1-15 minutos  
**Edge**: la magnitud del movimiento es proporcional al tamaño de la sorpresa

### 3. Anomaly Breakout Scalp

**Trigger**: `AnomalyAlert` con clasificación `INSTITUTIONAL_ACCUMULATION` + confianza ML > 0.75  
**Assets**: activos líquidos con anomalía confirmada  
**Timeframe**: 30s - 5 minutos  
**Edge**: detección de acumulación institucional silenciosa antes del breakout visible

### 4. Post-Event Mean Reversion

**Trigger**: spike violento > 2.5σ en 60s + `AnomalyAlert` clasificado como `NEWS_OVERREACTION`  
**Lógica**: el mercado sobre-reacciona en los primeros 60s, vuelve al 50-62% del movimiento  
**Timeframe**: 5-30 minutos

---

## Métricas de Éxito

| Métrica | Objetivo Paper Trading | Objetivo Live |
|---|---|---|
| Sharpe Ratio | > 1.2 | > 1.5 |
| Drawdown mensual máximo | < 5% | < 3% |
| Win Rate | > 52% | > 55% |
| Profit Factor | > 1.4 | > 1.6 |
| Latencia evento → orden | < 90s | < 60s |
| Precisión sentimiento (dirección) | > 60% | > 65% |
| Falsos positivos anomalía | < 20% | < 10% |
| Uptime del sistema | > 99% | > 99.5% |
| Coste diario Fireworks.ai | < $5 | < $10 |

El sistema no debe pasar a live trading hasta cumplir 60 días consecutivos de paper trading con Sharpe > 1.5 y drawdown máximo < 5%.

---

## Seguridad y Gestión de Riesgos

### Protecciones técnicas

- **API keys** exclusivamente en variables de entorno, nunca en código
- **Kill switch**: comando `/close_all` en Telegram cancela todo en < 2s
- **Reconnect automático** con backoff exponencial para todos los WebSockets
- **Validación Pydantic** en todos los modelos: datos corruptos = excepción antes de llegar a Risk Agent
- **Budget cap Fireworks.ai**: configurable por día, activa modo degradado automáticamente
- **Append-only** para el log de decisiones en PostgreSQL

### Protecciones operativas

- Circuit breaker: pérdida diaria > 2.5% → modo close-only hasta siguiente sesión
- Flash crash detector: movimiento > 2% en < 60s → cancela todas las órdenes pendientes
- Pre-event protection: ventana ±3 minutos alrededor de eventos max-impact → tamaño máximo 40%
- Nunca operar: 5 minutos antes/después de vencimientos de opciones (OpEx) sin ajuste de config
- Paper trading siempre activo en paralelo como entorno de testing continuo

---

## Fases de Implementación

| Fase | Contenido | Validación |
|---|---|---|
| **Fase 0** — Infraestructura | Docker + modelos Pydantic + DB factories | `make up && make test-unit` pasa |
| **Fase 1** — Monitor + Telegram | 4 sub-monitores + bot Telegram | Detecta declaración Trump en < 30s |
| **Fase 2** — Análisis | Sentiment + Market Data + Anomaly | Backtests > 60% precisión dirección |
| **Fase 3** — Decisión | Strategy + Risk + Execution | R/R > 1.5:1 en todos los trades propuestos |
| **Fase 4** — Orquestación | LangGraph StateGraph completo | Flujo end-to-end evento → orden < 90s |
| **Fase 5** — Paper Trading | 60 días Alpaca paper | Sharpe > 1.5, drawdown < 5% |
| **Fase 6** — Live | Capital real mínimo (< $5K) | Métricas estables 30 días adicionales |

---

## Coste Operativo Estimado

| Componente | Coste mensual |
|---|---|
| VPS (4 vCPU, 8GB RAM, SSD) | ~$20-40 |
| Fireworks.ai LLM | ~$60-240 (según actividad) |
| Todas las APIs de datos | $0 |
| Telegram Bot | $0 |
| PostgreSQL + Redis + Qdrant | $0 (self-hosted) |
| LangSmith (tier gratuito) | $0 |
| **TOTAL** | **~$80-280/mes** |

---

## Licencia

Este proyecto es de uso privado. No distribuir sin autorización explícita.

---

*TeamTrade v1.0 — Sistema Multi-Agente IA para Trading Event-Driven*
