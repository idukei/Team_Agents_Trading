# train_anomaly_models.py — Guía Técnica Completa

**Archivo:** `scripts/train_anomaly_models.py`  
**Propósito:** Entrena offline los modelos de ML que el Anomaly Agent usa en producción para detectar comportamientos anómalos en el mercado.

---

## 1. Objetivo y Contexto

El sistema TeamTrade detecta anomalías de mercado con un **ensemble de 3 modelos** que operan en tiempo real. Dos de esos modelos —IsolationForest y LSTM Autoencoder— requieren entrenamiento previo sobre datos históricos. Este script es el proceso offline que los entrena, guarda los artefactos en disco, y los deja listos para que los singletons de producción los carguen al arrancar.

```
Flujo general:
TimescaleDB → fetch_ohlcv() → build_features() → train() → models/
                                                                 ├── isolation_forest.pkl
                                                                 ├── lstm_autoencoder.pt
                                                                 ├── lstm_threshold.npy
                                                                 ├── lstm_scaler_min.npy
                                                                 └── lstm_scaler_scale.npy
```

El tercer modelo del ensemble (Adaptive Z-score) es **estadístico puro** —sin parámetros aprendidos— por lo que no necesita entrenamiento offline.

---

## 2. Los 4 Features de Entrada

Todo el pipeline de anomalías se construye sobre **4 dimensiones** extraídas de datos OHLCV:

```python
def build_features(df: pd.DataFrame) -> np.ndarray:
    price_return  = close.pct_change()                          # retorno porcentual tick-a-tick
    volume_ratio  = volume / rolling_mean(volume, 20)          # volumen relativo a su media 20p
    spread_approx = (high - low) / close * 10_000              # spread implícito en bps
    obi_approx    = (close - open) / (high - low + ε)          # Order Book Imbalance proxy
```

| Feature | Rango típico | Detecta |
|---|---|---|
| `price_return` | -0.05 a +0.05 | Movimientos bruscos de precio |
| `volume_ratio` | 0.1 a 5.0 | Acumulación silenciosa o ventas masivas |
| `spread_approx` | 1 a 200 bps | Iliquidez, spreads anormales |
| `obi_approx` | -1.0 a +1.0 | Presión compradora/vendedora neta |

> **Nota OBI:** Cuando `close ≈ high`, OBI → +1 (presión compradora). Cuando `close ≈ low`, OBI → -1 (presión vendedora). Un OBI de ±0.7 en combinación con volumen 2× la media es señal clásica de acumulación institucional silenciosa.

---

## 3. Modelo 1: IsolationForest

### ¿Qué es?

IsolationForest es un algoritmo de detección de anomalías basado en árboles de decisión aleatorios. Su premisa central: **los puntos anómalos son más fáciles de aislar** que los normales. Un punto que requiere pocas particiones para quedar solo en su propio subespacio es anómalo.

### Entrenamiento

```python
async def train_isolation_forest() -> None:
    # 1. Fetch datos diarios de los primeros 5 activos monitorizados
    for asset in settings.monitored_equities[:5]:
        df = await fetch_ohlcv(asset, timeframe="1d", limit=500)   # ~2 años de velas diarias
        X = build_features(df)
        all_features.append(X)

    # 2. Combinar todos los activos en una matriz unificada
    X_combined = np.vstack(all_features)                           # shape: (N, 4)
    X_combined = X_combined[~np.any(np.isnan(X_combined), axis=1)]

    # 3. Entrenar con contamination=0.03 (asumimos 3% de datos históricos son anómalos)
    detector.fit(X_combined, contamination=0.03)
```

### Configuración del modelo

```python
IsolationForest(
    n_estimators=200,       # 200 árboles → estabilidad estadística
    contamination=0.03,     # 3% de outliers esperados en datos de entrenamiento
    random_state=42,        # reproducibilidad
    n_jobs=-1,              # paralelizar en todos los cores disponibles
)
```

| Parámetro | Valor | Razonamiento |
|---|---|---|
| `n_estimators` | 200 | Más árboles = menor varianza. 100 es default, 200 mejora estabilidad sin coste excesivo |
| `contamination` | 0.03 | Mercados normales tienen ~3% de comportamientos extremos legítimos (earnings, macro events) |
| `n_jobs=-1` | todos los cores | El entrenamiento es paralelizable por árbol |

### Persistencia y carga

```python
# Entrenamiento → guarda
joblib.dump(self._model, "models/isolation_forest.pkl")

# Arranque del sistema → carga
isolation_forest_detector = IsolationForestDetector()   # singleton, llama _load() en __init__
```

### Predicción en producción

```python
def predict(self, features: np.ndarray) -> tuple[bool, float]:
    raw_score = model.decision_function(features_2d)[0]
    prediction = model.predict(features_2d)[0]      # -1 = anomalía, +1 = normal

    # Normalización a confianza [0, 1]
    # Scores negativos → anomalía. Más negativo = más confianza
    confidence = max(0.0, min(1.0, -raw_score / 0.5 + 0.5))
    return prediction == -1, confidence
```

---

## 4. Modelo 2: Adaptive Z-score (sin entrenamiento offline)

El Z-score adaptativo es **completamente estadístico** y se calcula en tiempo real sobre la ventana de precios disponible. No requiere entrenamiento previo.

```
z_score = (precio_actual - media_ventana) / std_ventana
```

### Umbral dinámico basado en ATR

El umbral no es fijo —se adapta a la volatilidad del mercado:

```python
atr_pct = (atr / price) * 100

if atr_pct > 2.0:   → threshold = 4.0σ   # mercado muy volátil → señal más fuerte requerida
elif atr_pct > 1.0: → threshold = 3.5σ   # volatilidad moderada
else:               → threshold = 3.0σ   # mercado tranquilo → sensible
```

Esto evita **falsos positivos en días de alta volatilidad** (ej: post-FOMC, earnings), donde movimientos de 3σ son frecuentes y normales dado el contexto.

El Z-score también tiene una verificación independiente sobre **volumen**:

```python
def check_volume_anomaly(volumes, window=20):
    z_vol = (volumes[-1] - mean) / std
    return z_vol, z_vol > 2.5      # umbral fijo: 2.5σ para volumen
```

---

## 5. Modelo 3: LSTM Autoencoder

### ¿Qué es y por qué LSTM?

El LSTM Autoencoder detecta anomalías en **secuencias temporales**. A diferencia de IsolationForest que evalúa cada punto de forma independiente (multivariado pero no temporal), el LSTM aprende los **patrones dinámicos** del mercado: cómo evolucionan los 4 features a lo largo del tiempo.

Es especialmente potente para detectar **acumulación institucional silenciosa**: patrones donde el precio apenas se mueve pero el volumen y el OBI evolucionan de forma atípica durante 30 velas consecutivas.

### Arquitectura

```
Encoder: LSTM(input_dim=4, hidden_dim=64)
         input:  (batch, 30, 4)     → secuencia de 30 velas, 4 features
         output: hidden state (1, batch, 64)

Decoder: LSTM(input_dim=64, hidden_dim=4)
         input:  hidden state repetido 30 veces → (batch, 30, 64)
         output: (batch, 30, 4)     → reconstrucción de la secuencia
```

```python
class LSTMAEModel(nn.Module):
    def forward(self, x):
        # Comprimir toda la secuencia a un vector de contexto
        _, (h, c) = self.encoder(x)
        # Expandir el vector de contexto de vuelta a longitud 30
        h_repeat = h.squeeze(0).unsqueeze(1).repeat(1, x.shape[1], 1)
        out, _ = self.decoder(h_repeat)
        return out
```

El modelo aprende a **comprimir y reconstruir** secuencias normales. Cuando le presentas una secuencia anómala, el error de reconstrucción es alto porque el modelo no sabe cómo representarla en su espacio latente comprimido.

### Entrenamiento

```python
async def train_lstm_autoencoder() -> None:
    # 1. Datos horarios (más granulares que daily → más secuencias)
    df = await fetch_ohlcv(asset, timeframe="1h", limit=2000)   # ~83 días de 1h

    # 2. Normalización MinMax [0, 1] por feature
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Guardar parámetros del scaler para normalizar en producción
    np.save("models/lstm_scaler_min.npy", scaler.data_min_)
    np.save("models/lstm_scaler_scale.npy", scaler.scale_)

    # 3. Construcción de secuencias deslizantes
    # Para N=2000 puntos y SEQ_LEN=30: genera 1970 secuencias
    sequences = [X_scaled[i:i+30] for i in range(len(X_scaled) - 30)]

    # 4. Entrenamiento
    optimizer = Adam(lr=1e-3)
    criterion = MSELoss()
    for epoch in range(20):
        loss = criterion(model(batch), batch)    # minimizar error de reconstrucción
        loss.backward()
        optimizer.step()

    # 5. Umbral calibrado en el percentil 95 del error sobre datos de entrenamiento
    errors = [mean((batch - model(batch))**2) for batch in loader]
    threshold = np.percentile(errors, 95)
    np.save("models/lstm_threshold.npy", threshold)
```

### Por qué percentil 95 como umbral

El umbral no es arbitrario: se calcula sobre los **datos de entrenamiento normales**. Si el error de reconstrucción de una nueva secuencia supera el 95º percentil de los errores "normales", es porque la secuencia es suficientemente inusual como para que el modelo no la conozca bien.

```
Error de reconstrucción en producción:
    error = mean((x - reconstructed)²)
    is_anomaly = error > threshold_95p
    confidence = min(1.0, error / 0.1)
```

### Secuencias deslizantes

```
Datos: [t1, t2, t3, ... t2000]

Secuencia 1:  [t1  ... t30]    → ¿normal?
Secuencia 2:  [t2  ... t31]    → ¿normal?
...
Secuencia 1970: [t1971 ... t2000]

Total: 1970 secuencias de entrenamiento de 2000 velas horarias
```

---

## 6. El Ensemble: 2 de 3 votos

El `ensemble.py` coordina los 3 modelos con **votación mayoritaria**:

```python
VOTES_REQUIRED = 2      # al menos 2 de 3 modelos deben detectar anomalía

votes = 0
if isolation_forest.predict(features)[0]:   votes += 1
if adaptive_zscore.compute(prices)[1]:      votes += 1
if lstm_detector.predict(sequence)[0]:      votes += 1

is_anomaly = votes >= 2
confidence = mean([conf_if, conf_z, conf_lstm])
```

### ¿Por qué 2/3 y no unanimidad?

| Criterio | 3/3 (unanimidad) | 2/3 (mayoría) |
|---|---|---|
| Falsos positivos | Muy bajo | Bajo |
| Falsos negativos | Alto (perdemos señales reales) | Bajo |
| Robustez si un modelo no está cargado | Falla | Funciona con 2 |

El LSTM puede no estar cargado si PyTorch no está disponible (el script lo hace opcional). Con 2/3, el sistema funciona degradado pero operativo.

### Clasificación del tipo de anomalía

Tras confirmar que hay anomalía, se clasifica heurísticamente:

```python
def classify_anomaly_type(price_return, volume_ratio, obi, z_score):
    if volume_ratio > 2.0 and abs(price_return) < 0.003:
        return VOLUME_ACCUMULATION_SILENT    # volumen 2× pero precio quieto → institucional
    elif abs(price_return) > 0.01:
        return PRICE_SPIKE_SUDDEN            # movimiento >1% → spike
    elif abs(obi) > 0.7:
        return OBI_IMBALANCE_EXTREME         # desequilibrio extremo de book
    else:
        return MULTIVARIATE_OUTLIER          # combinación anómala sin patrón claro
```

### Severidad por confianza y votos

```python
def severity_from_confidence(confidence, votes):
    if confidence > 0.85 or votes == 3:  → "CRITICAL"
    elif confidence > 0.70 or votes == 2: → "HIGH"
    elif confidence > 0.50:               → "MEDIUM"
    else:                                 → "LOW"
```

---

## 7. Flujo Completo: Entrenamiento → Producción

```
ENTRENAMIENTO (offline, ejecutar una vez)
─────────────────────────────────────────
uv run python scripts/train_anomaly_models.py

    TimescaleDB
         │
         ▼
    fetch_ohlcv(asset, "1d", limit=500)     ← IsolationForest usa velas diarias
    fetch_ohlcv(asset, "1h", limit=2000)    ← LSTM usa velas horarias
         │
         ▼
    build_features() → (price_return, volume_ratio, spread_bps, obi)
         │
    ┌────┴──────────────────┐
    ▼                       ▼
IsolationForest         LSTM Autoencoder
fit(X, contamination=0.03)  train 20 epochs MSE
    │                       │  compute threshold@P95
    ▼                       ▼
models/isolation_forest.pkl  models/lstm_autoencoder.pt
                             models/lstm_threshold.npy
                             models/lstm_scaler_*.npy


PRODUCCIÓN (tiempo real, al arrancar el sistema)
────────────────────────────────────────────────
main.py → singletons se auto-cargan:
    isolation_forest_detector = IsolationForestDetector()   # carga .pkl
    lstm_detector = LSTMAutoencoder()                        # carga .pt + threshold
    adaptive_zscore = AdaptiveZScore()                       # sin carga, estadístico

Anomaly Agent en cada tick:
    detect_anomaly(asset, prices, volumes, spread_bps, obi, atr)
         │
    ┌────┼────────────────────┐
    ▼    ▼                    ▼
   IF   Z-score             LSTM
  vote   vote               vote
    └────┼────────────────────┘
         ▼
    votes >= 2 → AnomalyAlert → CTS → TradeProposal
```

---

## 8. Datos de Entrenamiento

| Modelo | Timeframe | Velas | Assets | Secuencias approx. |
|---|---|---|---|---|
| IsolationForest | 1d (diario) | 500 | 5 equities | 2500 puntos multivariados |
| LSTM Autoencoder | 1h (horario) | 2000 | 5 equities | ~9850 secuencias de 30 |

Solo se usan los **primeros 5 activos** de `settings.monitored_equities` para mantener el tiempo de entrenamiento manejable. El modelo aprendido es generalizable a todos los activos porque los features son **relativos** (ratios, retornos porcentuales) no absolutos.

---

## 9. Cómo Ejecutar

### Prerequisito: datos en TimescaleDB

El script necesita datos históricos. Cárgalos primero con:

```bash
make seed           # ejecuta scripts/seed_historical.py → descarga 2 años de OHLCV
make train          # ejecuta este script
```

O directamente:

```bash
# Con uv (recomendado)
uv run python scripts/train_anomaly_models.py

# Con Python directamente
cd teamtrade
python scripts/train_anomaly_models.py
```

### Salida esperada

```json
{"event": "training_isolation_forest", "level": "info"}
{"event": "asset_features", "asset": "AAPL", "rows": 498, "level": "debug"}
{"event": "isolation_forest_trained", "samples": 2490, "level": "info"}
{"event": "training_lstm_autoencoder", "level": "info"}
{"event": "lstm_epoch", "epoch": 0, "loss": 0.0842, "level": "info"}
{"event": "lstm_epoch", "epoch": 5, "loss": 0.0231, "level": "info"}
{"event": "lstm_epoch", "epoch": 10, "loss": 0.0198, "level": "info"}
{"event": "lstm_epoch", "epoch": 15, "loss": 0.0187, "level": "info"}
{"event": "lstm_autoencoder_trained", "threshold": 0.0412, "sequences": 9852, "level": "info"}
{"event": "anomaly_models_training_complete", "level": "info"}
```

### Reentrenamiento

Se recomienda reentrenar los modelos:
- **Mensualmente** en producción normal
- **Inmediatamente** tras cambios de régimen de mercado (ej: crisis, cambios estructurales post-election)
- **Tras añadir nuevos activos** al universo monitorizados

---

## 10. Dependencias del Módulo

```
train_anomaly_models.py
    ├── core.config.settings              # lista de activos, umbrales
    ├── core.db.postgres.get_pool         # conexión asyncpg
    ├── core.db.timescale.fetch_ohlcv     # query TimescaleDB → DataFrame
    ├── core.logging.configure_logging    # structlog JSON
    ├── agents.anomaly.isolation_forest   # IsolationForestDetector.fit()
    └── agents.anomaly.lstm_autoencoder   # build_lstm_autoencoder_model(), SEQ_LEN, INPUT_DIM
```

---

## 11. Consideraciones de Diseño

**¿Por qué entrenar en paralelo (`asyncio.gather`)?**  
IsolationForest y LSTM usan timeframes distintos (1d vs 1h) y son independientes. Ejecutarlos en paralelo reduce el tiempo total de entrenamiento.

**¿Por qué no usar datos de alta frecuencia (tick-by-tick)?**  
Los modelos de anomalía operan sobre velas agregadas por diseño. El Adaptive Z-score sí trabaja con ventanas de ticks en tiempo real, pero los modelos de ML se entrenan sobre OHLCV para estabilidad y reproducibilidad.

**¿Por qué no normalizar antes de IsolationForest?**  
IsolationForest es **invariante a escala** por diseño —las particiones aleatorias no dependen de la magnitud de los valores. Los features tienen rangos distintos (returns ≈ 0.01, volume_ratio ≈ 1.0, spread ≈ 50) pero esto no afecta la calidad del modelo.

**¿Por qué sí normalizar el LSTM?**  
Las LSTMs son **sensibles a la escala** de sus entradas. Sin normalización, features con mayor magnitud (spread en bps, ≈50) dominarían sobre features pequeños (price_return, ≈0.01), impidiendo que el modelo aprenda la dinámica conjunta.
