# Modelo de nivel para /predict-daily

Código: `src/models/daily_level_model.py` (entrenamiento/predicción), integración en
`src/api/main.py` (`train_model_if_needed(variant='daily')`) y `src/prediction/forecaster.py`
(`_predict_next_n_days_daily_level`).

## Idea
Los árboles sobre el TOTAL crudo no extrapolan el nivel (la demanda crece ~12% interanual). El modelo
predice el **ratio `TOTAL_t / ancla(origen)`**, donde el ancla es el promedio de los últimos
`ANCHOR_DAYS` = 21 días sin festivos ni navidad (23-dic a 6-ene) al origen; demanda = ratio * ancla.
(La ventana de 21 días resultó más reactiva al nivel en un mercado en crecimiento que la de 28 días:
MAPE del backtest 2.88% vs 3.05%.) Features: calendario (día de semana, mes, festivo,
día después/antes de festivo, puente, navidad, Semana Santa), clima del día (temperatura
min/media/max, sensación térmica, viento, lluvia, nubosidad), horizonte `h` y `m7`/`m14`
(media 7/14 días sobre el ancla). Ensamble simple de LightGBM + ExtraTrees + Ridge.
LightGBM entrena con pérdida **L1 (MAE)**: el objetivo es el error porcentual, da un sesgo casi nulo
y ~0.03pp menos de MAPE que L2 (validado en el backtest de 7 y de 23 cortes).

## Artefactos (`models/{ucp}/`)
- `trained_diario/daily_level.joblib`: payload `{model, feature_names, model_kind='daily_level_ratio_v1', trained_until, metrics}`.
- `registry/champion_model_diario.joblib`: copia usada por el forecaster.
- `registry/champion_model_diario.meta.json`: sidecar (model_kind, trained_until, MAPE de validación, tamaño) para leer el tipo sin `joblib.load`.
- `registry/champion_model_diario.legacy.joblib`: respaldo del champion legacy previo (se crea una sola vez al migrar).
- Escrituras atómicas (temporal + `os.replace`) y un lock por UCP para migración/entrenamiento.

## Migración automática
Un champion diario sin `model_kind` (árboles legacy) con >=120 filas se reemplaza UNA vez por el modelo de
nivel. Si la migración falla, o el champion no se puede leer, se conserva tal cual (sin reentrenos en bucle).
Con <120 filas, o si el primer entrenamiento falla, se usa el camino legacy.

## Perillas de negocio (`forecaster.py`)
- `DAILY_HOLIDAY_BLEND`: mezcla ponderada con el valor de hace 1 año (0.70 días muy especiales; 0.60 festivos de
  navidad/Semana Santa y fines de semana navideños), escalado por el crecimiento interanual `yoy_growth_factor`
  (clip [0.8, 1.3]); en Semana Santa se alinea al mismo día litúrgico.
- `DAILY_BLEND_ORDINARY_FESTIVOS` (False): mezclar también festivos ordinarios (en backtest empeora: MAPE 3.29 vs 3.20 y festivos 7.9 vs 6.4). Se deja como perilla de negocio.

## Operación
- **Sin política de reentrenamiento automático**: solo `force_retrain=true` reentrena; el ancla se recalcula
  en vivo en cada predicción con la historia disponible, así que el nivel sigue la demanda sin reentrenar.
- Un champion de nivel vigente se devuelve sin tomar el lock. El lock (por UCP) es solo de proceso: asume un
  worker; con varios workers las escrituras atómicas evitan archivos corruptos, pero pueden entrenar en paralelo.
- **Backoff**: si la migración o el primer entrenamiento fallan, no se reintenta antes de 10 minutos
  (`_DAILY_RETRY_INTERVAL_S`, en memoria por UCP) salvo `force_retrain`; mientras tanto se usa el champion legacy
  (o el entrenamiento legacy). Un `force_retrain` fallido con champion de nivel vigente lo conserva.
- **Champion ilegible**: se copia (sin cargarlo) a `champion_model_diario.corrupt.joblib` (si no hay respaldo) y se
  reentrena bajo el lock y el mismo backoff. Si el reentrenamiento falla, `/predict-daily` responde 500 con
  "champion ilegible, reintente con force_retrain=true" hasta que pase el backoff o se use `force_retrain`.
- El fallback legacy del diario (<120 filas, excepción o backoff) toma el mismo lock por UCP.
- Cada request usa un archivo temporal único en `data/features/{ucp}/` y lo borra siempre.
- Entrenar requiere ~1-2 GB libres (LightGBM + ExtraTrees con n_jobs<=4); ante un MemoryError el entrenamiento
  cae al backoff/legacy como cualquier otra excepción.
- `metricas_modelo` incluye `columnas_clima`, `cobertura_clima` (promedio), `cobertura_clima_por_columna`,
  `columnas_clima_descartadas` y `clima_descartado` (todo el clima). La cobertura se mide POR COLUMNA sobre los días
  de entrenamiento: las columnas con cobertura <50% se descartan solas; si cae `tmean` se descarta todo el clima.
  `columnas_clima` también queda en el sidecar `.meta.json`.
- **Compatibilidad**: `comparacion_modelos` ahora trae los miembros `lightgbm`, `extratrees` y `ridge` (antes
  `xgboost`, `lightgbm`, `randomforest`) y `modelo_seleccionado` es `daily_level_ensemble`; los clientes que lean
  esas claves deben tolerarlo.

## Requisitos
- `data/raw/{ucp}/clima_new.csv` (opcional: sin clima, o con cobertura <50% de los días de entrenamiento, se entrena solo con calendario).
- >=120 días de demanda válida (> 0).
- Clima real hasta h=12; para h mayor (y días sin dato) se usa la climatología mensual, igual en train y predict.
- Humedad corrupta (`p_h == p_t`) descartada: solo filas y periodos confiables.

## Limitaciones
- El MAPE de validación reportado no incluye la mezcla de días especiales (se aplica después, en el forecaster).
- El clima de entrenamiento es el observado; en producción los días futuros usan pronóstico (sesgo leve).
- Para h > 30 (n_days > 30) el horizonte se satura en 30 y el ancla queda congelada en el origen: sin tendencia más allá del día 30.
- La mezcla de navidad no está validada con menos de 1 año de datos.
- Los festivos ordinarios no se mezclan con el año anterior.

## Backtest (Atlantico Norte, 7 cortes de 30 días)
MAPE 2.84% con clima conocido 10 días después del corte y 2.82% con 30 días (código original: 4.70%),
con sesgo casi nulo. Jueves/Viernes Santo 2026: 3.1% y 2.8% de error.
En un backtest ampliado (23 cortes cada 20 días desde 2025-05, 690 obs) el ancla de 21 días + pérdida L1
dan 4.11% (28 días + L2: 4.24%). El diseño de ventana de ancla y de pérdida se validó en AMBOS backtests
para evitar el sobreajuste que mostraban otras variantes (pesos NNLS, recencia, etc.).
