# Fase ASR — comparación ASR actual vs mesa de ajustes vs real (sub-plan)

Worktree `EPM-wt-asr`, rama `feat/ajustes-asr-comparacion`. Objetivo: preparar la revisión con el analista
del viernes 2026-11-13 (REQUERIMIENTOS.md R1): reproducir el ASR de Sphere en Python, compararlo con la mesa
y con el real en ~52 semanas por UCP, y diagnosticar con números por qué falla.

## Hallazgos de lectura del front (Sphere `Pronosticos.tsx`, rama `test`, commit 57eb4fb)
- `handleASR` (~l. 5174): por hora `desvi1 = pred − h1`, `desvi2 = h2 − h3`, `delta = Math.round(desvi2 − desvi1)`;
  payload `{ucp, fecha, tipo_dia: "laboral", predicciones_actuales: pred.toFixed(1), ajustes_solicitados: delta}`.
  Objetivo antes del validador = `pred + delta ≈ h1 + (h2 − h3)`: **no depende del pronóstico base**.
- `historicosCombinados` (h1, h2, h3) = `selectedHistoricosOrdenados`, que se reordena **por fecha descendente**
  en cada clic (h1 = el más reciente seleccionado).
- Pero `desvi2` normalmente sale de `valoresDesviDeltaOriginales.desvi2`, **congelado** (toFixed(1)) cuando
  hay ≥ 3 seleccionados en modo `diario_ajustado`, con otro orden: `[...selectedHistoricos (orden de clic),
  ...selectedHistoricosPronosticos]`. Si el orden de clic ≠ orden por fecha, desvi2 mezcla otros días.
  Además no se recalcula si cambia la selección (solo se limpia al salir del modo).
- El gráfico de áreas del modo ajustado etiqueta H1/H2/H3 en orden de clic (≠ el h1 que usa el ASR).
- La lista de históricos se muestra por fecha descendente; el usuario puede cargar rangos arbitrarios
  (con filtros de día de semana/festivo) y también "históricos pronóstico" (pronósticos pasados).
- Endpoint `/validate-hourly-adjustments`: **ignora `tipo_dia`**; elige cluster por `is_special_day`
  (festivo por API o 24-dic/2-ene) → perfil especial por mm-dd o promedio de festivos; si no, el cluster
  moda del día de semana (KMeans entrenado con los últimos 3 meses). Tolerancia ±15 % por periodo sobre
  la proporción del cluster, 3 iteraciones, no conserva energía.

## Entregables
1. `src/adjustments/asr.py`: reproducción fiel (redondeos JS `Math.round`, `toFixed(1)`, desvi2 congelado
   según orden de clic) + `senda_cluster`/`validar` con la misma lógica del endpoint; variantes:
   sin validador, `tipo_dia` (se documenta que no afecta), selección de h1/h2/h3 por escenario.
2. `src/adjustments/comparar_asr.py`: núcleo del comparador (una semana: base, ASR, mesa, real y,
   opcional, ajuste del analista; métricas por día). `scripts/comparar_asr.py`: modo semana y modo
   histórico (N semanas por UCP, bases `ingenuo`/`archivo`; `modelo` enchufable). `POST
   /adjustments/comparar-asr` en `src/adjustments/router_asr.py` (+ `include_router` en main.py).
3. Tests: `tests/test_asr.py` (redondeos, orden, endpoint vía TestClient con motor falso y con modelos
   reales si existen), `tests/test_comparar_asr.py` (comparador y endpoint con datos sintéticos).
4. Corrida histórica y `reports/ajustes/asr_comparacion_20261112.md` (CSV fuera de git) + `REPORTE.md`.

## Escenarios de selección de históricos (d = día objetivo, corte = sábado anterior)
- `anual` (supuesto principal, coherente con "escenario de referencia" y con el orden por fecha):
  h1 = último mismo día de semana real (ancla con el desfase del corte), h2 = día equivalente del año
  anterior (d − 364), h3 = equivalente del ancla (h1 − 364). Clic de arriba abajo (fecha desc).
- `anual_alineado`: igual pero con el desplazamiento de la semana de referencia de la mesa (festivo
  homólogo) en vez de −364 fijo. Mide el efecto de la alineación por festivo.
- `anual_desfase`: el analista asume h1 = d − 7 (h3 = h2 − 7) aunque el ancla real sea d − 14
  (domingo). Mide el efecto del desfase.
- `anual_clic_asc`: misma selección que `anual` pero clic en orden ascendente → desvi2 congelado = h2 − h1
  → objetivo ≈ h2 (el día del año pasado crudo).
- `recientes`: h1, h2, h3 = los 3 últimos días reales del mismo día de semana (modo "similares").
Para cada uno: con y sin validador. Descomposición nivel/forma del ASR (energía del ASR con forma base y
viceversa) para probar "mezcla nivel y forma".

## Métricas
WAPE horario, energía diaria (MAPE, sesgo), energía semanal (MAPE, sesgo), pico (|err %| y hora),
error de forma = 100·Σ|p/Σp − r/Σr| (WAPE si la energía fuera perfecta). Por UCP y tipo de día.

## Límites
No tocar `nivel.py`, `forma.py`, `router.py`, `feature_engineering.py`, `backtest_adjustments.py`
(otras fases). Sin red en el backtest (festivos de librería; `is_special_day` sin API). No escribir
data/raw, models, data/features.
