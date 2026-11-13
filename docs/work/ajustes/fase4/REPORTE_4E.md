# Reporte 4E — aprendizaje de intervenciones (F09)

## Alcance
Módulo de diagnóstico de **solo lectura** que agrega las ediciones del analista comparando la curva
calculada (`curvas_calculadas`) contra la editada (`curves`) por UCP × paso × tipo de día, más
`Store.list_runs` para leer corridas históricas. No se tocó `router.py`, ni escrituras/esquema SQL,
ni `data/raw`, `models` ni `data/features`. El endpoint `GET /adjustments/aprendizaje` lo cablea el
agente de integración.

## Archivos
- `src/adjustments/aprendizaje.py` (nuevo).
- `src/adjustments/store.py`: método `list_runs` en el `Store` Protocol y en `InMemoryStore` y
  `PostgresStore` (solo lectura); helpers `_fecha_o_none`/`_limite`. Sin cambios en escrituras.
- `tests/test_adjustments_aprendizaje.py` (nuevo).
- `docs/work/ajustes/fase4/REPORTE_4E.md` (este archivo).

## API pública
### `aprendizaje.resumir(runs, festivos_por_run=None, min_casos=3)`
- `runs`: iterable de `{"meta": {...}, "records": [StepRecord | dict, ...]}` (formato `store.get_run` /
  `store.list_runs`).
- `festivos_por_run`: opcional; callback `fn(meta) -> [fechas ISO]` o mapping `run_id -> [fechas]`.
  Si se omite se usan `meta.extras.festivos` (los que persiste el pipeline).
- Por cada record `estado == "editado"`: toma `result.diagnostics.fechas_editadas` (fallback: claves
  presentes en `curvas_calculadas` ∩ `curves`), y por fecha calcula
  `delta_pct = energia(editada)/energia(calculada) − 1` (energía = suma de los 24 periodos).
- Devuelve:
  - `grupos`: lista ordenada por `(ucp, paso, tipo_dia)` con `ucp`, `paso`, `tipo_dia`, `n_casos`,
    `n_dias` (fechas distintas), `n_corridas` (runs distintos), `delta_pct_medio`,
    `delta_pct_mediano`, `pct_positivo`, `p50_abs`, `p90_abs`, `fechas` (top 10 por `|delta|`,
    desempate por fecha) y `recurrente` (`n_dias >= min_casos`; default 3, grupos con menos días
    quedan `false`).
  - `resumen_global`: mismas métricas sobre todos los casos (sin `fechas`).
  - `por_dia_semana`: dict `lunes..domingo` (solo días con casos) con las mismas métricas y el mismo
    `min_casos`/`recurrente`.
  - Los porcentajes (`delta_pct_*`, `p50_abs`, `p90_abs`, `pct_positivo`) se devuelven **×100**
    (0.10 → 10.0; `pct_positivo` 0..100), coherentes con el resto de la API.
  - `omitidos_sin_calculada`: nº de fechas editadas que no se pudieron comparar por falta de
    `curvas_calculadas` (records viejos).
  - `min_casos`: valor efectivo usado.
- Nunca lanza por records viejos o valores no finitos: los días inválidos se omiten.
- Flotantes redondeados a 6 decimales; `None` cuando no hay casos (JSON-safe vía `to_jsonable`).
- `a_dict(resumen) -> Any`: envuelve `to_jsonable` para la respuesta del endpoint.

### `Store.list_runs(ucp=None, desde=None, hasta=None, limite=200) -> list[{"meta","records"}]`
- Más reciente primero por `semana_inicio DESC` (desempate `created_at DESC, id DESC`).
- `desde`/`hasta` son inclusivos sobre `semana_inicio`; `limite` se acota a **1..500** (default 200;
  no acepta `1e9`).
- `InMemoryStore`: filtra el dict interno bajo lock y devuelve los records en copia superficial
  (`copy.copy`; el consumidor es de solo lectura) para evitar el clon JSON.
- `PostgresStore`: **2 queries** — corridas con filtros/orden/límite y todos sus pasos con
  `corrida_id = ANY(%s::uuid[])` agrupados en Python; no lee `corrida_evento`.
- Si un store no implementa `list_runs`, el endpoint de integración debe responder 501 (queda en
  manos del agente de integración).

## Tests
`tests/test_adjustments_aprendizaje.py` (sin red ni base):
- Edición en nivel (100→110) y en forma (100→90): medias/medianas, `pct_positivo`, `p50_abs`,
  agrupación por `(paso, tipo_dia)`, `resumen_global` y `por_dia_semana`.
- `tipo_dia` festivo usando `meta.extras.festivos`.
- Records sin `curvas_calculadas` → `omitidos_sin_calculada` sin excepción; record editado sin
  fechas ni calculada no rompe.
- `min_casos`/`recurrente`, `n_dias`, `n_corridas` y métricas con varios días/corridas.
- `fechas` top 10 por `|delta|`.
- `a_dict` con `json.dumps(..., allow_nan=False)`.
- `InMemoryStore.list_runs`: orden desc, filtros `ucp`/`desde`/`hasta` (inclusivos), `limite`, ucp
  inexistente, ausencia de `eventos` y aislamiento de mutaciones.

Verificación ejecutada:
```
.venv\Scripts\python.exe -m pytest tests/test_adjustments_aprendizaje.py tests/test_adjustments_router.py -q -p no:cacheprovider
# 73 passed, 4 skipped (Postgres sin AJUSTES_DATABASE_URL)
.venv\Scripts\python.exe -c "import src.api.main"
# import ok (solo warning preexistente de pydantic schema_extra)
```

## Decisiones y riesgos
- `omitidos_sin_calculada` cuenta **fechas** editadas sin calculada (un record viejo con 2 fechas
  suma 2), no records; es la granularidad de los demás agregados.
- `fechas` se expone como lista de strings ISO (top 10 por magnitud de la intervención), sin
  duplicados.
- Días con energía calculada 0, curvas no finitas o fecha no parseable se descartan en silencio
  (no son "sin calculada"); no afectan los agregados.
- `list_runs` no devuelve `eventos` para mantener el contrato pedido `{"meta", "records"}`.
- En Postgres el filtro se compara con objetos `date` (adaptación psycopg2), no con texto.
- `list_runs` de Postgres usa 2 queries (sin N+1) y no toca el esquema; el límite máximo de 500
  acota el costo. Los records de `InMemoryStore.list_runs` comparten estructuras anidadas con el
  store: tratarlos como inmutables.
- Riesgo de integración: el nombre de las claves del resumen (`grupos`, `resumen_global`,
  `por_dia_semana`, `omitidos_sin_calculada`, `min_casos`) no estaba fijado por el plan; el agente
  del endpoint debe devolver el dict tal cual.
