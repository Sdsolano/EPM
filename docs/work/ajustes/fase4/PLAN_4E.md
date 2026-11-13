# Sub-plan 4E — aprendizaje de intervenciones (F09)

## Objetivo
Diagnóstico de **solo lectura** de las intervenciones recurrentes del analista: comparar, por día
editado, la curva **calculada** (`curvas_calculadas`, ya persistida desde R1-C) contra la **editada**
(`curves`), agrupado por UCP × tipo de día × paso. Sin reentrenar ni cambiar reglas automáticamente.

## Módulo nuevo `src/adjustments/aprendizaje.py`
- `resumir(runs, festivos_por_run=None, min_casos=3)`:
  - Entrada: iterable de `{"meta": {...}, "records": [StepRecord,...]}` (formato de `store.get_run`).
  - Por cada record con `estado == "editado"` y `fechas_editadas`: para cada fecha, calcular
    `energia_calculada` (de `curvas_calculadas[f]`) y `energia_editada` (de `curves[f]`);
    `delta_pct = editada/calculada − 1`; `tipo_dia` con `data.tipo_dia(f, festivos)` (festivos del
    run o `meta.extras.festivos`), `dia_semana`, `paso`.
  - Agrupa por `(ucp, paso, tipo_dia)` y devuelve: `n_dias`, `n_corridas`, `delta_pct_medio`,
    `delta_pct_mediano`, `pct_positivo`, `p50_abs`, `p90_abs`, `fechas` (top 10), `recurrente`
    (`n_dias ≥ min_casos`; grupos con menos días quedan `false`). Incluye `resumen_global` y
    `por_dia_semana`, ambos con el mismo `min_casos`.
  - Los porcentajes se devuelven **×100** (0.10 → 10.0; `pct_positivo` de 0 a 100), coherentes con
    el resto de la API.
  - Nunca lanza por registros viejos sin `curvas_calculadas` (los ignora y los cuenta en
    `omitidos_sin_calculada`).
- `a_dict(...)` JSON-safe (`to_jsonable`).

## Store (solo lectura)
- Añadir `list_runs(ucp: Optional[str], desde: Optional[date], hasta: Optional[date], limite: int)`
  → lista de `{"meta", "records"}` (más reciente primero) al `Store` Protocol.
  - `InMemoryStore`: filtra el dict interno por ucp/semana y ordena; records en copia superficial
    (consumidor de solo lectura, sin clon JSON).
  - `PostgresStore`: 2 queries — corridas con filtros/orden/límite y todos sus pasos con
    `corrida_id = ANY(%s)` agrupados en Python; **no** lee `corrida_evento`.
  - Límite acotado a 1..500 en el store (no acepta `1e9`).
  - Si el store no implementa `list_runs`, el endpoint responde 501.

## Endpoint `GET /adjustments/aprendizaje`
- Query: `ucp` (pattern UCP), `desde`, `hasta` (ISO, opcionales), `limite` (1..500, default 200),
  `min_casos` (1..50, default 3). `list_runs`, `resumir` y `a_dict` corren en threadpool (no
  bloquean el event loop).
- Requiere store (`_require_store`); token igual que el resto. Devuelve el dict de `resumir`.
- 404 si la UCP no existe (usa `_validar_ucp`), 422 fechas inválidas.

## Tests (`tests/test_adjustments_aprendizaje.py`)
- Corrida sintética con un día editado en nivel (calculada 100, editada 110) y otro en forma
  (calculada 100, editada 90): medias/medianas, signos y agrupación por UCP/tipo/paso.
- Registros sin `curvas_calculadas` → `omitidos_sin_calculada`, sin excepción.
- `min_casos` marca `recurrente` (default 3; también en `por_dia_semana`).
- Porcentajes ×100 y límite del store acotado a 500.
- `InMemoryStore.list_runs` con 2 corridas: filtros ucp/desde/hasta/limite y orden.
- Endpoint con `InMemoryStore` (upload de 2 corridas vía `create_run`) → 200 y claves esperadas;
  sin store → 503; store sin `list_runs` → 501; `desde>hasta` → 422.
- Postgres: test opcional con la base temporal existente del fixture (skip sin DSN).

## Verificación
Suite completa + `import src.api.main` + router con Postgres si está disponible. Documentar en
`docs/work/ajustes/fase4/LOG.md`.
