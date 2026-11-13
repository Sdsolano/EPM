# Reporte de integración fase 4 (4B/4C/4D/4E)

Rama `feat/ajustes-fase4`, worktree `EPM-wt-4`. Integración de los módulos nuevos en el router y el
pipeline, más tests de integración. Sin commit ni push. No se tocaron `data/raw`, `models` ni
`data/features`.

## Archivos

- `src/adjustments/router.py`
  - Importa `aprendizaje`, `automatico` y `eventos` (módulos, no funciones: los tests y la degradación
    parchean `router.eventos.eventos_de_semana`).
  - `RunRequest.aplicar_automatico: bool = False`.
  - `_compute`: `ctx.extras["aplicar_automatico"]`, `ctx.extras["modos_paso"]`,
    `ctx.extras["eventos"]` (una vez por request, fuera del threadpool del pipeline,
    `asyncio.wait_for(..., timeout=5)`, `except → {}`), y
    `meta.extras.{aplicar_automatico,modos_paso,eventos}`.
  - `meta.extras["modos_paso"]` viaja **siempre** (también con `aplicar_automatico=false`):
    `automatico.modos_paso(ucp)` es barato y no necesita store. R2 es una clasificación de revisión
    para el front; el backend aplica siempre los pasos (decisión sin usuario, ver LOG).
  - `_context_from_meta`: restaura el flag (default `False`), `modos_paso` (recalculado si la corrida
    vieja no lo trae) y reusa `extras["eventos"]`; solo consulta si falta (corridas viejas).
  - `_run_out`: filtra `eventos` de `meta.extras` igual que `festivos`; migra los records
    de corridas viejas de 5 pasos con `pipeline.migrar_records` (idempotente) para que el
    GET devuelva siempre 6 pasos.
  - `patch_step`: arma `by_paso` sobre `pipeline.migrar_records(saved["records"])`, así un
    PATCH de `especiales` sobre una corrida vieja funciona sin un PATCH previo.
  - Constantes: `HTTP_501`, `EVENTOS_TIMEOUT = 5`.
  - Docstring de cabecera con los endpoints nuevos y el uso de `meta.extras`.
- `src/adjustments/pipeline.py`
  - Importa `automatico`.
  - `migrar_records(records)`: público e idempotente (5 pasos → 6; 6 pasos → tal cual).
  - `_ejecutar`: si `estado != omitido`, `paso in ("nivel","forma")` y
    `ctx.extras.get("aplicar_automatico")`, añade `result.diagnostics["automatico"]` con los 7 días
    (`tipo_dia` vía `data.tipo_dia(..., ctx.festivos)` y `automatico.estado`). No toca curvas.
  - `_diagnostico_automatico`: cada día trae `{"tipo_dia", "estado", "paso"}` (contrato PLAN_4C).
  - `especiales` queda fuera de `_AUTOMATICO_PASOS`: no hay reglas R2 para él, así que no recibe
    `diagnostics["automatico"]` (solo `nivel`/`forma`).
- `src/adjustments/automatico.py`: `modos_paso(ucp)` (resumen por paso para el front).
- `src/adjustments/aprendizaje.py`: `min_casos` default 3; porcentajes ×100; `por_dia_semana` con el
  mismo `min_casos`.
- `src/adjustments/store.py`: `_limite` acotado a 1..500; `list_runs` de Postgres en 2 queries
  (`corrida` + `corrida_paso ... ANY`) sin N+1 ni `corrida_evento`; InMemory con copia superficial
  para lectura.
- `tests/test_adjustments_fase4_router.py` (nuevo, 24 tests): reusa las fixtures/fakes de
  `tests/test_adjustments_router.py` (`env`, `client`, `client_sin_store`, `store`, `_crear`, `_steps`).

## Endpoints nuevos (registrados antes de `GET /{run_id}`)

- `GET /adjustments/automatico?ucp=...` → `automatico.configuracion(ucp)`. Valida formato y existencia
  de la UCP (`_validar_ucp`: 422/404); no requiere store ni modelo.
- `GET /adjustments/aprendizaje?ucp=&desde=&hasta=&limite=&min_casos=` → `aprendizaje.a_dict(
  aprendizaje.resumir(runs, min_casos=...))`. Requiere store (503), 501 si el store no tiene
  `list_runs`, 422 si `desde > hasta`, `_validar_ucp` si llega `ucp`; `min_casos` default **3**;
  `list_runs`, `resumir` y `a_dict` corren en threadpool.

## Tests

- `tests/test_adjustments_fase4_router.py`: 24 passed.
- Suite completa:
  `.venv\Scripts\python.exe -m pytest tests -q --ignore=tests/test_hourly_disaggregation.py -p no:cacheprovider`
  → **639 passed, 5 skipped, 1 failed**. El único fallo es el preexistente
  `tests/test_holiday_homology.py::TestHolidayProfileSelection::test_christmas_profile_uses_prior_christmases`
  (`'Navidad' != 'Christmas Day'`), ajeno a esta integración. `test_hourly_disaggregation.py`
  no colecta (imports relativos, preexistente).
- `python -c "import src.api.main"` → OK.
- Postgres: `AJUSTES_DATABASE_URL=postgresql:///postgres` no está disponible en este entorno
  (`password authentication failed for user "samue"`), así que los 4 tests Postgres no se pudieron
  ejecutar (no es un fallo del código; el fixture conecta antes de los tests).
- `git status --short`: solo cambios en `src/adjustments/*`, `tests/*`, `src/api/main.py`, `.env.example`
  y documentos; sin cambios en `data/raw`, `models` ni `data/features`.

## Cobertura de los tests de integración

- `/automatico`: 200 con UCP real, 404 desconocida, 422 patrón inválido, 401 con
  `AJUSTES_SERVICE_TOKEN`, 200 sin store (no lo captura `/{run_id}`).
- Eventos: alerta `EVENTOS_SEMANA` en `control` con el detalle, `eventos` ausente de
  `meta.extras`, fallo y `TimeoutError` → corrida 200 sin alerta, y reuso en PATCH (no reconsulta).
- `aplicar_automatico=true`: `diagnostics.automatico` por día (con `paso`) en `nivel`/`forma`,
  `final_curves` idénticas a la corrida con `false`, y el flag se conserva en PATCH.
- `modos_paso` en `meta.extras` con y sin el flag, y conservado en PATCH.
- `/aprendizaje`: 200 con 2 corridas y una edición (grupo nivel, `recurrente` según `min_casos`,
  métricas ×100), default `min_casos=3`, `min_casos=1` marca recurrente, filtros de fechas,
  503 sin store, 501 con store sin `list_runs`, 422 `desde>hasta`, 404 UCP inexistente.

## Problemas / notas

- Postgres no verificable localmente (autenticación); pendiente correr con un DSN válido. El test
  `test_postgres_store` ya ejercita `list_runs` (2 queries, límite 500, pasos agrupados).
- Falla preexistente de `test_holiday_homology` (christmas), no relacionada.
- `InMemoryStore.list_runs` devuelve copia superficial de los records (sin clon JSON): el consumidor
  (`aprendizaje`) es de solo lectura y no debe mutarlos.
- `especiales` no recibe `diagnostics.automatico`: el backtest R2 solo clasifica `nivel`/`forma`.
- Limitaciones aceptadas (ronda 2): el clima observado no se persiste (se relee cacheado);
  `aprendizaje` agrega corridas en cualquier estado (`borrador`/`aprobada`/`reemplazada`) sin
  filtrarlas, y sus ediciones de todos los pasos; R2 solo es clasificación de revisión para el
  front (no activa pasos).
