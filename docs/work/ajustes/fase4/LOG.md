# Log Fase 4 + Paso 0 + RESUMEN_NOCHE (sesión 2026-11-13)

## Estado actual (inicio)
- main = `ded1df6` (ASR, 2a, 2b integradas). Fase 3 en Sphere `test` = `22ef6d1`.
- Worktree `../EPM-wt-4`, rama `feat/ajustes-fase4`, base main; datos/modelos copiados (sin red).
- **4A (R1-C) ya está implementado en main**: `StepResult.curvas_calculadas`, `_aplicar_edicion`,
  persistencia (`store.py`, `002_versionado.sql`) y tests existen. No requiere trabajo nuevo;
  se valida con la suite y se documenta como cumplido.
- Pendiente: Paso 0 (re-backtest con base `modelo_k1`) en el repo principal; 4B, 4C, 4D, 4E en el
  worktree; ronda de revisión multi-ángulo + regresión; RESUMEN_NOCHE.md.

## Decisiones tomadas sin el usuario
- 4B: el paso `especiales` se implementa con `aplicar=false` por defecto (sugerencias visibles,
  curvas idénticas). El candidato `alpha=0.5` + `ancla="analogos"` queda **refutado** por el
  re-backtest `modelo_k1`; el default sigue `aplicar=false, alpha=0.25, ancla="actual"` y, si se
  activa `aplicar=true`, se emite el warning `ESPECIALES_SIN_BACKTEST` (no bloquea).
- 4C: `aplicar_automatico` es un flag de request apagado por defecto; la configuración R2 se expone
  por endpoint y solo añade diagnósticos cuando se activa (no cambia curvas).
- 4D: cache de eventos en `.cache/eventos_ajustes` (gitignored), TTL 7 días; sin key/fallo degrada
  en silencio. La consulta se hace una vez por (UCP, semana) fuera del pipeline.
- 4E: se añade `Store.list_runs` de solo lectura para agregar intervenciones; sin cambiar escrituras.

## Archivos previstos
- `src/adjustments/steps/especiales.py` (nuevo), `pipeline.py`, `steps/__init__.py`.
- `src/adjustments/automatico.py`, `src/adjustments/eventos.py`, `src/adjustments/aprendizaje.py`.
- `steps/control.py`, `router.py`, `store.py`, `src/api/main.py`, `.env.example`.
- `tests/test_adjustments_especiales.py`, `_automatico.py`, `_eventos.py`, `_aprendizaje.py`.
- `scripts/backtest_adjustments.py`, `scripts/backtest_analogos.py`, `scripts/comparar_asr.py`
  (Paso 0, repo principal).

## Progreso real
- [x] Paso 0 (commit local `6ad685d` en main; reportes `*_20261113_modelo.md`)
- [x] 4B paso `especiales` (default sin efecto) + migración 5→6 pasos
- [x] 4C módulo `automatico.py` (config R2, off por defecto)
- [x] 4D `eventos.py` (cache/TTL/degradación) + alertas en control + delegación en main.py
- [x] 4E `aprendizaje.py` + `Store.list_runs` (solo lectura)
- [x] Integración router (endpoints `/automatico`, `/aprendizaje`, `aplicar_automatico`, eventos en `_compute`)
- [x] Corrección de hallazgos 4C/4E y contrato (modos_paso, unidades, min_casos, Postgres)
- [ ] Ronda 1 de revisión (8 ángulos + regresión)
- [ ] RESUMEN_NOCHE.md, commit y push

## Verificación actual (pre-revisión)
- Suite worktree: 595 passed, 5 skipped, 1 failed (christmas preexistente). Import API OK.
- Postgres local no autenticó (`password authentication failed for user "samue"`); tests de Postgres
  se saltan. Anotado como pendiente de verificar en el entorno de despliegue.
- `git status` del worktree: solo los 8 archivos modificados y 9 nuevos de la feature; datos intactos.

## Archivos tocados (sin commitear en el worktree)
`src/adjustments/{pipeline,router,store}.py`, `src/adjustments/steps/{control,especiales}.py`,
`src/api/main.py`, `src/adjustments/{automatico,eventos,aprendizaje}.py`, `.env.example`,
`tests/test_adjustments_{especiales,automatico,eventos,aprendizaje,fase4_router}.py`,
`tests/test_adjustments_{pipeline,analogos_pasos}.py` (ajuste de orden de pasos).

## Decisiones de la corrección 4C/4E (2026-11-13)

- **4C — R2 no activa nada (altitud).** El backend aplica siempre los pasos del pipeline; R2
  `"automatico"` es una *clasificación de revisión para el front*, no una instrucción de ejecución.
  `aplicar_automatico` solo añade `diagnostics["automatico"]` por día (con `paso`) y **no altera
  curvas**. Se añade `automatico.modos_paso(ucp)` (`{paso: "automatico"|"sugerido"}`; un paso es
  automático solo si todos los tipos con regla lo son) y el router lo expone **siempre** en
  `meta.extras["modos_paso"]` (sin store). `_context_from_meta` lo restaura para PATCH. Decisión
  tomada sin el usuario y anotada en PLAN_4C/REPORTE_4C/REPORTE_INTEGRACION.
- **4E — `min_casos` default 3.** Antes 1 marcaba `recurrente` con un solo día. Ahora el default
  (módulo, endpoint y docs) es 3 y también se aplica a `por_dia_semana`; los grupos con menos de
  `min_casos` quedan `recurrente=false`.
- **4E — unidades ×100.** `delta_pct_medio`, `delta_pct_mediano`, `p50_abs`, `p90_abs` y
  `pct_positivo` salen en porcentaje (0.10 → 10.0; `pct_positivo` 0..100), como el resto de la API.
- **4E — event loop.** `list_runs`, `resumir` y `a_dict` corren en `run_in_threadpool`.
- **Postgres `list_runs` — 2 queries.** Corridas con filtros/orden/límite + pasos con
  `corrida_id = ANY(%s::uuid[])` agrupados en Python; sin N+1 ni `corrida_evento`.
  `InMemoryStore` devuelve copia superficial (consumidor de solo lectura, sin clon JSON).
- **Store `_limite`.** Se acota a 1..500 también en la validación del store (no acepta `1e9`).
- **Docstrings de 5 pasos.** Otro agente ya actualizó `__init__.py`/`types.py`; solo queda la mención
  legítima a la migración de corridas viejas de 5 pasos en `pipeline._migrar_records_5_pasos`.

## Correcciones de la ronda 2 (ALTITUDE/MEDIO/BAJO)

- R2 (`modos_paso`): un paso es `automatico` solo si **los 4 tipos** lo son; cualquier tipo sin
  regla cuenta como `sugerido`. Atlántico/nivel y Antioquia/forma pasan a `sugerido`.
- Eventos: `consultar_openai` lanza `EventosNoDisponibles` ante falta de `API_KEY`/fallo; un fallo
  no sobrescribe la cache buena (se devuelve la vieja) y un `{}` exitoso sí se cachea corto.
- Clima: cache LRU de hasta 4 entradas por `(ruta, mtime_ns, tamaño)` sin `clear()` global.
- OpenAI: cliente con `timeout=5, max_retries=0`.
- Migración 5→6: `pipeline.migrar_records` es público e idempotente y lo usan `_run_out` (GET) y
  `patch_step` (`by_paso`), así el GET de una corrida vieja devuelve 6 pasos y el PATCH de
  `especiales` funciona sin otro PATCH.
- R2 no incluye `especiales`: sale de `_AUTOMATICO_PASOS` y del docstring de
  `RunRequest.aplicar_automatico`.
- `especiales` con `aplicar=true` emite `ESPECIALES_SIN_BACKTEST` (una vez, warning) y con
  `analogos`/`filtros_analogos` + `ancla != "analogos"` emite `PARAMS_ANALOGOS_IGNORADOS` (info).
- Docs: EVENTOS_TIMEOUT=5; α0.5+análogos refutado; docstring "Paso 5 (orden 4)"; comentario
  `# Utilidades internas` en `aprendizaje.py`.
