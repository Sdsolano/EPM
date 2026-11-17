# Plan: re-backtest con base modelo + Fase 4 + RESUMEN_NOCHE

## Contexto
Fases ASR, 2a, 2b (main = ded1df6) y 3 (Sphere test = 22ef6d1) ya integradas. Quedan: (0) re-backtest análogos/ASR con base modelo, (1) Fase 4, (2) RESUMEN_NOCHE.md. El usuario eligió los 4 ítems de Fase 4: paso `especiales`, automático por segmento (R2), alertas clima/eventos (F06) y aprendizaje F09. Todo lo no medido por backtest queda desactivado por defecto.

## Paso 0 — Re-backtest (agente propio, sin tocar src/)
- `--base modelo` actual lee `models/` real (fuga): no usar. Reusar curvas sin fuga de 2a: `EPM-wt-2a/reports/ajustes/curvas_modelo_k1_{Antioquia,Atlantico}.csv` (52 sem, reentreno semanal k=1); copiarlas a `reports/ajustes/` sin versionarlas.
- Añadir base `modelo_k1` (lee el CSV por (ucp, corte)) a `scripts/backtest_analogos.py` y `scripts/comparar_asr.py`, más `--base-regla` en el primero; tests en `tests/`.
- Correr con `--semanas 52`, `--fecha 20261113_modelo`, bases modelo_k1/ingenuo/archivo; reportes con tablas antes/después (WAPE, MAPE energía semanal) por UCP y segmento; segmentos <15 días = no concluyentes. Sin cambiar defaults.
- Verificar git status/hashes de data/ y models/, suite completa, commit local en main.

## Paso 1 — Fase 4 (worktree EPM-wt-4, rama feat/ajustes-fase4, base main; docs en docs/work/ajustes/fase4/)
Orden por dependencias (agentes por parte, ronda 1 de revisión multi-ángulo con sonnet; solo repetir con críticos/altos):
- **4A R1-C (prerrequisito de F09):** el paso `editado` conserva `curvas_calculadas` de días editados (`types.StepResult`, `pipeline.py`, `sql/` ya guarda `curvas_calculadas`).
- **4B Paso `especiales`:** `src/adjustments/steps/especiales.py` tras `forma`, antes de `control`. STEP_ORDER pasa a 6, añadir a `_MODULOS` y `_PASOS_QUE_MODIFICAN` (`pipeline.py:46-56`); router valida contra STEP_ORDER (`router.py:268,776`). Cubre Semana Santa, 24/31 dic, post-festivo, usando `holiday_homology.py`, `analogos.py` y `data.py` (festivos). Tolerar corridas viejas de 5 pasos (migración en `_records_por_paso`). Default `omitido`/sin efecto hasta que el backtest (modelo_k1, α 0.5 + `ancla: analogos` en festivos/Semana Santa) lo respalde.
- **4C Automático por segmento (R2):** configuración por UCP × tipo de día (forma auto en laborales ambas UCP y domingos Atlántico; resto `sugerido`), derivada de `backtest_modelo_20261113.md`; activable, off por defecto.
- **4D Alertas F06:** alertas info/warning en `steps/control.py`. Sin que el usuario entregue nada: eventos/elecciones y clima esperado se obtienen con la consulta a OpenAI que ya existe (`get_future_events_from_openai`, `src/api/main.py:765`, Responses API con búsqueda web, key en `.env`). Se extrae a un módulo `src/adjustments/eventos.py` (cliente inyectable para tests), se llama una vez por (UCP, semana) fuera del cálculo del pipeline, se cachea el resultado (JSON en disco) y se pasa por `RunContext.extras`; el paso `control` solo lee. Sin API key o con fallo → degrada en silencio (solo alertas de festivos/clima observado `analogos.cargar_clima`). Los eventos se marcan como "sugeridos por IA, no verificados" en el detalle de la alerta; nunca modifican curvas.
- **4E F09:** diagnóstico de intervenciones recurrentes (diferencia curva calculada vs editada por UCP × tipo de día × paso) como reporte/endpoint de solo lectura; sin reentrenar ni cambiar reglas automáticamente.
- Verificación por parte: tests nuevos + suite completa (solo falla christmas), `import src.api.main`, revisión multi-ángulo + regresión. Si cambia el contrato (6 pasos), avisar al front de Sphere (mock/tipos de la mesa) como pendiente documentado, no implementarlo aquí.
- Integración: rebase sobre main, suite, push de rama y `<rama>:main`, ff en `../EPM`.

## Paso 2 — RESUMEN_NOCHE.md
`docs/work/ajustes/RESUMEN_NOCHE.md`: tabla antes/después (ASR, 2a, 2b, re-backtest, Fase 4), decisiones tomadas sin el usuario, pendientes (campo `usuario` en aprobar, módulo en BD, front para paso especiales), qué revisar al despertar. `git add -f`, commit y push a main.

## Verificación end-to-end
`.venv\Scripts\python.exe -m pytest tests -q --ignore=tests/test_hourly_disaggregation.py -p no:cacheprovider`, `python -c "import src.api.main"`, hashes de data/ y models/ intactos tras backtests, reportes revisados.

## Riesgos
Evidencia con n pequeño en festivos/Semana Santa (n≈8-18) → recomendaciones "no concluyente"; los eventos vienen de OpenAI (no deterministas, costo por llamada, requieren API key) → cache por semana, tests con cliente falso, sin pronóstico de clima archivado para backtest (4D no se puede medir retroactivamente).
