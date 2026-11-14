# B1 — API de insights (sub-plan)
Fuente: noche2/CONTRATO_B.md (EXACTO). Rama feat/noche2-backend, worktree EPM-wt-n2. No commitear.

## Entregables
1. `scripts/generar_historial_pronosticos.py` -> `src/adjustments/precision_data/{ucp}_k1.csv` (fecha,semana_inicio,P1..P24, 1 decimal; solo lunes..domingo de cada semana).
2. `src/adjustments/insights.py`: logica pura (sin FastAPI): carga/cache de CSV, precision, banda, sesgo, recomendaciones, resumen, series.
3. `src/adjustments/router_insights.py`: 5 GET bajo /adjustments/insights (token), registrado en src/api/main.py.
4. Tests: tests/test_adjustments_insights.py (puro) y tests/test_adjustments_insights_router.py (TestClient+InMemoryStore).

## Decisiones de diseno
- PRECISION_DIR (variable de modulo) para poder apuntar a datos sinteticos en tests; cache por (ruta, mtime).
- Parche de recomendacion: paso = ultimo paso que modifica y es editable ("especiales") para que final_curves == curva_sugerida exacta (editar "forma" dejaria que especiales/control la vuelvan a tocar). Documentado en CONTRATO "Cambios".
- Extension para B2: `sesgo_por_segmento(...)`, `PROVEEDORES_SESGO` (lista de funciones) en insights.py.
- Nunca 500 por falta de historial: degradar a null/[]/"sin_datos".
- Calculos pesados con run_in_threadpool.

## Hecho = 5 endpoints con forma del contrato + tests + suite sin fallos nuevos.
