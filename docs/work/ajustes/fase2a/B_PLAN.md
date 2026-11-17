# Fase 2a — Parte B: sub-plan (backtest con el modelo real + recalibración + reporte)

Agente B, 2026-11-13. Worktree `EPM-wt-2a`, rama `feat/ajustes-fase2a`. Sin commits.

## Archivos
- `scripts/recalibrar_ajustes.py` (nuevo). No se modifica `scripts/backtest_adjustments.py`, `nivel.py` ni `forma.py`
  (fase 2b los toca): se importan sus funciones.
  - `generar`: curvas base del modelo con `SandboxModelo` (k=1 principal, k=4 sensibilidad) → cache CSV largo
    `reports/ajustes/curvas_modelo_k{k}.csv` (`ucp, k, semana_inicio, corte, origen, fecha, P1..P24`), reanudable
    (salta semanas ya cacheadas), hashes de `data/raw`, `models`, `data/features`, `data/processed` antes/después en
    `reports/ajustes/curvas_modelo_k{k}_hash.json`.
  - `evaluar`: backtest sobre el cache (proveedor `proveedor_desde_curvas`) y sobre `ingenuo`/`archivo` en las MISMAS
    semanas, con configs generales (α, fuente, n, `alpha_por_tipo`) → `reports/ajustes/backtest_modelo_<fecha>.csv`.
    Evaluador propio `evaluar_semana_cfg` = `evaluar_semana` con params arbitrarios (mismas columnas + `config`, `n`).
  - `recalibrar`: holdout temporal 60/40 por UCP, selección en calibración, reporte en prueba y total; banda del control;
    escribe `reports/ajustes/backtest_modelo_20261113.md`.
- `tests/test_recalibrar_ajustes.py` (nuevo): rápidos, sintéticos, sin entrenar.
- Docs: `B_PLAN.md`, LOG.md (hitos), `B_REPORTE.md`.

## Grid
α ∈ {0, .25, .5, .75, 1} × fuente ∈ {reciente, referencia, promedio_n(n ∈ {2, 3, 4, 6, 8})} (forma valida n ∈ [1, 60]).

## Selección (solo calibración)
- Config global: min WAPE horario del paso final.
- `alpha_por_tipo`: con la fuente/n elegidas, mejor α por tipo de día filtrando días; festivo solo si n_festivos_cal ≥ 10
  (si no, usa el α global). Luego se CORRE la config resultante.
- Banda control: {1, 1.5, 2, 2.5, 3, 4}; criterio principal lift = precision − tasa_base, desempate F1.
- Cambio de DEFAULTS solo con la regla del prompt (mejora WAPE final en prueba en ambas UCP y sin empeorar > 0.05 pp el
  MAPE de energía semanal en ninguna).
