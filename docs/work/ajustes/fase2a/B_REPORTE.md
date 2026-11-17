# Fase 2a — Parte B: reporte (backtest con el modelo real + recalibración con holdout)

Agente B, 2026-11-13. Sin commits. No se tocaron `nivel.py`, `forma.py`, `control.py` ni `scripts/backtest_adjustments.py`.

## Archivos
- `scripts/recalibrar_ajustes.py` (nuevo): `generar` (curvas base del modelo con `SandboxModelo`, rolling origin k,
  cache por UCP reanudable + hashes), `evaluar` (grid 35 configs sobre modelo_k1/k4, ingenuo, archivo en las mismas
  semanas, 4 procesos), `recalibrar` (holdout 60/40, selección en calibración, `alpha_por_tipo` corrido de verdad,
  banda del control, R2, reporte markdown con bloque manual preservado). Reutiliza `backtest_adjustments`
  (`construir_contexto`, `metricas_dia`, `semanas_objetivo`, proveedores, `resumen_por`, `resumen_semanal`, `tabla_md`).
- `tests/test_recalibrar_ajustes.py` (nuevo, 10 tests sintéticos, sin red ni entrenamiento, ~9 s).
- `reports/ajustes/backtest_modelo_20261113.md` (reporte; `/reports` está en .gitignore → `git add -f` si se versiona).
- No versionados (en `reports/ajustes/`): `curvas_modelo_k{1,4}_{UCP}.csv`, `hash_k*_*.json`, `progreso_k*_*.log`,
  `generar_k*.out`, `backtest_modelo_grid.csv.gz` (245k filas), `backtest_modelo_extra.csv.gz`, `evaluar_recalibrar.out`.

## Reproducir (cwd = worktree; Python del venv de EPM)
```
python scripts/recalibrar_ajustes.py generar --k 1 --ucp Antioquia   # ~30 min; en paralelo con --ucp Atlantico
python scripts/recalibrar_ajustes.py generar --k 4 --ucp Antioquia   # ~11 min
python scripts/recalibrar_ajustes.py evaluar --procesos 4            # ~5 min
python scripts/recalibrar_ajustes.py recalibrar --procesos 4         # ~1 min (--reusar-extra para no re-correr apt)
```

## No-fuga y hashes
Cada semana: `datos.csv` del sandbox recortado a FECHA ≤ corte (sábado previo) y `end_date=corte`; k=1 reentrena en
cada semana, k=4 cada 4 (intermedias con `predecir`, que rechaza modelos con corte posterior). sha256 de `data/raw`,
`models`, `data/features`, `data/processed` (54 archivos) idéntico antes/después en los 4 procesos de generación.
Sandboxes temporales borrados. `git status`: sin cambios en datos.

## Números clave (prueba = últimas 21 semanas por UCP; WAPE horario %, default α 0.25 + promedio_n n 4)
| | Antioquia | Atlántico |
|---|---|---|
| modelo_k1 base → nivel → forma | 2.44 → 2.44 → 2.33 | 5.86 → 5.77 → 5.46 |
| MAPE energía semanal base → final | 1.52 → 1.37 | 4.57 → 4.41 |
| ingenuo base → final | 3.91 → 3.56 | 6.21 → 6.06 |
| archivo base → final (17 sem) | 2.86 → 2.52 | — |
| modelo_k4 base → final | 2.58 → 2.42 | 5.88 → 5.47 |
En las 21 semanas del archivo (Antioquia): modelo_k1 final 2.61, archivo 2.79, ingenuo 3.82.

## Selección (calibración) → prueba
- Mejor conjunta `a0.25|promedio_n6`: prueba ΔWAPE vs default +0.016 (Ant) / −0.007 (Atl) → no cumple la regla.
- Mejor por UCP: Antioquia `promedio_n2`, Atlántico `promedio_n8` → +0.017 / +0.013 pp en prueba (peores).
- `alpha_por_tipo` conjunto = `{}` (todos los tipos óptimos en 0.25; festivo excluido: Atlántico 9 < 10 festivos).
  Por UCP: Antioquia `{domingo:0, festivo:0}` +0.007 pp, Atlántico `{laboral:0}` +0.014 pp en prueba (peores).
- Banda control: calibración elige 1.0 pp (lift 0.06/0.03); prueba lift 0.00/−0.04 vs 2.0 pp −0.05/+0.01 → ≈ 0.
- k=4 elige en calibración exactamente las mismas configs.

## DEFAULTS: NO cambiados
Ninguna candidata cumple la regla (mejora WAPE en prueba en ambas UCP sin empeorar MAPE semanal > 0.05 pp). α 0.25 y
`promedio_n` quedan confirmados fuera de muestra con el modelo real. Banda 2.0 pp se mantiene (sin evidencia de mejora).
`alpha_por_tipo` no se recomienda; además, ponerlo en DEFAULTS cambiaría corridas guardadas al re-ejecutarlas
(`rerun_from` re-valida params persistidos y `_params.completar` rellena claves ausentes con los DEFAULTS actuales).

## R2 (ver § "R2" del reporte)
- forma automático: laboral (ambas UCP) y domingo de Atlántico; sugerido: sábados, domingo de Antioquia, festivos.
- nivel automático: solo sábado de Atlántico (borde); "no aplicar" por WAPE: domingo Antioquia, laboral Atlántico;
  resto sugerido (ruidoso por semana, pero mejora la energía semanal).
- control: informativo, banda 2.0 pp.

## Verificación
- Suite: `pytest tests -q --ignore=tests/test_hourly_disaggregation.py` → 280 passed, 5 skipped, 1 failed
  (`test_christmas_profile_uses_prior_christmases`, permitido). `python -c "import src.api.main"` OK.

## Decisiones tomadas sin el usuario
- Cache por UCP (un archivo por k × UCP) para correr 4 procesos en paralelo sin escrituras cruzadas.
- Evaluador propio `evaluar_semana_cfg` (copia de `evaluar_semana` con params arbitrarios: n, alpha_por_tipo) en el
  script nuevo, en vez de modificar `backtest_adjustments.py` (fase 2b lo toca).
- n de `promedio_n` ∈ {2, 3, 4, 6, 8} (forma admite 1..60).
- Candidata a default global = "mejor conjunta" (min del promedio de WAPE entre UCP en calibración).
- `alpha_por_tipo`: festivo con α propio solo si cada UCP tiene ≥ 10 festivos en calibración; claves iguales al α
  global se omiten.
- Banda: criterio max lift (precision − tasa_base) promedio entre UCP, desempate F1; NO se cambia el default porque en
  prueba no mejora (lift ≈ 0 en todas las bandas).
- R2: "automático" = mejora WAPE en calibración y prueba, ≤ 35 % de semanas peores en prueba y ≥ 10 días de prueba.

## Riesgos / pendientes
- Clima observado (optimista); Antioquia sin clima antes de 2025-12-01 (calibración casi entera sin clima).
- 21 semanas de prueba: diferencias < 0.05 pp son ruido.
- La alerta de tendencia de referencia no predice el error semanal: revisar su propósito con el analista.
- R2 de nivel debería revisarse con ediciones reales del analista (Mesa de ajustes) y con la energía semanal como métrica.
