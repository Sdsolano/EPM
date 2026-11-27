# Agente 2 — REPORTE: `scripts/backtest_mesa_vs_jamer.py`

## Qué se hizo
- `scripts/backtest_mesa_vs_jamer.py`: backtest reproducible (parte C del PLAN). Entradas solo por CLI
  (`--historico --festivos --envios [--desde] [--partes 1,2] [--seed] [--desde-ampliada] [--n-control]
  [--salida] [--salida-detalle] [--fecha]`), sin credenciales ni rutas fijas. Configs como datos (`CONFIGS`);
  si un paso rechaza una clave (ValueError "params desconocidos") la config se salta con aviso y queda listada
  en el reporte. Parte 1 (vs Jamer: VF y final, corte = envío − 1, base ingenua, assert sin fuga, métricas,
  tipo de día + post_festivo, lead, pareado bootstrap IC95 con semilla, 60/40), parte 2 (2022→último dato,
  semanas con festivo / Semana Santa / R1 saltada + control con semilla, corte = lunes − 6), veredictos de las
  3 reglas por config (PASA / NO PASA con números; sin datos → NO PASA).
- `tests/test_backtest_mesa_vs_jamer.py` (22 tests, sintéticos con `tests/ajustes_sinteticos.py`): VF/final y
  descarte de envíos retroactivos, cortes, días a evaluar, sin fuga + config no soportada, métricas,
  segmentos, bootstrap determinista, criterio (cada regla), `tipo_festivo`, clasificador R1, selección de la
  parte 2, `main` extremo a extremo.
- Salida: `reports/ajustes/backtest_mesa_vs_jamer_<fecha>.md` (agregados; `/reports` está en `.gitignore`,
  se versiona con `git add -f` como los demás) y detalle CSV en `reports/ajustes/jamer/` (ignorado; trae energía real).

## Decisiones tomadas sin el usuario
- Segmento de la regla 1: `festivo_por_tipo` → festivos; `r1_limpia` → días no festivos cuyo R1 de `jamer`
  saltaría el clasificador del PLAN B (implementado en el script, independiente del código del Agente 1);
  `saltar_puente` → días cuyo R1 cambia respecto a `jamer`; `ambas` → festivos ∪ R1 saltada. Sensibilidad
  `r1_umbral_atipico` 0.04/0.08 sin veredicto.
- Regla 2 = media pareada ≤ 0 y IC95 no enteramente > 0. Regla 3 se evalúa en la parte 1 VF.
- `--desde` default = todas las semanas con VF (48 desde 2025-12-01). **El criterio del PLAN habla de 33
  semanas: para aplicarlo literalmente correr con `--desde 2026-04-06`** (ver diferencia abajo).
- VF puede enviarse dentro de la semana (p. ej. lunes 2025-12-01), como en `bt_multi.py`.

## Corrida real (código del Agente 1 EN CURSO durante la corrida → cifras de variantes PROVISIONALES)
- Todas las semanas (48 sem parte 1 + 176 sem parte 2): 229 s. `--desde 2026-04-06 --partes 1`: 58 s.
- Reproducción del backtest previo (33 sem, VF): `jamer` 2,57 % vs `jamer_envio` 2,50 % MAPE h; pareado
  +0,074 pp IC95 [−0,184, +0,359] → idéntico al del PLAN.
- 48 semanas: `jamer_envio` 2,81 %; `jamer` vs envío +0,246 pp [−0,115, +0,595]; `ambas` vs envío +0,107 pp.
- Veredictos (48 sem + ampliada): festivo_por_tipo NO PASA (festivos ampliada +0,15 pp: fijos 6,63→7,96 %,
  Semana Santa 6,81→5,28 %); r1_limpia PASA (R1 saltada −1,16 pp en 181 d, prueba −1,19 pp en 2 d; global
  −0,058 [−0,199, +0,063]); saltar_puente NO PASA; ambas PASA (−0,139 [−0,302, +0,011]).
- Con las 33 semanas del PLAN (solo parte 1): r1_limpia y ambas NO PASAN regla 3 (domingo +0,15 pp).
- Hay que re-correr cuando el Agente 1 termine; los veredictos dependen de la muestra (33 vs 48 semanas).

## Suite
`pytest tests -q --ignore=tests/test_hourly_disaggregation.py`: 994 passed, 16 skipped, 2 failed —
`test_christmas_profile_uses_prior_christmases` (preexistente) y
`test_adjustments_nivel_jamer::test_params_default_no_cambian_ni_se_persisten` (dict exacto del Agente 1,
previsto en el PLAN). Los 22 tests nuevos pasan.
