# Backtest Mesa (perfil jamer + variantes) vs Jamer y vs real — 20261127

Generado por `scripts/backtest_mesa_vs_jamer.py --conjunto post_festivo` (fase `docs/work/ajustes/post_festivo/`). Solo agregados (errores %, no curvas ni demanda real). El detalle por día está en un CSV ignorado por git.

Entradas: `historico_antioquia.csv`, `festivos_antioquia.csv`, `envios_jamer.csv` (fuera del repo). Semilla 7. Duración 171 s. Configs evaluadas: `jamer`, `jamer+pf_homologo`, `jamer+pf_recientes`, `jamer+pf_combinado`.

## Criterio de aceptación (PLAN, fijado antes de correr; Enmienda 2026-11-27)

Veredictos SOLO con la muestra del PLAN (parte 1 desde `--desde`); la muestra informativa no decide. Regla 1: el segmento objetivo mejora (diferencia de MAPE h < 0) en CADA parte listada (con sus días). Festivos por tipo (segmento = festivos `fijo`/`semana_santa`; los `lunes` no cambian): parte 2 partida por fecha objetivo, calibración 2022-2024 y prueba 2025-2026. R1 saltada / R1 cambia: parte 2 completa y semanas de prueba de la parte 1 (VF). `jamer+ambas`: deben pasar las dos mitades (festivos por tipo; días con R1 distinta fuera de ese segmento). Regla 2 (lectura estricta, Enmienda 2026-11-27): pareado global parte 1 VF (config - jamer) con media <= 0 Y IC95 no enteramente > 0. Regla 3: ningún tipo de día empeora > 0.1 pp (parte 1, VF). Sin datos para una regla: NO PASA.

Factor post-festivo (segmento = días `post_festivo`): de las variantes `jamer+pf_*` se elige UNA, la de menor MAPE h medio en post-festivos de la parte 2 con fecha objetivo 2022-2024 (calibración); solo ella recibe veredicto y las otras son informativas. Regla 1: parte 2 2022-2024 y 2025-2026 (por fecha objetivo); los post-festivos de las semanas de prueba de la parte 1 (VF) se reportan como información. Base = perfil `jamer` actual (con `festivo_por_tipo`).

### Elección de variante (calibración, parte 2)

| grupo | variante | dias_calibracion | mape_h_calibracion | elegida |
|---|---|---|---|---|
| post_festivo | jamer+pf_homologo | 45 | 5.220 | **sí** |
| post_festivo | jamer+pf_recientes | 45 | 5.616 | no |
| post_festivo | jamer+pf_combinado | 45 | 5.627 | no |

| config | segmento | regla | valor | veredicto |
|---|---|---|---|---|
| jamer+pf_homologo | días post_festivo | 1 segmento mejora (en cada parte) | post-festivos 2022-2024 (ampliada, calibración) -0.11 pp (45 d); post-festivos 2025-2026 (ampliada, prueba) +0.43 pp (28 d) | NO PASA |
| jamer+pf_homologo | días post_festivo | 2 global no empeora (pareado, lectura estricta) | media -0.025 pp, IC95 [-0.077, +0.017], 33 sem | PASA |
| jamer+pf_homologo | días post_festivo | 3 ningún tipo de día empeora > 0.1 pp | laboral -0.04, sabado +0.00, domingo +0.00, festivo +0.00 | PASA |
| jamer+pf_homologo |  | informativo (no decide) | post-festivos prueba parte 1 (VF, informativo) -1.46 pp (3 d) | — |
| jamer+pf_homologo |  | **global** |  | **NO PASA** |
| jamer+pf_recientes | días post_festivo | 1 segmento mejora (en cada parte) | post-festivos 2022-2024 (ampliada, calibración) +0.29 pp (45 d); post-festivos 2025-2026 (ampliada, prueba) +0.28 pp (28 d) | (NO PASA, informativo) |
| jamer+pf_recientes | días post_festivo | 2 global no empeora (pareado, lectura estricta) | media -0.012 pp, IC95 [-0.074, +0.045], 33 sem | (PASA, informativo) |
| jamer+pf_recientes | días post_festivo | 3 ningún tipo de día empeora > 0.1 pp | laboral -0.02, sabado +0.00, domingo +0.00, festivo +0.00 | (PASA, informativo) |
| jamer+pf_recientes |  | informativo (no decide) | post-festivos prueba parte 1 (VF, informativo) -1.64 pp (3 d) | — |
| jamer+pf_recientes |  | **global** |  | **informativa (variante no elegida, sin veredicto)** |
| jamer+pf_combinado | días post_festivo | 1 segmento mejora (en cada parte) | post-festivos 2022-2024 (ampliada, calibración) +0.30 pp (45 d); post-festivos 2025-2026 (ampliada, prueba) +0.53 pp (28 d) | (NO PASA, informativo) |
| jamer+pf_combinado | días post_festivo | 2 global no empeora (pareado, lectura estricta) | media -0.007 pp, IC95 [-0.066, +0.048], 33 sem | (PASA, informativo) |
| jamer+pf_combinado | días post_festivo | 3 ningún tipo de día empeora > 0.1 pp | laboral -0.01, sabado +0.00, domingo +0.00, festivo +0.00 | (PASA, informativo) |
| jamer+pf_combinado |  | informativo (no decide) | post-festivos prueba parte 1 (VF, informativo) -1.44 pp (3 d) | — |
| jamer+pf_combinado |  | **global** |  | **informativa (variante no elegida, sin veredicto)** |

## Parte 1 — Mesa vs envíos de Jamer (muestra del PLAN: decide)

Semanas: 33 (2026-04-06 .. 2026-11-16); calibración 20 (hasta 2026-08-17), prueba 13 (desde 2026-08-24). Base `ingenuo`; corte = envío - 1. VF evalúa solo días > corte: 0 día(s) VF con fecha <= corte descartados.

### VF, todas las semanas

| serie | dias | mape_h | abs_err_dia | sesgo_energia | p90_abs_err_dia |
|---|---|---|---|---|---|
| jamer_envio | 230 | 2.50 | 2.15 | 0.56 | 4.47 |
| jamer | 230 | 2.53 | 2.10 | -0.08 | 4.13 |
| jamer+pf_homologo | 230 | 2.50 | 2.06 | -0.01 | 4.02 |
| jamer+pf_recientes | 230 | 2.51 | 2.09 | -0.09 | 4.00 |
| jamer+pf_combinado | 230 | 2.52 | 2.08 | -0.05 | 4.00 |

### VF, calibración

| serie | dias | mape_h | abs_err_dia | sesgo_energia | p90_abs_err_dia |
|---|---|---|---|---|---|
| jamer_envio | 140 | 2.52 | 2.15 | -0.03 | 4.25 |
| jamer | 140 | 2.68 | 2.24 | -0.41 | 4.61 |
| jamer+pf_homologo | 140 | 2.66 | 2.22 | -0.32 | 4.55 |
| jamer+pf_recientes | 140 | 2.69 | 2.25 | -0.39 | 4.55 |
| jamer+pf_combinado | 140 | 2.69 | 2.23 | -0.34 | 4.55 |

### VF, prueba (fuera de muestra)

| serie | dias | mape_h | abs_err_dia | sesgo_energia | p90_abs_err_dia |
|---|---|---|---|---|---|
| jamer_envio | 90 | 2.47 | 2.15 | 1.46 | 4.56 |
| jamer | 90 | 2.29 | 1.88 | 0.43 | 3.60 |
| jamer+pf_homologo | 90 | 2.24 | 1.82 | 0.47 | 3.54 |
| jamer+pf_recientes | 90 | 2.24 | 1.83 | 0.38 | 3.52 |
| jamer+pf_combinado | 90 | 2.24 | 1.84 | 0.39 | 3.52 |

### final (días posteriores al último envío)

| serie | dias | mape_h | abs_err_dia | sesgo_energia | p90_abs_err_dia |
|---|---|---|---|---|---|
| jamer_envio | 170 | 2.38 | 2.02 | 0.26 | 4.51 |
| jamer | 170 | 2.37 | 1.99 | -0.00 | 4.13 |
| jamer+pf_homologo | 170 | 2.34 | 1.95 | 0.03 | 4.02 |
| jamer+pf_recientes | 170 | 2.33 | 1.95 | -0.01 | 3.91 |
| jamer+pf_combinado | 170 | 2.33 | 1.95 | -0.00 | 3.91 |

### Pareado por semana (MAPE h semanal, a - b; negativo = a mejor; IC95 bootstrap)

| muestra | a | b | semanas | dif_media_pp | ic95_inf | ic95_sup | semanas_gana_a |
|---|---|---|---|---|---|---|---|
| VF todas | jamer | jamer_envio | 33 | 0.028 | -0.220 | 0.301 | 17 |
| VF todas | jamer+pf_homologo | jamer | 33 | -0.025 | -0.077 | 0.017 | 4 |
| VF todas | jamer+pf_homologo | jamer_envio | 33 | 0.002 | -0.253 | 0.284 | 18 |
| VF todas | jamer+pf_recientes | jamer | 33 | -0.012 | -0.074 | 0.045 | 5 |
| VF todas | jamer+pf_recientes | jamer_envio | 33 | 0.016 | -0.237 | 0.296 | 17 |
| VF todas | jamer+pf_combinado | jamer | 33 | -0.007 | -0.066 | 0.048 | 4 |
| VF todas | jamer+pf_combinado | jamer_envio | 33 | 0.021 | -0.226 | 0.296 | 17 |
| VF prueba | jamer | jamer_envio | 13 | -0.171 | -0.453 | 0.100 | 7 |
| VF prueba | jamer+pf_homologo | jamer | 13 | -0.047 | -0.145 | 0.021 | 2 |
| VF prueba | jamer+pf_homologo | jamer_envio | 13 | -0.217 | -0.510 | 0.071 | 8 |
| VF prueba | jamer+pf_recientes | jamer | 13 | -0.051 | -0.157 | 0.040 | 2 |
| VF prueba | jamer+pf_recientes | jamer_envio | 13 | -0.222 | -0.510 | 0.071 | 8 |
| VF prueba | jamer+pf_combinado | jamer | 13 | -0.045 | -0.139 | 0.038 | 2 |
| VF prueba | jamer+pf_combinado | jamer_envio | 13 | -0.216 | -0.502 | 0.073 | 8 |
| final | jamer | jamer_envio | 33 | 0.053 | -0.230 | 0.335 | 15 |
| final | jamer+pf_homologo | jamer | 33 | -0.026 | -0.066 | 0.000 | 3 |
| final | jamer+pf_homologo | jamer_envio | 33 | 0.028 | -0.256 | 0.313 | 16 |
| final | jamer+pf_recientes | jamer | 33 | -0.030 | -0.075 | 0.000 | 3 |
| final | jamer+pf_recientes | jamer_envio | 33 | 0.023 | -0.261 | 0.309 | 16 |
| final | jamer+pf_combinado | jamer | 33 | -0.028 | -0.067 | 0.000 | 3 |
| final | jamer+pf_combinado | jamer_envio | 33 | 0.025 | -0.260 | 0.311 | 16 |

### VF por tipo de día — MAPE h %

| tipo | dias | jamer_envio | jamer | jamer+pf_homologo | jamer+pf_recientes | jamer+pf_combinado |
|---|---|---|---|---|---|---|
| domingo | 32 | 3.04 | 2.79 | 2.79 | 2.79 | 2.79 |
| festivo | 12 | 3.57 | 4.46 | 4.46 | 4.46 | 4.46 |
| laboral | 143 | 2.14 | 2.14 | 2.14 | 2.14 | 2.14 |
| laboral+post_festivo | 10 | 3.43 | 3.71 | 3.11 | 3.42 | 3.53 |
| sabado | 33 | 2.88 | 2.89 | 2.89 | 2.89 | 2.89 |

### VF por tipo de día — sesgo de energía %

| tipo_dia | jamer_envio | jamer | jamer+pf_homologo | jamer+pf_recientes | jamer+pf_combinado |
|---|---|---|---|---|---|
| domingo | 0.54 | -0.03 | -0.03 | -0.03 | -0.03 |
| festivo | 0.73 | -0.91 | -0.91 | -0.91 | -0.91 |
| laboral | 0.56 | 0.01 | 0.11 | -0.00 | 0.05 |
| sabado | 0.49 | -0.31 | -0.31 | -0.31 | -0.31 |

### VF por lead (días desde el corte) — MAPE h %

| lead | dias | jamer_envio | jamer | jamer+pf_homologo | jamer+pf_recientes | jamer+pf_combinado |
|---|---|---|---|---|---|---|
| 6 | 32 | 2.72 | 3.03 | 3.03 | 3.03 | 3.03 |
| 7 | 33 | 2.37 | 2.36 | 2.18 | 2.27 | 2.30 |
| 8 | 33 | 2.29 | 2.34 | 2.34 | 2.34 | 2.34 |
| 9 | 33 | 2.05 | 1.97 | 1.97 | 1.97 | 1.97 |
| 10 | 33 | 2.12 | 2.28 | 2.28 | 2.28 | 2.28 |
| 11 | 33 | 2.88 | 2.88 | 2.88 | 2.88 | 2.88 |
| 12 | 32 | 2.90 | 2.79 | 2.79 | 2.79 | 2.79 |
| 13 | 1 | 8.71 | 4.03 | 4.03 | 4.03 | 4.03 |

### Casos revisados día a día (VF) — error de energía del día %

| fecha | jamer_envio | jamer | jamer+pf_homologo | jamer+pf_recientes | jamer+pf_combinado |
|---|---|---|---|---|---|
| 2026-04-13 | -2.00 | -2.42 | -2.42 | -2.42 | -2.42 |
| 2026-04-18 | -2.46 | -10.93 | -10.93 | -10.93 | -10.93 |
| 2026-05-01 | -1.16 | -4.96 | -4.96 | -4.96 | -4.96 |
| 2026-08-07 | -1.65 | -2.99 | -2.99 | -2.99 | -2.99 |
| 2026-11-03 | -1.04 | -5.51 | 0.77 | -3.45 | -3.36 |

## Parte 2 — muestra ampliada contra real (sin Jamer)

Desde 2022-01-01; corte = martes anterior (lunes - 6). Semanas: 220 (festivo 82, Semana Santa 10, con R1 saltada 63, con post-festivo 73, control 40; una semana puede estar en varios grupos). R1 saltada = la R1 de nivel de `jamer+r1_limpia` difiere de la de `jamer`.

### MAPE h % por segmento

| segmento | dias | jamer | jamer+pf_homologo | jamer+pf_recientes | jamer+pf_combinado |
|---|---|---|---|---|---|
| festivo lunes | 56 | 4.85 | 4.85 | 4.85 | 4.85 |
| festivo fijo | 23 | 5.52 | 5.52 | 5.52 | 5.52 |
| festivo semana_santa | 10 | 5.27 | 5.27 | 5.27 | 5.27 |
| festivo fijo+semana_santa 2022-2024 (calibración) | 20 | 5.49 | 5.49 | 5.49 | 5.49 |
| festivo fijo+semana_santa 2025-2026 (prueba) | 13 | 5.38 | 5.38 | 5.38 | 5.38 |
| festivos (todos) | 89 | 5.07 | 5.07 | 5.07 | 5.07 |
| post-festivos | 73 | 5.29 | 5.39 | 5.58 | 5.68 |
| post-festivos 2022-2024 (calibración) | 45 | 5.33 | 5.22 | 5.62 | 5.63 |
| post-festivos 2025-2026 (prueba) | 28 | 5.24 | 5.66 | 5.52 | 5.77 |
| semanas de Semana Santa | 70 | 4.30 | 4.38 | 4.24 | 4.32 |
| control (semanas normales) | 280 | 2.92 | 2.92 | 2.92 | 2.92 |
| todo (muestra sesgada) | 1538 | 3.78 | 3.79 | 3.80 | 3.80 |

### Sesgo de energía % por segmento

| segmento | jamer | jamer+pf_homologo | jamer+pf_recientes | jamer+pf_combinado |
|---|---|---|---|---|
| festivo lunes | -0.08 | -0.08 | -0.08 | -0.08 |
| festivo fijo | -1.62 | -1.62 | -1.62 | -1.62 |
| festivo semana_santa | 0.18 | 0.18 | 0.18 | 0.18 |
| festivo fijo+semana_santa 2022-2024 (calibración) | 0.45 | 0.45 | 0.45 | 0.45 |
| festivo fijo+semana_santa 2025-2026 (prueba) | -3.23 | -3.23 | -3.23 | -3.23 |
| festivos (todos) | -0.45 | -0.45 | -0.45 | -0.45 |
| post-festivos | 0.82 | 0.57 | 0.32 | 0.34 |
| post-festivos 2022-2024 (calibración) | 0.24 | -0.18 | -0.01 | -0.15 |
| post-festivos 2025-2026 (prueba) | 1.67 | 1.67 | 0.80 | 1.07 |
| semanas de Semana Santa | 1.16 | 1.49 | 1.31 | 1.43 |
| control (semanas normales) | -0.63 | -0.63 | -0.63 | -0.63 |
| todo (muestra sesgada) | -0.06 | -0.07 | -0.08 | -0.08 |

### Por tipo de día — MAPE h %

| tipo_dia | dias | jamer | jamer+pf_homologo | jamer+pf_recientes | jamer+pf_combinado |
|---|---|---|---|---|---|
| domingo | 213 | 3.90 | 3.90 | 3.90 | 3.90 |
| festivo | 89 | 5.07 | 5.07 | 5.07 | 5.07 |
| laboral | 1018 | 3.62 | 3.63 | 3.64 | 3.65 |
| sabado | 218 | 3.89 | 3.89 | 3.89 | 3.89 |

