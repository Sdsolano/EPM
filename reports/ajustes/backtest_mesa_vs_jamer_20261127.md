# Backtest Mesa (perfil jamer + variantes) vs Jamer y vs real — 20261127

Generado por `scripts/backtest_mesa_vs_jamer.py` (fase `docs/work/ajustes/festivos_r1/`). Solo agregados (errores %, no curvas ni demanda real). El detalle por día está en un CSV ignorado por git.

Entradas: `historico_antioquia.csv`, `festivos_antioquia.csv`, `envios_jamer.csv` (fuera del repo). Semilla 7. Duración 461 s. Configs evaluadas: `jamer`, `jamer+festivo_por_tipo`, `jamer+r1_limpia`, `jamer+saltar_puente`, `jamer+ambas`, `jamer+r1_limpia_u04`, `jamer+r1_limpia_u08`.

## Criterio de aceptación (PLAN, fijado antes de correr; Enmienda 2026-11-27)

Veredictos SOLO con la muestra del PLAN (parte 1 desde `--desde`); la muestra informativa no decide. Regla 1: el segmento objetivo mejora (diferencia de MAPE h < 0) en CADA parte listada (con sus días). Festivos por tipo (segmento = festivos `fijo`/`semana_santa`; los `lunes` no cambian): parte 2 partida por fecha objetivo, calibración 2022-2024 y prueba 2025-2026. R1 saltada / R1 cambia: parte 2 completa y semanas de prueba de la parte 1 (VF). `jamer+ambas`: deben pasar las dos mitades (festivos por tipo; días con R1 distinta fuera de ese segmento). Regla 2 (lectura estricta, Enmienda 2026-11-27): pareado global parte 1 VF (config - jamer) con media <= 0 Y IC95 no enteramente > 0. Regla 3: ningún tipo de día empeora > 0.1 pp (parte 1, VF). Sin datos para una regla: NO PASA.

Semanas de prueba de la parte 1 (13): solo contienen festivos lunes (3 día(s) festivo(s) evaluado(s), todos de tipo `lunes`, que `festivo_por_tipo` no cambia): por eso la regla 1 del segmento de festivos usa el corte temporal de la parte 2 (Enmienda 2026-11-27).

| config | segmento | regla | valor | veredicto |
|---|---|---|---|---|
| jamer+festivo_por_tipo | festivos de tipo fijo o semana_santa | 1 segmento mejora (en cada parte) | festivos 2022-2024 (ampliada, calibración) -0.76 pp (20 d); festivos 2025-2026 (ampliada, prueba) -1.86 pp (13 d) | PASA |
| jamer+festivo_por_tipo | festivos de tipo fijo o semana_santa | 2 global no empeora (pareado, lectura estricta) | media -0.046 pp, IC95 [-0.131, +0.000], 33 sem | PASA |
| jamer+festivo_por_tipo | festivos de tipo fijo o semana_santa | 3 ningún tipo de día empeora > 0.1 pp | laboral +0.00, sabado +0.00, domingo +0.00, festivo -0.90 | PASA |
| jamer+festivo_por_tipo |  | **global** |  | **PASA** |
| jamer+r1_limpia | días cuya R1 de nivel (corrida de la config) difiere de la de `jamer` (R1 saltada) | 1 segmento mejora (en cada parte) | ampliada -1.84 pp (138 d); prueba parte 1 — pp (0 d) | NO PASA |
| jamer+r1_limpia | días cuya R1 de nivel (corrida de la config) difiere de la de `jamer` (R1 saltada) | 2 global no empeora (pareado, lectura estricta) | media -0.035 pp, IC95 [-0.141, +0.038], 33 sem | PASA |
| jamer+r1_limpia | días cuya R1 de nivel (corrida de la config) difiere de la de `jamer` (R1 saltada) | 3 ningún tipo de día empeora > 0.1 pp | laboral -0.03, sabado -0.26, domingo +0.15, festivo +0.00 | NO PASA |
| jamer+r1_limpia |  | **global** |  | **NO PASA** |
| jamer+saltar_puente | días cuya R1 de nivel difiere de la de `jamer` | 1 segmento mejora (en cada parte) | ampliada +0.26 pp (190 d); prueba parte 1 +0.44 pp (8 d) | NO PASA |
| jamer+saltar_puente | días cuya R1 de nivel difiere de la de `jamer` | 2 global no empeora (pareado, lectura estricta) | media +0.012 pp, IC95 [-0.085, +0.129], 33 sem | NO PASA |
| jamer+saltar_puente | días cuya R1 de nivel difiere de la de `jamer` | 3 ningún tipo de día empeora > 0.1 pp | laboral +0.00, sabado -0.03, domingo +0.11, festivo +0.00 | NO PASA |
| jamer+saltar_puente |  | **global** |  | **NO PASA** |
| jamer+ambas | festivos fijo/semana_santa + (aparte) días fuera de ese segmento con R1 de nivel distinta | 1 segmento mejora (en cada parte) | festivos 2022-2024 (ampliada, calibración) -0.76 pp (20 d); festivos 2025-2026 (ampliada, prueba) -1.86 pp (13 d); R1 distinta ampliada -1.84 pp (138 d); R1 distinta prueba parte 1 — pp (0 d) | NO PASA |
| jamer+ambas | festivos fijo/semana_santa + (aparte) días fuera de ese segmento con R1 de nivel distinta | 2 global no empeora (pareado, lectura estricta) | media -0.081 pp, IC95 [-0.214, +0.020], 33 sem | PASA |
| jamer+ambas | festivos fijo/semana_santa + (aparte) días fuera de ese segmento con R1 de nivel distinta | 3 ningún tipo de día empeora > 0.1 pp | laboral -0.03, sabado -0.26, domingo +0.15, festivo -0.90 | NO PASA |
| jamer+ambas |  | **global** |  | **NO PASA** |

## Parte 1 — Mesa vs envíos de Jamer (muestra del PLAN: decide)

Semanas: 33 (2026-04-06 .. 2026-11-16); calibración 20 (hasta 2026-08-17), prueba 13 (desde 2026-08-24). Base `ingenuo`; corte = envío - 1. VF evalúa solo días > corte: 0 día(s) VF con fecha <= corte descartados.

### VF, todas las semanas

| serie | dias | mape_h | abs_err_dia | sesgo_energia | p90_abs_err_dia |
|---|---|---|---|---|---|
| jamer_envio | 230 | 2.50 | 2.15 | 0.56 | 4.47 |
| jamer | 230 | 2.57 | 2.15 | -0.13 | 4.17 |
| jamer+festivo_por_tipo | 230 | 2.53 | 2.10 | -0.08 | 4.13 |
| jamer+r1_limpia | 230 | 2.54 | 2.12 | -0.10 | 4.13 |
| jamer+saltar_puente | 230 | 2.58 | 2.17 | -0.13 | 4.42 |
| jamer+ambas | 230 | 2.49 | 2.07 | -0.05 | 4.05 |
| jamer+r1_limpia_u04 | 230 | 2.54 | 2.14 | -0.14 | 4.05 |
| jamer+r1_limpia_u08 | 230 | 2.55 | 2.13 | -0.09 | 4.13 |

### VF, calibración

| serie | dias | mape_h | abs_err_dia | sesgo_energia | p90_abs_err_dia |
|---|---|---|---|---|---|
| jamer_envio | 140 | 2.52 | 2.15 | -0.03 | 4.25 |
| jamer | 140 | 2.75 | 2.32 | -0.49 | 4.98 |
| jamer+festivo_por_tipo | 140 | 2.68 | 2.24 | -0.41 | 4.61 |
| jamer+r1_limpia | 140 | 2.70 | 2.28 | -0.45 | 4.98 |
| jamer+saltar_puente | 140 | 2.75 | 2.34 | -0.48 | 5.02 |
| jamer+ambas | 140 | 2.62 | 2.20 | -0.37 | 4.61 |
| jamer+r1_limpia_u04 | 140 | 2.71 | 2.30 | -0.51 | 4.98 |
| jamer+r1_limpia_u08 | 140 | 2.71 | 2.30 | -0.43 | 4.98 |

### VF, prueba (fuera de muestra)

| serie | dias | mape_h | abs_err_dia | sesgo_energia | p90_abs_err_dia |
|---|---|---|---|---|---|
| jamer_envio | 90 | 2.47 | 2.15 | 1.46 | 4.56 |
| jamer | 90 | 2.29 | 1.88 | 0.43 | 3.60 |
| jamer+festivo_por_tipo | 90 | 2.29 | 1.88 | 0.43 | 3.60 |
| jamer+r1_limpia | 90 | 2.29 | 1.88 | 0.43 | 3.60 |
| jamer+saltar_puente | 90 | 2.33 | 1.90 | 0.42 | 3.75 |
| jamer+ambas | 90 | 2.29 | 1.88 | 0.43 | 3.60 |
| jamer+r1_limpia_u04 | 90 | 2.29 | 1.88 | 0.43 | 3.60 |
| jamer+r1_limpia_u08 | 90 | 2.29 | 1.88 | 0.43 | 3.60 |

### final (días posteriores al último envío)

| serie | dias | mape_h | abs_err_dia | sesgo_energia | p90_abs_err_dia |
|---|---|---|---|---|---|
| jamer_envio | 170 | 2.38 | 2.02 | 0.26 | 4.51 |
| jamer | 170 | 2.44 | 2.06 | -0.06 | 4.19 |
| jamer+festivo_por_tipo | 170 | 2.37 | 1.99 | -0.00 | 4.13 |
| jamer+r1_limpia | 170 | 2.40 | 2.02 | -0.03 | 4.13 |
| jamer+saltar_puente | 170 | 2.58 | 2.24 | -0.19 | 4.91 |
| jamer+ambas | 170 | 2.33 | 1.95 | 0.03 | 3.92 |
| jamer+r1_limpia_u04 | 170 | 2.41 | 2.04 | -0.09 | 4.13 |
| jamer+r1_limpia_u08 | 170 | 2.41 | 2.03 | -0.02 | 4.13 |

### Pareado por semana (MAPE h semanal, a - b; negativo = a mejor; IC95 bootstrap)

| muestra | a | b | semanas | dif_media_pp | ic95_inf | ic95_sup | semanas_gana_a |
|---|---|---|---|---|---|---|---|
| VF todas | jamer | jamer_envio | 33 | 0.074 | -0.184 | 0.359 | 17 |
| VF todas | jamer+festivo_por_tipo | jamer | 33 | -0.046 | -0.131 | 0.000 | 2 |
| VF todas | jamer+festivo_por_tipo | jamer_envio | 33 | 0.028 | -0.220 | 0.301 | 17 |
| VF todas | jamer+r1_limpia | jamer | 33 | -0.035 | -0.141 | 0.038 | 2 |
| VF todas | jamer+r1_limpia | jamer_envio | 33 | 0.039 | -0.179 | 0.280 | 17 |
| VF todas | jamer+saltar_puente | jamer | 33 | 0.012 | -0.085 | 0.129 | 8 |
| VF todas | jamer+saltar_puente | jamer_envio | 33 | 0.086 | -0.170 | 0.370 | 17 |
| VF todas | jamer+ambas | jamer | 33 | -0.081 | -0.214 | 0.020 | 4 |
| VF todas | jamer+ambas | jamer_envio | 33 | -0.007 | -0.212 | 0.214 | 17 |
| VF todas | jamer+r1_limpia_u04 | jamer | 33 | -0.027 | -0.160 | 0.100 | 4 |
| VF todas | jamer+r1_limpia_u04 | jamer_envio | 33 | 0.047 | -0.178 | 0.288 | 16 |
| VF todas | jamer+r1_limpia_u08 | jamer | 33 | -0.024 | -0.130 | 0.057 | 1 |
| VF todas | jamer+r1_limpia_u08 | jamer_envio | 33 | 0.050 | -0.170 | 0.288 | 17 |
| VF prueba | jamer | jamer_envio | 13 | -0.171 | -0.453 | 0.100 | 7 |
| VF prueba | jamer+festivo_por_tipo | jamer | 13 | 0.000 | 0.000 | 0.000 | 0 |
| VF prueba | jamer+festivo_por_tipo | jamer_envio | 13 | -0.171 | -0.453 | 0.100 | 7 |
| VF prueba | jamer+r1_limpia | jamer | 13 | 0.000 | 0.000 | 0.000 | 0 |
| VF prueba | jamer+r1_limpia | jamer_envio | 13 | -0.171 | -0.453 | 0.100 | 7 |
| VF prueba | jamer+saltar_puente | jamer | 13 | 0.039 | -0.154 | 0.309 | 3 |
| VF prueba | jamer+saltar_puente | jamer_envio | 13 | -0.132 | -0.427 | 0.137 | 7 |
| VF prueba | jamer+ambas | jamer | 13 | 0.000 | 0.000 | 0.000 | 0 |
| VF prueba | jamer+ambas | jamer_envio | 13 | -0.171 | -0.453 | 0.100 | 7 |
| VF prueba | jamer+r1_limpia_u04 | jamer | 13 | 0.000 | 0.000 | 0.000 | 0 |
| VF prueba | jamer+r1_limpia_u04 | jamer_envio | 13 | -0.171 | -0.453 | 0.100 | 7 |
| VF prueba | jamer+r1_limpia_u08 | jamer | 13 | 0.000 | 0.000 | 0.000 | 0 |
| VF prueba | jamer+r1_limpia_u08 | jamer_envio | 13 | -0.171 | -0.453 | 0.100 | 7 |
| final | jamer | jamer_envio | 33 | 0.100 | -0.198 | 0.393 | 14 |
| final | jamer+festivo_por_tipo | jamer | 33 | -0.046 | -0.131 | 0.000 | 2 |
| final | jamer+festivo_por_tipo | jamer_envio | 33 | 0.053 | -0.230 | 0.335 | 15 |
| final | jamer+r1_limpia | jamer | 33 | -0.036 | -0.108 | 0.016 | 2 |
| final | jamer+r1_limpia | jamer_envio | 33 | 0.063 | -0.218 | 0.331 | 15 |
| final | jamer+saltar_puente | jamer | 33 | 0.286 | -0.031 | 0.701 | 5 |
| final | jamer+saltar_puente | jamer_envio | 33 | 0.386 | -0.031 | 0.856 | 13 |
| final | jamer+ambas | jamer | 33 | -0.083 | -0.193 | 0.000 | 4 |
| final | jamer+ambas | jamer_envio | 33 | 0.017 | -0.253 | 0.273 | 16 |
| final | jamer+r1_limpia_u04 | jamer | 33 | -0.048 | -0.171 | 0.068 | 4 |
| final | jamer+r1_limpia_u04 | jamer_envio | 33 | 0.052 | -0.283 | 0.345 | 14 |
| final | jamer+r1_limpia_u08 | jamer | 33 | -0.018 | -0.078 | 0.025 | 1 |
| final | jamer+r1_limpia_u08 | jamer_envio | 33 | 0.082 | -0.200 | 0.346 | 14 |

### VF por tipo de día — MAPE h %

| tipo | dias | jamer_envio | jamer | jamer+festivo_por_tipo | jamer+r1_limpia | jamer+saltar_puente | jamer+ambas | jamer+r1_limpia_u04 | jamer+r1_limpia_u08 |
|---|---|---|---|---|---|---|---|---|---|
| domingo | 32 | 3.04 | 2.79 | 2.79 | 2.94 | 2.90 | 2.94 | 3.10 | 3.01 |
| festivo | 12 | 3.57 | 5.36 | 4.46 | 5.36 | 5.36 | 4.46 | 5.36 | 5.36 |
| laboral | 143 | 2.14 | 2.14 | 2.14 | 2.11 | 2.14 | 2.11 | 2.12 | 2.11 |
| laboral+post_festivo | 10 | 3.43 | 3.71 | 3.71 | 3.71 | 3.71 | 3.71 | 3.71 | 3.71 |
| sabado | 33 | 2.88 | 2.89 | 2.89 | 2.63 | 2.86 | 2.63 | 2.45 | 2.63 |

### VF por tipo de día — sesgo de energía %

| tipo_dia | jamer_envio | jamer | jamer+festivo_por_tipo | jamer+r1_limpia | jamer+saltar_puente | jamer+ambas | jamer+r1_limpia_u04 | jamer+r1_limpia_u08 |
|---|---|---|---|---|---|---|---|---|
| domingo | 0.54 | -0.03 | -0.03 | -0.33 | -0.07 | -0.33 | -0.59 | -0.25 |
| festivo | 0.73 | -1.94 | -0.91 | -1.94 | -1.94 | -0.91 | -1.94 | -1.94 |
| laboral | 0.56 | 0.01 | 0.01 | 0.05 | 0.01 | 0.05 | -0.01 | 0.05 |
| sabado | 0.49 | -0.31 | -0.31 | -0.06 | -0.30 | -0.06 | 0.17 | -0.06 |

### VF por lead (días desde el corte) — MAPE h %

| lead | dias | jamer_envio | jamer | jamer+festivo_por_tipo | jamer+r1_limpia | jamer+saltar_puente | jamer+ambas | jamer+r1_limpia_u04 | jamer+r1_limpia_u08 |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 32 | 2.72 | 3.03 | 3.03 | 3.03 | 3.03 | 3.03 | 2.94 | 3.03 |
| 7 | 33 | 2.37 | 2.36 | 2.36 | 2.36 | 2.36 | 2.36 | 2.37 | 2.36 |
| 8 | 33 | 2.29 | 2.34 | 2.34 | 2.22 | 2.34 | 2.22 | 2.27 | 2.22 |
| 9 | 33 | 2.05 | 1.97 | 1.97 | 1.97 | 1.97 | 1.97 | 2.02 | 1.97 |
| 10 | 33 | 2.12 | 2.60 | 2.28 | 2.60 | 2.60 | 2.28 | 2.66 | 2.60 |
| 11 | 33 | 2.88 | 2.88 | 2.88 | 2.62 | 2.86 | 2.62 | 2.45 | 2.62 |
| 12 | 32 | 2.90 | 2.79 | 2.79 | 2.79 | 2.90 | 2.79 | 2.96 | 2.87 |
| 13 | 1 | 8.71 | 4.03 | 4.03 | 8.49 | 4.03 | 8.49 | 8.49 | 8.49 |

### Casos revisados día a día (VF) — error de energía del día %

| fecha | jamer_envio | jamer | jamer+festivo_por_tipo | jamer+r1_limpia | jamer+saltar_puente | jamer+ambas | jamer+r1_limpia_u04 | jamer+r1_limpia_u08 |
|---|---|---|---|---|---|---|---|---|
| 2026-04-13 | -2.00 | -2.42 | -2.42 | -2.42 | -2.42 | -2.42 | -2.42 | -2.42 |
| 2026-04-18 | -2.46 | -10.93 | -10.93 | -2.53 | -10.93 | -2.53 | -2.53 | -2.53 |
| 2026-05-01 | -1.16 | -14.34 | -4.96 | -14.34 | -14.34 | -4.96 | -14.34 | -14.34 |
| 2026-08-07 | -1.65 | -5.33 | -2.99 | -5.33 | -5.33 | -2.99 | -5.33 | -5.33 |

## Parte 2 — muestra ampliada contra real (sin Jamer)

Desde 2022-01-01; corte = martes anterior (lunes - 6). Semanas: 218 (festivo 82, Semana Santa 10, con R1 saltada 63, control 40; una semana puede estar en varios grupos). R1 saltada = la R1 de nivel de `jamer+r1_limpia` difiere de la de `jamer`.

### MAPE h % por segmento

| segmento | dias | jamer | jamer+festivo_por_tipo | jamer+r1_limpia | jamer+saltar_puente | jamer+ambas | jamer+r1_limpia_u04 | jamer+r1_limpia_u08 |
|---|---|---|---|---|---|---|---|---|
| festivo lunes | 56 | 4.85 | 4.85 | 4.85 | 4.85 | 4.85 | 4.85 | 4.85 |
| festivo fijo | 23 | 6.63 | 5.52 | 6.63 | 6.63 | 5.52 | 6.63 | 6.63 |
| festivo semana_santa | 10 | 6.66 | 5.27 | 6.66 | 6.66 | 5.27 | 6.66 | 6.66 |
| festivo fijo+semana_santa 2022-2024 (calibración) | 20 | 6.25 | 5.49 | 6.25 | 6.25 | 5.49 | 6.25 | 6.25 |
| festivo fijo+semana_santa 2025-2026 (prueba) | 13 | 7.24 | 5.38 | 7.24 | 7.24 | 5.38 | 7.24 | 7.24 |
| festivos (todos) | 89 | 5.51 | 5.07 | 5.51 | 5.51 | 5.07 | 5.51 | 5.51 |
| semanas de Semana Santa | 70 | 4.50 | 4.30 | 4.47 | 4.74 | 4.27 | 4.29 | 4.50 |
| días con R1 saltada (R1 de jamer+r1_limpia != jamer) | 138 | 6.23 | 6.23 | 4.39 | 6.17 | 4.39 | 4.61 | 4.41 |
| control (semanas normales) | 280 | 2.84 | 2.84 | 2.84 | 2.84 | 2.84 | 2.91 | 2.84 |
| todo (muestra sesgada) | 1524 | 3.80 | 3.77 | 3.63 | 3.83 | 3.60 | 3.68 | 3.63 |

### Sesgo de energía % por segmento

| segmento | jamer | jamer+festivo_por_tipo | jamer+r1_limpia | jamer+saltar_puente | jamer+ambas | jamer+r1_limpia_u04 | jamer+r1_limpia_u08 |
|---|---|---|---|---|---|---|---|
| festivo lunes | -0.08 | -0.08 | -0.08 | -0.08 | -0.08 | -0.08 | -0.08 |
| festivo fijo | -3.08 | -1.62 | -3.08 | -3.08 | -1.62 | -3.08 | -3.08 |
| festivo semana_santa | -4.58 | 0.18 | -4.58 | -4.58 | 0.18 | -4.58 | -4.58 |
| festivo fijo+semana_santa 2022-2024 (calibración) | -1.93 | 0.45 | -1.93 | -1.93 | 0.45 | -1.93 | -1.93 |
| festivo fijo+semana_santa 2025-2026 (prueba) | -5.75 | -3.23 | -5.75 | -5.75 | -3.23 | -5.75 | -5.75 |
| festivos (todos) | -1.34 | -0.45 | -1.34 | -1.34 | -0.45 | -1.34 | -1.34 |
| semanas de Semana Santa | 0.58 | 1.16 | 0.81 | 0.45 | 1.39 | 0.78 | 0.58 |
| días con R1 saltada (R1 de jamer+r1_limpia != jamer) | -3.22 | -3.22 | 1.73 | -2.91 | 1.73 | 1.65 | 1.08 |
| control (semanas normales) | -0.46 | -0.46 | -0.46 | -0.46 | -0.46 | -0.57 | -0.46 |
| todo (muestra sesgada) | -0.05 | -0.01 | 0.38 | -0.08 | 0.42 | 0.45 | 0.32 |

Días con R1 saltada por motivo: atipico 65, temporada 46, semana_santa 27.

### Por tipo de día — MAPE h %

| tipo_dia | dias | jamer | jamer+festivo_por_tipo | jamer+r1_limpia | jamer+saltar_puente | jamer+ambas | jamer+r1_limpia_u04 | jamer+r1_limpia_u08 |
|---|---|---|---|---|---|---|---|---|
| domingo | 211 | 3.83 | 3.83 | 3.67 | 3.85 | 3.67 | 3.78 | 3.65 |
| festivo | 89 | 5.51 | 5.07 | 5.51 | 5.51 | 5.07 | 5.51 | 5.51 |
| laboral | 1008 | 3.62 | 3.62 | 3.48 | 3.62 | 3.48 | 3.52 | 3.49 |
| sabado | 216 | 3.87 | 3.87 | 3.52 | 4.08 | 3.52 | 3.58 | 3.47 |

## Parte 1 INFORMATIVO — no decide (muestra desde 2025-12-01)

Semanas: 48 (2025-12-01 .. 2026-11-16); calibración 29 (hasta 2026-07-06), prueba 19 (desde 2026-07-13). Base `ingenuo`; corte = envío - 1. VF evalúa solo días > corte: 0 día(s) VF con fecha <= corte descartados.

### VF, todas las semanas

| serie | dias | mape_h | abs_err_dia | sesgo_energia | p90_abs_err_dia |
|---|---|---|---|---|---|
| jamer_envio | 335 | 2.81 | 2.44 | 0.65 | 4.76 |
| jamer | 335 | 3.06 | 2.66 | -0.18 | 5.64 |
| jamer+festivo_por_tipo | 335 | 2.99 | 2.58 | -0.10 | 5.53 |
| jamer+r1_limpia | 335 | 2.99 | 2.58 | 0.03 | 5.53 |
| jamer+saltar_puente | 335 | 3.10 | 2.70 | -0.21 | 5.86 |
| jamer+ambas | 335 | 2.92 | 2.50 | 0.12 | 5.45 |
| jamer+r1_limpia_u04 | 335 | 2.97 | 2.57 | 0.02 | 5.52 |
| jamer+r1_limpia_u08 | 335 | 3.01 | 2.60 | 0.00 | 5.58 |

### VF, calibración

| serie | dias | mape_h | abs_err_dia | sesgo_energia | p90_abs_err_dia |
|---|---|---|---|---|---|
| jamer_envio | 203 | 3.04 | 2.64 | 0.27 | 5.02 |
| jamer | 203 | 3.58 | 3.19 | -0.55 | 7.05 |
| jamer+festivo_por_tipo | 203 | 3.46 | 3.07 | -0.42 | 6.66 |
| jamer+r1_limpia | 203 | 3.47 | 3.08 | -0.18 | 6.45 |
| jamer+saltar_puente | 203 | 3.63 | 3.26 | -0.58 | 7.05 |
| jamer+ambas | 203 | 3.36 | 2.96 | -0.05 | 6.26 |
| jamer+r1_limpia_u04 | 203 | 3.45 | 3.07 | -0.21 | 6.35 |
| jamer+r1_limpia_u08 | 203 | 3.49 | 3.10 | -0.24 | 6.66 |

### VF, prueba (fuera de muestra)

| serie | dias | mape_h | abs_err_dia | sesgo_energia | p90_abs_err_dia |
|---|---|---|---|---|---|
| jamer_envio | 132 | 2.47 | 2.14 | 1.20 | 4.50 |
| jamer | 132 | 2.27 | 1.84 | 0.36 | 3.63 |
| jamer+festivo_por_tipo | 132 | 2.25 | 1.82 | 0.37 | 3.59 |
| jamer+r1_limpia | 132 | 2.25 | 1.82 | 0.34 | 3.63 |
| jamer+saltar_puente | 132 | 2.28 | 1.84 | 0.33 | 3.73 |
| jamer+ambas | 132 | 2.23 | 1.80 | 0.36 | 3.59 |
| jamer+r1_limpia_u04 | 132 | 2.24 | 1.80 | 0.35 | 3.59 |
| jamer+r1_limpia_u08 | 132 | 2.27 | 1.84 | 0.36 | 3.63 |

### final (días posteriores al último envío)

| serie | dias | mape_h | abs_err_dia | sesgo_energia | p90_abs_err_dia |
|---|---|---|---|---|---|
| jamer_envio | 253 | 2.80 | 2.47 | 0.56 | 5.21 |
| jamer | 253 | 2.81 | 2.43 | -0.01 | 4.62 |
| jamer+festivo_por_tipo | 253 | 2.78 | 2.40 | 0.04 | 4.53 |
| jamer+r1_limpia | 253 | 2.76 | 2.37 | 0.17 | 4.62 |
| jamer+saltar_puente | 253 | 2.91 | 2.55 | -0.04 | 5.30 |
| jamer+ambas | 253 | 2.69 | 2.28 | 0.25 | 4.53 |
| jamer+r1_limpia_u04 | 253 | 2.81 | 2.42 | 0.11 | 4.73 |
| jamer+r1_limpia_u08 | 253 | 2.77 | 2.38 | 0.17 | 4.62 |

### Pareado por semana (MAPE h semanal, a - b; negativo = a mejor; IC95 bootstrap)

| muestra | a | b | semanas | dif_media_pp | ic95_inf | ic95_sup | semanas_gana_a |
|---|---|---|---|---|---|---|---|
| VF todas | jamer | jamer_envio | 48 | 0.246 | -0.115 | 0.595 | 23 |
| VF todas | jamer+festivo_por_tipo | jamer | 48 | -0.073 | -0.175 | -0.004 | 4 |
| VF todas | jamer+festivo_por_tipo | jamer_envio | 48 | 0.173 | -0.169 | 0.507 | 23 |
| VF todas | jamer+r1_limpia | jamer | 48 | -0.070 | -0.207 | 0.040 | 5 |
| VF todas | jamer+r1_limpia | jamer_envio | 48 | 0.175 | -0.142 | 0.478 | 22 |
| VF todas | jamer+saltar_puente | jamer | 48 | 0.041 | -0.044 | 0.141 | 9 |
| VF todas | jamer+saltar_puente | jamer_envio | 48 | 0.287 | -0.090 | 0.661 | 22 |
| VF todas | jamer+ambas | jamer | 48 | -0.143 | -0.302 | -0.002 | 9 |
| VF todas | jamer+ambas | jamer_envio | 48 | 0.103 | -0.210 | 0.392 | 22 |
| VF todas | jamer+r1_limpia_u04 | jamer | 48 | -0.084 | -0.288 | 0.085 | 8 |
| VF todas | jamer+r1_limpia_u04 | jamer_envio | 48 | 0.162 | -0.147 | 0.454 | 20 |
| VF todas | jamer+r1_limpia_u08 | jamer | 48 | -0.054 | -0.165 | 0.041 | 3 |
| VF todas | jamer+r1_limpia_u08 | jamer_envio | 48 | 0.192 | -0.136 | 0.506 | 23 |
| VF prueba | jamer | jamer_envio | 19 | -0.202 | -0.460 | 0.037 | 12 |
| VF prueba | jamer+festivo_por_tipo | jamer | 19 | -0.014 | -0.042 | 0.000 | 1 |
| VF prueba | jamer+festivo_por_tipo | jamer_envio | 19 | -0.216 | -0.473 | 0.023 | 12 |
| VF prueba | jamer+r1_limpia | jamer | 19 | -0.018 | -0.054 | 0.000 | 1 |
| VF prueba | jamer+r1_limpia | jamer_envio | 19 | -0.220 | -0.476 | 0.026 | 12 |
| VF prueba | jamer+saltar_puente | jamer | 19 | 0.014 | -0.124 | 0.211 | 5 |
| VF prueba | jamer+saltar_puente | jamer_envio | 19 | -0.188 | -0.445 | 0.066 | 12 |
| VF prueba | jamer+ambas | jamer | 19 | -0.032 | -0.078 | 0.000 | 2 |
| VF prueba | jamer+ambas | jamer_envio | 19 | -0.234 | -0.490 | 0.010 | 12 |
| VF prueba | jamer+r1_limpia_u04 | jamer | 19 | -0.028 | -0.088 | 0.023 | 2 |
| VF prueba | jamer+r1_limpia_u04 | jamer_envio | 19 | -0.229 | -0.490 | 0.019 | 11 |
| VF prueba | jamer+r1_limpia_u08 | jamer | 19 | 0.000 | 0.000 | 0.000 | 0 |
| VF prueba | jamer+r1_limpia_u08 | jamer_envio | 19 | -0.202 | -0.460 | 0.037 | 12 |
| final | jamer | jamer_envio | 48 | 0.079 | -0.277 | 0.412 | 22 |
| final | jamer+festivo_por_tipo | jamer | 48 | -0.021 | -0.149 | 0.106 | 3 |
| final | jamer+festivo_por_tipo | jamer_envio | 48 | 0.057 | -0.229 | 0.349 | 23 |
| final | jamer+r1_limpia | jamer | 48 | -0.037 | -0.113 | 0.027 | 5 |
| final | jamer+r1_limpia | jamer_envio | 48 | 0.042 | -0.313 | 0.373 | 22 |
| final | jamer+saltar_puente | jamer | 48 | 0.184 | -0.045 | 0.481 | 7 |
| final | jamer+saltar_puente | jamer_envio | 48 | 0.262 | -0.156 | 0.673 | 21 |
| final | jamer+ambas | jamer | 48 | -0.093 | -0.218 | 0.010 | 8 |
| final | jamer+ambas | jamer_envio | 48 | -0.014 | -0.329 | 0.286 | 23 |
| final | jamer+r1_limpia_u04 | jamer | 48 | -0.013 | -0.132 | 0.100 | 7 |
| final | jamer+r1_limpia_u04 | jamer_envio | 48 | 0.066 | -0.298 | 0.394 | 20 |
| final | jamer+r1_limpia_u08 | jamer | 48 | -0.024 | -0.092 | 0.028 | 2 |
| final | jamer+r1_limpia_u08 | jamer_envio | 48 | 0.055 | -0.300 | 0.384 | 22 |

### VF por tipo de día — MAPE h %

| tipo | dias | jamer_envio | jamer | jamer+festivo_por_tipo | jamer+r1_limpia | jamer+saltar_puente | jamer+ambas | jamer+r1_limpia_u04 | jamer+r1_limpia_u08 |
|---|---|---|---|---|---|---|---|---|---|
| domingo | 47 | 3.35 | 3.26 | 3.26 | 3.33 | 3.32 | 3.33 | 3.46 | 3.32 |
| festivo | 18 | 5.88 | 6.40 | 5.05 | 6.40 | 6.40 | 5.05 | 6.40 | 6.40 |
| laboral | 209 | 2.30 | 2.58 | 2.58 | 2.51 | 2.58 | 2.51 | 2.49 | 2.54 |
| laboral+post_festivo | 13 | 4.20 | 5.06 | 5.06 | 5.06 | 5.06 | 5.06 | 5.06 | 5.06 |
| sabado | 48 | 3.01 | 3.17 | 3.17 | 2.88 | 3.40 | 2.88 | 2.76 | 2.88 |

### VF por tipo de día — sesgo de energía %

| tipo_dia | jamer_envio | jamer | jamer+festivo_por_tipo | jamer+r1_limpia | jamer+saltar_puente | jamer+ambas | jamer+r1_limpia_u04 | jamer+r1_limpia_u08 |
|---|---|---|---|---|---|---|---|---|
| domingo | 0.43 | 0.03 | 0.03 | -0.06 | -0.11 | -0.06 | -0.07 | 0.07 |
| festivo | 3.52 | -3.69 | -1.90 | -3.69 | -3.69 | -1.90 | -3.69 | -3.69 |
| laboral | 0.51 | 0.02 | 0.02 | 0.26 | 0.02 | 0.26 | 0.21 | 0.20 |
| sabado | 0.58 | -0.18 | -0.18 | 0.21 | -0.29 | 0.21 | 0.38 | 0.21 |

### VF por lead (días desde el corte) — MAPE h %

| lead | dias | jamer_envio | jamer | jamer+festivo_por_tipo | jamer+r1_limpia | jamer+saltar_puente | jamer+ambas | jamer+r1_limpia_u04 | jamer+r1_limpia_u08 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 3.11 | 2.24 | 2.24 | 2.24 | 2.24 | 2.24 | 2.24 | 2.24 |
| 2 | 1 | 1.74 | 3.40 | 3.40 | 3.40 | 3.40 | 3.40 | 3.40 | 3.40 |
| 3 | 1 | 1.10 | 3.40 | 3.40 | 3.40 | 3.40 | 3.40 | 3.40 | 3.40 |
| 4 | 1 | 1.91 | 2.64 | 2.64 | 2.64 | 2.64 | 2.64 | 2.64 | 2.64 |
| 5 | 7 | 2.80 | 4.39 | 4.39 | 3.61 | 4.39 | 3.61 | 3.61 | 3.72 |
| 6 | 46 | 2.86 | 3.21 | 3.21 | 3.09 | 3.23 | 3.09 | 3.01 | 3.09 |
| 7 | 48 | 2.51 | 2.58 | 2.58 | 2.68 | 2.58 | 2.68 | 2.61 | 2.68 |
| 8 | 47 | 2.72 | 3.02 | 3.01 | 2.82 | 3.02 | 2.81 | 2.80 | 2.94 |
| 9 | 47 | 2.84 | 2.73 | 2.46 | 2.80 | 2.73 | 2.53 | 2.83 | 2.80 |
| 10 | 47 | 2.43 | 3.26 | 3.02 | 3.14 | 3.45 | 2.90 | 3.17 | 3.14 |
| 11 | 47 | 3.29 | 3.26 | 3.26 | 2.95 | 3.32 | 2.95 | 2.84 | 2.95 |
| 12 | 40 | 3.03 | 3.12 | 3.12 | 3.22 | 3.23 | 3.22 | 3.35 | 3.20 |
| 13 | 2 | 5.86 | 4.42 | 4.42 | 7.41 | 3.29 | 7.41 | 7.41 | 7.41 |

### Casos revisados día a día (VF) — error de energía del día %

| fecha | jamer_envio | jamer | jamer+festivo_por_tipo | jamer+r1_limpia | jamer+saltar_puente | jamer+ambas | jamer+r1_limpia_u04 | jamer+r1_limpia_u08 |
|---|---|---|---|---|---|---|---|---|
| 2026-04-13 | -2.00 | -2.42 | -2.42 | -2.42 | -2.42 | -2.42 | -2.42 | -2.42 |
| 2026-04-18 | -2.46 | -10.93 | -10.93 | -2.53 | -10.93 | -2.53 | -2.53 | -2.53 |
| 2026-05-01 | -1.16 | -14.34 | -4.96 | -14.34 | -14.34 | -4.96 | -14.34 | -14.34 |
| 2026-08-07 | -1.65 | -5.33 | -2.99 | -5.33 | -5.33 | -2.99 | -5.33 | -5.33 |

