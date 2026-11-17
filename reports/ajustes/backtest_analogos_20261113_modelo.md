# Backtest de días análogos (R3) — 20261113_modelo

Generado por `scripts/backtest_analogos.py` (importa `scripts/backtest_adjustments.py`). Detalle en `backtest_analogos_20261113_modelo.csv` (fuera de git; una fila por UCP × semana × día × paso × config).

## Configuración

- Bases: modelo_k1, ingenuo, archivo. `modelo_k1` = pronóstico del modelo real de la Fase 2a, sin fuga (curvas generadas con corte previo y verificadas; ver `docs/work/ajustes/fase2a/`).
- Semanas: últimas 52 completas por UCP; corte = sábado anterior (fecha_emision = domingo). Calibración = mitad más antigua, holdout = mitad más reciente. Base `archivo`: partición `archivo` aparte (no se elige configuración ahí).
- Sin fuga: `construir_contexto` verifica `max(historico.fecha) <= fecha_corte <= sábado`; además se verifica que toda fecha análoga usada por forma/nivel sea <= corte (aserción por día y config).
- Configs (11): `default`, `forma_referencia`, `forma_analogos_n3`, `forma_analogos_n3_t45`, `forma_analogos_n5`, `forma_analogos_n5_t45`, `forma_analogos_n8`, `forma_analogos_n8_t45`, `nivel_actual_a050`, `nivel_analogos_a025`, `nivel_analogos_a050`.
  - `default` = forma `promedio_n` n=4 + nivel ancla actual α=0.25.
  - `forma_*` cambian solo la forma (nivel default); `_nK` = k análogos; `_t45` = filtro `temporada_dias` 45 además de los automáticos (mismo día de semana y misma relación/distancia con festivo).
  - `nivel_*` cambian solo el nivel (forma default): ancla `actual`/`analogos`, α en centésimas.
  - `combo[...]` = mejor forma + mejor nivel elegidas en calibración (si ambas mejoran).
- Regla para proponer default por segmento: gana en calibración y holdout, en ambas UCP con base `modelo_k1`, por >= 2.0 % relativo del WAPE de cada paso afectado (forma; nivel y forma para configs de nivel/combo) y sin empeorar el error de pico (|error %| del máximo) en esos pasos. La base de la regla es `modelo_k1` (no cambia defaults del producto).
- Tiempo de corrida: nan s.

### Semanas evaluadas

| base | ucp | particion | primera_semana | ultima_semana | n_semanas |
|---|---|---|---|---|---|
| archivo | Antioquia | archivo | 2026-01-05 | 2026-05-25 | 21 |
| ingenuo | Antioquia | calibracion | 2025-06-30 | 2025-12-22 | 26 |
| ingenuo | Antioquia | holdout | 2025-12-29 | 2026-06-22 | 26 |
| ingenuo | Atlantico | calibracion | 2025-05-05 | 2025-10-27 | 26 |
| ingenuo | Atlantico | holdout | 2025-11-03 | 2026-04-27 | 26 |
| modelo_k1 | Antioquia | calibracion | 2025-06-30 | 2025-12-22 | 26 |
| modelo_k1 | Antioquia | holdout | 2025-12-29 | 2026-06-22 | 26 |
| modelo_k1 | Atlantico | calibracion | 2025-05-05 | 2025-10-27 | 26 |
| modelo_k1 | Atlantico | holdout | 2025-11-03 | 2026-04-27 | 26 |

## 1. Resultado global (todos los días)

WAPE horario % del paso final (forma) por config; columnas = base·UCP·partición.

| config | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo | modelo_k1·Antioquia·calibracion | modelo_k1·Antioquia·holdout | modelo_k1·Atlantico·calibracion | modelo_k1·Atlantico·holdout |
|---|---|---|---|---|---|---|---|---|---|
| default | 3.58 | 3.82 | 5.93 | 5.86 | 2.79 | 2.57 | 2.60 | 4.96 | 5.15 |
| forma_referencia | 3.68 | 3.99 | 6.31 | 6.46 | 2.86 | 2.67 | 2.75 | 5.37 | 5.83 |
| forma_analogos_n3 | 3.64 | 3.94 | 6.22 | 6.23 | 2.82 | 2.60 | 2.71 | 5.27 | 5.49 |
| forma_analogos_n3_t45 | 3.68 | 3.91 | 6.22 | 6.21 | 2.80 | 2.63 | 2.68 | 5.28 | 5.49 |
| forma_analogos_n5 | 3.66 | 3.96 | 6.20 | 6.20 | 2.85 | 2.62 | 2.75 | 5.32 | 5.47 |
| forma_analogos_n5_t45 | 3.68 | 3.94 | 6.20 | 6.18 | 2.83 | 2.65 | 2.72 | 5.33 | 5.48 |
| forma_analogos_n8 | 3.71 | 4.04 | 6.26 | 6.23 | 2.90 | 2.67 | 2.82 | 5.41 | 5.49 |
| forma_analogos_n8_t45 | 3.73 | 4.01 | 6.27 | 6.22 | 2.88 | 2.69 | 2.78 | 5.42 | 5.50 |
| nivel_actual_a050 | 3.51 | 3.63 | 6.12 | 5.96 | 2.87 | 2.81 | 2.81 | 5.40 | 5.35 |
| nivel_analogos_a025 | 3.48 | 3.73 | 5.97 | 5.85 | 2.52 | 2.39 | 2.45 | 4.97 | 5.09 |
| nivel_analogos_a050 | 3.22 | 3.36 | 5.91 | 5.76 | 2.41 | 2.41 | 2.54 | 5.14 | 5.10 |

WAPE % del paso nivel (solo cambia en configs de nivel/combo):

| config | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo | modelo_k1·Antioquia·calibracion | modelo_k1·Antioquia·holdout | modelo_k1·Atlantico·calibracion | modelo_k1·Atlantico·holdout |
|---|---|---|---|---|---|---|---|---|---|
| default | 3.64 | 3.88 | 6.29 | 5.99 | 2.87 | 2.62 | 2.71 | 5.18 | 5.44 |
| nivel_actual_a050 | 3.62 | 3.76 | 6.48 | 6.10 | 2.95 | 2.86 | 2.90 | 5.64 | 5.63 |
| nivel_analogos_a025 | 3.56 | 3.80 | 6.35 | 5.98 | 2.58 | 2.46 | 2.57 | 5.15 | 5.37 |
| nivel_analogos_a050 | 3.35 | 3.51 | 6.32 | 5.92 | 2.47 | 2.48 | 2.63 | 5.31 | 5.38 |

Mejora relativa % del WAPE vs default (positivo = mejor):

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo | modelo_k1·Antioquia·calibracion | modelo_k1·Antioquia·holdout | modelo_k1·Atlantico·calibracion | modelo_k1·Atlantico·holdout |
|---|---|---|---|---|---|---|---|---|---|---|
| forma_referencia | forma | -2.56 | -4.30 | -6.35 | -10.23 | -2.36 | -3.89 | -5.72 | -8.15 | -13.21 |
| forma_analogos_n3 | forma | -1.60 | -2.99 | -4.78 | -6.30 | -1.07 | -1.37 | -4.03 | -6.20 | -6.55 |
| forma_analogos_n3_t45 | forma | -2.64 | -2.25 | -4.85 | -5.86 | -0.35 | -2.45 | -2.97 | -6.39 | -6.54 |
| forma_analogos_n5 | forma | -2.08 | -3.70 | -4.44 | -5.65 | -2.17 | -1.99 | -5.43 | -7.25 | -6.18 |
| forma_analogos_n5_t45 | forma | -2.76 | -2.93 | -4.50 | -5.43 | -1.46 | -2.95 | -4.31 | -7.40 | -6.38 |
| forma_analogos_n8 | forma | -3.58 | -5.72 | -5.56 | -6.24 | -3.80 | -3.97 | -8.23 | -9.08 | -6.58 |
| forma_analogos_n8_t45 | forma | -3.99 | -4.81 | -5.65 | -5.98 | -3.10 | -4.78 | -6.84 | -9.32 | -6.75 |
| nivel_actual_a050 | nivel | 0.74 | 3.17 | -2.97 | -1.74 | -2.92 | -9.01 | -6.93 | -8.76 | -3.44 |
| nivel_actual_a050 | forma | 1.94 | 5.04 | -3.23 | -1.67 | -2.83 | -9.45 | -7.90 | -8.82 | -3.84 |
| nivel_analogos_a025 | nivel | 2.18 | 2.07 | -0.86 | 0.21 | 9.83 | 6.17 | 5.11 | 0.61 | 1.27 |
| nivel_analogos_a025 | forma | 2.89 | 2.40 | -0.58 | 0.25 | 9.81 | 6.97 | 5.82 | -0.19 | 1.18 |
| nivel_analogos_a050 | nivel | 7.98 | 9.63 | -0.47 | 1.16 | 13.77 | 5.54 | 2.71 | -2.40 | 1.17 |
| nivel_analogos_a050 | forma | 10.15 | 12.09 | 0.37 | 1.77 | 13.65 | 6.18 | 2.56 | -3.53 | 0.99 |

### Todas las métricas del paso afectado

WAPE horario; energía diaria (MAPE y sesgo %); pico (|error %| y diferencia de hora); energía semanal (MAPE y sesgo %).

| base | ucp | particion | config | paso | n_dias | wape_pct | mape_energia_dia_pct | sesgo_energia_dia_pct | err_pico_pct | dif_hora_pico_h | n_semanas | mape_energia_semana_pct | sesgo_energia_semana_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| modelo_k1 | Antioquia | calibracion | default | base | 182.00 | 2.67 | 2.35 | -1.59 | 2.43 | 1.85 | 26 | 1.83 | -1.61 |
| modelo_k1 | Antioquia | calibracion | default | nivel | 182.00 | 2.62 | 2.35 | -1.19 | 2.33 | 1.85 | 26 | 1.66 | -1.21 |
| modelo_k1 | Antioquia | calibracion | default | forma | 182.00 | 2.57 | 2.35 | -1.19 | 2.25 | 1.48 | 26 | 1.66 | -1.21 |
| modelo_k1 | Antioquia | calibracion | forma_referencia | forma | 182.00 | 2.67 | 2.35 | -1.19 | 2.22 | 2.65 | 26 | 1.66 | -1.21 |
| modelo_k1 | Antioquia | calibracion | forma_analogos_n3 | forma | 182.00 | 2.60 | 2.35 | -1.19 | 2.27 | 2.45 | 26 | 1.66 | -1.21 |
| modelo_k1 | Antioquia | calibracion | forma_analogos_n3_t45 | forma | 182.00 | 2.63 | 2.35 | -1.19 | 2.26 | 2.42 | 26 | 1.66 | -1.21 |
| modelo_k1 | Antioquia | calibracion | forma_analogos_n5 | forma | 182.00 | 2.62 | 2.35 | -1.19 | 2.18 | 2.43 | 26 | 1.66 | -1.21 |
| modelo_k1 | Antioquia | calibracion | forma_analogos_n5_t45 | forma | 182.00 | 2.65 | 2.35 | -1.19 | 2.21 | 2.45 | 26 | 1.66 | -1.21 |
| modelo_k1 | Antioquia | calibracion | forma_analogos_n8 | forma | 182.00 | 2.67 | 2.35 | -1.19 | 2.10 | 2.71 | 26 | 1.66 | -1.21 |
| modelo_k1 | Antioquia | calibracion | forma_analogos_n8_t45 | forma | 182.00 | 2.69 | 2.35 | -1.19 | 2.14 | 2.74 | 26 | 1.66 | -1.21 |
| modelo_k1 | Antioquia | calibracion | nivel_actual_a050 | nivel | 182.00 | 2.86 | 2.66 | -0.79 | 2.52 | 1.85 | 26 | 1.65 | -0.81 |
| modelo_k1 | Antioquia | calibracion | nivel_actual_a050 | forma | 182.00 | 2.81 | 2.66 | -0.79 | 2.45 | 1.48 | 26 | 1.65 | -0.81 |
| modelo_k1 | Antioquia | calibracion | nivel_analogos_a025 | nivel | 182.00 | 2.46 | 2.10 | -1.12 | 2.23 | 1.85 | 26 | 1.63 | -1.15 |
| modelo_k1 | Antioquia | calibracion | nivel_analogos_a025 | forma | 182.00 | 2.39 | 2.10 | -1.12 | 2.10 | 1.48 | 26 | 1.63 | -1.15 |
| modelo_k1 | Antioquia | calibracion | nivel_analogos_a050 | nivel | 182.00 | 2.48 | 2.17 | -0.66 | 2.25 | 1.85 | 26 | 1.59 | -0.68 |
| modelo_k1 | Antioquia | calibracion | nivel_analogos_a050 | forma | 182.00 | 2.41 | 2.17 | -0.66 | 2.06 | 1.48 | 26 | 1.59 | -0.68 |
| modelo_k1 | Antioquia | holdout | default | base | 182.00 | 2.74 | 2.40 | -0.12 | 2.41 | 2.15 | 26 | 1.78 | -0.14 |
| modelo_k1 | Antioquia | holdout | default | nivel | 182.00 | 2.71 | 2.37 | -0.14 | 2.43 | 2.15 | 26 | 1.64 | -0.16 |
| modelo_k1 | Antioquia | holdout | default | forma | 182.00 | 2.60 | 2.37 | -0.14 | 2.32 | 2.03 | 26 | 1.64 | -0.16 |
| modelo_k1 | Antioquia | holdout | forma_referencia | forma | 182.00 | 2.75 | 2.37 | -0.14 | 2.33 | 2.55 | 26 | 1.64 | -0.16 |
| modelo_k1 | Antioquia | holdout | forma_analogos_n3 | forma | 182.00 | 2.71 | 2.37 | -0.14 | 2.33 | 2.26 | 26 | 1.64 | -0.16 |
| modelo_k1 | Antioquia | holdout | forma_analogos_n3_t45 | forma | 182.00 | 2.68 | 2.37 | -0.14 | 2.34 | 2.20 | 26 | 1.64 | -0.16 |
| modelo_k1 | Antioquia | holdout | forma_analogos_n5 | forma | 182.00 | 2.75 | 2.37 | -0.14 | 2.38 | 2.26 | 26 | 1.64 | -0.16 |
| modelo_k1 | Antioquia | holdout | forma_analogos_n5_t45 | forma | 182.00 | 2.72 | 2.37 | -0.14 | 2.39 | 2.11 | 26 | 1.64 | -0.16 |
| modelo_k1 | Antioquia | holdout | forma_analogos_n8 | forma | 182.00 | 2.82 | 2.37 | -0.14 | 2.44 | 2.81 | 26 | 1.64 | -0.16 |
| modelo_k1 | Antioquia | holdout | forma_analogos_n8_t45 | forma | 182.00 | 2.78 | 2.37 | -0.14 | 2.43 | 2.68 | 26 | 1.64 | -0.16 |
| modelo_k1 | Antioquia | holdout | nivel_actual_a050 | nivel | 182.00 | 2.90 | 2.58 | -0.17 | 2.64 | 2.15 | 26 | 1.65 | -0.19 |
| modelo_k1 | Antioquia | holdout | nivel_actual_a050 | forma | 182.00 | 2.81 | 2.58 | -0.17 | 2.57 | 2.03 | 26 | 1.65 | -0.19 |
| modelo_k1 | Antioquia | holdout | nivel_analogos_a025 | nivel | 182.00 | 2.57 | 2.17 | -0.18 | 2.26 | 2.15 | 26 | 1.60 | -0.21 |
| modelo_k1 | Antioquia | holdout | nivel_analogos_a025 | forma | 182.00 | 2.45 | 2.17 | -0.18 | 2.18 | 2.03 | 26 | 1.60 | -0.21 |
| modelo_k1 | Antioquia | holdout | nivel_analogos_a050 | nivel | 182.00 | 2.63 | 2.26 | -0.24 | 2.42 | 2.15 | 26 | 1.52 | -0.27 |
| modelo_k1 | Antioquia | holdout | nivel_analogos_a050 | forma | 182.00 | 2.54 | 2.26 | -0.24 | 2.30 | 2.03 | 26 | 1.52 | -0.27 |
| modelo_k1 | Atlantico | calibracion | default | base | 182.00 | 5.22 | 4.37 | 0.09 | 4.66 | 0.77 | 26 | 2.39 | -0.10 |
| modelo_k1 | Atlantico | calibracion | default | nivel | 182.00 | 5.18 | 4.31 | 0.02 | 4.63 | 0.77 | 26 | 2.31 | -0.16 |
| modelo_k1 | Atlantico | calibracion | default | forma | 182.00 | 4.96 | 4.31 | 0.02 | 4.42 | 0.74 | 26 | 2.31 | -0.16 |
| modelo_k1 | Atlantico | calibracion | forma_referencia | forma | 182.00 | 5.37 | 4.31 | 0.02 | 4.47 | 1.10 | 26 | 2.31 | -0.16 |
| modelo_k1 | Atlantico | calibracion | forma_analogos_n3 | forma | 182.00 | 5.27 | 4.31 | 0.02 | 4.50 | 1.28 | 26 | 2.31 | -0.16 |
| modelo_k1 | Atlantico | calibracion | forma_analogos_n3_t45 | forma | 182.00 | 5.28 | 4.31 | 0.02 | 4.49 | 1.41 | 26 | 2.31 | -0.16 |
| modelo_k1 | Atlantico | calibracion | forma_analogos_n5 | forma | 182.00 | 5.32 | 4.31 | 0.02 | 4.74 | 1.25 | 26 | 2.31 | -0.16 |
| modelo_k1 | Atlantico | calibracion | forma_analogos_n5_t45 | forma | 182.00 | 5.33 | 4.31 | 0.02 | 4.72 | 1.35 | 26 | 2.31 | -0.16 |
| modelo_k1 | Atlantico | calibracion | forma_analogos_n8 | forma | 182.00 | 5.41 | 4.31 | 0.02 | 4.89 | 1.12 | 26 | 2.31 | -0.16 |
| modelo_k1 | Atlantico | calibracion | forma_analogos_n8_t45 | forma | 182.00 | 5.42 | 4.31 | 0.02 | 4.89 | 1.21 | 26 | 2.31 | -0.16 |
| modelo_k1 | Atlantico | calibracion | nivel_actual_a050 | nivel | 182.00 | 5.64 | 4.80 | -0.04 | 5.13 | 0.77 | 26 | 2.51 | -0.21 |
| modelo_k1 | Atlantico | calibracion | nivel_actual_a050 | forma | 182.00 | 5.40 | 4.80 | -0.04 | 4.98 | 0.74 | 26 | 2.51 | -0.21 |
| modelo_k1 | Atlantico | calibracion | nivel_analogos_a025 | nivel | 182.00 | 5.15 | 4.36 | 0.17 | 4.56 | 0.77 | 26 | 2.41 | -0.01 |
| modelo_k1 | Atlantico | calibracion | nivel_analogos_a025 | forma | 182.00 | 4.97 | 4.36 | 0.17 | 4.43 | 0.74 | 26 | 2.41 | -0.01 |
| modelo_k1 | Atlantico | calibracion | nivel_analogos_a050 | nivel | 182.00 | 5.31 | 4.58 | 0.24 | 4.71 | 0.77 | 26 | 2.43 | 0.08 |
| modelo_k1 | Atlantico | calibracion | nivel_analogos_a050 | forma | 182.00 | 5.14 | 4.58 | 0.24 | 4.62 | 0.74 | 26 | 2.43 | 0.08 |
| modelo_k1 | Atlantico | holdout | default | base | 182.00 | 5.57 | 4.86 | -0.22 | 5.01 | 0.90 | 26 | 3.81 | -0.34 |
| modelo_k1 | Atlantico | holdout | default | nivel | 182.00 | 5.44 | 4.63 | -0.10 | 4.84 | 0.90 | 26 | 3.66 | -0.22 |
| modelo_k1 | Atlantico | holdout | default | forma | 182.00 | 5.15 | 4.63 | -0.10 | 4.65 | 0.79 | 26 | 3.66 | -0.22 |
| modelo_k1 | Atlantico | holdout | forma_referencia | forma | 182.00 | 5.83 | 4.63 | -0.10 | 5.11 | 0.97 | 26 | 3.66 | -0.22 |
| modelo_k1 | Atlantico | holdout | forma_analogos_n3 | forma | 182.00 | 5.49 | 4.63 | -0.10 | 4.63 | 1.10 | 26 | 3.66 | -0.22 |
| modelo_k1 | Atlantico | holdout | forma_analogos_n3_t45 | forma | 182.00 | 5.49 | 4.63 | -0.10 | 4.63 | 1.12 | 26 | 3.66 | -0.22 |
| modelo_k1 | Atlantico | holdout | forma_analogos_n5 | forma | 182.00 | 5.47 | 4.63 | -0.10 | 4.79 | 1.08 | 26 | 3.66 | -0.22 |
| modelo_k1 | Atlantico | holdout | forma_analogos_n5_t45 | forma | 182.00 | 5.48 | 4.63 | -0.10 | 4.80 | 1.17 | 26 | 3.66 | -0.22 |
| modelo_k1 | Atlantico | holdout | forma_analogos_n8 | forma | 182.00 | 5.49 | 4.63 | -0.10 | 4.96 | 1.12 | 26 | 3.66 | -0.22 |
| modelo_k1 | Atlantico | holdout | forma_analogos_n8_t45 | forma | 182.00 | 5.50 | 4.63 | -0.10 | 4.99 | 1.16 | 26 | 3.66 | -0.22 |
| modelo_k1 | Atlantico | holdout | nivel_actual_a050 | nivel | 182.00 | 5.63 | 4.83 | 0.02 | 5.00 | 0.90 | 26 | 3.58 | -0.10 |
| modelo_k1 | Atlantico | holdout | nivel_actual_a050 | forma | 182.00 | 5.35 | 4.83 | 0.02 | 4.88 | 0.79 | 26 | 3.58 | -0.10 |
| modelo_k1 | Atlantico | holdout | nivel_analogos_a025 | nivel | 182.00 | 5.37 | 4.60 | -0.28 | 4.73 | 0.90 | 26 | 3.68 | -0.39 |
| modelo_k1 | Atlantico | holdout | nivel_analogos_a025 | forma | 182.00 | 5.09 | 4.60 | -0.28 | 4.59 | 0.79 | 26 | 3.68 | -0.39 |
| modelo_k1 | Atlantico | holdout | nivel_analogos_a050 | nivel | 182.00 | 5.38 | 4.58 | -0.33 | 4.68 | 0.90 | 26 | 3.66 | -0.43 |
| modelo_k1 | Atlantico | holdout | nivel_analogos_a050 | forma | 182.00 | 5.10 | 4.58 | -0.33 | 4.61 | 0.79 | 26 | 3.66 | -0.43 |
| ingenuo | Antioquia | calibracion | default | base | 182.00 | 3.87 | 3.77 | 0.19 | 3.30 | 1.42 | 26 | 2.57 | 0.06 |
| ingenuo | Antioquia | calibracion | default | nivel | 182.00 | 3.64 | 3.51 | 0.15 | 2.99 | 1.42 | 26 | 2.32 | 0.04 |
| ingenuo | Antioquia | calibracion | default | forma | 182.00 | 3.58 | 3.51 | 0.15 | 3.27 | 1.48 | 26 | 2.32 | 0.04 |
| ingenuo | Antioquia | calibracion | forma_referencia | forma | 182.00 | 3.68 | 3.51 | 0.15 | 3.37 | 2.65 | 26 | 2.32 | 0.04 |
| ingenuo | Antioquia | calibracion | forma_analogos_n3 | forma | 182.00 | 3.64 | 3.51 | 0.15 | 3.40 | 2.45 | 26 | 2.32 | 0.04 |
| ingenuo | Antioquia | calibracion | forma_analogos_n3_t45 | forma | 182.00 | 3.68 | 3.51 | 0.15 | 3.48 | 2.42 | 26 | 2.32 | 0.04 |
| ingenuo | Antioquia | calibracion | forma_analogos_n5 | forma | 182.00 | 3.66 | 3.51 | 0.15 | 3.37 | 2.43 | 26 | 2.32 | 0.04 |
| ingenuo | Antioquia | calibracion | forma_analogos_n5_t45 | forma | 182.00 | 3.68 | 3.51 | 0.15 | 3.43 | 2.45 | 26 | 2.32 | 0.04 |
| ingenuo | Antioquia | calibracion | forma_analogos_n8 | forma | 182.00 | 3.71 | 3.51 | 0.15 | 3.40 | 2.71 | 26 | 2.32 | 0.04 |
| ingenuo | Antioquia | calibracion | forma_analogos_n8_t45 | forma | 182.00 | 3.73 | 3.51 | 0.15 | 3.45 | 2.74 | 26 | 2.32 | 0.04 |
| ingenuo | Antioquia | calibracion | nivel_actual_a050 | nivel | 182.00 | 3.62 | 3.44 | 0.11 | 2.92 | 1.42 | 26 | 2.22 | 0.02 |
| ingenuo | Antioquia | calibracion | nivel_actual_a050 | forma | 182.00 | 3.51 | 3.44 | 0.11 | 3.16 | 1.48 | 26 | 2.22 | 0.02 |
| ingenuo | Antioquia | calibracion | nivel_analogos_a025 | nivel | 182.00 | 3.56 | 3.41 | 0.21 | 2.95 | 1.42 | 26 | 2.31 | 0.11 |
| ingenuo | Antioquia | calibracion | nivel_analogos_a025 | forma | 182.00 | 3.48 | 3.41 | 0.21 | 3.20 | 1.48 | 26 | 2.31 | 0.11 |
| ingenuo | Antioquia | calibracion | nivel_analogos_a050 | nivel | 182.00 | 3.35 | 3.13 | 0.23 | 2.66 | 1.42 | 26 | 2.16 | 0.15 |
| ingenuo | Antioquia | calibracion | nivel_analogos_a050 | forma | 182.00 | 3.22 | 3.13 | 0.23 | 2.94 | 1.48 | 26 | 2.16 | 0.15 |
| ingenuo | Antioquia | holdout | default | base | 182.00 | 4.22 | 4.07 | -0.42 | 3.80 | 2.27 | 26 | 2.80 | -0.56 |
| ingenuo | Antioquia | holdout | default | nivel | 182.00 | 3.88 | 3.67 | -0.37 | 3.43 | 2.27 | 26 | 2.39 | -0.48 |
| ingenuo | Antioquia | holdout | default | forma | 182.00 | 3.82 | 3.67 | -0.37 | 3.66 | 2.03 | 26 | 2.39 | -0.48 |
| ingenuo | Antioquia | holdout | forma_referencia | forma | 182.00 | 3.99 | 3.67 | -0.37 | 3.68 | 2.55 | 26 | 2.39 | -0.48 |
| ingenuo | Antioquia | holdout | forma_analogos_n3 | forma | 182.00 | 3.94 | 3.67 | -0.37 | 3.72 | 2.26 | 26 | 2.39 | -0.48 |
| ingenuo | Antioquia | holdout | forma_analogos_n3_t45 | forma | 182.00 | 3.91 | 3.67 | -0.37 | 3.74 | 2.20 | 26 | 2.39 | -0.48 |
| ingenuo | Antioquia | holdout | forma_analogos_n5 | forma | 182.00 | 3.96 | 3.67 | -0.37 | 3.73 | 2.26 | 26 | 2.39 | -0.48 |
| ingenuo | Antioquia | holdout | forma_analogos_n5_t45 | forma | 182.00 | 3.94 | 3.67 | -0.37 | 3.75 | 2.11 | 26 | 2.39 | -0.48 |
| ingenuo | Antioquia | holdout | forma_analogos_n8 | forma | 182.00 | 4.04 | 3.67 | -0.37 | 3.78 | 2.81 | 26 | 2.39 | -0.48 |
| ingenuo | Antioquia | holdout | forma_analogos_n8_t45 | forma | 182.00 | 4.01 | 3.67 | -0.37 | 3.78 | 2.68 | 26 | 2.39 | -0.48 |
| ingenuo | Antioquia | holdout | nivel_actual_a050 | nivel | 182.00 | 3.76 | 3.44 | -0.32 | 3.28 | 2.27 | 26 | 2.15 | -0.40 |
| ingenuo | Antioquia | holdout | nivel_actual_a050 | forma | 182.00 | 3.63 | 3.44 | -0.32 | 3.49 | 2.03 | 26 | 2.15 | -0.40 |
| ingenuo | Antioquia | holdout | nivel_analogos_a025 | nivel | 182.00 | 3.80 | 3.58 | -0.41 | 3.35 | 2.27 | 26 | 2.47 | -0.52 |
| ingenuo | Antioquia | holdout | nivel_analogos_a025 | forma | 182.00 | 3.73 | 3.58 | -0.41 | 3.60 | 2.03 | 26 | 2.47 | -0.52 |
| ingenuo | Antioquia | holdout | nivel_analogos_a050 | nivel | 182.00 | 3.51 | 3.18 | -0.39 | 2.99 | 2.27 | 26 | 2.22 | -0.48 |
| ingenuo | Antioquia | holdout | nivel_analogos_a050 | forma | 182.00 | 3.36 | 3.18 | -0.39 | 3.23 | 2.03 | 26 | 2.22 | -0.48 |
| ingenuo | Atlantico | calibracion | default | base | 182.00 | 6.42 | 5.55 | -0.12 | 5.54 | 1.05 | 26 | 2.89 | -0.29 |
| ingenuo | Atlantico | calibracion | default | nivel | 182.00 | 6.29 | 5.37 | -0.09 | 5.41 | 1.05 | 26 | 2.65 | -0.24 |
| ingenuo | Atlantico | calibracion | default | forma | 182.00 | 5.93 | 5.37 | -0.09 | 5.34 | 0.74 | 26 | 2.65 | -0.24 |
| ingenuo | Atlantico | calibracion | forma_referencia | forma | 182.00 | 6.31 | 5.37 | -0.09 | 5.48 | 1.10 | 26 | 2.65 | -0.24 |
| ingenuo | Atlantico | calibracion | forma_analogos_n3 | forma | 182.00 | 6.22 | 5.37 | -0.09 | 5.47 | 1.28 | 26 | 2.65 | -0.24 |
| ingenuo | Atlantico | calibracion | forma_analogos_n3_t45 | forma | 182.00 | 6.22 | 5.37 | -0.09 | 5.46 | 1.41 | 26 | 2.65 | -0.24 |
| ingenuo | Atlantico | calibracion | forma_analogos_n5 | forma | 182.00 | 6.20 | 5.37 | -0.09 | 5.55 | 1.25 | 26 | 2.65 | -0.24 |
| ingenuo | Atlantico | calibracion | forma_analogos_n5_t45 | forma | 182.00 | 6.20 | 5.37 | -0.09 | 5.54 | 1.35 | 26 | 2.65 | -0.24 |
| ingenuo | Atlantico | calibracion | forma_analogos_n8 | forma | 182.00 | 6.26 | 5.37 | -0.09 | 5.67 | 1.12 | 26 | 2.65 | -0.24 |
| ingenuo | Atlantico | calibracion | forma_analogos_n8_t45 | forma | 182.00 | 6.27 | 5.37 | -0.09 | 5.67 | 1.21 | 26 | 2.65 | -0.24 |
| ingenuo | Atlantico | calibracion | nivel_actual_a050 | nivel | 182.00 | 6.48 | 5.54 | -0.05 | 5.68 | 1.05 | 26 | 2.73 | -0.20 |
| ingenuo | Atlantico | calibracion | nivel_actual_a050 | forma | 182.00 | 6.12 | 5.54 | -0.05 | 5.61 | 0.74 | 26 | 2.73 | -0.20 |
| ingenuo | Atlantico | calibracion | nivel_analogos_a025 | nivel | 182.00 | 6.35 | 5.45 | 0.00 | 5.48 | 1.05 | 26 | 2.73 | -0.16 |
| ingenuo | Atlantico | calibracion | nivel_analogos_a025 | forma | 182.00 | 5.97 | 5.45 | 0.00 | 5.40 | 0.74 | 26 | 2.73 | -0.16 |
| ingenuo | Atlantico | calibracion | nivel_analogos_a050 | nivel | 182.00 | 6.32 | 5.41 | 0.13 | 5.49 | 1.05 | 26 | 2.64 | -0.02 |
| ingenuo | Atlantico | calibracion | nivel_analogos_a050 | forma | 182.00 | 5.91 | 5.41 | 0.13 | 5.35 | 0.74 | 26 | 2.64 | -0.02 |
| ingenuo | Atlantico | holdout | default | base | 182.00 | 6.12 | 5.51 | -0.30 | 5.44 | 0.98 | 26 | 4.28 | -0.45 |
| ingenuo | Atlantico | holdout | default | nivel | 182.00 | 5.99 | 5.33 | -0.07 | 5.37 | 0.98 | 26 | 3.98 | -0.21 |
| ingenuo | Atlantico | holdout | default | forma | 182.00 | 5.86 | 5.33 | -0.07 | 5.43 | 0.79 | 26 | 3.98 | -0.21 |
| ingenuo | Atlantico | holdout | forma_referencia | forma | 182.00 | 6.46 | 5.33 | -0.07 | 5.87 | 0.97 | 26 | 3.98 | -0.21 |
| ingenuo | Atlantico | holdout | forma_analogos_n3 | forma | 182.00 | 6.23 | 5.33 | -0.07 | 5.37 | 1.10 | 26 | 3.98 | -0.21 |
| ingenuo | Atlantico | holdout | forma_analogos_n3_t45 | forma | 182.00 | 6.21 | 5.33 | -0.07 | 5.26 | 1.12 | 26 | 3.98 | -0.21 |
| ingenuo | Atlantico | holdout | forma_analogos_n5 | forma | 182.00 | 6.20 | 5.33 | -0.07 | 5.51 | 1.08 | 26 | 3.98 | -0.21 |
| ingenuo | Atlantico | holdout | forma_analogos_n5_t45 | forma | 182.00 | 6.18 | 5.33 | -0.07 | 5.42 | 1.17 | 26 | 3.98 | -0.21 |
| ingenuo | Atlantico | holdout | forma_analogos_n8 | forma | 182.00 | 6.23 | 5.33 | -0.07 | 5.70 | 1.12 | 26 | 3.98 | -0.21 |
| ingenuo | Atlantico | holdout | forma_analogos_n8_t45 | forma | 182.00 | 6.22 | 5.33 | -0.07 | 5.63 | 1.16 | 26 | 3.98 | -0.21 |
| ingenuo | Atlantico | holdout | nivel_actual_a050 | nivel | 182.00 | 6.10 | 5.48 | 0.16 | 5.63 | 0.98 | 26 | 3.82 | 0.03 |
| ingenuo | Atlantico | holdout | nivel_actual_a050 | forma | 182.00 | 5.96 | 5.48 | 0.16 | 5.54 | 0.79 | 26 | 3.82 | 0.03 |
| ingenuo | Atlantico | holdout | nivel_analogos_a025 | nivel | 182.00 | 5.98 | 5.33 | -0.33 | 5.37 | 0.98 | 26 | 4.16 | -0.47 |
| ingenuo | Atlantico | holdout | nivel_analogos_a025 | forma | 182.00 | 5.85 | 5.33 | -0.33 | 5.40 | 0.79 | 26 | 4.16 | -0.47 |
| ingenuo | Atlantico | holdout | nivel_analogos_a050 | nivel | 182.00 | 5.92 | 5.24 | -0.37 | 5.41 | 0.98 | 26 | 4.05 | -0.49 |
| ingenuo | Atlantico | holdout | nivel_analogos_a050 | forma | 182.00 | 5.76 | 5.24 | -0.37 | 5.30 | 0.79 | 26 | 4.05 | -0.49 |
| archivo | Antioquia | archivo | default | base | 147.00 | 3.03 | 2.75 | -0.73 | 2.56 | 2.31 | 21 | 2.11 | -0.79 |
| archivo | Antioquia | archivo | default | nivel | 147.00 | 2.87 | 2.55 | -0.56 | 2.45 | 2.31 | 21 | 1.90 | -0.60 |
| archivo | Antioquia | archivo | default | forma | 147.00 | 2.79 | 2.55 | -0.56 | 2.60 | 2.11 | 21 | 1.90 | -0.60 |
| archivo | Antioquia | archivo | forma_referencia | forma | 147.00 | 2.86 | 2.55 | -0.56 | 2.40 | 2.43 | 21 | 1.90 | -0.60 |
| archivo | Antioquia | archivo | forma_analogos_n3 | forma | 147.00 | 2.82 | 2.55 | -0.56 | 2.54 | 2.08 | 21 | 1.90 | -0.60 |
| archivo | Antioquia | archivo | forma_analogos_n3_t45 | forma | 147.00 | 2.80 | 2.55 | -0.56 | 2.57 | 2.00 | 21 | 1.90 | -0.60 |
| archivo | Antioquia | archivo | forma_analogos_n5 | forma | 147.00 | 2.85 | 2.55 | -0.56 | 2.51 | 2.06 | 21 | 1.90 | -0.60 |
| archivo | Antioquia | archivo | forma_analogos_n5_t45 | forma | 147.00 | 2.83 | 2.55 | -0.56 | 2.54 | 1.95 | 21 | 1.90 | -0.60 |
| archivo | Antioquia | archivo | forma_analogos_n8 | forma | 147.00 | 2.90 | 2.55 | -0.56 | 2.55 | 2.61 | 21 | 1.90 | -0.60 |
| archivo | Antioquia | archivo | forma_analogos_n8_t45 | forma | 147.00 | 2.88 | 2.55 | -0.56 | 2.57 | 2.53 | 21 | 1.90 | -0.60 |
| archivo | Antioquia | archivo | nivel_actual_a050 | nivel | 147.00 | 2.95 | 2.65 | -0.38 | 2.62 | 2.31 | 21 | 1.84 | -0.41 |
| archivo | Antioquia | archivo | nivel_actual_a050 | forma | 147.00 | 2.87 | 2.65 | -0.38 | 2.69 | 2.11 | 21 | 1.84 | -0.41 |
| archivo | Antioquia | archivo | nivel_analogos_a025 | nivel | 147.00 | 2.58 | 2.26 | -0.74 | 2.14 | 2.31 | 21 | 1.67 | -0.78 |
| archivo | Antioquia | archivo | nivel_analogos_a025 | forma | 147.00 | 2.52 | 2.26 | -0.74 | 2.28 | 2.11 | 21 | 1.67 | -0.78 |
| archivo | Antioquia | archivo | nivel_analogos_a050 | nivel | 147.00 | 2.47 | 2.09 | -0.75 | 2.08 | 2.31 | 21 | 1.33 | -0.77 |
| archivo | Antioquia | archivo | nivel_analogos_a050 | forma | 147.00 | 2.41 | 2.09 | -0.75 | 2.18 | 2.11 | 21 | 1.33 | -0.77 |

### Uso real de la fuente pedida (fallbacks)

| base | ucp | config | dias | pct_forma_analogos | pct_forma_otra | pct_nivel_regla_analogos |
|---|---|---|---|---|---|---|
| modelo_k1 | Antioquia | forma_analogos_n3 | 364 | 95.1 | 4.9 | — |
| modelo_k1 | Antioquia | forma_analogos_n3_t45 | 364 | 94.2 | 5.8 | — |
| modelo_k1 | Antioquia | forma_analogos_n5 | 364 | 95.1 | 4.9 | — |
| modelo_k1 | Antioquia | forma_analogos_n5_t45 | 364 | 94.2 | 5.8 | — |
| modelo_k1 | Antioquia | forma_analogos_n8 | 364 | 95.1 | 4.9 | — |
| modelo_k1 | Antioquia | forma_analogos_n8_t45 | 364 | 94.2 | 5.8 | — |
| modelo_k1 | Atlantico | forma_analogos_n3 | 364 | 95.3 | 4.7 | — |
| modelo_k1 | Atlantico | forma_analogos_n3_t45 | 364 | 94.5 | 5.5 | — |
| modelo_k1 | Atlantico | forma_analogos_n5 | 364 | 95.3 | 4.7 | — |
| modelo_k1 | Atlantico | forma_analogos_n5_t45 | 364 | 94.5 | 5.5 | — |
| modelo_k1 | Atlantico | forma_analogos_n8 | 364 | 95.3 | 4.7 | — |
| modelo_k1 | Atlantico | forma_analogos_n8_t45 | 364 | 94.5 | 5.5 | — |
| ingenuo | Antioquia | forma_analogos_n3 | 364 | 95.1 | 4.9 | — |
| ingenuo | Antioquia | forma_analogos_n3_t45 | 364 | 94.2 | 5.8 | — |
| ingenuo | Antioquia | forma_analogos_n5 | 364 | 95.1 | 4.9 | — |
| ingenuo | Antioquia | forma_analogos_n5_t45 | 364 | 94.2 | 5.8 | — |
| ingenuo | Antioquia | forma_analogos_n8 | 364 | 95.1 | 4.9 | — |
| ingenuo | Antioquia | forma_analogos_n8_t45 | 364 | 94.2 | 5.8 | — |
| ingenuo | Atlantico | forma_analogos_n3 | 364 | 95.3 | 4.7 | — |
| ingenuo | Atlantico | forma_analogos_n3_t45 | 364 | 94.5 | 5.5 | — |
| ingenuo | Atlantico | forma_analogos_n5 | 364 | 95.3 | 4.7 | — |
| ingenuo | Atlantico | forma_analogos_n5_t45 | 364 | 94.5 | 5.5 | — |
| ingenuo | Atlantico | forma_analogos_n8 | 364 | 95.3 | 4.7 | — |
| ingenuo | Atlantico | forma_analogos_n8_t45 | 364 | 94.5 | 5.5 | — |
| archivo | Antioquia | forma_analogos_n3 | 147 | 95.9 | 4.1 | — |
| archivo | Antioquia | forma_analogos_n3_t45 | 147 | 93.9 | 6.1 | — |
| archivo | Antioquia | forma_analogos_n5 | 147 | 95.9 | 4.1 | — |
| archivo | Antioquia | forma_analogos_n5_t45 | 147 | 93.9 | 6.1 | — |
| archivo | Antioquia | forma_analogos_n8 | 147 | 95.9 | 4.1 | — |
| archivo | Antioquia | forma_analogos_n8_t45 | 147 | 93.9 | 6.1 | — |
| modelo_k1 | Antioquia | nivel_analogos_a025 | 364 | — | — | 97.3 |
| modelo_k1 | Antioquia | nivel_analogos_a050 | 364 | — | — | 97.3 |
| modelo_k1 | Atlantico | nivel_analogos_a025 | 364 | — | — | 97.3 |
| modelo_k1 | Atlantico | nivel_analogos_a050 | 364 | — | — | 97.3 |
| ingenuo | Antioquia | nivel_analogos_a025 | 364 | — | — | 97.3 |
| ingenuo | Antioquia | nivel_analogos_a050 | 364 | — | — | 97.3 |
| ingenuo | Atlantico | nivel_analogos_a025 | 364 | — | — | 97.3 |
| ingenuo | Atlantico | nivel_analogos_a050 | 364 | — | — | 97.3 |
| archivo | Antioquia | nivel_analogos_a025 | 147 | — | — | 98.0 |
| archivo | Antioquia | nivel_analogos_a050 | 147 | — | — | 98.0 |

## 2. Elección en calibración y resultado en holdout

- **modelo_k1 / Antioquia**: mejor config en calibración = `nivel_analogos_a025` (WAPE forma 2.390 %, mejora +6.97 % vs default) → holdout 2.453 % vs default 2.605 % (mejora +5.82 %).
- **modelo_k1 / Atlantico**: mejor config en calibración = `default` (WAPE forma 4.962 %, mejora +0.00 % vs default) → holdout 5.153 % vs default 5.153 % (mejora +0.00 %).
- Mejor variante de **forma** en calibración (media de mejora relativa sobre las UCP, modelo_k1): ninguna mejora al default.
- Mejor variante de **nivel** en calibración (media de mejora relativa sobre las UCP, modelo_k1): `nivel_analogos_a025`.
- No se corrió combinación (alguna de las dos mejores variantes no mejora al default).

## 3. Por segmento

Días por segmento (config default; los segmentos se solapan):

| segmento | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo | modelo_k1·Antioquia·calibracion | modelo_k1·Antioquia·holdout | modelo_k1·Atlantico·calibracion | modelo_k1·Atlantico·holdout |
|---|---|---|---|---|---|---|---|---|---|
| todos | 182 | 182 | 182 | 182 | 147 | 182 | 182 | 182 | 182 |
| tipo:laboral | 122 | 121 | 124 | 120 | 99 | 122 | 121 | 124 | 120 |
| tipo:sabado | 26 | 26 | 26 | 26 | 21 | 26 | 26 | 26 | 26 |
| tipo:domingo | 25 | 26 | 25 | 26 | 21 | 25 | 26 | 25 | 26 |
| tipo:festivo | 9 | 9 | 7 | 10 | 6 | 9 | 9 | 7 | 10 |
| rel:festivo | 9 | 9 | 7 | 10 | 6 | 9 | 9 | 7 | 10 |
| rel:puente | 13 | 17 | 14 | 14 | 10 | 13 | 17 | 14 | 14 |
| rel:antes | 13 | 15 | 11 | 16 | 9 | 13 | 15 | 11 | 16 |
| rel:despues | 25 | 19 | 20 | 20 | 11 | 25 | 19 | 20 | 20 |
| rel:ninguna | 122 | 122 | 130 | 122 | 111 | 122 | 122 | 130 | 122 |
| despues_dist1 | 7 | 5 | 6 | 5 | 3 | 7 | 5 | 6 | 5 |
| semana_santa | — | 8 | — | 8 | 8 | — | 8 | — | 8 |
| diciembre | 28 | 3 | — | 31 | — | 28 | 3 | — | 31 |
| normal | 108 | 121 | 130 | 107 | 110 | 108 | 121 | 130 | 107 |

WAPE % del paso forma con el default por segmento:

| segmento | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo | modelo_k1·Antioquia·calibracion | modelo_k1·Antioquia·holdout | modelo_k1·Atlantico·calibracion | modelo_k1·Atlantico·holdout |
|---|---|---|---|---|---|---|---|---|---|
| todos | 3.58 | 3.82 | 5.93 | 5.86 | 2.79 | 2.57 | 2.60 | 4.96 | 5.15 |
| tipo:laboral | 3.25 | 3.52 | 5.63 | 5.64 | 2.76 | 2.50 | 2.53 | 4.73 | 5.24 |
| tipo:sabado | 2.62 | 2.94 | 5.81 | 5.71 | 2.49 | 2.24 | 2.53 | 5.32 | 5.10 |
| tipo:domingo | 3.41 | 3.60 | 6.51 | 5.95 | 2.89 | 2.83 | 2.80 | 5.31 | 4.65 |
| tipo:festivo | 12.83 | 12.33 | 10.34 | 8.96 | 4.34 | 4.00 | 3.51 | 6.92 | 5.48 |
| rel:festivo | 12.83 | 12.33 | 10.34 | 8.96 | 4.34 | 4.00 | 3.51 | 6.92 | 5.48 |
| rel:puente | 3.51 | 3.45 | 6.08 | 5.49 | 3.24 | 2.74 | 2.91 | 6.31 | 5.20 |
| rel:antes | 3.55 | 3.30 | 5.53 | 4.08 | 2.41 | 3.10 | 2.29 | 4.49 | 4.45 |
| rel:despues | 2.24 | 3.00 | 4.69 | 3.74 | 3.56 | 2.21 | 2.79 | 3.96 | 4.05 |
| rel:ninguna | 3.30 | 3.53 | 5.93 | 6.25 | 2.64 | 2.48 | 2.52 | 4.93 | 5.39 |
| despues_dist1 | 2.54 | 4.71 | 4.98 | 3.78 | 3.90 | 2.22 | 3.84 | 3.82 | 4.26 |
| semana_santa | — | 8.37 | — | 8.13 | 3.60 | — | 3.10 | — | 4.19 |
| diciembre | 5.13 | 3.42 | — | 6.51 | — | 2.95 | 4.56 | — | 5.44 |
| normal | 3.21 | 3.55 | 5.93 | 6.22 | 2.62 | 2.45 | 2.51 | 4.93 | 5.50 |

### Segmento `tipo:laboral` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo | modelo_k1·Antioquia·calibracion | modelo_k1·Antioquia·holdout | modelo_k1·Atlantico·calibracion | modelo_k1·Atlantico·holdout |
|---|---|---|---|---|---|---|---|---|---|---|
| forma_referencia | forma | -2.8 | -3.4 | -7.7 | -10.4 | -2.3 | -4.4 | -4.7 | -9.3 | -11.1 |
| forma_analogos_n3 | forma | -1.1 | -2.7 | -5.3 | -7.0 | -0.7 | -1.0 | -3.0 | -6.2 | -6.5 |
| forma_analogos_n3_t45 | forma | -2.4 | -1.5 | -5.5 | -5.9 | 0.3 | -2.2 | -1.4 | -6.5 | -6.1 |
| forma_analogos_n5 | forma | -1.7 | -3.2 | -5.0 | -6.9 | -1.5 | -1.2 | -4.3 | -7.5 | -6.8 |
| forma_analogos_n5_t45 | forma | -2.6 | -2.0 | -5.1 | -6.1 | -0.5 | -2.3 | -2.6 | -7.7 | -6.6 |
| forma_analogos_n8 | forma | -3.6 | -5.4 | -6.7 | -7.7 | -3.3 | -4.0 | -7.5 | -10.0 | -7.4 |
| forma_analogos_n8_t45 | forma | -4.1 | -3.9 | -6.9 | -6.8 | -2.3 | -4.9 | -5.4 | -10.3 | -7.1 |
| nivel_actual_a050 | nivel | -0.7 | 0.3 | -3.8 | -3.2 | -1.2 | -6.1 | -6.6 | -8.4 | -4.5 |
| nivel_actual_a050 | forma | -0.1 | 1.8 | -3.9 | -3.2 | -1.4 | -6.5 | -7.7 | -8.6 | -4.5 |
| nivel_analogos_a025 | nivel | 1.0 | 2.5 | -0.6 | 1.0 | 9.8 | 5.2 | 6.3 | 1.1 | 3.0 |
| nivel_analogos_a025 | forma | 1.8 | 2.8 | -0.6 | 0.6 | 9.6 | 6.0 | 6.8 | 0.4 | 3.3 |
| nivel_analogos_a050 | nivel | 4.2 | 8.0 | -1.8 | 1.7 | 15.2 | 5.3 | 5.0 | -3.9 | 2.8 |
| nivel_analogos_a050 | forma | 6.0 | 9.8 | -1.6 | 2.0 | 15.1 | 5.7 | 4.5 | -5.2 | 3.6 |

### Segmento `tipo:sabado` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo | modelo_k1·Antioquia·calibracion | modelo_k1·Antioquia·holdout | modelo_k1·Atlantico·calibracion | modelo_k1·Atlantico·holdout |
|---|---|---|---|---|---|---|---|---|---|---|
| forma_referencia | forma | -1.9 | -14.2 | -6.2 | -12.1 | -2.2 | -4.9 | -12.7 | -8.5 | -17.1 |
| forma_analogos_n3 | forma | -3.6 | -5.0 | -5.0 | -7.2 | 3.1 | -3.8 | -7.5 | -7.4 | -7.4 |
| forma_analogos_n3_t45 | forma | -4.5 | -5.0 | -5.0 | -7.6 | 3.1 | -5.0 | -7.5 | -7.4 | -7.4 |
| forma_analogos_n5 | forma | -3.6 | -5.2 | -4.6 | -5.7 | 2.3 | -5.1 | -7.0 | -6.8 | -5.9 |
| forma_analogos_n5_t45 | forma | -4.0 | -5.2 | -4.6 | -5.9 | 2.3 | -5.6 | -7.0 | -6.8 | -5.6 |
| forma_analogos_n8 | forma | -5.4 | -9.7 | -5.0 | -6.5 | -1.0 | -6.5 | -10.9 | -7.8 | -5.8 |
| forma_analogos_n8_t45 | forma | -5.8 | -9.7 | -5.0 | -6.6 | -1.0 | -7.0 | -10.9 | -7.8 | -5.5 |
| nivel_actual_a050 | nivel | -6.0 | -2.3 | 2.0 | -2.7 | -11.0 | -7.0 | -9.8 | -4.1 | -2.4 |
| nivel_actual_a050 | forma | -6.5 | -1.3 | 1.9 | -3.2 | -8.9 | -8.1 | -9.5 | -4.8 | -4.0 |
| nivel_analogos_a025 | nivel | 3.5 | -0.1 | -5.0 | -3.6 | 12.8 | 1.1 | 7.2 | 1.9 | 0.5 |
| nivel_analogos_a025 | forma | 3.0 | -0.5 | -5.8 | -1.4 | 13.6 | 1.8 | 8.5 | 1.3 | 0.0 |
| nivel_analogos_a050 | nivel | 2.9 | 5.8 | -4.5 | -3.7 | 10.7 | -0.6 | 8.9 | 2.5 | 5.0 |
| nivel_analogos_a050 | forma | 2.2 | 6.5 | -5.0 | -1.4 | 10.4 | -0.3 | 9.3 | 0.8 | 2.4 |

### Segmento `tipo:domingo` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo | modelo_k1·Antioquia·calibracion | modelo_k1·Antioquia·holdout | modelo_k1·Atlantico·calibracion | modelo_k1·Atlantico·holdout |
|---|---|---|---|---|---|---|---|---|---|---|
| forma_referencia | forma | -5.1 | -5.2 | -2.9 | -13.6 | -3.7 | -2.4 | -6.2 | -5.3 | -26.8 |
| forma_analogos_n3 | forma | -4.9 | -6.5 | -4.0 | -5.5 | -7.4 | -1.6 | -7.4 | -7.2 | -8.9 |
| forma_analogos_n3_t45 | forma | -5.9 | -6.5 | -3.8 | -7.3 | -7.4 | -2.8 | -7.8 | -7.2 | -11.1 |
| forma_analogos_n5 | forma | -5.7 | -9.4 | -3.7 | -2.7 | -11.1 | -4.1 | -11.8 | -9.1 | -5.5 |
| forma_analogos_n5_t45 | forma | -6.5 | -9.4 | -3.6 | -5.2 | -11.1 | -5.2 | -12.3 | -9.1 | -8.7 |
| forma_analogos_n8 | forma | -6.4 | -10.7 | -3.1 | -2.6 | -10.8 | -3.6 | -12.7 | -9.4 | -5.9 |
| forma_analogos_n8_t45 | forma | -7.3 | -10.7 | -2.9 | -5.0 | -10.8 | -4.6 | -13.2 | -9.4 | -9.0 |
| nivel_actual_a050 | nivel | -15.8 | 0.8 | -9.6 | 0.1 | -7.3 | -19.7 | -8.0 | -17.2 | -0.5 |
| nivel_actual_a050 | forma | -15.8 | 1.5 | -11.3 | 0.5 | -8.3 | -20.0 | -8.9 | -16.2 | -2.1 |
| nivel_analogos_a025 | nivel | 10.0 | 1.1 | 0.5 | -0.9 | 6.4 | 13.6 | 2.4 | -4.9 | -3.1 |
| nivel_analogos_a025 | forma | 12.2 | 1.3 | 2.8 | -0.3 | 6.4 | 13.6 | 3.4 | -6.5 | -3.6 |
| nivel_analogos_a050 | nivel | 11.8 | 3.9 | 2.9 | -0.3 | 6.0 | 15.6 | -2.1 | -3.7 | -4.5 |
| nivel_analogos_a050 | forma | 13.8 | 4.6 | 7.0 | 0.7 | 5.4 | 16.8 | -0.4 | -3.1 | -6.3 |

### Segmento `tipo:festivo` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo | modelo_k1·Antioquia·calibracion | modelo_k1·Antioquia·holdout | modelo_k1·Atlantico·calibracion | modelo_k1·Atlantico·holdout |
|---|---|---|---|---|---|---|---|---|---|---|
| forma_referencia | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n3 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n3_t45 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n5 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n5_t45 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n8 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n8_t45 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| nivel_actual_a050 | nivel | 23.2 | 22.7 | 10.2 | 8.5 | 3.4 | -22.1 | -2.7 | -5.3 | 0.4 |
| nivel_actual_a050 | forma | 29.2 | 28.0 | 10.4 | 9.5 | 5.7 | -21.4 | -3.7 | -4.0 | 1.2 |
| nivel_analogos_a025 | nivel | 0.3 | 2.8 | 2.5 | 2.2 | 12.2 | 11.5 | -7.1 | 5.0 | -7.4 |
| nivel_analogos_a025 | forma | 0.3 | 3.9 | 4.0 | 1.4 | 12.2 | 13.2 | -6.6 | 4.0 | -11.7 |
| nivel_analogos_a050 | nivel | 24.5 | 25.1 | 15.5 | 7.5 | 20.5 | -0.5 | -27.0 | 6.6 | -15.0 |
| nivel_analogos_a050 | forma | 30.2 | 33.3 | 18.5 | 7.0 | 21.4 | 1.9 | -29.3 | 5.0 | -18.8 |

### Segmento `rel:festivo` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo | modelo_k1·Antioquia·calibracion | modelo_k1·Antioquia·holdout | modelo_k1·Atlantico·calibracion | modelo_k1·Atlantico·holdout |
|---|---|---|---|---|---|---|---|---|---|---|
| forma_referencia | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n3 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n3_t45 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n5 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n5_t45 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n8 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n8_t45 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| nivel_actual_a050 | nivel | 23.2 | 22.7 | 10.2 | 8.5 | 3.4 | -22.1 | -2.7 | -5.3 | 0.4 |
| nivel_actual_a050 | forma | 29.2 | 28.0 | 10.4 | 9.5 | 5.7 | -21.4 | -3.7 | -4.0 | 1.2 |
| nivel_analogos_a025 | nivel | 0.3 | 2.8 | 2.5 | 2.2 | 12.2 | 11.5 | -7.1 | 5.0 | -7.4 |
| nivel_analogos_a025 | forma | 0.3 | 3.9 | 4.0 | 1.4 | 12.2 | 13.2 | -6.6 | 4.0 | -11.7 |
| nivel_analogos_a050 | nivel | 24.5 | 25.1 | 15.5 | 7.5 | 20.5 | -0.5 | -27.0 | 6.6 | -15.0 |
| nivel_analogos_a050 | forma | 30.2 | 33.3 | 18.5 | 7.0 | 21.4 | 1.9 | -29.3 | 5.0 | -18.8 |

### Segmento `rel:puente` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo | modelo_k1·Antioquia·calibracion | modelo_k1·Antioquia·holdout | modelo_k1·Atlantico·calibracion | modelo_k1·Atlantico·holdout |
|---|---|---|---|---|---|---|---|---|---|---|
| forma_referencia | forma | -6.3 | -8.9 | -2.3 | -18.0 | -1.7 | -5.5 | -12.3 | -2.6 | -22.2 |
| forma_analogos_n3 | forma | -8.3 | -8.5 | -9.0 | -6.9 | -8.0 | -1.7 | -12.7 | -8.7 | -4.1 |
| forma_analogos_n3_t45 | forma | -9.1 | -8.9 | -9.0 | -7.3 | -8.0 | -2.6 | -12.8 | -8.7 | -5.2 |
| forma_analogos_n5 | forma | -8.5 | -13.0 | -6.5 | -11.3 | -13.0 | -4.0 | -17.1 | -7.7 | -8.8 |
| forma_analogos_n5_t45 | forma | -9.1 | -13.0 | -6.5 | -12.2 | -13.0 | -3.8 | -16.7 | -7.7 | -9.9 |
| forma_analogos_n8 | forma | -9.5 | -15.1 | -7.7 | -12.6 | -15.0 | -3.9 | -20.2 | -9.9 | -10.1 |
| forma_analogos_n8_t45 | forma | -10.1 | -14.8 | -7.7 | -13.4 | -15.0 | -3.7 | -18.8 | -9.9 | -11.3 |
| nivel_actual_a050 | nivel | -4.6 | 2.1 | -2.6 | -5.8 | 3.6 | -16.1 | -7.4 | -8.6 | -5.6 |
| nivel_actual_a050 | forma | -6.0 | 2.9 | -4.7 | -5.5 | 2.8 | -16.5 | -8.2 | -7.8 | -5.0 |
| nivel_analogos_a025 | nivel | 3.6 | 0.4 | -0.2 | 3.1 | 13.8 | 4.0 | 12.1 | 10.1 | 5.7 |
| nivel_analogos_a025 | forma | 3.9 | 1.3 | 0.6 | 3.3 | 13.9 | 4.9 | 13.4 | 10.4 | 6.8 |
| nivel_analogos_a050 | nivel | 4.9 | 2.4 | -0.4 | 0.1 | 6.4 | -4.2 | 4.8 | 8.8 | 10.9 |
| nivel_analogos_a050 | forma | 4.3 | 3.9 | 0.5 | 0.6 | 5.7 | -2.6 | 5.3 | 10.0 | 10.0 |

### Segmento `rel:antes` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo | modelo_k1·Antioquia·calibracion | modelo_k1·Antioquia·holdout | modelo_k1·Atlantico·calibracion | modelo_k1·Atlantico·holdout |
|---|---|---|---|---|---|---|---|---|---|---|
| forma_referencia | forma | -7.0 | -2.0 | -14.4 | -6.6 | 4.9 | -1.9 | -2.9 | -16.4 | -3.5 |
| forma_analogos_n3 | forma | -4.2 | -15.1 | -19.4 | -29.6 | -9.0 | -4.8 | -18.5 | -24.2 | -19.1 |
| forma_analogos_n3_t45 | forma | -10.9 | -4.6 | -21.6 | -17.3 | 4.0 | -11.3 | -3.5 | -28.2 | -14.3 |
| forma_analogos_n5 | forma | -6.3 | -16.3 | -13.5 | -32.7 | -14.8 | -4.6 | -23.2 | -17.9 | -19.7 |
| forma_analogos_n5_t45 | forma | -12.3 | -5.8 | -16.8 | -22.6 | -1.9 | -12.7 | -8.1 | -23.2 | -16.8 |
| forma_analogos_n8 | forma | -9.0 | -21.9 | -16.9 | -34.3 | -17.0 | -7.8 | -30.3 | -21.2 | -21.6 |
| forma_analogos_n8_t45 | forma | -14.6 | -9.8 | -19.6 | -22.2 | -4.3 | -15.4 | -12.7 | -25.8 | -17.1 |
| nivel_actual_a050 | nivel | -12.1 | 9.9 | -0.7 | -1.7 | 22.6 | -23.1 | 7.2 | -8.1 | -0.9 |
| nivel_actual_a050 | forma | -11.8 | 13.2 | -1.7 | 0.0 | 27.4 | -20.9 | 8.6 | -8.8 | -0.4 |
| nivel_analogos_a025 | nivel | -4.9 | -11.9 | -3.1 | -2.7 | 14.0 | 10.7 | -3.0 | -1.1 | 6.0 |
| nivel_analogos_a025 | forma | -1.5 | -12.0 | -2.5 | -5.2 | 14.5 | 10.7 | -2.4 | -2.1 | 8.4 |
| nivel_analogos_a050 | nivel | -7.1 | -10.6 | -4.8 | -4.5 | -2.1 | -1.1 | -17.2 | -9.8 | 6.3 |
| nivel_analogos_a050 | forma | -4.9 | -10.0 | -4.2 | -5.1 | -1.5 | -0.0 | -18.6 | -10.1 | 10.8 |

### Segmento `rel:despues` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo | modelo_k1·Antioquia·calibracion | modelo_k1·Antioquia·holdout | modelo_k1·Atlantico·calibracion | modelo_k1·Atlantico·holdout |
|---|---|---|---|---|---|---|---|---|---|---|
| forma_referencia | forma | -4.0 | -0.7 | -13.4 | -24.3 | -2.4 | -3.4 | -5.5 | -14.1 | -20.7 |
| forma_analogos_n3 | forma | -5.2 | -6.1 | -3.8 | -23.6 | -5.8 | -2.6 | -6.6 | -9.0 | -19.0 |
| forma_analogos_n3_t45 | forma | -10.9 | -6.1 | -3.2 | -27.8 | -5.8 | -6.2 | -7.1 | -8.6 | -22.2 |
| forma_analogos_n5 | forma | -8.2 | -7.9 | -7.4 | -27.0 | -9.9 | -3.9 | -9.1 | -15.9 | -22.8 |
| forma_analogos_n5_t45 | forma | -10.6 | -7.9 | -5.9 | -32.1 | -9.9 | -6.0 | -9.7 | -14.2 | -26.7 |
| forma_analogos_n8 | forma | -14.4 | -9.9 | -9.7 | -32.5 | -10.1 | -8.0 | -12.9 | -19.7 | -27.3 |
| forma_analogos_n8_t45 | forma | -14.0 | -9.9 | -8.9 | -38.7 | -10.1 | -9.2 | -13.4 | -19.3 | -32.3 |
| nivel_actual_a050 | nivel | -8.0 | -10.1 | -4.7 | -5.7 | -9.4 | -2.7 | -10.5 | -8.7 | -5.2 |
| nivel_actual_a050 | forma | -8.5 | -10.4 | -4.2 | -4.7 | -9.3 | -2.6 | -12.0 | -6.0 | -4.1 |
| nivel_analogos_a025 | nivel | -2.8 | -10.5 | 4.1 | -0.9 | 24.5 | 3.4 | 8.2 | 5.7 | -0.8 |
| nivel_analogos_a025 | forma | -2.8 | -10.8 | 3.5 | -1.1 | 21.8 | 4.6 | 7.1 | 3.8 | 0.8 |
| nivel_analogos_a050 | nivel | -15.5 | -18.3 | 4.1 | -0.9 | 28.2 | 2.0 | -0.8 | 6.1 | -0.4 |
| nivel_analogos_a050 | forma | -14.5 | -19.0 | 4.2 | 0.3 | 26.1 | 0.9 | -3.8 | 3.8 | 5.0 |

### Segmento `rel:ninguna` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo | modelo_k1·Antioquia·calibracion | modelo_k1·Antioquia·holdout | modelo_k1·Atlantico·calibracion | modelo_k1·Atlantico·holdout |
|---|---|---|---|---|---|---|---|---|---|---|
| forma_referencia | forma | -2.1 | -5.4 | -5.8 | -9.6 | -3.2 | -4.5 | -5.6 | -8.0 | -13.4 |
| forma_analogos_n3 | forma | -0.5 | -1.1 | -3.7 | -3.3 | 0.8 | -0.8 | -1.0 | -4.6 | -4.4 |
| forma_analogos_n3_t45 | forma | -0.5 | -1.1 | -3.7 | -3.3 | 0.8 | -0.8 | -1.0 | -4.6 | -4.4 |
| forma_analogos_n5 | forma | -0.6 | -1.3 | -3.5 | -1.3 | 0.8 | -1.3 | -1.5 | -5.8 | -2.9 |
| forma_analogos_n5_t45 | forma | -0.6 | -1.3 | -3.5 | -1.3 | 0.8 | -1.3 | -1.5 | -5.8 | -2.9 |
| forma_analogos_n8 | forma | -1.7 | -3.4 | -4.4 | -1.4 | -1.1 | -3.1 | -3.9 | -7.4 | -2.6 |
| forma_analogos_n8_t45 | forma | -1.7 | -3.4 | -4.4 | -1.4 | -1.1 | -3.1 | -3.9 | -7.4 | -2.6 |
| nivel_actual_a050 | nivel | -1.5 | 0.1 | -4.1 | -2.1 | -5.1 | -6.5 | -8.4 | -9.1 | -3.6 |
| nivel_actual_a050 | forma | -0.8 | 1.4 | -4.2 | -2.4 | -5.5 | -7.2 | -9.4 | -9.6 | -4.4 |
| nivel_analogos_a025 | nivel | 4.0 | 5.4 | -1.7 | 0.1 | 7.0 | 5.8 | 5.7 | -1.3 | 1.3 |
| nivel_analogos_a025 | forma | 4.7 | 5.7 | -1.4 | 0.4 | 7.2 | 6.5 | 6.5 | -2.2 | 0.8 |
| nivel_analogos_a050 | nivel | 9.3 | 13.2 | -2.0 | 1.3 | 13.3 | 8.6 | 8.1 | -4.8 | 1.1 |
| nivel_analogos_a050 | forma | 11.1 | 15.2 | -1.3 | 2.0 | 13.3 | 9.4 | 8.5 | -6.2 | 0.0 |

### Segmento `despues_dist1` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo | modelo_k1·Antioquia·calibracion | modelo_k1·Antioquia·holdout | modelo_k1·Atlantico·calibracion | modelo_k1·Atlantico·holdout |
|---|---|---|---|---|---|---|---|---|---|---|
| forma_referencia | forma | -9.9 | -1.7 | -15.7 | -23.7 | -2.4 | -2.8 | -3.6 | -16.8 | -8.3 |
| forma_analogos_n3 | forma | -3.7 | -4.8 | -2.1 | -18.0 | -12.7 | 3.6 | -6.8 | -8.8 | -7.0 |
| forma_analogos_n3_t45 | forma | -8.4 | -4.8 | -2.3 | -18.0 | -12.7 | 0.2 | -6.8 | -9.2 | -7.0 |
| forma_analogos_n5 | forma | -8.6 | -6.2 | -5.0 | -16.3 | -20.2 | 1.5 | -9.4 | -19.0 | -6.3 |
| forma_analogos_n5_t45 | forma | -8.1 | -6.2 | -3.3 | -16.3 | -20.2 | 0.5 | -9.4 | -16.3 | -6.3 |
| forma_analogos_n8 | forma | -13.7 | -8.4 | -4.7 | -23.8 | -21.7 | -3.1 | -11.2 | -20.3 | -12.2 |
| forma_analogos_n8_t45 | forma | -9.9 | -8.4 | -3.8 | -23.8 | -21.7 | -2.6 | -11.2 | -19.1 | -12.2 |
| nivel_actual_a050 | nivel | -19.2 | -20.6 | -3.6 | -17.5 | -32.2 | -13.4 | -28.0 | -13.7 | -11.7 |
| nivel_actual_a050 | forma | -18.5 | -20.6 | -1.8 | -21.6 | -31.3 | -16.5 | -30.6 | -8.6 | -10.3 |
| nivel_analogos_a025 | nivel | 7.8 | -1.9 | 2.7 | 8.5 | 45.0 | 14.3 | 27.4 | -0.4 | 3.4 |
| nivel_analogos_a025 | forma | 8.6 | -2.6 | -1.5 | 10.4 | 42.3 | 19.0 | 26.9 | -0.5 | 8.0 |
| nivel_analogos_a050 | nivel | -6.8 | -8.1 | 3.2 | 5.5 | 49.1 | 3.7 | 13.7 | -8.6 | 2.5 |
| nivel_analogos_a050 | forma | -5.8 | -7.9 | -1.0 | 7.2 | 52.7 | 9.1 | 8.9 | -8.6 | 11.7 |

### Segmento `semana_santa` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·holdout | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo | modelo_k1·Antioquia·holdout | modelo_k1·Atlantico·holdout |
|---|---|---|---|---|---|---|
| forma_referencia | forma | 0.8 | -0.3 | 8.6 | 9.2 | 1.0 |
| forma_analogos_n3 | forma | -0.3 | -2.1 | 7.5 | 6.8 | -2.7 |
| forma_analogos_n3_t45 | forma | -0.3 | -2.1 | 7.5 | 6.8 | -2.7 |
| forma_analogos_n5 | forma | -0.3 | -4.5 | 3.2 | 3.1 | -4.6 |
| forma_analogos_n5_t45 | forma | -0.3 | -4.5 | 3.2 | 3.1 | -4.6 |
| forma_analogos_n8 | forma | -2.1 | -1.7 | 2.1 | 0.3 | 0.3 |
| forma_analogos_n8_t45 | forma | -2.1 | -3.5 | 2.1 | 0.3 | -3.2 |
| nivel_actual_a050 | nivel | 24.1 | 11.1 | 14.3 | 12.6 | -6.2 |
| nivel_actual_a050 | forma | 26.3 | 11.8 | 15.4 | 12.7 | -7.4 |
| nivel_analogos_a025 | nivel | 2.1 | 1.4 | 1.6 | -0.4 | 2.1 |
| nivel_analogos_a025 | forma | 1.8 | 0.6 | 0.8 | -1.0 | -0.5 |
| nivel_analogos_a050 | nivel | 27.5 | 14.2 | 19.4 | 12.2 | 1.1 |
| nivel_analogos_a050 | forma | 29.9 | 15.5 | 18.7 | 13.0 | 1.2 |

### Segmento `diciembre` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·holdout | modelo_k1·Antioquia·calibracion | modelo_k1·Antioquia·holdout | modelo_k1·Atlantico·holdout |
|---|---|---|---|---|---|---|---|
| forma_referencia | forma | -1.8 | 21.3 | -2.7 | 0.2 | 8.4 | -7.5 |
| forma_analogos_n3 | forma | -0.1 | -7.2 | -7.1 | -0.6 | -3.9 | -10.5 |
| forma_analogos_n3_t45 | forma | -2.6 | -8.7 | -8.8 | -3.9 | -3.6 | -13.6 |
| forma_analogos_n5 | forma | -0.6 | -8.1 | -5.8 | -0.8 | -5.1 | -10.5 |
| forma_analogos_n5_t45 | forma | -3.1 | -8.7 | -8.6 | -4.3 | -3.6 | -15.0 |
| forma_analogos_n8 | forma | -2.1 | -15.4 | -8.0 | -2.8 | -10.5 | -12.8 |
| forma_analogos_n8_t45 | forma | -4.4 | -8.7 | -9.5 | -5.9 | -3.6 | -15.9 |
| nivel_actual_a050 | nivel | 5.4 | 8.0 | -1.6 | -18.8 | 16.2 | -1.5 |
| nivel_actual_a050 | forma | 6.2 | 13.8 | -1.5 | -17.8 | 17.9 | -1.2 |
| nivel_analogos_a025 | nivel | -2.3 | -22.1 | 0.8 | 11.0 | -6.0 | -0.8 |
| nivel_analogos_a025 | forma | -0.8 | -10.1 | 0.7 | 10.9 | -6.8 | -0.4 |
| nivel_analogos_a050 | nivel | 5.3 | -1.2 | -0.6 | -2.3 | 7.3 | -5.3 |
| nivel_analogos_a050 | forma | 7.5 | 0.6 | -0.4 | -1.6 | 10.0 | -6.6 |

### Segmento `normal` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo | modelo_k1·Antioquia·calibracion | modelo_k1·Antioquia·holdout | modelo_k1·Atlantico·calibracion | modelo_k1·Atlantico·holdout |
|---|---|---|---|---|---|---|---|---|---|---|
| forma_referencia | forma | -2.6 | -5.4 | -5.8 | -10.4 | -3.2 | -5.3 | -5.7 | -8.0 | -13.2 |
| forma_analogos_n3 | forma | -0.9 | -1.1 | -3.7 | -3.2 | 0.8 | -1.0 | -1.0 | -4.6 | -3.6 |
| forma_analogos_n3_t45 | forma | -0.9 | -1.1 | -3.7 | -3.2 | 0.8 | -1.0 | -1.0 | -4.6 | -3.6 |
| forma_analogos_n5 | forma | -0.6 | -1.4 | -3.5 | -1.1 | 0.8 | -1.3 | -1.5 | -5.8 | -1.7 |
| forma_analogos_n5_t45 | forma | -0.6 | -1.4 | -3.5 | -1.1 | 0.8 | -1.3 | -1.5 | -5.8 | -1.7 |
| forma_analogos_n8 | forma | -1.4 | -3.4 | -4.4 | -0.9 | -1.1 | -2.9 | -3.9 | -7.4 | -1.1 |
| forma_analogos_n8_t45 | forma | -1.4 | -3.4 | -4.4 | -0.9 | -1.1 | -2.9 | -3.9 | -7.4 | -1.1 |
| nivel_actual_a050 | nivel | -1.9 | 0.0 | -4.1 | -2.7 | -5.5 | -5.5 | -8.7 | -9.1 | -4.2 |
| nivel_actual_a050 | forma | -1.1 | 1.4 | -4.2 | -2.9 | -5.8 | -6.3 | -9.7 | -9.6 | -5.3 |
| nivel_analogos_a025 | nivel | 3.7 | 5.5 | -1.7 | 0.8 | 6.9 | 4.6 | 5.6 | -1.3 | 2.6 |
| nivel_analogos_a025 | forma | 4.5 | 5.7 | -1.4 | 1.2 | 7.1 | 5.3 | 6.4 | -2.2 | 2.4 |
| nivel_analogos_a050 | nivel | 9.1 | 13.3 | -2.0 | 2.6 | 12.8 | 8.4 | 7.6 | -4.8 | 3.6 |
| nivel_analogos_a050 | forma | 11.1 | 15.3 | -1.3 | 3.4 | 12.8 | 9.4 | 7.9 | -6.2 | 3.1 |

## 4. Días difíciles

Muestras chicas: con 52 semanas por partición hay pocos festivos, puentes y días de Semana Santa por UCP (ver n). Diferencias en estos segmentos son ruidosas; un solo día atípico mueve el WAPE. Con base `ingenuo` los festivos arrancan de un día no festivo (error artificial grande).

- **rel:festivo**: modelo_k1·Antioquia·calibracion: n=9, default 4.00 %, mejor `nivel_analogos_a025` 3.48 % (+13.2 %); modelo_k1·Antioquia·holdout: n=9, default 3.51 %, mejor `default` 3.51 % (+0.0 %); modelo_k1·Atlantico·calibracion: n=7, default 6.92 %, mejor `nivel_analogos_a050` 6.58 % (+5.0 %); modelo_k1·Atlantico·holdout: n=10, default 5.48 %, mejor `nivel_actual_a050` 5.41 % (+1.2 %); ingenuo·Antioquia·calibracion: n=9, default 12.83 %, mejor `nivel_analogos_a050` 8.96 % (+30.2 %); ingenuo·Antioquia·holdout: n=9, default 12.33 %, mejor `nivel_analogos_a050` 8.23 % (+33.3 %); ingenuo·Atlantico·calibracion: n=7, default 10.34 %, mejor `nivel_analogos_a050` 8.43 % (+18.5 %); ingenuo·Atlantico·holdout: n=10, default 8.96 %, mejor `nivel_actual_a050` 8.11 % (+9.5 %); archivo·Antioquia·archivo: n=6, default 4.34 %, mejor `nivel_analogos_a050` 3.41 % (+21.4 %).
- **rel:despues**: modelo_k1·Antioquia·calibracion: n=25, default 2.21 %, mejor `nivel_analogos_a025` 2.11 % (+4.6 %); modelo_k1·Antioquia·holdout: n=19, default 2.79 %, mejor `nivel_analogos_a025` 2.59 % (+7.1 %); modelo_k1·Atlantico·calibracion: n=20, default 3.96 %, mejor `nivel_analogos_a050` 3.81 % (+3.8 %); modelo_k1·Atlantico·holdout: n=20, default 4.05 %, mejor `nivel_analogos_a050` 3.85 % (+5.0 %); ingenuo·Antioquia·calibracion: n=25, default 2.24 %, mejor `default` 2.24 % (+0.0 %); ingenuo·Antioquia·holdout: n=19, default 3.00 %, mejor `default` 3.00 % (+0.0 %); ingenuo·Atlantico·calibracion: n=20, default 4.69 %, mejor `nivel_analogos_a050` 4.49 % (+4.2 %); ingenuo·Atlantico·holdout: n=20, default 3.74 %, mejor `nivel_analogos_a050` 3.73 % (+0.3 %); archivo·Antioquia·archivo: n=11, default 3.56 %, mejor `nivel_analogos_a050` 2.63 % (+26.1 %).
- **despues_dist1**: modelo_k1·Antioquia·calibracion: n=7, default 2.22 %, mejor `nivel_analogos_a025` 1.79 % (+19.0 %); modelo_k1·Antioquia·holdout: n=5, default 3.84 %, mejor `nivel_analogos_a025` 2.81 % (+26.9 %); modelo_k1·Atlantico·calibracion: n=6, default 3.82 %, mejor `default` 3.82 % (+0.0 %); modelo_k1·Atlantico·holdout: n=5, default 4.26 %, mejor `nivel_analogos_a050` 3.76 % (+11.7 %); ingenuo·Antioquia·calibracion: n=7, default 2.54 %, mejor `nivel_analogos_a025` 2.32 % (+8.6 %); ingenuo·Antioquia·holdout: n=5, default 4.71 %, mejor `default` 4.71 % (+0.0 %); ingenuo·Atlantico·calibracion: n=6, default 4.98 %, mejor `default` 4.98 % (+0.0 %); ingenuo·Atlantico·holdout: n=5, default 3.78 %, mejor `nivel_analogos_a025` 3.39 % (+10.4 %); archivo·Antioquia·archivo: n=3, default 3.90 %, mejor `nivel_analogos_a050` 1.84 % (+52.7 %).
- **rel:antes**: modelo_k1·Antioquia·calibracion: n=13, default 3.10 %, mejor `nivel_analogos_a025` 2.77 % (+10.7 %); modelo_k1·Antioquia·holdout: n=15, default 2.29 %, mejor `nivel_actual_a050` 2.09 % (+8.6 %); modelo_k1·Atlantico·calibracion: n=11, default 4.49 %, mejor `default` 4.49 % (+0.0 %); modelo_k1·Atlantico·holdout: n=16, default 4.45 %, mejor `nivel_analogos_a050` 3.96 % (+10.8 %); ingenuo·Antioquia·calibracion: n=13, default 3.55 %, mejor `default` 3.55 % (+0.0 %); ingenuo·Antioquia·holdout: n=15, default 3.30 %, mejor `nivel_actual_a050` 2.86 % (+13.2 %); ingenuo·Atlantico·calibracion: n=11, default 5.53 %, mejor `default` 5.53 % (+0.0 %); ingenuo·Atlantico·holdout: n=16, default 4.08 %, mejor `nivel_actual_a050` 4.08 % (+0.0 %); archivo·Antioquia·archivo: n=9, default 2.41 %, mejor `nivel_actual_a050` 1.75 % (+27.4 %).
- **rel:puente**: modelo_k1·Antioquia·calibracion: n=13, default 2.74 %, mejor `nivel_analogos_a025` 2.61 % (+4.9 %); modelo_k1·Antioquia·holdout: n=17, default 2.91 %, mejor `nivel_analogos_a025` 2.52 % (+13.4 %); modelo_k1·Atlantico·calibracion: n=14, default 6.31 %, mejor `nivel_analogos_a025` 5.65 % (+10.4 %); modelo_k1·Atlantico·holdout: n=14, default 5.20 %, mejor `nivel_analogos_a050` 4.68 % (+10.0 %); ingenuo·Antioquia·calibracion: n=13, default 3.51 %, mejor `nivel_analogos_a050` 3.37 % (+4.3 %); ingenuo·Antioquia·holdout: n=17, default 3.45 %, mejor `nivel_analogos_a050` 3.31 % (+3.9 %); ingenuo·Atlantico·calibracion: n=14, default 6.08 %, mejor `nivel_analogos_a025` 6.04 % (+0.6 %); ingenuo·Atlantico·holdout: n=14, default 5.49 %, mejor `nivel_analogos_a025` 5.31 % (+3.3 %); archivo·Antioquia·archivo: n=10, default 3.24 %, mejor `nivel_analogos_a025` 2.79 % (+13.9 %).
- **semana_santa**: modelo_k1·Antioquia·holdout: n=8, default 3.10 %, mejor `nivel_analogos_a050` 2.70 % (+13.0 %); modelo_k1·Atlantico·holdout: n=8, default 4.19 %, mejor `nivel_analogos_a050` 4.14 % (+1.2 %); ingenuo·Antioquia·holdout: n=8, default 8.37 %, mejor `nivel_analogos_a050` 5.87 % (+29.9 %); ingenuo·Atlantico·holdout: n=8, default 8.13 %, mejor `nivel_analogos_a050` 6.87 % (+15.5 %); archivo·Antioquia·archivo: n=8, default 3.60 %, mejor `nivel_analogos_a050` 2.93 % (+18.7 %).
- **diciembre**: modelo_k1·Antioquia·calibracion: n=28, default 2.95 %, mejor `nivel_analogos_a025` 2.63 % (+10.9 %); modelo_k1·Antioquia·holdout: n=3, default 4.56 %, mejor `nivel_actual_a050` 3.74 % (+17.9 %); modelo_k1·Atlantico·holdout: n=31, default 5.44 %, mejor `default` 5.44 % (+0.0 %); ingenuo·Antioquia·calibracion: n=28, default 5.13 %, mejor `nivel_analogos_a050` 4.74 % (+7.5 %); ingenuo·Antioquia·holdout: n=3, default 3.42 %, mejor `forma_referencia` 2.69 % (+21.3 %); ingenuo·Atlantico·holdout: n=31, default 6.51 %, mejor `nivel_analogos_a025` 6.47 % (+0.7 %).

## 5. Regla de cambio de default (base `modelo_k1`, ambas UCP, calibración y holdout)

Una fila por segmento × variante. `celdas_que_ganan` = UCP × partición × paso afectado que mejoran >= 2.0 % sin empeorar el pico. Solo se listan las variantes con al menos la mitad de las celdas ganadas; `cumple_regla` = todas.

| segmento | config | pasos | n_dias_min | celdas | celdas_que_ganan | mejora_min_pct | mejora_media_pct | delta_pico_max_pp | cumple_regla |
|---|---|---|---|---|---|---|---|---|---|
| rel:puente | nivel_analogos_a025 | nivel+forma | 13 | 8/8 | 8 | 3.97 | 8.43 | -0.10 | sí |
| tipo:laboral | nivel_analogos_a025 | nivel+forma | 120 | 8/8 | 6 | 0.45 | 4.02 | -0.03 | no |
| normal | nivel_analogos_a025 | nivel+forma | 107 | 8/8 | 6 | -2.16 | 2.91 | 0.03 | no |
| tipo:laboral | nivel_analogos_a050 | nivel+forma | 120 | 8/8 | 6 | -5.22 | 2.22 | 0.20 | no |
| normal | nivel_analogos_a050 | nivel+forma | 107 | 8/8 | 6 | -6.24 | 3.64 | 0.25 | no |
| todos | nivel_analogos_a025 | nivel+forma | 182 | 8/8 | 4 | -0.19 | 3.37 | 0.01 | no |
| despues_dist1 | nivel_analogos_a025 | nivel+forma | 5 | 8/8 | 4 | -0.50 | 12.28 | 0.43 | no |
| tipo:sabado | nivel_analogos_a050 | nivel+forma | 26 | 8/8 | 4 | -0.58 | 3.51 | 0.39 | no |
| rel:despues | nivel_analogos_a025 | nivel+forma | 19 | 8/8 | 4 | -0.83 | 4.10 | 0.13 | no |
| rel:ninguna | nivel_analogos_a025 | nivel+forma | 122 | 8/8 | 4 | -2.16 | 2.89 | 0.03 | no |
| rel:antes | nivel_analogos_a025 | nivel+forma | 11 | 8/8 | 4 | -3.04 | 3.39 | 0.57 | no |
| todos | nivel_analogos_a050 | nivel+forma | 182 | 8/8 | 4 | -3.53 | 1.65 | 0.20 | no |
| rel:puente | nivel_analogos_a050 | nivel+forma | 13 | 8/8 | 4 | -4.24 | 5.35 | 0.25 | no |
| rel:ninguna | nivel_analogos_a050 | nivel+forma | 122 | 8/8 | 4 | -6.24 | 3.08 | 0.25 | no |
| tipo:domingo | nivel_analogos_a025 | nivel+forma | 25 | 8/8 | 4 | -6.47 | 1.85 | 0.24 | no |

## 6. Conclusión

**Variantes que cumplen la regla (candidatas a default por segmento; decide el coordinador):**

- Segmento `rel:puente`: `nivel_analogos_a025` (pasos nivel+forma, n mínimo 13 días, mejora mínima 4.0 %, media 8.4 %, Δ pico máx -0.10 pp).

Si se aplica, debe ser por segmento y retrocompatible (p. ej. un default condicionado al tipo de día), y conviene revalidar con la base `modelo_k1` antes de fijarlo. Segmentos con n chico (< 15 días por celda) deben tomarse como indicio, no como evidencia.

## 7. Cómo correr con la base modelo_k1 (Fase 2a)

```
.venv/Scripts/python.exe scripts/backtest_analogos.py --semanas 52 --base modelo_k1 --base-regla modelo_k1 --fecha <AAAAMMDD>
```
`modelo_k1` usa las curvas `reports/ajustes/curvas_modelo_k1_<ucp>.csv` de la Fase 2a (sin fuga, corte previo verificado). `--base-regla` elige la base con la que se evalúa la regla de la §5; `--modelo-dir` cambia el directorio de las curvas.

## 8. Limitaciones

- **Base ingenua ≠ modelo real**: la base `ingenuo` (mismo día de la semana anterior) es débil y en festivos
  parte de un día no festivo; las mejoras medidas sobre ella pueden no trasladarse al pronóstico del modelo.
  La base `modelo_k1` (Fase 2a) sí es el pronóstico del modelo real sin fuga, pero solo cubre las 52 semanas
  generadas y su corrida es más cara; la regla se evalúa con `--base-regla` (default `ingenuo`).
- **Base `modelo_k1`**: curvas pronosticadas con corte previo (sábado) por la Fase 2a; no se regeneran aquí.
  Si faltan los CSV, la base se omite para esa UCP.
- **Base `archivo`**: una sola emisión con corte 2025-12-31 y ajustes post-hoc; solo Antioquia, ~21 semanas
  (ene–may 2026). Se reporta como contraste, sin elegir nada ahí.
- **Clima no usado** en la búsqueda automática (`seleccionar` no puntúa clima; además el clima real solo
  cubre 2025-12..2026-07). El filtro de clima del endpoint queda sin evaluar.
- **Muestras chicas** en festivos, puentes, Semana Santa y diciembre (ver n por segmento): diferencias de
  pocos puntos en esos segmentos no son concluyentes (n < 15 días por celda).
- **Corte siempre en sábado** (1 día de brecha); otras emisiones no se evalúan.
- **Calibración y holdout contiguos** (la mitad de las semanas cada una): un cambio de régimen en el holdout
  afecta a todas las configs por igual, pero la comparación es de una sola partición temporal (sin validación
  cruzada).
- La elección de la mejor variante y la combinación se hacen solo en calibración con la base de la regla
  (`--base-regla`, default `ingenuo`); el holdout no se usa para elegir.

## 9. Antes/después (base `modelo_k1` sin fuga)

Comparación con `backtest_analogos_20261113.md` (corrida previa con `ingenuo` + `archivo` y 104 semanas por UCP; la regla se evaluó con `ingenuo`). La corrida nueva usa 52 semanas por UCP (`--semanas 52`), así que el "antes" de 104 semanas no es estrictamente comparable; el mismo `ingenuo` en esta ventana de 52 semanas queda en la tabla.

| reporte | base de la regla | ventana | default WAPE forma Antioquia (calib/holdout) | Atlantico (calib/holdout) | variantes que cumplen la regla |
|---|---|---|---|---|---|
| 20261113 (antes) | ingenuo | 104 sem (52+52) | 3.66 / 3.71 | 5.46 / 5.90 | `nivel_analogos_a050` y `nivel_actual_a050` en festivos y Semana Santa (n=17 y 8) |
| 20261113_modelo (después) | modelo_k1 | 52 sem (26+26) | 2.57 / 2.60 | 4.96 / 5.15 | solo `rel:puente` con `nivel_analogos_a025` (n=13, <15 → no concluyente) |

WAPE % del paso forma con el default, por UCP y segmento (misma ventana de 52 semanas):

| segmento | modelo_k1 Antioquia (calib/holdout) | modelo_k1 Atlantico (calib/holdout) | ingenuo Antioquia (calib/holdout) | ingenuo Atlantico (calib/holdout) |
|---|---|---|---|---|
| todos | 2.57 / 2.60 | 4.96 / 5.15 | 3.58 / 3.82 | 5.93 / 5.86 |
| tipo:laboral | 2.50 / 2.53 | 4.73 / 5.24 | 3.25 / 3.52 | 5.63 / 5.64 |
| tipo:sabado | 2.24 / 2.53 | 5.32 / 5.10 | 2.62 / 2.94 | 5.81 / 5.71 |
| tipo:domingo | 2.83 / 2.80 | 5.31 / 4.65 | 3.41 / 3.60 | 6.51 / 5.95 |
| tipo:festivo | 4.00 / 3.51 | 6.92 / 5.48 | 12.83 / 12.33 | 10.34 / 8.96 |
| rel:puente | 2.74 / 2.91 | 6.31 / 5.20 | 3.51 / 3.45 | 6.08 / 5.49 |
| normal | 2.45 / 2.51 | 4.93 / 5.50 | 3.21 / 3.55 | 5.93 / 6.22 |

Hallazgos:

- Con `modelo_k1` el default mejora al `ingenuo` en casi todos los segmentos (la excepción es `rel:puente` en calibración de Atlántico); la brecha es mayor en festivos porque el `ingenuo` pronostica un día no festivo.
- La candidata de la corrida anterior (α alto con ancla por análogos en festivos / Semana Santa) **no se sostiene** sobre el modelo real: con `modelo_k1` el default ya baja el WAPE festivo a 4.00/3.51 (Antioquia) y 6.92/5.48 (Atlántico), y ninguna variante de festivos cumple la regla.
- La única variante que cumple la regla es `nivel_analogos_a025` en `rel:puente` (mejora mínima 3.97 %, media 8.43 %, Δ pico −0.10 pp), con n=13 días → **no concluyente** (<15).
- En Antioquia `nivel_analogos_a025` mejora `todos` +6.97 % en calibración y +5.82 % en holdout, pero en Atlántico no mejora (0 %); por eso no cumple la regla global.
- La forma por análogos no mejora al default en ninguna UCP con `modelo_k1` (igual que con `ingenuo`).

