# Backtest de días análogos (R3) — 20261113

Generado por `scripts/backtest_analogos.py` (importa `scripts/backtest_adjustments.py`). Detalle en `backtest_analogos_20261113.csv` (fuera de git; una fila por UCP × semana × día × paso × config).

## Configuración

- Bases: ingenuo, archivo. `modelo` no se corrió (depende de la Fase 2a).
- Semanas: últimas 104 completas por UCP; corte = sábado anterior (fecha_emision = domingo). Calibración = mitad más antigua, holdout = mitad más reciente. Base `archivo`: partición `archivo` aparte (no se elige configuración ahí).
- Sin fuga: `construir_contexto` verifica `max(historico.fecha) <= fecha_corte <= sábado`; además se verifica que toda fecha análoga usada por forma/nivel sea <= corte (aserción por día y config).
- Configs (11): `default`, `forma_referencia`, `forma_analogos_n3`, `forma_analogos_n3_t45`, `forma_analogos_n5`, `forma_analogos_n5_t45`, `forma_analogos_n8`, `forma_analogos_n8_t45`, `nivel_actual_a050`, `nivel_analogos_a025`, `nivel_analogos_a050`.
  - `default` = forma `promedio_n` n=4 + nivel ancla actual α=0.25.
  - `forma_*` cambian solo la forma (nivel default); `_nK` = k análogos; `_t45` = filtro `temporada_dias` 45 además de los automáticos (mismo día de semana y misma relación/distancia con festivo).
  - `nivel_*` cambian solo el nivel (forma default): ancla `actual`/`analogos`, α en centésimas.
  - `combo[...]` = mejor forma + mejor nivel elegidas en calibración (si ambas mejoran).
- Regla para proponer default por segmento: gana en calibración y holdout, en ambas UCP con base ingenuo, por >= 2.0 % relativo del WAPE de cada paso afectado (forma; nivel y forma para configs de nivel/combo) y sin empeorar el error de pico (|error %| del máximo) en esos pasos.
- Tiempo de corrida: 655 s.

### Semanas evaluadas

| base | ucp | particion | primera_semana | ultima_semana | n_semanas |
|---|---|---|---|---|---|
| archivo | Antioquia | archivo | 2026-01-05 | 2026-05-25 | 21 |
| ingenuo | Antioquia | calibracion | 2024-07-01 | 2025-06-23 | 52 |
| ingenuo | Antioquia | holdout | 2025-06-30 | 2026-06-22 | 52 |
| ingenuo | Atlantico | calibracion | 2023-12-25 | 2025-04-28 | 52 |
| ingenuo | Atlantico | holdout | 2025-05-05 | 2026-04-27 | 52 |

## 1. Resultado global (todos los días)

WAPE horario % del paso final (forma) por config; columnas = base·UCP·partición.

| config | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo |
|---|---|---|---|---|---|
| default | 3.66 | 3.71 | 5.46 | 5.90 | 2.79 |
| forma_referencia | 3.74 | 3.83 | 6.81 | 6.39 | 2.86 |
| forma_analogos_n3 | 3.72 | 3.79 | 6.37 | 6.22 | 2.82 |
| forma_analogos_n3_t45 | 3.72 | 3.80 | 6.40 | 6.21 | 2.80 |
| forma_analogos_n5 | 3.72 | 3.81 | 6.46 | 6.20 | 2.85 |
| forma_analogos_n5_t45 | 3.73 | 3.81 | 6.49 | 6.19 | 2.83 |
| forma_analogos_n8 | 3.76 | 3.88 | 6.58 | 6.25 | 2.90 |
| forma_analogos_n8_t45 | 3.77 | 3.87 | 6.64 | 6.24 | 2.88 |
| nivel_actual_a050 | 3.57 | 3.57 | 5.59 | 6.04 | 2.87 |
| nivel_analogos_a025 | 3.61 | 3.61 | 5.46 | 5.91 | 2.52 |
| nivel_analogos_a050 | 3.35 | 3.29 | 5.44 | 5.83 | 2.41 |

WAPE % del paso nivel (solo cambia en configs de nivel/combo):

| config | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo |
|---|---|---|---|---|---|
| default | 3.72 | 3.76 | 5.74 | 6.14 | 2.87 |
| nivel_actual_a050 | 3.68 | 3.69 | 5.88 | 6.29 | 2.95 |
| nivel_analogos_a025 | 3.68 | 3.68 | 5.75 | 6.16 | 2.58 |
| nivel_analogos_a050 | 3.49 | 3.43 | 5.73 | 6.12 | 2.47 |

Mejora relativa % del WAPE vs default (positivo = mejor):

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo |
|---|---|---|---|---|---|---|
| forma_referencia | forma | -2.37 | -3.47 | -24.60 | -8.29 | -2.36 |
| forma_analogos_n3 | forma | -1.69 | -2.33 | -16.67 | -5.54 | -1.07 |
| forma_analogos_n3_t45 | forma | -1.70 | -2.43 | -17.10 | -5.35 | -0.35 |
| forma_analogos_n5 | forma | -1.83 | -2.93 | -18.25 | -5.05 | -2.17 |
| forma_analogos_n5_t45 | forma | -2.03 | -2.85 | -18.74 | -4.97 | -1.46 |
| forma_analogos_n8 | forma | -2.84 | -4.70 | -20.53 | -5.90 | -3.80 |
| forma_analogos_n8_t45 | forma | -3.03 | -4.42 | -21.53 | -5.82 | -3.10 |
| nivel_actual_a050 | nivel | 1.14 | 2.02 | -2.40 | -2.36 | -2.92 |
| nivel_actual_a050 | forma | 2.47 | 3.57 | -2.41 | -2.45 | -2.83 |
| nivel_analogos_a025 | nivel | 0.99 | 2.13 | -0.01 | -0.33 | 9.83 |
| nivel_analogos_a025 | forma | 1.29 | 2.64 | 0.07 | -0.17 | 9.81 |
| nivel_analogos_a050 | nivel | 6.22 | 8.84 | 0.18 | 0.33 | 13.77 |
| nivel_analogos_a050 | forma | 8.36 | 11.16 | 0.45 | 1.07 | 13.65 |

### Todas las métricas del paso afectado

WAPE horario; energía diaria (MAPE y sesgo %); pico (|error %| y diferencia de hora); energía semanal (MAPE y sesgo %).

| base | ucp | particion | config | paso | n_dias | wape_pct | mape_energia_dia_pct | sesgo_energia_dia_pct | err_pico_pct | dif_hora_pico_h | n_semanas | mape_energia_semana_pct | sesgo_energia_semana_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ingenuo | Antioquia | calibracion | default | base | 364.00 | 3.99 | 3.90 | 0.15 | 3.67 | 2.33 | 52 | 2.81 | 0.03 |
| ingenuo | Antioquia | calibracion | default | nivel | 364.00 | 3.72 | 3.58 | 0.09 | 3.36 | 2.33 | 52 | 2.42 | -0.01 |
| ingenuo | Antioquia | calibracion | default | forma | 364.00 | 3.66 | 3.58 | 0.09 | 3.60 | 1.70 | 52 | 2.42 | -0.01 |
| ingenuo | Antioquia | calibracion | forma_referencia | forma | 364.00 | 3.74 | 3.58 | 0.09 | 3.65 | 2.04 | 52 | 2.42 | -0.01 |
| ingenuo | Antioquia | calibracion | forma_analogos_n3 | forma | 364.00 | 3.72 | 3.58 | 0.09 | 3.70 | 1.84 | 52 | 2.42 | -0.01 |
| ingenuo | Antioquia | calibracion | forma_analogos_n3_t45 | forma | 364.00 | 3.72 | 3.58 | 0.09 | 3.71 | 1.89 | 52 | 2.42 | -0.01 |
| ingenuo | Antioquia | calibracion | forma_analogos_n5 | forma | 364.00 | 3.72 | 3.58 | 0.09 | 3.67 | 1.82 | 52 | 2.42 | -0.01 |
| ingenuo | Antioquia | calibracion | forma_analogos_n5_t45 | forma | 364.00 | 3.73 | 3.58 | 0.09 | 3.70 | 1.84 | 52 | 2.42 | -0.01 |
| ingenuo | Antioquia | calibracion | forma_analogos_n8 | forma | 364.00 | 3.76 | 3.58 | 0.09 | 3.65 | 1.86 | 52 | 2.42 | -0.01 |
| ingenuo | Antioquia | calibracion | forma_analogos_n8_t45 | forma | 364.00 | 3.77 | 3.58 | 0.09 | 3.69 | 1.88 | 52 | 2.42 | -0.01 |
| ingenuo | Antioquia | calibracion | nivel_actual_a050 | nivel | 364.00 | 3.68 | 3.47 | 0.03 | 3.24 | 2.33 | 52 | 2.30 | -0.04 |
| ingenuo | Antioquia | calibracion | nivel_actual_a050 | forma | 364.00 | 3.57 | 3.47 | 0.03 | 3.46 | 1.70 | 52 | 2.30 | -0.04 |
| ingenuo | Antioquia | calibracion | nivel_analogos_a025 | nivel | 364.00 | 3.68 | 3.54 | 0.12 | 3.32 | 2.33 | 52 | 2.48 | 0.01 |
| ingenuo | Antioquia | calibracion | nivel_analogos_a025 | forma | 364.00 | 3.61 | 3.54 | 0.12 | 3.55 | 1.70 | 52 | 2.48 | 0.01 |
| ingenuo | Antioquia | calibracion | nivel_analogos_a050 | nivel | 364.00 | 3.49 | 3.25 | 0.08 | 3.06 | 2.33 | 52 | 2.20 | -0.00 |
| ingenuo | Antioquia | calibracion | nivel_analogos_a050 | forma | 364.00 | 3.35 | 3.25 | 0.08 | 3.26 | 1.70 | 52 | 2.20 | -0.00 |
| ingenuo | Antioquia | holdout | default | base | 364.00 | 4.05 | 3.92 | -0.11 | 3.55 | 1.85 | 52 | 2.69 | -0.25 |
| ingenuo | Antioquia | holdout | default | nivel | 364.00 | 3.76 | 3.59 | -0.11 | 3.21 | 1.85 | 52 | 2.35 | -0.22 |
| ingenuo | Antioquia | holdout | default | forma | 364.00 | 3.71 | 3.59 | -0.11 | 3.47 | 1.76 | 52 | 2.35 | -0.22 |
| ingenuo | Antioquia | holdout | forma_referencia | forma | 364.00 | 3.83 | 3.59 | -0.11 | 3.53 | 2.60 | 52 | 2.35 | -0.22 |
| ingenuo | Antioquia | holdout | forma_analogos_n3 | forma | 364.00 | 3.79 | 3.59 | -0.11 | 3.56 | 2.35 | 52 | 2.35 | -0.22 |
| ingenuo | Antioquia | holdout | forma_analogos_n3_t45 | forma | 364.00 | 3.80 | 3.59 | -0.11 | 3.61 | 2.31 | 52 | 2.35 | -0.22 |
| ingenuo | Antioquia | holdout | forma_analogos_n5 | forma | 364.00 | 3.81 | 3.59 | -0.11 | 3.55 | 2.34 | 52 | 2.35 | -0.22 |
| ingenuo | Antioquia | holdout | forma_analogos_n5_t45 | forma | 364.00 | 3.81 | 3.59 | -0.11 | 3.59 | 2.28 | 52 | 2.35 | -0.22 |
| ingenuo | Antioquia | holdout | forma_analogos_n8 | forma | 364.00 | 3.88 | 3.59 | -0.11 | 3.59 | 2.76 | 52 | 2.35 | -0.22 |
| ingenuo | Antioquia | holdout | forma_analogos_n8_t45 | forma | 364.00 | 3.87 | 3.59 | -0.11 | 3.62 | 2.71 | 52 | 2.35 | -0.22 |
| ingenuo | Antioquia | holdout | nivel_actual_a050 | nivel | 364.00 | 3.69 | 3.44 | -0.11 | 3.10 | 1.85 | 52 | 2.19 | -0.19 |
| ingenuo | Antioquia | holdout | nivel_actual_a050 | forma | 364.00 | 3.57 | 3.44 | -0.11 | 3.32 | 1.76 | 52 | 2.19 | -0.19 |
| ingenuo | Antioquia | holdout | nivel_analogos_a025 | nivel | 364.00 | 3.68 | 3.50 | -0.10 | 3.15 | 1.85 | 52 | 2.39 | -0.21 |
| ingenuo | Antioquia | holdout | nivel_analogos_a025 | forma | 364.00 | 3.61 | 3.50 | -0.10 | 3.40 | 1.76 | 52 | 2.39 | -0.21 |
| ingenuo | Antioquia | holdout | nivel_analogos_a050 | nivel | 364.00 | 3.43 | 3.16 | -0.08 | 2.83 | 1.85 | 52 | 2.19 | -0.16 |
| ingenuo | Antioquia | holdout | nivel_analogos_a050 | forma | 364.00 | 3.29 | 3.16 | -0.08 | 3.08 | 1.76 | 52 | 2.19 | -0.16 |
| ingenuo | Atlantico | calibracion | default | base | 364.00 | 5.84 | 4.85 | -0.06 | 4.81 | 0.83 | 52 | 3.55 | -0.17 |
| ingenuo | Atlantico | calibracion | default | nivel | 364.00 | 5.74 | 4.74 | -0.00 | 4.64 | 0.83 | 52 | 3.22 | -0.11 |
| ingenuo | Atlantico | calibracion | default | forma | 364.00 | 5.46 | 4.74 | -0.00 | 4.90 | 0.59 | 52 | 3.22 | -0.11 |
| ingenuo | Atlantico | calibracion | forma_referencia | forma | 364.00 | 6.81 | 4.74 | -0.00 | 5.61 | 3.47 | 52 | 3.22 | -0.11 |
| ingenuo | Atlantico | calibracion | forma_analogos_n3 | forma | 364.00 | 6.37 | 4.74 | -0.00 | 5.63 | 2.51 | 52 | 3.22 | -0.11 |
| ingenuo | Atlantico | calibracion | forma_analogos_n3_t45 | forma | 364.00 | 6.40 | 4.74 | -0.00 | 5.64 | 2.60 | 52 | 3.22 | -0.11 |
| ingenuo | Atlantico | calibracion | forma_analogos_n5 | forma | 364.00 | 6.46 | 4.74 | -0.00 | 5.82 | 2.46 | 52 | 3.22 | -0.11 |
| ingenuo | Atlantico | calibracion | forma_analogos_n5_t45 | forma | 364.00 | 6.49 | 4.74 | -0.00 | 5.82 | 2.54 | 52 | 3.22 | -0.11 |
| ingenuo | Atlantico | calibracion | forma_analogos_n8 | forma | 364.00 | 6.58 | 4.74 | -0.00 | 5.90 | 2.34 | 52 | 3.22 | -0.11 |
| ingenuo | Atlantico | calibracion | forma_analogos_n8_t45 | forma | 364.00 | 6.64 | 4.74 | -0.00 | 5.88 | 2.56 | 52 | 3.22 | -0.11 |
| ingenuo | Atlantico | calibracion | nivel_actual_a050 | nivel | 364.00 | 5.88 | 4.89 | 0.05 | 4.75 | 0.83 | 52 | 3.18 | -0.05 |
| ingenuo | Atlantico | calibracion | nivel_actual_a050 | forma | 364.00 | 5.59 | 4.89 | 0.05 | 5.00 | 0.59 | 52 | 3.18 | -0.05 |
| ingenuo | Atlantico | calibracion | nivel_analogos_a025 | nivel | 364.00 | 5.75 | 4.73 | 0.01 | 4.74 | 0.83 | 52 | 3.46 | -0.10 |
| ingenuo | Atlantico | calibracion | nivel_analogos_a025 | forma | 364.00 | 5.46 | 4.73 | 0.01 | 4.99 | 0.59 | 52 | 3.46 | -0.10 |
| ingenuo | Atlantico | calibracion | nivel_analogos_a050 | nivel | 364.00 | 5.73 | 4.67 | 0.08 | 4.75 | 0.83 | 52 | 3.40 | -0.03 |
| ingenuo | Atlantico | calibracion | nivel_analogos_a050 | forma | 364.00 | 5.44 | 4.67 | 0.08 | 4.98 | 0.59 | 52 | 3.40 | -0.03 |
| ingenuo | Atlantico | holdout | default | base | 364.00 | 6.27 | 5.53 | -0.21 | 5.49 | 1.02 | 52 | 3.58 | -0.37 |
| ingenuo | Atlantico | holdout | default | nivel | 364.00 | 6.14 | 5.35 | -0.08 | 5.39 | 1.02 | 52 | 3.31 | -0.23 |
| ingenuo | Atlantico | holdout | default | forma | 364.00 | 5.90 | 5.35 | -0.08 | 5.39 | 0.76 | 52 | 3.31 | -0.23 |
| ingenuo | Atlantico | holdout | forma_referencia | forma | 364.00 | 6.39 | 5.35 | -0.08 | 5.67 | 1.04 | 52 | 3.31 | -0.23 |
| ingenuo | Atlantico | holdout | forma_analogos_n3 | forma | 364.00 | 6.22 | 5.35 | -0.08 | 5.42 | 1.19 | 52 | 3.31 | -0.23 |
| ingenuo | Atlantico | holdout | forma_analogos_n3_t45 | forma | 364.00 | 6.21 | 5.35 | -0.08 | 5.36 | 1.26 | 52 | 3.31 | -0.23 |
| ingenuo | Atlantico | holdout | forma_analogos_n5 | forma | 364.00 | 6.20 | 5.35 | -0.08 | 5.53 | 1.16 | 52 | 3.31 | -0.23 |
| ingenuo | Atlantico | holdout | forma_analogos_n5_t45 | forma | 364.00 | 6.19 | 5.35 | -0.08 | 5.48 | 1.26 | 52 | 3.31 | -0.23 |
| ingenuo | Atlantico | holdout | forma_analogos_n8 | forma | 364.00 | 6.25 | 5.35 | -0.08 | 5.69 | 1.12 | 52 | 3.31 | -0.23 |
| ingenuo | Atlantico | holdout | forma_analogos_n8_t45 | forma | 364.00 | 6.24 | 5.35 | -0.08 | 5.65 | 1.19 | 52 | 3.31 | -0.23 |
| ingenuo | Atlantico | holdout | nivel_actual_a050 | nivel | 364.00 | 6.29 | 5.51 | 0.05 | 5.65 | 1.02 | 52 | 3.28 | -0.08 |
| ingenuo | Atlantico | holdout | nivel_actual_a050 | forma | 364.00 | 6.04 | 5.51 | 0.05 | 5.58 | 0.76 | 52 | 3.28 | -0.08 |
| ingenuo | Atlantico | holdout | nivel_analogos_a025 | nivel | 364.00 | 6.16 | 5.39 | -0.17 | 5.43 | 1.02 | 52 | 3.45 | -0.31 |
| ingenuo | Atlantico | holdout | nivel_analogos_a025 | forma | 364.00 | 5.91 | 5.39 | -0.17 | 5.40 | 0.76 | 52 | 3.45 | -0.31 |
| ingenuo | Atlantico | holdout | nivel_analogos_a050 | nivel | 364.00 | 6.12 | 5.32 | -0.12 | 5.45 | 1.02 | 52 | 3.35 | -0.25 |
| ingenuo | Atlantico | holdout | nivel_analogos_a050 | forma | 364.00 | 5.83 | 5.32 | -0.12 | 5.32 | 0.76 | 52 | 3.35 | -0.25 |
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
| ingenuo | Antioquia | forma_analogos_n3 | 728 | 95.2 | 4.8 | — |
| ingenuo | Antioquia | forma_analogos_n3_t45 | 728 | 94.4 | 5.6 | — |
| ingenuo | Antioquia | forma_analogos_n5 | 728 | 95.2 | 4.8 | — |
| ingenuo | Antioquia | forma_analogos_n5_t45 | 728 | 94.4 | 5.6 | — |
| ingenuo | Antioquia | forma_analogos_n8 | 728 | 95.2 | 4.8 | — |
| ingenuo | Antioquia | forma_analogos_n8_t45 | 728 | 94.4 | 5.6 | — |
| ingenuo | Atlantico | forma_analogos_n3 | 728 | 95.1 | 4.9 | — |
| ingenuo | Atlantico | forma_analogos_n3_t45 | 728 | 94.2 | 5.8 | — |
| ingenuo | Atlantico | forma_analogos_n5 | 728 | 95.1 | 4.9 | — |
| ingenuo | Atlantico | forma_analogos_n5_t45 | 728 | 94.2 | 5.8 | — |
| ingenuo | Atlantico | forma_analogos_n8 | 728 | 95.1 | 4.9 | — |
| ingenuo | Atlantico | forma_analogos_n8_t45 | 728 | 94.2 | 5.8 | — |
| archivo | Antioquia | forma_analogos_n3 | 147 | 95.9 | 4.1 | — |
| archivo | Antioquia | forma_analogos_n3_t45 | 147 | 93.9 | 6.1 | — |
| archivo | Antioquia | forma_analogos_n5 | 147 | 95.9 | 4.1 | — |
| archivo | Antioquia | forma_analogos_n5_t45 | 147 | 93.9 | 6.1 | — |
| archivo | Antioquia | forma_analogos_n8 | 147 | 95.9 | 4.1 | — |
| archivo | Antioquia | forma_analogos_n8_t45 | 147 | 93.9 | 6.1 | — |
| ingenuo | Antioquia | nivel_analogos_a025 | 728 | — | — | 97.8 |
| ingenuo | Antioquia | nivel_analogos_a050 | 728 | — | — | 97.8 |
| ingenuo | Atlantico | nivel_analogos_a025 | 728 | — | — | 98.2 |
| ingenuo | Atlantico | nivel_analogos_a050 | 728 | — | — | 98.2 |
| archivo | Antioquia | nivel_analogos_a025 | 147 | — | — | 98.0 |
| archivo | Antioquia | nivel_analogos_a050 | 147 | — | — | 98.0 |

## 2. Elección en calibración y resultado en holdout

- **ingenuo / Antioquia**: mejor config en calibración = `nivel_analogos_a050` (WAPE forma 3.351 %, mejora +8.36 % vs default) → holdout 3.292 % vs default 3.705 % (mejora +11.16 %).
- **ingenuo / Atlantico**: mejor config en calibración = `nivel_analogos_a050` (WAPE forma 5.438 %, mejora +0.45 % vs default) → holdout 5.835 % vs default 5.898 % (mejora +1.07 %).
- Mejor variante de **forma** en calibración (media de mejora relativa sobre las UCP, ingenuo): ninguna mejora al default.
- Mejor variante de **nivel** en calibración (media de mejora relativa sobre las UCP, ingenuo): `nivel_analogos_a050`.
- No se corrió combinación (alguna de las dos mejores variantes no mejora al default).

## 3. Por segmento

Días por segmento (config default; los segmentos se solapan):

| segmento | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo |
|---|---|---|---|---|---|
| todos | 364 | 364 | 364 | 364 | 147 |
| tipo:laboral | 245 | 243 | 242 | 244 | 99 |
| tipo:sabado | 51 | 52 | 51 | 52 | 21 |
| tipo:domingo | 51 | 51 | 52 | 51 | 21 |
| tipo:festivo | 17 | 18 | 19 | 17 | 6 |
| rel:festivo | 17 | 18 | 19 | 17 | 6 |
| rel:puente | 23 | 30 | 24 | 28 | 10 |
| rel:antes | 28 | 28 | 25 | 27 | 9 |
| rel:despues | 43 | 44 | 43 | 40 | 11 |
| rel:ninguna | 253 | 244 | 253 | 252 | 111 |
| despues_dist1 | 13 | 12 | 13 | 11 | 3 |
| semana_santa | 8 | 8 | 16 | 8 | 8 |
| diciembre | 31 | 31 | 7 | 31 | — |
| normal | 238 | 229 | 252 | 237 | 110 |

WAPE % del paso forma con el default por segmento:

| segmento | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo |
|---|---|---|---|---|---|
| todos | 3.66 | 3.71 | 5.46 | 5.90 | 2.79 |
| tipo:laboral | 3.34 | 3.38 | 5.07 | 5.64 | 2.76 |
| tipo:sabado | 2.91 | 2.78 | 5.84 | 5.76 | 2.49 |
| tipo:domingo | 3.29 | 3.51 | 6.39 | 6.22 | 2.89 |
| tipo:festivo | 12.94 | 12.57 | 7.21 | 9.51 | 4.34 |
| rel:festivo | 12.94 | 12.57 | 7.21 | 9.51 | 4.34 |
| rel:puente | 3.85 | 3.48 | 5.85 | 5.78 | 3.24 |
| rel:antes | 3.61 | 3.41 | 4.85 | 4.67 | 2.41 |
| rel:despues | 3.72 | 2.57 | 4.68 | 4.22 | 3.56 |
| rel:ninguna | 3.11 | 3.42 | 5.50 | 6.08 | 2.64 |
| despues_dist1 | 4.72 | 3.46 | 4.82 | 4.42 | 3.90 |
| semana_santa | 5.72 | 8.37 | 6.93 | 8.13 | 3.60 |
| diciembre | 4.19 | 4.97 | 3.98 | 6.51 | — |
| normal | 3.19 | 3.39 | 5.50 | 6.06 | 2.62 |

### Segmento `tipo:laboral` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo |
|---|---|---|---|---|---|---|
| forma_referencia | forma | -2.4 | -3.1 | -29.9 | -9.0 | -2.3 |
| forma_analogos_n3 | forma | -2.2 | -1.9 | -19.3 | -6.2 | -0.7 |
| forma_analogos_n3_t45 | forma | -2.2 | -1.9 | -19.9 | -5.7 | 0.3 |
| forma_analogos_n5 | forma | -1.9 | -2.5 | -21.4 | -5.9 | -1.5 |
| forma_analogos_n5_t45 | forma | -2.3 | -2.3 | -22.2 | -5.6 | -0.5 |
| forma_analogos_n8 | forma | -3.0 | -4.5 | -24.5 | -7.2 | -3.3 |
| forma_analogos_n8_t45 | forma | -3.3 | -4.0 | -25.9 | -6.8 | -2.3 |
| nivel_actual_a050 | nivel | -0.0 | -0.2 | -2.0 | -3.5 | -1.2 |
| nivel_actual_a050 | forma | 0.7 | 0.9 | -1.9 | -3.5 | -1.4 |
| nivel_analogos_a025 | nivel | 1.4 | 1.8 | -1.5 | 0.2 | 9.8 |
| nivel_analogos_a025 | forma | 1.9 | 2.3 | -1.4 | -0.0 | 9.6 |
| nivel_analogos_a050 | nivel | 4.4 | 6.2 | -1.9 | -0.1 | 15.2 |
| nivel_analogos_a050 | forma | 6.3 | 8.0 | -1.6 | 0.2 | 15.1 |

### Segmento `tipo:sabado` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo |
|---|---|---|---|---|---|---|
| forma_referencia | forma | -3.7 | -8.5 | -20.6 | -9.1 | -2.2 |
| forma_analogos_n3 | forma | -0.3 | -4.4 | -15.5 | -6.1 | 3.1 |
| forma_analogos_n3_t45 | forma | -0.4 | -4.8 | -15.7 | -6.3 | 3.1 |
| forma_analogos_n5 | forma | -1.8 | -4.5 | -16.2 | -5.1 | 2.3 |
| forma_analogos_n5_t45 | forma | -1.7 | -4.7 | -16.4 | -5.2 | 2.3 |
| forma_analogos_n8 | forma | -3.3 | -7.7 | -18.2 | -5.7 | -1.0 |
| forma_analogos_n8_t45 | forma | -3.3 | -7.9 | -18.8 | -5.8 | -1.0 |
| nivel_actual_a050 | nivel | -11.4 | -4.0 | -5.9 | -0.3 | -11.0 |
| nivel_actual_a050 | forma | -12.3 | -3.7 | -5.4 | -0.7 | -8.9 |
| nivel_analogos_a025 | nivel | 8.3 | 1.6 | 4.8 | -4.3 | 12.8 |
| nivel_analogos_a025 | forma | 8.4 | 1.1 | 5.0 | -3.6 | 13.6 |
| nivel_analogos_a050 | nivel | 11.5 | 4.5 | 5.4 | -4.2 | 10.7 |
| nivel_analogos_a050 | forma | 11.8 | 4.5 | 6.1 | -3.2 | 10.4 |

### Segmento `tipo:domingo` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo |
|---|---|---|---|---|---|---|
| forma_referencia | forma | -4.0 | -5.1 | -17.3 | -8.2 | -3.7 |
| forma_analogos_n3 | forma | -2.4 | -5.7 | -14.2 | -4.7 | -7.4 |
| forma_analogos_n3_t45 | forma | -2.5 | -6.2 | -14.3 | -5.5 | -7.4 |
| forma_analogos_n5 | forma | -3.5 | -7.6 | -15.0 | -3.2 | -11.1 |
| forma_analogos_n5_t45 | forma | -3.5 | -8.0 | -14.9 | -4.3 | -11.1 |
| forma_analogos_n8 | forma | -5.2 | -8.7 | -15.2 | -2.8 | -10.8 |
| forma_analogos_n8_t45 | forma | -5.2 | -9.1 | -15.4 | -3.9 | -10.8 |
| nivel_actual_a050 | nivel | -5.9 | -6.9 | -6.1 | -4.9 | -7.3 |
| nivel_actual_a050 | forma | -6.2 | -6.6 | -7.4 | -5.4 | -8.3 |
| nivel_analogos_a025 | nivel | -6.1 | 5.3 | 1.2 | -0.2 | 6.4 |
| nivel_analogos_a025 | forma | -6.5 | 6.4 | 1.1 | 1.2 | 6.4 |
| nivel_analogos_a050 | nivel | -5.1 | 7.6 | -1.1 | 1.3 | 6.0 |
| nivel_analogos_a050 | forma | -5.9 | 8.9 | -1.5 | 3.9 | 5.4 |

### Segmento `tipo:festivo` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo |
|---|---|---|---|---|---|---|
| forma_referencia | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n3 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n3_t45 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n5 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n5_t45 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n8 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n8_t45 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| nivel_actual_a050 | nivel | 21.4 | 23.0 | 9.7 | 9.2 | 3.4 |
| nivel_actual_a050 | forma | 29.0 | 28.6 | 11.2 | 9.9 | 5.7 |
| nivel_analogos_a025 | nivel | -1.0 | 1.5 | 0.6 | 2.3 | 12.2 |
| nivel_analogos_a025 | forma | -1.0 | 2.1 | 0.3 | 2.5 | 12.2 |
| nivel_analogos_a050 | nivel | 19.1 | 24.8 | 10.7 | 11.0 | 20.5 |
| nivel_analogos_a050 | forma | 26.4 | 31.7 | 12.3 | 12.1 | 21.4 |

### Segmento `rel:festivo` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo |
|---|---|---|---|---|---|---|
| forma_referencia | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n3 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n3_t45 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n5 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n5_t45 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n8 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| forma_analogos_n8_t45 | forma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| nivel_actual_a050 | nivel | 21.4 | 23.0 | 9.7 | 9.2 | 3.4 |
| nivel_actual_a050 | forma | 29.0 | 28.6 | 11.2 | 9.9 | 5.7 |
| nivel_analogos_a025 | nivel | -1.0 | 1.5 | 0.6 | 2.3 | 12.2 |
| nivel_analogos_a025 | forma | -1.0 | 2.1 | 0.3 | 2.5 | 12.2 |
| nivel_analogos_a050 | nivel | 19.1 | 24.8 | 10.7 | 11.0 | 20.5 |
| nivel_analogos_a050 | forma | 26.4 | 31.7 | 12.3 | 12.1 | 21.4 |

### Segmento `rel:puente` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo |
|---|---|---|---|---|---|---|
| forma_referencia | forma | -5.4 | -7.8 | -23.5 | -9.9 | -1.7 |
| forma_analogos_n3 | forma | -3.5 | -8.4 | -26.8 | -7.9 | -8.0 |
| forma_analogos_n3_t45 | forma | -3.1 | -9.0 | -25.5 | -8.2 | -8.0 |
| forma_analogos_n5 | forma | -5.7 | -11.1 | -21.5 | -8.8 | -13.0 |
| forma_analogos_n5_t45 | forma | -5.3 | -11.4 | -21.0 | -9.2 | -13.0 |
| forma_analogos_n8 | forma | -7.3 | -12.7 | -18.8 | -10.1 | -15.0 |
| forma_analogos_n8_t45 | forma | -6.8 | -12.8 | -18.7 | -10.5 | -15.0 |
| nivel_actual_a050 | nivel | -3.4 | -0.8 | -4.0 | -4.1 | 3.6 |
| nivel_actual_a050 | forma | -4.5 | -0.9 | -3.5 | -5.1 | 2.8 |
| nivel_analogos_a025 | nivel | 6.9 | 1.7 | 2.6 | 1.4 | 13.8 |
| nivel_analogos_a025 | forma | 6.6 | 2.4 | 2.6 | 1.9 | 13.9 |
| nivel_analogos_a050 | nivel | 10.1 | 3.5 | 3.3 | -0.2 | 6.4 |
| nivel_analogos_a050 | forma | 9.2 | 4.1 | 2.8 | 0.5 | 5.7 |

### Segmento `rel:antes` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo |
|---|---|---|---|---|---|---|
| forma_referencia | forma | -2.9 | -4.4 | -22.1 | -10.4 | 4.9 |
| forma_analogos_n3 | forma | -8.8 | -9.9 | -29.2 | -24.7 | -9.0 |
| forma_analogos_n3_t45 | forma | -9.3 | -7.6 | -35.0 | -19.3 | 4.0 |
| forma_analogos_n5 | forma | -6.9 | -11.6 | -30.4 | -23.5 | -14.8 |
| forma_analogos_n5_t45 | forma | -9.4 | -8.9 | -36.5 | -19.8 | -1.9 |
| forma_analogos_n8 | forma | -8.3 | -15.8 | -30.0 | -25.9 | -17.0 |
| forma_analogos_n8_t45 | forma | -10.5 | -12.0 | -37.4 | -20.9 | -4.3 |
| nivel_actual_a050 | nivel | -1.8 | -0.9 | -3.5 | -1.2 | 22.6 |
| nivel_actual_a050 | forma | -1.5 | 1.3 | -3.3 | -0.8 | 27.4 |
| nivel_analogos_a025 | nivel | 1.2 | -8.5 | -5.3 | -2.9 | 14.0 |
| nivel_analogos_a025 | forma | 1.5 | -7.0 | -5.8 | -3.9 | 14.5 |
| nivel_analogos_a050 | nivel | 3.6 | -8.9 | -13.6 | -4.6 | -2.1 |
| nivel_analogos_a050 | forma | 4.2 | -7.6 | -14.3 | -4.7 | -1.5 |

### Segmento `rel:despues` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo |
|---|---|---|---|---|---|---|
| forma_referencia | forma | -2.8 | -2.3 | -26.5 | -18.2 | -2.4 |
| forma_analogos_n3 | forma | -2.6 | -5.7 | -28.6 | -12.6 | -5.8 |
| forma_analogos_n3_t45 | forma | -2.7 | -8.5 | -30.2 | -14.1 | -5.8 |
| forma_analogos_n5 | forma | -4.3 | -8.0 | -30.9 | -16.1 | -9.9 |
| forma_analogos_n5_t45 | forma | -4.6 | -9.2 | -32.5 | -17.5 | -9.9 |
| forma_analogos_n8 | forma | -5.6 | -12.1 | -29.6 | -19.8 | -10.1 |
| forma_analogos_n8_t45 | forma | -6.1 | -11.9 | -34.9 | -22.1 | -10.1 |
| nivel_actual_a050 | nivel | 3.0 | -9.1 | -1.7 | -5.1 | -9.4 |
| nivel_actual_a050 | forma | 2.9 | -9.5 | -1.8 | -4.5 | -9.3 |
| nivel_analogos_a025 | nivel | -7.7 | -6.8 | -1.9 | 2.0 | 24.5 |
| nivel_analogos_a025 | forma | -6.6 | -6.9 | -1.9 | 1.5 | 21.8 |
| nivel_analogos_a050 | nivel | -12.1 | -17.0 | -3.9 | 1.9 | 28.2 |
| nivel_analogos_a050 | forma | -10.4 | -16.8 | -4.1 | 2.5 | 26.1 |

### Segmento `rel:ninguna` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo |
|---|---|---|---|---|---|---|
| forma_referencia | forma | -2.5 | -3.8 | -26.9 | -7.7 | -3.2 |
| forma_analogos_n3 | forma | -0.8 | -0.8 | -14.3 | -3.5 | 0.8 |
| forma_analogos_n3_t45 | forma | -0.8 | -0.8 | -14.3 | -3.5 | 0.8 |
| forma_analogos_n5 | forma | -0.7 | -1.0 | -16.7 | -2.4 | 0.8 |
| forma_analogos_n5_t45 | forma | -0.7 | -1.0 | -16.7 | -2.4 | 0.8 |
| forma_analogos_n8 | forma | -1.8 | -2.6 | -20.4 | -2.9 | -1.1 |
| forma_analogos_n8_t45 | forma | -1.8 | -2.6 | -20.4 | -2.9 | -1.1 |
| nivel_actual_a050 | nivel | -3.2 | -0.7 | -3.5 | -3.1 | -5.1 |
| nivel_actual_a050 | forma | -2.6 | 0.4 | -3.6 | -3.3 | -5.5 |
| nivel_analogos_a025 | nivel | 2.6 | 4.7 | 0.4 | -0.8 | 7.0 |
| nivel_analogos_a025 | forma | 2.9 | 5.2 | 0.6 | -0.5 | 7.2 |
| nivel_analogos_a050 | nivel | 6.9 | 11.3 | 0.7 | -0.4 | 13.3 |
| nivel_analogos_a050 | forma | 8.4 | 13.3 | 1.1 | 0.4 | 13.3 |

### Segmento `despues_dist1` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo |
|---|---|---|---|---|---|---|
| forma_referencia | forma | 1.0 | -5.1 | -25.2 | -18.8 | -2.4 |
| forma_analogos_n3 | forma | -0.6 | -4.3 | -21.4 | -8.4 | -12.7 |
| forma_analogos_n3_t45 | forma | -1.1 | -6.3 | -23.2 | -8.5 | -12.7 |
| forma_analogos_n5 | forma | -1.4 | -7.2 | -24.0 | -9.4 | -20.2 |
| forma_analogos_n5_t45 | forma | -2.4 | -7.0 | -26.1 | -8.4 | -20.2 |
| forma_analogos_n8 | forma | -1.9 | -10.6 | -25.2 | -12.2 | -21.7 |
| forma_analogos_n8_t45 | forma | -2.9 | -9.1 | -28.7 | -11.6 | -21.7 |
| nivel_actual_a050 | nivel | -3.8 | -20.1 | -1.2 | -8.8 | -32.2 |
| nivel_actual_a050 | forma | -3.5 | -19.7 | -2.1 | -9.6 | -31.3 |
| nivel_analogos_a025 | nivel | -4.6 | 2.1 | -1.4 | 4.9 | 45.0 |
| nivel_analogos_a025 | forma | -3.1 | 2.1 | -0.7 | 3.2 | 42.3 |
| nivel_analogos_a050 | nivel | -13.4 | -7.6 | -3.5 | 4.1 | 49.1 |
| nivel_analogos_a050 | forma | -12.1 | -7.0 | -2.7 | 2.2 | 52.7 |

### Segmento `semana_santa` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo |
|---|---|---|---|---|---|---|
| forma_referencia | forma | 2.8 | 0.8 | 7.7 | -0.3 | 8.6 |
| forma_analogos_n3 | forma | -1.1 | -0.3 | -3.6 | -2.1 | 7.5 |
| forma_analogos_n3_t45 | forma | -1.1 | -0.3 | -3.6 | -2.1 | 7.5 |
| forma_analogos_n5 | forma | -1.8 | -0.3 | -1.9 | -4.5 | 3.2 |
| forma_analogos_n5_t45 | forma | -1.8 | -0.3 | -1.9 | -4.5 | 3.2 |
| forma_analogos_n8 | forma | -2.4 | -2.1 | -0.8 | -1.7 | 2.1 |
| forma_analogos_n8_t45 | forma | -2.6 | -2.1 | -1.3 | -3.5 | 2.1 |
| nivel_actual_a050 | nivel | 16.7 | 24.1 | 9.7 | 11.1 | 14.3 |
| nivel_actual_a050 | forma | 25.2 | 26.3 | 12.3 | 11.8 | 15.4 |
| nivel_analogos_a025 | nivel | -8.1 | 2.1 | -0.1 | 1.4 | 1.6 |
| nivel_analogos_a025 | forma | -9.4 | 1.8 | -0.6 | 0.6 | 0.8 |
| nivel_analogos_a050 | nivel | 13.2 | 27.5 | 13.1 | 14.2 | 19.4 |
| nivel_analogos_a050 | forma | 17.0 | 29.9 | 13.9 | 15.5 | 18.7 |

### Segmento `diciembre` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout |
|---|---|---|---|---|---|
| forma_referencia | forma | -3.6 | -0.3 | -14.1 | -2.7 |
| forma_analogos_n3 | forma | -10.6 | -0.5 | -16.5 | -7.1 |
| forma_analogos_n3_t45 | forma | -10.6 | -3.0 | -16.5 | -8.8 |
| forma_analogos_n5 | forma | -8.6 | -1.1 | -12.9 | -5.8 |
| forma_analogos_n5_t45 | forma | -10.2 | -3.5 | -12.9 | -8.6 |
| forma_analogos_n8 | forma | -8.3 | -2.9 | -7.6 | -8.0 |
| forma_analogos_n8_t45 | forma | -10.2 | -4.7 | -9.4 | -9.5 |
| nivel_actual_a050 | nivel | 12.7 | 5.5 | 7.9 | -1.6 |
| nivel_actual_a050 | forma | 12.5 | 6.7 | 10.2 | -1.5 |
| nivel_analogos_a025 | nivel | -11.3 | -3.2 | -2.3 | 0.8 |
| nivel_analogos_a025 | forma | -10.8 | -1.4 | -2.0 | 0.7 |
| nivel_analogos_a050 | nivel | -11.5 | 5.0 | 1.5 | -0.6 |
| nivel_analogos_a050 | forma | -10.1 | 7.0 | 4.2 | -0.4 |

### Segmento `normal` — mejora relativa % del WAPE vs default

| config | paso | ingenuo·Antioquia·calibracion | ingenuo·Antioquia·holdout | ingenuo·Atlantico·calibracion | ingenuo·Atlantico·holdout | archivo·Antioquia·archivo |
|---|---|---|---|---|---|---|
| forma_referencia | forma | -2.5 | -4.2 | -27.1 | -8.0 | -3.2 |
| forma_analogos_n3 | forma | 0.2 | -1.0 | -14.4 | -3.5 | 0.8 |
| forma_analogos_n3_t45 | forma | 0.2 | -1.0 | -14.4 | -3.5 | 0.8 |
| forma_analogos_n5 | forma | -0.1 | -1.0 | -16.7 | -2.4 | 0.8 |
| forma_analogos_n5_t45 | forma | -0.1 | -1.0 | -16.7 | -2.4 | 0.8 |
| forma_analogos_n8 | forma | -1.2 | -2.5 | -20.5 | -2.8 | -1.1 |
| forma_analogos_n8_t45 | forma | -1.2 | -2.5 | -20.5 | -2.8 | -1.1 |
| nivel_actual_a050 | nivel | -3.4 | -0.8 | -3.5 | -3.4 | -5.5 |
| nivel_actual_a050 | forma | -2.8 | 0.3 | -3.6 | -3.6 | -5.8 |
| nivel_analogos_a025 | nivel | 3.6 | 4.7 | 0.4 | -0.5 | 6.9 |
| nivel_analogos_a025 | forma | 3.9 | 5.2 | 0.6 | -0.2 | 7.1 |
| nivel_analogos_a050 | nivel | 8.2 | 11.4 | 0.6 | 0.1 | 12.8 |
| nivel_analogos_a050 | forma | 9.9 | 13.4 | 1.1 | 0.9 | 12.8 |

## 4. Días difíciles

Muestras chicas: con 52 semanas por partición hay pocos festivos, puentes y días de Semana Santa por UCP (ver n). Diferencias en estos segmentos son ruidosas; un solo día atípico mueve el WAPE. Con base `ingenuo` los festivos arrancan de un día no festivo (error artificial grande).

- **rel:festivo**: ingenuo·Antioquia·calibracion: n=17, default 12.94 %, mejor `nivel_actual_a050` 9.18 % (+29.0 %); ingenuo·Antioquia·holdout: n=18, default 12.57 %, mejor `nivel_analogos_a050` 8.58 % (+31.7 %); ingenuo·Atlantico·calibracion: n=19, default 7.21 %, mejor `nivel_analogos_a050` 6.33 % (+12.3 %); ingenuo·Atlantico·holdout: n=17, default 9.51 %, mejor `nivel_analogos_a050` 8.37 % (+12.1 %); archivo·Antioquia·archivo: n=6, default 4.34 %, mejor `nivel_analogos_a050` 3.41 % (+21.4 %).
- **rel:despues**: ingenuo·Antioquia·calibracion: n=43, default 3.72 %, mejor `nivel_actual_a050` 3.61 % (+2.9 %); ingenuo·Antioquia·holdout: n=44, default 2.57 %, mejor `default` 2.57 % (+0.0 %); ingenuo·Atlantico·calibracion: n=43, default 4.68 %, mejor `default` 4.68 % (+0.0 %); ingenuo·Atlantico·holdout: n=40, default 4.22 %, mejor `nivel_analogos_a050` 4.11 % (+2.5 %); archivo·Antioquia·archivo: n=11, default 3.56 %, mejor `nivel_analogos_a050` 2.63 % (+26.1 %).
- **despues_dist1**: ingenuo·Antioquia·calibracion: n=13, default 4.72 %, mejor `forma_referencia` 4.67 % (+1.0 %); ingenuo·Antioquia·holdout: n=12, default 3.46 %, mejor `nivel_analogos_a025` 3.38 % (+2.1 %); ingenuo·Atlantico·calibracion: n=13, default 4.82 %, mejor `default` 4.82 % (+0.0 %); ingenuo·Atlantico·holdout: n=11, default 4.42 %, mejor `nivel_analogos_a025` 4.28 % (+3.2 %); archivo·Antioquia·archivo: n=3, default 3.90 %, mejor `nivel_analogos_a050` 1.84 % (+52.7 %).
- **rel:antes**: ingenuo·Antioquia·calibracion: n=28, default 3.61 %, mejor `nivel_analogos_a050` 3.46 % (+4.2 %); ingenuo·Antioquia·holdout: n=28, default 3.41 %, mejor `nivel_actual_a050` 3.37 % (+1.3 %); ingenuo·Atlantico·calibracion: n=25, default 4.85 %, mejor `default` 4.85 % (+0.0 %); ingenuo·Atlantico·holdout: n=27, default 4.67 %, mejor `default` 4.67 % (+0.0 %); archivo·Antioquia·archivo: n=9, default 2.41 %, mejor `nivel_actual_a050` 1.75 % (+27.4 %).
- **rel:puente**: ingenuo·Antioquia·calibracion: n=23, default 3.85 %, mejor `nivel_analogos_a050` 3.50 % (+9.2 %); ingenuo·Antioquia·holdout: n=30, default 3.48 %, mejor `nivel_analogos_a050` 3.33 % (+4.1 %); ingenuo·Atlantico·calibracion: n=24, default 5.85 %, mejor `nivel_analogos_a050` 5.69 % (+2.8 %); ingenuo·Atlantico·holdout: n=28, default 5.78 %, mejor `nivel_analogos_a025` 5.67 % (+1.9 %); archivo·Antioquia·archivo: n=10, default 3.24 %, mejor `nivel_analogos_a025` 2.79 % (+13.9 %).
- **semana_santa**: ingenuo·Antioquia·calibracion: n=8, default 5.72 %, mejor `nivel_actual_a050` 4.28 % (+25.2 %); ingenuo·Antioquia·holdout: n=8, default 8.37 %, mejor `nivel_analogos_a050` 5.87 % (+29.9 %); ingenuo·Atlantico·calibracion: n=16, default 6.93 %, mejor `nivel_analogos_a050` 5.97 % (+13.9 %); ingenuo·Atlantico·holdout: n=8, default 8.13 %, mejor `nivel_analogos_a050` 6.87 % (+15.5 %); archivo·Antioquia·archivo: n=8, default 3.60 %, mejor `nivel_analogos_a050` 2.93 % (+18.7 %).
- **diciembre**: ingenuo·Antioquia·calibracion: n=31, default 4.19 %, mejor `nivel_actual_a050` 3.67 % (+12.5 %); ingenuo·Antioquia·holdout: n=31, default 4.97 %, mejor `nivel_analogos_a050` 4.62 % (+7.0 %); ingenuo·Atlantico·calibracion: n=7, default 3.98 %, mejor `nivel_actual_a050` 3.57 % (+10.2 %); ingenuo·Atlantico·holdout: n=31, default 6.51 %, mejor `nivel_analogos_a025` 6.47 % (+0.7 %).

## 5. Regla de cambio de default (base ingenuo, ambas UCP, calibración y holdout)

Una fila por segmento × variante. `celdas_que_ganan` = UCP × partición × paso afectado que mejoran >= 2.0 % sin empeorar el pico. Solo se listan las variantes con al menos la mitad de las celdas ganadas; `cumple_regla` = todas.

| segmento | config | pasos | n_dias_min | celdas | celdas_que_ganan | mejora_min_pct | mejora_media_pct | delta_pico_max_pp | cumple_regla |
|---|---|---|---|---|---|---|---|---|---|
| semana_santa | nivel_analogos_a050 | nivel+forma | 8 | 8/8 | 8 | 13.12 | 18.04 | -0.61 | sí |
| tipo:festivo | nivel_analogos_a050 | nivel+forma | 17 | 8/8 | 8 | 10.74 | 18.52 | -0.31 | sí |
| rel:festivo | nivel_analogos_a050 | nivel+forma | 17 | 8/8 | 8 | 10.74 | 18.52 | -0.31 | sí |
| semana_santa | nivel_actual_a050 | nivel+forma | 8 | 8/8 | 8 | 9.70 | 17.16 | -0.73 | sí |
| tipo:festivo | nivel_actual_a050 | nivel+forma | 17 | 8/8 | 8 | 9.24 | 17.75 | -0.37 | sí |
| rel:festivo | nivel_actual_a050 | nivel+forma | 17 | 8/8 | 8 | 9.24 | 17.75 | -0.37 | sí |
| diciembre | nivel_actual_a050 | nivel+forma | 7 | 8/8 | 6 | -1.57 | 6.55 | 0.30 | no |
| tipo:sabado | nivel_analogos_a050 | nivel+forma | 51 | 8/8 | 6 | -4.15 | 4.55 | 0.51 | no |
| rel:puente | nivel_analogos_a025 | nivel+forma | 23 | 8/8 | 5 | 1.38 | 3.27 | 0.02 | no |
| rel:puente | nivel_analogos_a050 | nivel+forma | 23 | 8/8 | 5 | -0.18 | 4.16 | 0.13 | no |
| todos | nivel_analogos_a050 | nivel+forma | 364 | 8/8 | 4 | 0.18 | 4.58 | 0.11 | no |
| normal | nivel_analogos_a050 | nivel+forma | 229 | 8/8 | 4 | 0.07 | 5.70 | 0.16 | no |
| rel:ninguna | nivel_analogos_a050 | nivel+forma | 244 | 8/8 | 4 | -0.44 | 5.20 | 0.16 | no |
| normal | nivel_analogos_a025 | nivel+forma | 229 | 8/8 | 4 | -0.52 | 2.20 | 0.13 | no |
| rel:ninguna | nivel_analogos_a025 | nivel+forma | 244 | 8/8 | 4 | -0.81 | 1.90 | 0.13 | no |
| tipo:laboral | nivel_analogos_a050 | nivel+forma | 242 | 8/8 | 4 | -1.87 | 2.68 | 0.20 | no |
| tipo:sabado | nivel_analogos_a025 | nivel+forma | 51 | 8/8 | 4 | -4.31 | 2.66 | 0.49 | no |

## 6. Conclusión

**Variantes que cumplen la regla (candidatas a default por segmento; decide el coordinador):**

- Segmento `rel:festivo`: `nivel_analogos_a050` (pasos nivel+forma, n mínimo 17 días, mejora mínima 10.7 %, media 18.5 %, Δ pico máx -0.31 pp).
- Segmento `rel:festivo`: `nivel_actual_a050` (pasos nivel+forma, n mínimo 17 días, mejora mínima 9.2 %, media 17.7 %, Δ pico máx -0.37 pp).
- Segmento `semana_santa`: `nivel_analogos_a050` (pasos nivel+forma, n mínimo 8 días, mejora mínima 13.1 %, media 18.0 %, Δ pico máx -0.61 pp).
- Segmento `semana_santa`: `nivel_actual_a050` (pasos nivel+forma, n mínimo 8 días, mejora mínima 9.7 %, media 17.2 %, Δ pico máx -0.73 pp).
- Segmento `tipo:festivo`: `nivel_analogos_a050` (pasos nivel+forma, n mínimo 17 días, mejora mínima 10.7 %, media 18.5 %, Δ pico máx -0.31 pp).
- Segmento `tipo:festivo`: `nivel_actual_a050` (pasos nivel+forma, n mínimo 17 días, mejora mínima 9.2 %, media 17.7 %, Δ pico máx -0.37 pp).

Si se aplica, debe ser por segmento y retrocompatible (p. ej. un default condicionado al tipo de día), y conviene revalidar con la base `modelo` antes de fijarlo. Segmentos con n chico (< 15 días por celda) deben tomarse como indicio, no como evidencia.

**Lectura: la ganancia viene sobre todo de α, no del ancla por análogos.** En estos segmentos también cumple la variante con ancla actual y el mismo α:
- `tipo:festivo`: `nivel_analogos_a050` mejora media 18.5 % vs `nivel_actual_a050` 17.7 % → aporte propio del ancla por análogos ≈ +0.8 pp.
- `rel:festivo`: `nivel_analogos_a050` mejora media 18.5 % vs `nivel_actual_a050` 17.7 % → aporte propio del ancla por análogos ≈ +0.8 pp.
- `semana_santa`: `nivel_analogos_a050` mejora media 18.0 % vs `nivel_actual_a050` 17.2 % → aporte propio del ancla por análogos ≈ +0.9 pp.

**Festivos y Semana Santa con base `ingenuo` no son interpretables por sí solos**: la base ingenua pronostica cada festivo con un día NO festivo, así que cualquier α más alto "gana" porque corrige ese error artificial (mismo hallazgo de `backtest_20261112_v3.md`). La base `archivo` (Antioquia, n chico) apunta en la misma dirección, pero es una sola emisión con ajustes post-hoc. Recomendación: NO cambiar defaults todavía; dejar α = 0.5 con ancla por análogos en festivos / Semana Santa como candidato a revalidar con `--base modelo` (y coordinarlo con el α por tipo de día de la Fase 2a).

## 7. Cómo correr con la base modelo (Fase 2a)

```
.venv/Scripts/python.exe scripts/backtest_analogos.py --base modelo ingenuo --fecha <AAAAMMDD>
```
Requiere que `run_predict_flow` funcione offline (Fase 2a). La regla de la §5 se evalúa solo con `ingenuo`; con `modelo` se debe repetir el análisis cambiando la base en `evaluar_regla`.

## 8. Limitaciones

- **Base ingenua ≠ modelo real**: la base `ingenuo` (mismo día de la semana anterior) es débil y en festivos
  parte de un día no festivo; las mejoras medidas sobre ella pueden no trasladarse al pronóstico del modelo
  (probablemente serán menores). La regla se evalúa con `ingenuo` porque es la única base con dos UCP y 104
  semanas; repetir con `--base modelo` cuando la Fase 2a lo habilite.
- **Base `archivo`**: una sola emisión con corte 2025-12-31 y ajustes post-hoc; solo Antioquia, ~21 semanas
  (ene–may 2026). Se reporta como contraste, sin elegir nada ahí.
- **Clima no usado** en la búsqueda automática (`seleccionar` no puntúa clima; además el clima real solo
  cubre 2025-12..2026-07). El filtro de clima del endpoint queda sin evaluar.
- **Muestras chicas** en festivos, puentes, Semana Santa y diciembre (ver n por segmento): diferencias de
  pocos puntos en esos segmentos no son concluyentes.
- **Corte siempre en sábado** (1 día de brecha); otras emisiones no se evalúan.
- **Calibración y holdout contiguos** (52 + 52 semanas): un cambio de régimen en el holdout afecta a todas
  las configs por igual, pero la comparación es de una sola partición temporal (sin validación cruzada).
- La elección de la mejor variante y la combinación se hacen solo en calibración (ingenuo); el holdout no se
  usa para elegir.
