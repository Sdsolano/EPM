# Backtest de ajustes con el modelo real y recalibración — 20261113

Generado por `scripts/recalibrar_ajustes.py` (`generar` → `evaluar` → `recalibrar`). Detalle en `reports/ajustes/backtest_modelo_grid.csv.gz` y `backtest_modelo_extra.csv.gz` (no versionados).

<!-- manual:inicio -->
## Resumen y decisiones

**Qué es nuevo**: por primera vez la base es el **modelo real sin fuga** (sandbox, entrenado solo con datos ≤ corte en
cada semana), no la base ingenua ni el archivo histórico. Holdout temporal 60/40 por UCP: todo se elige en las 31
semanas antiguas (calibración) y se reporta en las 21 recientes (prueba).

### Números clave (prueba, config default α 0.25 + `promedio_n` n 4, WAPE horario %)
| UCP | base | modelo_k1 base → nivel → forma | MAPE energía semanal base → final | ingenuo final | archivo final* |
|---|---|---|---|---|---|
| Antioquia | modelo | 2.44 → 2.44 → **2.33** | 1.52 → 1.37 | 3.56 | 2.52 |
| Atlántico | modelo | 5.86 → 5.77 → **5.46** | 4.57 → 4.41 | 6.06 | — |

\* `archivo` solo cubre 21 semanas de Antioquia (17 en prueba). En esas mismas 21 semanas: modelo_k1 final 2.61 %,
archivo final 2.79 %, ingenuo final 3.82 % (WAPE); el modelo real es la mejor base y los ajustes la mejoran ~0.1 pp.

- Los ajustes aportan menos sobre el modelo que sobre la base ingenua (Antioquia prueba −0.11 pp vs −0.35 pp con
  ingenuo; Atlántico −0.40 pp vs −0.15 pp), como se anticipaba. El aporte viene sobre todo de **forma**; **nivel**
  casi no cambia el WAPE horario (Antioquia prueba 2.44 → 2.44) pero sí mejora la **energía semanal** (MAPE −0.15 pp
  en Antioquia y −0.16 pp en Atlántico, prueba).

### DEFAULTS: no se cambian (decisión con evidencia)
- α: en calibración el mejor α es 0.25 en ambas UCP (todas las fuentes); en prueba Antioquia queda 0.00 ≈ 0.25
  (2.32 vs 2.33) y Atlántico 0.25. **Se confirma α = 0.25** fuera de muestra con el modelo real.
- Fuente/n: `promedio_n` gana a `reciente` y `referencia` en las dos UCP y en las dos particiones; n ∈ {2..8}
  difiere en centésimas. La mejor conjunta en calibración (`promedio_n` n = 6) **no cumple la regla** en prueba
  (Antioquia +0.016 pp, Atlántico −0.007 pp). Mejor por UCP: Antioquia n = 2, Atlántico n = 8, ninguna mejora al
  default en prueba (+0.017 y +0.013 pp). → se mantiene `promedio_n`, n = 4.
- `alpha_por_tipo`: con la regla (festivo solo si ≥ 10 festivos en calibración por UCP; Atlántico tiene 9) el α
  conjunto óptimo por tipo es 0.25 en todos los tipos → `{}` (nada que proponer como default). Por UCP:
  Antioquia `{domingo: 0, festivo: 0}` (calibración −0.045 pp) y Atlántico `{laboral: 0}` (calibración −0.09 pp)
  **empeoran en prueba** (+0.007 y +0.014 pp, y el MAPE semanal +0.02 / +0.07 pp). No se recomienda ni como default
  ni por UCP. (Aunque hubiera ganado, meterlo en DEFAULTS cambiaría el resultado de corridas guardadas al
  re-ejecutarse: `rerun_from` re-valida los params persistidos con `_params.completar`, que rellena claves ausentes
  con los DEFAULTS actuales. Habría que versionar params antes de hacerlo.)
- Banda del control `banda_referencia_pp`: en calibración gana 1.0 pp por lift promedio (0.06 / 0.03), pero el lift
  es ≈ 0 en todas las bandas y en prueba 1.0 pp da lift 0.00 (Antioquia) y −0.04 (Atlántico) vs 2.0 pp −0.05 / +0.01.
  **La alerta no predice el error semanal del final** a ninguna banda (21 semanas de prueba: evidencia débil).
  Se mantiene 2.0 pp: cambiarla solo cambiaría cuántas veces suena (~70 % de las semanas con 1.0 pp) sin ganar
  información. Es una alerta de coherencia de proceso (info), no un predictor de error.

### R2 — qué puede pasar de "sugerido" a "automático" (modelo_k1, default; criterio en § 8)
Criterio: el paso baja el WAPE del segmento en calibración **y** en prueba, empeora en ≤ 35 % de las semanas de
prueba y hay ≥ 10 días de prueba.
- **Forma — automático**: laboral (Antioquia y Atlántico; prueba −0.14 / −0.29 pp, empeora en 10 % / 14 % de las
  semanas) y domingo de Atlántico (−0.55 pp). **Sugerido**: sábados (neutro en prueba en ambas UCP), domingo de
  Antioquia (neutro/levemente peor) y festivos (n < 10 en prueba; mejora −0.3 / −0.7 pp pero no concluyente).
- **Nivel — automático** solo sábado de Atlántico (−0.36 pp en ambas particiones, 33 % de semanas peores; borde).
  **No aplicar por defecto** (empeora WAPE en ambas particiones): domingo de Antioquia (+0.13 / +0.10 pp) y laboral
  de Atlántico (+0.09 / +0.02 pp). Resto **sugerido**: el efecto medio es pequeño y el paso empeora el WAPE horario
  en 40–60 % de las semanas (es ruidoso día a día) aunque mejora la energía semanal. Nota: "no aplicar" se refiere
  al WAPE horario; la energía semanal mejora con nivel en ambas UCP, así que antes de apagarlo por segmento conviene
  medirlo con ediciones reales del analista.
- **Control**: sigue informativo (no modifica curvas). Banda 2.0 pp sin cambio.

### Sensibilidad k = 1 vs k = 4
Con reentreno cada 4 semanas la base empeora un poco (Antioquia prueba 2.44 → 2.58 %, Atlántico 5.86 → 5.88 %), los
ajustes aportan algo más (Antioquia base → forma −0.16 pp con k = 4 vs −0.11 con k = 1) y las conclusiones de
selección son las mismas (α 0.25, `promedio_n`, mejores n idénticos). k = 1 es la variante más favorable al modelo.

### Limitaciones
- **Clima observado** como "pronóstico" de clima (optimista para la base modelo). Antioquia solo tiene clima desde
  2025-12-01: las semanas de calibración de Antioquia (hasta 2026-01-26) entrenaron casi todas con clima en 0 y
  predicen con promedios históricos (igual que haría producción con esos archivos); Atlántico tiene clima desde 2018.
- **Festivos** del calendario oficial (`holidays`), no de la API de festivos por UCP.
- **Holdout pequeño**: 21 semanas de prueba por UCP (7–8 festivos); diferencias < 0.05 pp no son distinguibles del
  ruido. Las diferencias entre n de `promedio_n` son de centésimas.
- **Corte siempre sábado** (1 día de brecha); otras fechas de emisión no evaluadas.
- **k = 1 más favorable que la operación real** (el champion no se reentrena cada semana); k = 4 como sensibilidad.
- Sin ediciones del analista: el enclavamiento `TENDENCIA_FUERA_BANDA` no se puede evaluar.
- `archivo`: una sola emisión con corte 2025-12-31 y ajustes post-hoc; solo como contraste.

Decisiones tomadas sin el usuario: ver `docs/work/ajustes/fase2a/B_REPORTE.md`.
<!-- manual:fin -->

## 1. Configuración

- Base `modelo`: flujo real (`scripts/backtest_modelo.SandboxModelo`), entrenado SOLO con datos ≤ corte en un sandbox temporal sin red. `modelo_k1` = reentreno cada semana (corrida principal); `modelo_k4` = reentreno cada 4 semanas (las intermedias predicen con el modelo del último origen; sensibilidad).
- Corte = sábado anterior a la semana objetivo (como `construir_contexto`); `n_days` = 8 (domingo de brecha + semana).
- Holdout temporal por UCP: primeras 60 % de semanas = calibración (TODO se elige ahí), últimas = prueba (solo se reporta). Bases `ingenuo` y `archivo` en las mismas semanas.
- Grid: α ∈ [0.0, 0.25, 0.5, 0.75, 1.0] × fuente ∈ {reciente, referencia, promedio_n(n ∈ [2, 3, 4, 6, 8])}; default actual = `a0.25|promedio_n4` (α 0.25, `promedio_n`, n 4).

| ucp | semanas | calibracion | prueba |
|---|---|---|---|
| Antioquia | 52 | 2025-06-30 .. 2026-01-26 (31) | 2026-02-02 .. 2026-06-22 (21) |
| Atlantico | 52 | 2025-05-05 .. 2025-12-01 (31) | 2025-12-08 .. 2026-04-27 (21) |

### Cobertura por base

| base | ucp | primera | ultima | n_semanas |
|---|---|---|---|---|
| archivo | Antioquia | 2026-01-05 | 2026-05-25 | 21 |
| ingenuo | Antioquia | 2025-06-30 | 2026-06-22 | 52 |
| ingenuo | Atlantico | 2025-05-05 | 2026-04-27 | 52 |
| modelo_k1 | Antioquia | 2025-06-30 | 2026-06-22 | 52 |
| modelo_k1 | Atlantico | 2025-05-05 | 2026-04-27 | 52 |
| modelo_k4 | Antioquia | 2025-06-30 | 2026-06-22 | 52 |
| modelo_k4 | Atlantico | 2025-05-05 | 2026-04-27 | 52 |

### No-fuga y hashes

Cada semana: `datos.csv` del sandbox recortado a FECHA ≤ corte + `end_date=corte`; el contexto del pipeline verifica `max(historico) ≤ fecha_corte ≤ sábado`. Hash sha256 de `data/raw`, `models`, `data/features`, `data/processed` del worktree antes/después de cada proceso de generación:

| archivo | k | ucps | archivos_hasheados | intacto | segundos |
|---|---|---|---|---|---|
| hash_k1_Antioquia.json | 1 | Antioquia | 54 | sí | 1762.10 |
| hash_k1_Atlantico.json | 1 | Atlantico | 54 | sí | 1760.30 |
| hash_k4_Antioquia.json | 4 | Antioquia | 54 | sí | 690.10 |
| hash_k4_Atlantico.json | 4 | Atlantico | 54 | sí | 659.20 |

## 2. Resultado por paso con el default actual (modelo vs ingenuo vs archivo)

WAPE horario = Σ|pred−real|/Σreal. Energía diaria y semanal: MAPE y sesgo %. Pico: |error %| del máximo horario y diferencia de hora (circular). `archivo` solo Antioquia (semanas 2026-01..05, mayormente prueba).

| particion | base | ucp | paso | n_dias | wape_pct | mape_energia_dia_pct | sesgo_energia_dia_pct | err_pico_pct | dif_hora_pico_h | n_semanas | mape_energia_semana_pct | sesgo_energia_semana_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prueba | modelo_k1 | Antioquia | base | 147.00 | 2.44 | 2.10 | -0.50 | 2.10 | 2.19 | 21 | 1.52 | -0.52 |
| prueba | modelo_k1 | Antioquia | nivel | 147.00 | 2.44 | 2.09 | -0.55 | 2.15 | 2.19 | 21 | 1.37 | -0.57 |
| prueba | modelo_k1 | Antioquia | forma | 147.00 | 2.33 | 2.09 | -0.55 | 2.07 | 2.08 | 21 | 1.37 | -0.57 |
| prueba | modelo_k1 | Atlantico | base | 147.00 | 5.86 | 5.19 | -0.36 | 5.28 | 0.88 | 21 | 4.57 | -0.45 |
| prueba | modelo_k1 | Atlantico | nivel | 147.00 | 5.77 | 5.05 | -0.26 | 5.16 | 0.88 | 21 | 4.41 | -0.36 |
| prueba | modelo_k1 | Atlantico | forma | 147.00 | 5.46 | 5.05 | -0.26 | 5.04 | 0.77 | 21 | 4.41 | -0.36 |
| prueba | modelo_k4 | Antioquia | base | 147.00 | 2.58 | 2.23 | -0.71 | 2.21 | 2.67 | 21 | 1.59 | -0.73 |
| prueba | modelo_k4 | Antioquia | nivel | 147.00 | 2.53 | 2.19 | -0.70 | 2.19 | 2.67 | 21 | 1.44 | -0.73 |
| prueba | modelo_k4 | Antioquia | forma | 147.00 | 2.42 | 2.19 | -0.70 | 2.22 | 2.08 | 21 | 1.44 | -0.73 |
| prueba | modelo_k4 | Atlantico | base | 147.00 | 5.88 | 5.17 | -0.49 | 5.36 | 0.85 | 21 | 4.52 | -0.58 |
| prueba | modelo_k4 | Atlantico | nivel | 147.00 | 5.80 | 5.04 | -0.36 | 5.25 | 0.85 | 21 | 4.37 | -0.45 |
| prueba | modelo_k4 | Atlantico | forma | 147.00 | 5.47 | 5.04 | -0.36 | 5.06 | 0.77 | 21 | 4.37 | -0.45 |
| prueba | ingenuo | Antioquia | base | 147.00 | 3.91 | 3.73 | -0.15 | 3.49 | 2.31 | 21 | 2.50 | -0.29 |
| prueba | ingenuo | Antioquia | nivel | 147.00 | 3.66 | 3.44 | -0.28 | 3.26 | 2.31 | 21 | 2.22 | -0.39 |
| prueba | ingenuo | Antioquia | forma | 147.00 | 3.56 | 3.44 | -0.28 | 3.46 | 2.08 | 21 | 2.22 | -0.39 |
| prueba | ingenuo | Atlantico | base | 147.00 | 6.21 | 5.65 | -0.81 | 5.61 | 1.01 | 21 | 4.58 | -0.94 |
| prueba | ingenuo | Atlantico | nivel | 147.00 | 6.16 | 5.54 | -0.55 | 5.63 | 1.01 | 21 | 4.37 | -0.68 |
| prueba | ingenuo | Atlantico | forma | 147.00 | 6.06 | 5.54 | -0.55 | 5.63 | 0.77 | 21 | 4.37 | -0.68 |
| prueba | archivo | Antioquia | base | 119.00 | 2.86 | 2.54 | -1.44 | 2.31 | 2.34 | 17 | 1.82 | -1.50 |
| prueba | archivo | Antioquia | nivel | 119.00 | 2.64 | 2.28 | -1.24 | 2.14 | 2.34 | 17 | 1.58 | -1.29 |
| prueba | archivo | Antioquia | forma | 119.00 | 2.52 | 2.28 | -1.24 | 2.38 | 2.08 | 17 | 1.58 | -1.29 |
| total | modelo_k1 | Antioquia | base | 364.00 | 2.71 | 2.39 | -0.89 | 2.44 | 2.01 | 52 | 1.86 | -0.91 |
| total | modelo_k1 | Antioquia | nivel | 364.00 | 2.68 | 2.38 | -0.69 | 2.40 | 2.01 | 52 | 1.70 | -0.71 |
| total | modelo_k1 | Antioquia | forma | 364.00 | 2.61 | 2.38 | -0.69 | 2.30 | 1.76 | 52 | 1.70 | -0.71 |
| total | modelo_k1 | Atlantico | base | 364.00 | 5.46 | 4.72 | -0.14 | 4.85 | 0.84 | 52 | 3.44 | -0.29 |
| total | modelo_k1 | Atlantico | nivel | 364.00 | 5.40 | 4.60 | -0.10 | 4.79 | 0.84 | 52 | 3.23 | -0.24 |
| total | modelo_k1 | Atlantico | forma | 364.00 | 5.15 | 4.60 | -0.10 | 4.65 | 0.76 | 52 | 3.23 | -0.24 |
| total | modelo_k4 | Antioquia | base | 364.00 | 2.80 | 2.48 | -1.01 | 2.51 | 2.33 | 52 | 1.93 | -1.02 |
| total | modelo_k4 | Antioquia | nivel | 364.00 | 2.73 | 2.43 | -0.78 | 2.42 | 2.33 | 52 | 1.74 | -0.80 |
| total | modelo_k4 | Antioquia | forma | 364.00 | 2.65 | 2.43 | -0.78 | 2.37 | 1.76 | 52 | 1.74 | -0.80 |
| total | modelo_k4 | Atlantico | base | 364.00 | 5.54 | 4.67 | -0.06 | 4.99 | 0.84 | 52 | 3.39 | -0.20 |
| total | modelo_k4 | Atlantico | nivel | 364.00 | 5.48 | 4.58 | -0.03 | 4.92 | 0.84 | 52 | 3.15 | -0.17 |
| total | modelo_k4 | Atlantico | forma | 364.00 | 5.13 | 4.58 | -0.03 | 4.61 | 0.76 | 52 | 3.15 | -0.17 |
| total | ingenuo | Antioquia | base | 364.00 | 4.05 | 3.92 | -0.11 | 3.55 | 1.85 | 52 | 2.69 | -0.25 |
| total | ingenuo | Antioquia | nivel | 364.00 | 3.76 | 3.59 | -0.11 | 3.21 | 1.85 | 52 | 2.35 | -0.22 |
| total | ingenuo | Antioquia | forma | 364.00 | 3.71 | 3.59 | -0.11 | 3.47 | 1.76 | 52 | 2.35 | -0.22 |
| total | ingenuo | Atlantico | base | 364.00 | 6.27 | 5.53 | -0.21 | 5.49 | 1.02 | 52 | 3.58 | -0.37 |
| total | ingenuo | Atlantico | nivel | 364.00 | 6.14 | 5.35 | -0.08 | 5.39 | 1.02 | 52 | 3.31 | -0.23 |
| total | ingenuo | Atlantico | forma | 364.00 | 5.90 | 5.35 | -0.08 | 5.39 | 0.76 | 52 | 3.31 | -0.23 |
| total | archivo | Antioquia | base | 147.00 | 3.03 | 2.75 | -0.73 | 2.56 | 2.31 | 21 | 2.11 | -0.79 |
| total | archivo | Antioquia | nivel | 147.00 | 2.87 | 2.55 | -0.56 | 2.45 | 2.31 | 21 | 1.90 | -0.60 |
| total | archivo | Antioquia | forma | 147.00 | 2.79 | 2.55 | -0.56 | 2.60 | 2.11 | 21 | 1.90 | -0.60 |

### Calibración (referencia)

| base | ucp | paso | n_dias | wape_pct | mape_energia_dia_pct | sesgo_energia_dia_pct | err_pico_pct | dif_hora_pico_h | n_semanas | mape_energia_semana_pct | sesgo_energia_semana_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| modelo_k1 | Antioquia | base | 217.00 | 2.91 | 2.59 | -1.15 | 2.67 | 1.88 | 31 | 2.09 | -1.17 |
| modelo_k1 | Antioquia | nivel | 217.00 | 2.86 | 2.58 | -0.79 | 2.56 | 1.88 | 31 | 1.93 | -0.81 |
| modelo_k1 | Antioquia | forma | 217.00 | 2.80 | 2.58 | -0.79 | 2.46 | 1.54 | 31 | 1.93 | -0.81 |
| modelo_k1 | Atlantico | base | 217.00 | 5.19 | 4.40 | 0.01 | 4.56 | 0.81 | 31 | 2.68 | -0.18 |
| modelo_k1 | Atlantico | nivel | 217.00 | 5.15 | 4.31 | 0.01 | 4.54 | 0.81 | 31 | 2.44 | -0.16 |
| modelo_k1 | Atlantico | forma | 217.00 | 4.93 | 4.31 | 0.01 | 4.39 | 0.76 | 31 | 2.44 | -0.16 |
| modelo_k4 | Antioquia | base | 217.00 | 2.96 | 2.64 | -1.21 | 2.72 | 2.10 | 31 | 2.15 | -1.22 |
| modelo_k4 | Antioquia | nivel | 217.00 | 2.88 | 2.60 | -0.83 | 2.58 | 2.10 | 31 | 1.95 | -0.85 |
| modelo_k4 | Antioquia | forma | 217.00 | 2.82 | 2.60 | -0.83 | 2.47 | 1.54 | 31 | 1.95 | -0.85 |
| modelo_k4 | Atlantico | base | 217.00 | 5.30 | 4.34 | 0.23 | 4.75 | 0.84 | 31 | 2.62 | 0.05 |
| modelo_k4 | Atlantico | nivel | 217.00 | 5.26 | 4.27 | 0.20 | 4.70 | 0.84 | 31 | 2.33 | 0.03 |
| modelo_k4 | Atlantico | forma | 217.00 | 4.89 | 4.27 | 0.20 | 4.31 | 0.76 | 31 | 2.33 | 0.03 |
| ingenuo | Antioquia | base | 217.00 | 4.15 | 4.05 | -0.08 | 3.59 | 1.53 | 31 | 2.82 | -0.22 |
| ingenuo | Antioquia | nivel | 217.00 | 3.84 | 3.69 | 0.01 | 3.18 | 1.53 | 31 | 2.45 | -0.10 |
| ingenuo | Antioquia | forma | 217.00 | 3.81 | 3.69 | 0.01 | 3.47 | 1.54 | 31 | 2.45 | -0.10 |
| ingenuo | Atlantico | base | 217.00 | 6.31 | 5.45 | 0.19 | 5.41 | 1.02 | 31 | 2.91 | 0.02 |
| ingenuo | Atlantico | nivel | 217.00 | 6.13 | 5.22 | 0.24 | 5.23 | 1.02 | 31 | 2.60 | 0.08 |
| ingenuo | Atlantico | forma | 217.00 | 5.78 | 5.22 | 0.24 | 5.22 | 0.76 | 31 | 2.60 | 0.08 |
| archivo | Antioquia | base | 28.00 | 3.82 | 3.65 | 2.25 | 3.60 | 2.18 | 4 | 3.30 | 2.22 |
| archivo | Antioquia | nivel | 28.00 | 3.88 | 3.71 | 2.36 | 3.77 | 2.18 | 4 | 3.24 | 2.33 |
| archivo | Antioquia | forma | 28.00 | 4.01 | 3.71 | 2.36 | 3.56 | 2.21 | 4 | 3.24 | 2.33 |

### Por tipo de día — modelo_k1, default (prueba y total)

| particion | ucp | tipo_dia | paso | n_dias | wape_pct | mape_energia_dia_pct | sesgo_energia_dia_pct | err_pico_pct | dif_hora_pico_h |
|---|---|---|---|---|---|---|---|---|---|
| prueba | Antioquia | laboral | base | 98.00 | 2.29 | 1.93 | -0.49 | 1.99 | 2.70 |
| prueba | Antioquia | sabado | base | 21.00 | 2.62 | 2.44 | -0.39 | 2.15 | 2.67 |
| prueba | Antioquia | domingo | base | 21.00 | 2.56 | 2.15 | -0.84 | 2.32 | 0.05 |
| prueba | Antioquia | laboral | nivel | 98.00 | 2.31 | 1.94 | -0.51 | 2.10 | 2.70 |
| prueba | Antioquia | sabado | nivel | 21.00 | 2.41 | 2.26 | -0.72 | 1.85 | 2.67 |
| prueba | Antioquia | domingo | nivel | 21.00 | 2.66 | 2.24 | -0.76 | 2.39 | 0.05 |
| prueba | Antioquia | laboral | forma | 98.00 | 2.17 | 1.94 | -0.51 | 2.02 | 2.53 |
| prueba | Antioquia | sabado | forma | 21.00 | 2.41 | 2.26 | -0.72 | 1.72 | 2.76 |
| prueba | Antioquia | domingo | forma | 21.00 | 2.68 | 2.24 | -0.76 | 2.46 | 0.00 |
| prueba | Antioquia | festivo | base | 7.00 | 4.03 | 3.19 | 0.01 | 2.85 | 0.00 |
| prueba | Antioquia | festivo | nivel | 7.00 | 3.97 | 3.18 | 0.16 | 2.95 | 0.00 |
| prueba | Antioquia | festivo | forma | 7.00 | 3.67 | 3.18 | 0.16 | 2.81 | 0.00 |
| prueba | Atlantico | festivo | base | 8.00 | 6.06 | 4.76 | -1.94 | 6.39 | 0.62 |
| prueba | Atlantico | laboral | base | 97.00 | 5.83 | 5.18 | -0.34 | 5.26 | 0.75 |
| prueba | Atlantico | sabado | base | 21.00 | 5.56 | 5.07 | 0.42 | 4.29 | 1.62 |
| prueba | Atlantico | domingo | base | 21.00 | 6.23 | 5.55 | -0.59 | 5.96 | 0.86 |
| prueba | Atlantico | festivo | nivel | 8.00 | 5.48 | 3.83 | -0.78 | 5.71 | 0.62 |
| prueba | Atlantico | laboral | nivel | 97.00 | 5.85 | 5.13 | -0.22 | 5.30 | 0.75 |
| prueba | Atlantico | sabado | nivel | 21.00 | 5.21 | 4.82 | 0.30 | 3.87 | 1.62 |
| prueba | Atlantico | domingo | nivel | 21.00 | 6.05 | 5.37 | -0.86 | 5.61 | 0.86 |
| prueba | Atlantico | festivo | forma | 8.00 | 4.79 | 3.83 | -0.78 | 4.27 | 0.25 |
| prueba | Atlantico | laboral | forma | 97.00 | 5.56 | 5.13 | -0.22 | 5.30 | 0.64 |
| prueba | Atlantico | sabado | forma | 21.00 | 5.21 | 4.82 | 0.30 | 4.36 | 1.57 |
| prueba | Atlantico | domingo | forma | 21.00 | 5.50 | 5.37 | -0.86 | 4.83 | 0.76 |
| total | Antioquia | festivo | base | 18.00 | 3.60 | 3.00 | -0.20 | 2.41 | 0.22 |
| total | Antioquia | laboral | base | 243.00 | 2.66 | 2.30 | -0.90 | 2.40 | 2.36 |
| total | Antioquia | sabado | base | 52.00 | 2.60 | 2.39 | -0.67 | 2.36 | 2.58 |
| total | Antioquia | domingo | base | 51.00 | 2.86 | 2.60 | -1.27 | 2.71 | 0.37 |
| total | Antioquia | festivo | nivel | 18.00 | 3.88 | 3.41 | -0.33 | 2.86 | 0.22 |
| total | Antioquia | laboral | nivel | 243.00 | 2.61 | 2.27 | -0.67 | 2.35 | 2.36 |
| total | Antioquia | sabado | nivel | 52.00 | 2.42 | 2.21 | -0.59 | 2.08 | 2.58 |
| total | Antioquia | domingo | nivel | 51.00 | 2.98 | 2.74 | -1.01 | 2.78 | 0.37 |
| total | Antioquia | festivo | forma | 18.00 | 3.76 | 3.41 | -0.33 | 2.67 | 0.00 |
| total | Antioquia | laboral | forma | 243.00 | 2.52 | 2.27 | -0.67 | 2.20 | 2.13 |
| total | Antioquia | sabado | forma | 52.00 | 2.39 | 2.21 | -0.59 | 2.07 | 2.19 |
| total | Antioquia | domingo | forma | 51.00 | 2.97 | 2.74 | -1.01 | 2.88 | 0.16 |
| total | Atlantico | laboral | base | 244.00 | 5.16 | 4.37 | -0.84 | 4.62 | 0.75 |
| total | Atlantico | sabado | base | 52.00 | 5.78 | 5.12 | 2.30 | 4.85 | 1.46 |
| total | Atlantico | domingo | base | 51.00 | 6.31 | 5.65 | 0.77 | 5.70 | 0.76 |
| total | Atlantico | laboral | nivel | 244.00 | 5.22 | 4.42 | -0.72 | 4.66 | 0.75 |
| total | Atlantico | sabado | nivel | 52.00 | 5.42 | 4.66 | 1.89 | 4.48 | 1.46 |
| total | Atlantico | domingo | nivel | 51.00 | 5.98 | 5.18 | 0.64 | 5.47 | 0.76 |
| total | Atlantico | laboral | forma | 244.00 | 4.98 | 4.42 | -0.72 | 4.56 | 0.70 |
| total | Atlantico | sabado | forma | 52.00 | 5.21 | 4.66 | 1.89 | 4.65 | 1.15 |
| total | Atlantico | domingo | forma | 51.00 | 5.63 | 5.18 | 0.64 | 5.04 | 0.76 |
| total | Atlantico | festivo | base | 17.00 | 6.63 | 5.68 | -0.22 | 5.67 | 0.41 |
| total | Atlantico | festivo | nivel | 17.00 | 6.45 | 5.36 | 0.49 | 5.59 | 0.41 |
| total | Atlantico | festivo | forma | 17.00 | 6.08 | 5.36 | 0.49 | 4.80 | 0.53 |

## 3. Barrido α × fuente (WAPE % del paso final, modelo_k1)

### Antioquia — calibracion

| alpha | promedio_n2 | promedio_n3 | promedio_n4 | promedio_n6 | promedio_n8 | reciente | referencia |
|---|---|---|---|---|---|---|---|
| 0.00 | 2.84 | 2.84 | 2.85 | 2.84 | 2.84 | 2.86 | 2.91 |
| 0.25 | 2.79 | 2.80 | 2.80 | 2.80 | 2.80 | 2.81 | 2.86 |
| 0.50 | 3.02 | 3.03 | 3.02 | 3.02 | 3.02 | 3.03 | 3.07 |
| 0.75 | 3.49 | 3.50 | 3.49 | 3.48 | 3.48 | 3.50 | 3.52 |
| 1.00 | 4.12 | 4.12 | 4.13 | 4.11 | 4.11 | 4.13 | 4.14 |

### Antioquia — prueba

| alpha | promedio_n2 | promedio_n3 | promedio_n4 | promedio_n6 | promedio_n8 | reciente | referencia |
|---|---|---|---|---|---|---|---|
| 0.00 | 2.35 | 2.33 | 2.32 | 2.34 | 2.35 | 2.41 | 2.56 |
| 0.25 | 2.35 | 2.33 | 2.33 | 2.35 | 2.36 | 2.40 | 2.55 |
| 0.50 | 2.57 | 2.55 | 2.54 | 2.56 | 2.57 | 2.63 | 2.74 |
| 0.75 | 2.94 | 2.91 | 2.91 | 2.92 | 2.93 | 3.00 | 3.08 |
| 1.00 | 3.41 | 3.39 | 3.38 | 3.39 | 3.40 | 3.47 | 3.54 |

### Atlantico — calibracion

| alpha | promedio_n2 | promedio_n3 | promedio_n4 | promedio_n6 | promedio_n8 | reciente | referencia |
|---|---|---|---|---|---|---|---|
| 0.00 | 5.09 | 5.03 | 5.00 | 4.96 | 4.95 | 5.33 | 5.43 |
| 0.25 | 5.02 | 4.94 | 4.93 | 4.90 | 4.90 | 5.23 | 5.35 |
| 0.50 | 5.43 | 5.37 | 5.34 | 5.32 | 5.32 | 5.59 | 5.74 |
| 0.75 | 6.14 | 6.08 | 6.04 | 6.03 | 6.02 | 6.28 | 6.40 |
| 1.00 | 7.02 | 6.96 | 6.93 | 6.92 | 6.92 | 7.15 | 7.23 |

### Atlantico — prueba

| alpha | promedio_n2 | promedio_n3 | promedio_n4 | promedio_n6 | promedio_n8 | reciente | referencia |
|---|---|---|---|---|---|---|---|
| 0.00 | 5.61 | 5.59 | 5.58 | 5.57 | 5.60 | 5.77 | 6.21 |
| 0.25 | 5.49 | 5.48 | 5.46 | 5.46 | 5.48 | 5.63 | 6.10 |
| 0.50 | 5.65 | 5.62 | 5.62 | 5.60 | 5.62 | 5.73 | 6.22 |
| 0.75 | 6.05 | 6.02 | 6.03 | 6.01 | 6.03 | 6.11 | 6.61 |
| 1.00 | 6.64 | 6.63 | 6.63 | 6.63 | 6.64 | 6.71 | 7.19 |

### Paso nivel por α (energía, no depende de la forma) — prueba

| ucp | alpha | n_dias | wape_pct | mape_energia_dia_pct | sesgo_energia_dia_pct | err_pico_pct | dif_hora_pico_h | n_semanas | mape_energia_semana_pct | sesgo_energia_semana_pct |
|---|---|---|---|---|---|---|---|---|---|---|
| Antioquia | 0.00 | 147.00 | 2.44 | 2.10 | -0.50 | 2.10 | 2.19 | 21 | 1.52 | -0.52 |
| Antioquia | 0.25 | 147.00 | 2.44 | 2.09 | -0.55 | 2.15 | 2.19 | 21 | 1.37 | -0.57 |
| Antioquia | 0.50 | 147.00 | 2.64 | 2.33 | -0.59 | 2.39 | 2.19 | 21 | 1.39 | -0.62 |
| Antioquia | 0.75 | 147.00 | 2.98 | 2.69 | -0.64 | 2.77 | 2.19 | 21 | 1.73 | -0.67 |
| Antioquia | 1.00 | 147.00 | 3.44 | 3.22 | -0.69 | 3.26 | 2.19 | 21 | 2.09 | -0.72 |
| Atlantico | 0.00 | 147.00 | 5.86 | 5.19 | -0.36 | 5.28 | 0.88 | 21 | 4.57 | -0.45 |
| Atlantico | 0.25 | 147.00 | 5.77 | 5.05 | -0.26 | 5.16 | 0.88 | 21 | 4.41 | -0.36 |
| Atlantico | 0.50 | 147.00 | 5.94 | 5.16 | -0.17 | 5.39 | 0.88 | 21 | 4.34 | -0.27 |
| Atlantico | 0.75 | 147.00 | 6.33 | 5.59 | -0.08 | 5.80 | 0.88 | 21 | 4.34 | -0.18 |
| Atlantico | 1.00 | 147.00 | 6.93 | 6.22 | 0.02 | 6.39 | 0.88 | 21 | 4.47 | -0.09 |

## 4. Selección en calibración y resultado en prueba

Candidatas elegidas SOLO con calibración (min WAPE del paso final): `mejor_<UCP>` por UCP, `mejor_conjunta` = min del promedio de WAPE entre UCP (única candidata posible a default global, porque los DEFAULTS son globales), `apt_*` = la misma config + `alpha_por_tipo` elegido en calibración (corrida REAL del pipeline con ese parámetro, no reconstruida).

| candidata | config |
|---|---|
| default | a0.25|promedio_n4 |
| mejor_conjunta | a0.25|promedio_n6 |
| mejor_Antioquia | a0.25|promedio_n2 |
| mejor_Atlantico | a0.25|promedio_n8 |
| apt_Antioquia | a0.25|promedio_n2|apt(domingo=0,festivo=0) |
| apt_Atlantico | a0.25|promedio_n8|apt(laboral=0) |

| ucp | particion | candidata | config | wape_pct | dif_wape_vs_default_pp | mape_sem_pct | dif_mape_sem_vs_default_pp |
|---|---|---|---|---|---|---|---|
| Antioquia | calibracion | default | a0.25|promedio_n4 | 2.804 | 0.000 | 1.927 | 0.000 |
| Antioquia | calibracion | mejor_conjunta | a0.25|promedio_n6 | 2.799 | -0.005 | 1.927 | 0.000 |
| Antioquia | calibracion | mejor_Antioquia | a0.25|promedio_n2 | 2.795 | -0.009 | 1.927 | 0.000 |
| Antioquia | calibracion | mejor_Atlantico | a0.25|promedio_n8 | 2.801 | -0.003 | 1.927 | 0.000 |
| Antioquia | calibracion | apt_Antioquia | a0.25|promedio_n2|apt(domingo=0,festivo=0) | 2.758 | -0.045 | 1.935 | 0.008 |
| Antioquia | calibracion | apt_Atlantico | a0.25|promedio_n8|apt(laboral=0) | 2.860 | 0.056 | 2.041 | 0.114 |
| Antioquia | prueba | default | a0.25|promedio_n4 | 2.331 | 0.000 | 1.366 | 0.000 |
| Antioquia | prueba | mejor_conjunta | a0.25|promedio_n6 | 2.347 | 0.016 | 1.366 | 0.000 |
| Antioquia | prueba | mejor_Antioquia | a0.25|promedio_n2 | 2.348 | 0.017 | 1.366 | 0.000 |
| Antioquia | prueba | mejor_Atlantico | a0.25|promedio_n8 | 2.355 | 0.025 | 1.366 | 0.000 |
| Antioquia | prueba | apt_Antioquia | a0.25|promedio_n2|apt(domingo=0,festivo=0) | 2.338 | 0.007 | 1.384 | 0.017 |
| Antioquia | prueba | apt_Atlantico | a0.25|promedio_n8|apt(laboral=0) | 2.330 | -0.000 | 1.445 | 0.079 |
| Antioquia | total | default | a0.25|promedio_n4 | 2.607 | 0.000 | 1.701 | 0.000 |
| Antioquia | total | mejor_conjunta | a0.25|promedio_n6 | 2.611 | 0.004 | 1.701 | 0.000 |
| Antioquia | total | mejor_Antioquia | a0.25|promedio_n2 | 2.609 | 0.002 | 1.701 | 0.000 |
| Antioquia | total | mejor_Atlantico | a0.25|promedio_n8 | 2.616 | 0.009 | 1.701 | 0.000 |
| Antioquia | total | apt_Antioquia | a0.25|promedio_n2|apt(domingo=0,festivo=0) | 2.583 | -0.023 | 1.712 | 0.012 |
| Antioquia | total | apt_Atlantico | a0.25|promedio_n8|apt(laboral=0) | 2.640 | 0.033 | 1.800 | 0.100 |
| Atlantico | calibracion | default | a0.25|promedio_n4 | 4.928 | 0.000 | 2.438 | 0.000 |
| Atlantico | calibracion | mejor_conjunta | a0.25|promedio_n6 | 4.901 | -0.027 | 2.438 | 0.000 |
| Atlantico | calibracion | mejor_Antioquia | a0.25|promedio_n2 | 5.018 | 0.090 | 2.438 | 0.000 |
| Atlantico | calibracion | mejor_Atlantico | a0.25|promedio_n8 | 4.901 | -0.028 | 2.438 | -0.000 |
| Atlantico | calibracion | apt_Antioquia | a0.25|promedio_n2|apt(domingo=0,festivo=0) | 5.081 | 0.153 | 2.458 | 0.020 |
| Atlantico | calibracion | apt_Atlantico | a0.25|promedio_n8|apt(laboral=0) | 4.836 | -0.092 | 2.533 | 0.095 |
| Atlantico | prueba | default | a0.25|promedio_n4 | 5.464 | 0.000 | 4.405 | 0.000 |
| Atlantico | prueba | mejor_conjunta | a0.25|promedio_n6 | 5.457 | -0.007 | 4.405 | -0.000 |
| Atlantico | prueba | mejor_Antioquia | a0.25|promedio_n2 | 5.495 | 0.031 | 4.405 | 0.000 |
| Atlantico | prueba | mejor_Atlantico | a0.25|promedio_n8 | 5.477 | 0.013 | 4.405 | 0.000 |
| Atlantico | prueba | apt_Antioquia | a0.25|promedio_n2|apt(domingo=0,festivo=0) | 5.579 | 0.115 | 4.461 | 0.056 |
| Atlantico | prueba | apt_Atlantico | a0.25|promedio_n8|apt(laboral=0) | 5.478 | 0.014 | 4.474 | 0.069 |
| Atlantico | total | default | a0.25|promedio_n4 | 5.147 | 0.000 | 3.233 | 0.000 |
| Atlantico | total | mejor_conjunta | a0.25|promedio_n6 | 5.128 | -0.019 | 3.233 | 0.000 |
| Atlantico | total | mejor_Antioquia | a0.25|promedio_n2 | 5.212 | 0.066 | 3.233 | 0.000 |
| Atlantico | total | mejor_Atlantico | a0.25|promedio_n8 | 5.136 | -0.011 | 3.233 | 0.000 |
| Atlantico | total | apt_Antioquia | a0.25|promedio_n2|apt(domingo=0,festivo=0) | 5.284 | 0.137 | 3.267 | 0.034 |
| Atlantico | total | apt_Atlantico | a0.25|promedio_n8|apt(laboral=0) | 5.098 | -0.049 | 3.317 | 0.084 |

### Regla para cambiar DEFAULTS (en prueba)

Cumple si mejora el WAPE del final en TODAS las UCP y no empeora el MAPE semanal > 0.05 pp en ninguna. Solo `mejor_conjunta`/`apt_conjunta`/`apt_default` pueden convertirse en default global.

| candidata | cumple | mejora_wape_todas | mape_sem_ok | detalle (ΔWAPE, ΔMAPE sem) por UCP |
|---|---|---|---|---|
| mejor_conjunta | no | no | sí | {'Antioquia': (0.016, 0.0), 'Atlantico': (-0.007, -0.0)} |
| mejor_Antioquia | no | no | sí | {'Antioquia': (0.017, 0.0), 'Atlantico': (0.031, 0.0)} |
| mejor_Atlantico | no | no | sí | {'Antioquia': (0.025, 0.0), 'Atlantico': (0.013, 0.0)} |
| apt_Antioquia | no | no | no | {'Antioquia': (0.007, 0.017), 'Atlantico': (0.115, 0.056)} |
| apt_Atlantico | no | no | no | {'Antioquia': (-0.0, 0.079), 'Atlantico': (0.014, 0.069)} |

## 5. `alpha_por_tipo` (elegido en calibración)

Para cada candidata se fija su fuente/n y se elige el mejor α por tipo de día filtrando días del tipo (promedio de WAPE entre las UCP del grupo). Festivo: si alguna UCP tiene < 10 días festivos en calibración, NO se fija α propio (usa el global). Claves iguales al α global se omiten.

### default (sobre `a0.25|promedio_n4`, UCP Antioquia, Atlantico) → `{}`

| ucp | tipo_dia | n_dias | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|---|
| Antioquia | domingo | 30 | 3.07 | 3.19 | 3.67 | 4.50 | 5.45 |
| Antioquia | festivo | 11 | 3.24 | 3.83 | 4.60 | 5.50 | 6.42 |
| Antioquia | laboral | 145 | 2.85 | 2.76 | 2.92 | 3.32 | 3.92 |
| Antioquia | sabado | 31 | 2.53 | 2.37 | 2.52 | 2.85 | 3.32 |
| Atlantico | domingo | 30 | 6.27 | 5.73 | 6.39 | 7.41 | 8.53 |
| Atlantico | festivo | 9 | 7.17 | 7.26 | 7.55 | 8.14 | 8.92 |
| Atlantico | laboral | 147 | 4.53 | 4.60 | 5.00 | 5.63 | 6.42 |
| Atlantico | sabado | 31 | 5.61 | 5.21 | 5.47 | 6.27 | 7.49 |

### conjunta (sobre `a0.25|promedio_n6`, UCP Antioquia, Atlantico) → `{}`

| ucp | tipo_dia | n_dias | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|---|
| Antioquia | domingo | 30 | 3.06 | 3.16 | 3.66 | 4.50 | 5.45 |
| Antioquia | festivo | 11 | 3.24 | 3.83 | 4.60 | 5.50 | 6.42 |
| Antioquia | laboral | 145 | 2.84 | 2.76 | 2.91 | 3.30 | 3.90 |
| Antioquia | sabado | 31 | 2.53 | 2.37 | 2.53 | 2.86 | 3.31 |
| Atlantico | domingo | 30 | 6.28 | 5.79 | 6.46 | 7.45 | 8.56 |
| Atlantico | festivo | 9 | 7.17 | 7.26 | 7.55 | 8.14 | 8.92 |
| Atlantico | laboral | 147 | 4.47 | 4.55 | 4.96 | 5.60 | 6.39 |
| Atlantico | sabado | 31 | 5.60 | 5.20 | 5.44 | 6.26 | 7.47 |

### Antioquia (sobre `a0.25|promedio_n2`, UCP Antioquia) → `{"domingo": 0.0, "festivo": 0.0}`

| ucp | tipo_dia | n_dias | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|---|
| Antioquia | domingo | 30 | 3.08 | 3.17 | 3.66 | 4.50 | 5.45 |
| Antioquia | festivo | 11 | 3.24 | 3.83 | 4.60 | 5.50 | 6.42 |
| Antioquia | laboral | 145 | 2.84 | 2.76 | 2.92 | 3.33 | 3.91 |
| Antioquia | sabado | 31 | 2.50 | 2.34 | 2.51 | 2.84 | 3.28 |

### Atlantico (sobre `a0.25|promedio_n8`, UCP Atlantico) → `{"laboral": 0.0}`

| ucp | tipo_dia | n_dias | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|---|
| Atlantico | domingo | 30 | 6.37 | 5.92 | 6.55 | 7.52 | 8.66 |
| Atlantico | festivo | 9 | 7.17 | 7.26 | 7.55 | 8.14 | 8.92 |
| Atlantico | laboral | 147 | 4.44 | 4.54 | 4.96 | 5.60 | 6.39 |
| Atlantico | sabado | 31 | 5.54 | 5.14 | 5.40 | 6.19 | 7.43 |

### Efecto por tipo de día en prueba (WAPE del final)

| ucp | tipo_dia | config | n_dias | wape_pct | mape_energia_dia_pct | sesgo_energia_dia_pct |
|---|---|---|---|---|---|---|
| Antioquia | laboral | a0.25|promedio_n2 | 98.000 | 2.182 | 1.943 | -0.514 |
| Antioquia | sabado | a0.25|promedio_n2 | 21.000 | 2.421 | 2.262 | -0.721 |
| Antioquia | domingo | a0.25|promedio_n2 | 21.000 | 2.760 | 2.242 | -0.756 |
| Antioquia | laboral | a0.25|promedio_n4 | 98.000 | 2.172 | 1.943 | -0.514 |
| Antioquia | sabado | a0.25|promedio_n4 | 21.000 | 2.414 | 2.262 | -0.721 |
| Antioquia | domingo | a0.25|promedio_n4 | 21.000 | 2.683 | 2.242 | -0.756 |
| Antioquia | laboral | a0.25|promedio_n6 | 98.000 | 2.197 | 1.943 | -0.514 |
| Antioquia | sabado | a0.25|promedio_n6 | 21.000 | 2.416 | 2.262 | -0.721 |
| Antioquia | domingo | a0.25|promedio_n6 | 21.000 | 2.668 | 2.242 | -0.756 |
| Antioquia | laboral | a0.25|promedio_n8 | 98.000 | 2.208 | 1.943 | -0.514 |
| Antioquia | sabado | a0.25|promedio_n8 | 21.000 | 2.435 | 2.262 | -0.721 |
| Antioquia | domingo | a0.25|promedio_n8 | 21.000 | 2.658 | 2.242 | -0.756 |
| Antioquia | festivo | a0.25|promedio_n2 | 7.000 | 3.671 | 3.181 | 0.156 |
| Antioquia | festivo | a0.25|promedio_n4 | 7.000 | 3.671 | 3.181 | 0.156 |
| Antioquia | festivo | a0.25|promedio_n6 | 7.000 | 3.671 | 3.181 | 0.156 |
| Antioquia | festivo | a0.25|promedio_n8 | 7.000 | 3.671 | 3.181 | 0.156 |
| Atlantico | festivo | a0.25|promedio_n2 | 8.000 | 4.787 | 3.832 | -0.782 |
| Atlantico | laboral | a0.25|promedio_n2 | 97.000 | 5.592 | 5.127 | -0.215 |
| Atlantico | sabado | a0.25|promedio_n2 | 21.000 | 5.203 | 4.817 | 0.302 |
| Atlantico | domingo | a0.25|promedio_n2 | 21.000 | 5.585 | 5.368 | -0.858 |
| Atlantico | festivo | a0.25|promedio_n4 | 8.000 | 4.787 | 3.832 | -0.782 |
| Atlantico | laboral | a0.25|promedio_n4 | 97.000 | 5.561 | 5.127 | -0.215 |
| Atlantico | sabado | a0.25|promedio_n4 | 21.000 | 5.207 | 4.817 | 0.302 |
| Atlantico | domingo | a0.25|promedio_n4 | 21.000 | 5.503 | 5.368 | -0.858 |
| Atlantico | festivo | a0.25|promedio_n6 | 8.000 | 4.787 | 3.832 | -0.782 |
| Atlantico | laboral | a0.25|promedio_n6 | 97.000 | 5.555 | 5.127 | -0.215 |
| Atlantico | sabado | a0.25|promedio_n6 | 21.000 | 5.203 | 4.817 | 0.302 |
| Atlantico | domingo | a0.25|promedio_n6 | 21.000 | 5.487 | 5.368 | -0.858 |
| Atlantico | festivo | a0.25|promedio_n8 | 8.000 | 4.787 | 3.832 | -0.782 |
| Atlantico | laboral | a0.25|promedio_n8 | 97.000 | 5.586 | 5.127 | -0.215 |
| Atlantico | sabado | a0.25|promedio_n8 | 21.000 | 5.146 | 4.817 | 0.302 |
| Atlantico | domingo | a0.25|promedio_n8 | 21.000 | 5.542 | 5.368 | -0.858 |
| Antioquia | laboral | a0.25|promedio_n2|apt(domingo=0,festivo=0) | 98.000 | 2.182 | 1.943 | -0.514 |
| Antioquia | sabado | a0.25|promedio_n2|apt(domingo=0,festivo=0) | 21.000 | 2.421 | 2.262 | -0.721 |
| Antioquia | domingo | a0.25|promedio_n2|apt(domingo=0,festivo=0) | 21.000 | 2.661 | 2.150 | -0.844 |
| Antioquia | laboral | a0.25|promedio_n8|apt(laboral=0) | 98.000 | 2.172 | 1.933 | -0.486 |
| Antioquia | sabado | a0.25|promedio_n8|apt(laboral=0) | 21.000 | 2.435 | 2.262 | -0.721 |
| Antioquia | domingo | a0.25|promedio_n8|apt(laboral=0) | 21.000 | 2.658 | 2.242 | -0.756 |
| Antioquia | festivo | a0.25|promedio_n2|apt(domingo=0,festivo=0) | 7.000 | 3.726 | 3.187 | 0.013 |
| Antioquia | festivo | a0.25|promedio_n8|apt(laboral=0) | 7.000 | 3.671 | 3.181 | 0.156 |
| Atlantico | festivo | a0.25|promedio_n2|apt(domingo=0,festivo=0) | 8.000 | 5.632 | 4.760 | -1.936 |
| Atlantico | laboral | a0.25|promedio_n2|apt(domingo=0,festivo=0) | 97.000 | 5.592 | 5.127 | -0.215 |
| Atlantico | sabado | a0.25|promedio_n2|apt(domingo=0,festivo=0) | 21.000 | 5.203 | 4.817 | 0.302 |
| Atlantico | domingo | a0.25|promedio_n2|apt(domingo=0,festivo=0) | 21.000 | 5.893 | 5.549 | -0.589 |
| Atlantico | festivo | a0.25|promedio_n8|apt(laboral=0) | 8.000 | 4.787 | 3.832 | -0.782 |
| Atlantico | laboral | a0.25|promedio_n8|apt(laboral=0) | 97.000 | 5.587 | 5.181 | -0.345 |
| Atlantico | sabado | a0.25|promedio_n8|apt(laboral=0) | 21.000 | 5.146 | 4.817 | 0.302 |
| Atlantico | domingo | a0.25|promedio_n8|apt(laboral=0) | 21.000 | 5.542 | 5.368 | -0.858 |

## 6. Banda del control (`banda_referencia_pp`, alerta TENDENCIA_REFERENCIA_FUERA_BANDA)

Con la config default de modelo_k1 (el diagnóstico `diferencia_tendencia_pp` no depende de la banda): alerta = |diferencia_tendencia_pp| > b; evento real = |error de energía semanal del final| > b. precision = P(evento | alerta), recall = P(alerta | evento), tasa_base = P(evento), lift = precision − tasa_base. Se elige en calibración (promedio entre UCP: max lift, desempate F1).

**Banda elegida en calibración: 1.0 pp** (default actual 2.0 pp).

| ucp | particion | banda_pp | semanas | alertas | eventos | TP | precision | recall | tasa_base | lift | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Antioquia | calibracion | 1.00 | 31 | 24 | 20 | 17 | 0.71 | 0.85 | 0.65 | 0.06 | 0.77 |
| Antioquia | calibracion | 1.50 | 31 | 22 | 15 | 13 | 0.59 | 0.87 | 0.48 | 0.11 | 0.70 |
| Antioquia | calibracion | 2.00 | 31 | 12 | 13 | 6 | 0.50 | 0.46 | 0.42 | 0.08 | 0.48 |
| Antioquia | calibracion | 2.50 | 31 | 7 | 8 | 1 | 0.14 | 0.12 | 0.26 | -0.12 | 0.13 |
| Antioquia | calibracion | 3.00 | 31 | 6 | 6 | 1 | 0.17 | 0.17 | 0.19 | -0.03 | 0.17 |
| Antioquia | calibracion | 4.00 | 31 | 5 | 4 | 0 | 0.00 | 0.00 | 0.13 | -0.13 | 0.00 |
| Antioquia | prueba | 1.00 | 21 | 14 | 9 | 6 | 0.43 | 0.67 | 0.43 | 0.00 | 0.52 |
| Antioquia | prueba | 1.50 | 21 | 9 | 8 | 3 | 0.33 | 0.38 | 0.38 | -0.05 | 0.35 |
| Antioquia | prueba | 2.00 | 21 | 7 | 7 | 2 | 0.29 | 0.29 | 0.33 | -0.05 | 0.29 |
| Antioquia | prueba | 2.50 | 21 | 4 | 4 | 0 | 0.00 | 0.00 | 0.19 | -0.19 | 0.00 |
| Antioquia | prueba | 3.00 | 21 | 2 | 1 | 0 | 0.00 | 0.00 | 0.05 | -0.05 | 0.00 |
| Antioquia | prueba | 4.00 | 21 | 0 | 1 | 0 | — | 0.00 | 0.05 | — | 0.00 |
| Atlantico | calibracion | 1.00 | 31 | 18 | 25 | 15 | 0.83 | 0.60 | 0.81 | 0.03 | 0.70 |
| Atlantico | calibracion | 1.50 | 31 | 16 | 17 | 8 | 0.50 | 0.47 | 0.55 | -0.05 | 0.48 |
| Atlantico | calibracion | 2.00 | 31 | 13 | 12 | 4 | 0.31 | 0.33 | 0.39 | -0.08 | 0.32 |
| Atlantico | calibracion | 2.50 | 31 | 10 | 11 | 3 | 0.30 | 0.27 | 0.35 | -0.05 | 0.29 |
| Atlantico | calibracion | 3.00 | 31 | 7 | 10 | 1 | 0.14 | 0.10 | 0.32 | -0.18 | 0.12 |
| Atlantico | calibracion | 4.00 | 31 | 5 | 6 | 0 | 0.00 | 0.00 | 0.19 | -0.19 | 0.00 |
| Atlantico | prueba | 1.00 | 21 | 15 | 19 | 13 | 0.87 | 0.68 | 0.90 | -0.04 | 0.76 |
| Atlantico | prueba | 1.50 | 21 | 12 | 18 | 10 | 0.83 | 0.56 | 0.86 | -0.02 | 0.67 |
| Atlantico | prueba | 2.00 | 21 | 11 | 17 | 9 | 0.82 | 0.53 | 0.81 | 0.01 | 0.64 |
| Atlantico | prueba | 2.50 | 21 | 8 | 15 | 6 | 0.75 | 0.40 | 0.71 | 0.04 | 0.52 |
| Atlantico | prueba | 3.00 | 21 | 5 | 13 | 3 | 0.60 | 0.23 | 0.62 | -0.02 | 0.33 |
| Atlantico | prueba | 4.00 | 21 | 2 | 9 | 1 | 0.50 | 0.11 | 0.43 | 0.07 | 0.18 |

## 7. Sensibilidad: reentreno semanal (k=1) vs cada 4 semanas (k=4)

| ucp | particion | config | base | wape_base | wape_nivel | wape_forma | mape_sem_forma |
|---|---|---|---|---|---|---|---|
| Antioquia | calibracion | a0.25|promedio_n4 | modelo_k1 | 2.907 | 2.857 | 2.804 | 1.927 |
| Antioquia | calibracion | a0.25|promedio_n4 | modelo_k4 | 2.961 | 2.876 | 2.820 | 1.947 |
| Antioquia | calibracion | a0.25|promedio_n6 | modelo_k1 | 2.907 | 2.857 | 2.799 | 1.927 |
| Antioquia | calibracion | a0.25|promedio_n6 | modelo_k4 | 2.961 | 2.876 | 2.816 | 1.947 |
| Antioquia | prueba | a0.25|promedio_n4 | modelo_k1 | 2.444 | 2.438 | 2.331 | 1.366 |
| Antioquia | prueba | a0.25|promedio_n4 | modelo_k4 | 2.582 | 2.534 | 2.423 | 1.442 |
| Antioquia | prueba | a0.25|promedio_n6 | modelo_k1 | 2.444 | 2.438 | 2.347 | 1.366 |
| Antioquia | prueba | a0.25|promedio_n6 | modelo_k4 | 2.582 | 2.534 | 2.438 | 1.442 |
| Antioquia | total | a0.25|promedio_n4 | modelo_k1 | 2.715 | 2.683 | 2.607 | 1.701 |
| Antioquia | total | a0.25|promedio_n4 | modelo_k4 | 2.804 | 2.734 | 2.655 | 1.743 |
| Antioquia | total | a0.25|promedio_n6 | modelo_k1 | 2.715 | 2.683 | 2.611 | 1.701 |
| Antioquia | total | a0.25|promedio_n6 | modelo_k4 | 2.804 | 2.734 | 2.659 | 1.743 |
| Atlantico | calibracion | a0.25|promedio_n4 | modelo_k1 | 5.185 | 5.146 | 4.928 | 2.438 |
| Atlantico | calibracion | a0.25|promedio_n4 | modelo_k4 | 5.302 | 5.255 | 4.888 | 2.333 |
| Atlantico | calibracion | a0.25|promedio_n6 | modelo_k1 | 5.185 | 5.146 | 4.901 | 2.438 |
| Atlantico | calibracion | a0.25|promedio_n6 | modelo_k4 | 5.302 | 5.255 | 4.870 | 2.333 |
| Atlantico | prueba | a0.25|promedio_n4 | modelo_k1 | 5.857 | 5.769 | 5.464 | 4.405 |
| Atlantico | prueba | a0.25|promedio_n4 | modelo_k4 | 5.885 | 5.802 | 5.472 | 4.367 |
| Atlantico | prueba | a0.25|promedio_n6 | modelo_k1 | 5.857 | 5.769 | 5.457 | 4.405 |
| Atlantico | prueba | a0.25|promedio_n6 | modelo_k4 | 5.885 | 5.802 | 5.461 | 4.367 |
| Atlantico | total | a0.25|promedio_n4 | modelo_k1 | 5.459 | 5.400 | 5.147 | 3.233 |
| Atlantico | total | a0.25|promedio_n4 | modelo_k4 | 5.539 | 5.478 | 5.126 | 3.155 |
| Atlantico | total | a0.25|promedio_n6 | modelo_k1 | 5.459 | 5.400 | 5.128 | 3.233 |
| Atlantico | total | a0.25|promedio_n6 | modelo_k4 | 5.539 | 5.478 | 5.111 | 3.155 |

## 8. R2: qué pasos pueden ser automáticos por segmento (modelo_k1, config default)

ΔWAPE = WAPE(paso) − WAPE(paso previo) (negativo = el paso ayuda). `automático` si ayuda en calibración Y en prueba, empeora en ≤ 35 % de las semanas de prueba y hay ≥ 10 días de prueba; `no aplicar` si empeora en ambas; `sugerido` en otro caso.

| ucp | tipo_dia | paso | dwape_cal | n_dias_cal | dwape_pru | n_dias_pru | pct_semanas_peores_pru | recomendacion |
|---|---|---|---|---|---|---|---|---|
| Antioquia | domingo | nivel | 0.130 | 30 | 0.102 | 21 | 42.857 | no aplicar |
| Antioquia | domingo | forma | -0.024 | 30 | 0.022 | 21 | 33.333 | sugerido |
| Antioquia | festivo | nivel | 0.521 | 11 | -0.061 | 7 | 50.000 | sugerido |
| Antioquia | festivo | forma | 0.012 | 11 | -0.299 | 7 | 16.667 | sugerido |
| Antioquia | laboral | nivel | -0.094 | 145 | 0.017 | 98 | 61.905 | sugerido |
| Antioquia | laboral | forma | -0.061 | 145 | -0.140 | 98 | 9.524 | automático |
| Antioquia | sabado | nivel | -0.162 | 31 | -0.204 | 21 | 42.857 | sugerido |
| Antioquia | sabado | forma | -0.061 | 31 | 0.000 | 21 | 33.333 | sugerido |
| Atlantico | domingo | nivel | -0.428 | 30 | -0.177 | 21 | 38.095 | sugerido |
| Atlantico | domingo | forma | -0.203 | 30 | -0.550 | 21 | 23.810 | automático |
| Atlantico | festivo | nivel | 0.175 | 9 | -0.572 | 8 | 0.000 | sugerido |
| Atlantico | festivo | forma | -0.060 | 9 | -0.697 | 8 | 0.000 | sugerido |
| Atlantico | laboral | nivel | 0.085 | 147 | 0.023 | 97 | 42.857 | no aplicar |
| Atlantico | laboral | forma | -0.201 | 147 | -0.291 | 97 | 14.286 | automático |
| Atlantico | sabado | nivel | -0.356 | 31 | -0.355 | 21 | 33.333 | automático |
| Atlantico | sabado | forma | -0.365 | 31 | 0.001 | 21 | 19.048 | sugerido |

### Con `a0.25|promedio_n6` (mejor conjunta)

| ucp | tipo_dia | paso | dwape_cal | n_dias_cal | dwape_pru | n_dias_pru | pct_semanas_peores_pru | recomendacion |
|---|---|---|---|---|---|---|---|---|
| Antioquia | domingo | nivel | 0.130 | 30 | 0.102 | 21 | 42.857 | no aplicar |
| Antioquia | domingo | forma | -0.049 | 30 | 0.007 | 21 | 28.571 | sugerido |
| Antioquia | festivo | nivel | 0.521 | 11 | -0.061 | 7 | 50.000 | sugerido |
| Antioquia | festivo | forma | 0.012 | 11 | -0.299 | 7 | 16.667 | sugerido |
| Antioquia | laboral | nivel | -0.094 | 145 | 0.017 | 98 | 61.905 | sugerido |
| Antioquia | laboral | forma | -0.065 | 145 | -0.114 | 98 | 14.286 | automático |
| Antioquia | sabado | nivel | -0.162 | 31 | -0.204 | 21 | 42.857 | sugerido |
| Antioquia | sabado | forma | -0.057 | 31 | 0.002 | 21 | 28.571 | sugerido |
| Atlantico | domingo | nivel | -0.428 | 30 | -0.177 | 21 | 38.095 | sugerido |
| Atlantico | domingo | forma | -0.141 | 30 | -0.565 | 21 | 14.286 | automático |
| Atlantico | festivo | nivel | 0.175 | 9 | -0.572 | 8 | 0.000 | sugerido |
| Atlantico | festivo | forma | -0.060 | 9 | -0.697 | 8 | 0.000 | sugerido |
| Atlantico | laboral | nivel | 0.085 | 147 | 0.023 | 97 | 42.857 | no aplicar |
| Atlantico | laboral | forma | -0.249 | 147 | -0.298 | 97 | 9.524 | automático |
| Atlantico | sabado | nivel | -0.356 | 31 | -0.355 | 21 | 33.333 | automático |
| Atlantico | sabado | forma | -0.374 | 31 | -0.002 | 21 | 28.571 | automático |
