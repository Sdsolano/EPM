# Comparación ASR actual vs mesa de ajustes vs real — 20261113_modelo

Generado por `scripts/comparar_asr.py --historico`. Detalle en `asr_comparacion_20261113_modelo.csv` (no versionado; una fila por UCP × base × semana × día × serie).

## 1. Configuración

- Bases: modelo_k1. `modelo_k1` = pronóstico del modelo real de la Fase 2a, sin fuga (corte previo verificado). Semanas: últimas 52 completas por UCP; corte = sábado anterior (el domingo es brecha). Contexto sin fuga (histórico ≤ corte; los históricos del ASR salen de ese mismo histórico).
- `mesa` = `run_pipeline` con los defaults vigentes sobre la misma base.
- `asr_<escenario>` = ASR de Sphere reproducido (`src/adjustments/asr.py`) + validador del endpoint (models/{ucp}, festivos de librería para `is_special_day`). `_sin_validador` = objetivo h1 + desvi2 sin el validador. `_solo_nivel` = forma de la base con la energía del ASR; `_solo_forma` = forma del ASR con la energía de la base.
- Escenarios (orden de clic): `anual` h1 = último mismo día de semana real, h2 = d−364, h3 = h1−364; `anual_alineado` = con el desplazamiento de la semana de referencia de la mesa (festivo homólogo); `anual_desfase` = h3 = d−371 aunque h1 sea d−14; `anual_clic_asc` = misma selección marcada de abajo arriba (desvi2 congelado = h2 − h1); `recientes` = los 3 últimos mismos días de semana.

### Semanas evaluadas

| base | ucp | primera_semana | ultima_semana | n_semanas |
|---|---|---|---|---|
| modelo_k1 | Antioquia | 2025-06-30 | 2026-06-22 | 52 |
| modelo_k1 | Atlantico | 2025-05-05 | 2026-04-27 | 52 |

### Cobertura

Las secciones 2, 3 y 5 usan solo los días en que todas las series comparadas tienen dato (el ASR necesita h2/h3 de hace un año; Atlántico tiene huecos en 2024).

| base | ucp | dias_total | dias_comunes |
|---|---|---|---|
| modelo_k1 | Antioquia | 364 | 364 |
| modelo_k1 | Atlantico | 364 | 231 |

## 2. Error por serie (días comunes)

WAPE = Σ|pred−real|/Σreal. Energía día/semana: MAPE y sesgo. err_forma = 100·Σ|p/Σp − r/Σr| (media diaria). Pico: |error %| del máximo y diferencia de hora.

| base | ucp | serie | n_dias | wape_pct | mape_energia_dia_pct | sesgo_energia_dia_pct | err_forma_pct | err_pico_pct | dif_hora_pico_h | n_semanas | mape_energia_semana_pct | sesgo_energia_semana_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| modelo_k1 | Antioquia | base | 364.00 | 2.70 | 2.37 | -0.85 | 1.29 | 2.42 | 2.00 | 52 | 1.80 | -0.88 |
| modelo_k1 | Antioquia | mesa | 364.00 | 2.59 | 2.36 | -0.66 | 1.16 | 2.28 | 1.76 | 52 | 1.65 | -0.69 |
| modelo_k1 | Antioquia | asr_anual | 364.00 | 5.26 | 5.02 | -0.11 | 2.15 | 4.80 | 2.20 | 52 | 3.18 | -0.23 |
| modelo_k1 | Antioquia | asr_anual_alineado | 364.00 | 5.02 | 4.71 | -0.49 | 2.07 | 4.53 | 2.17 | 52 | 3.06 | -0.56 |
| modelo_k1 | Antioquia | asr_anual_desfase | 364.00 | 5.27 | 5.01 | -0.12 | 2.16 | 4.77 | 2.20 | 52 | 3.07 | -0.24 |
| modelo_k1 | Antioquia | asr_anual_clic_asc | 364.00 | 6.02 | 5.78 | -4.36 | 1.73 | 5.49 | 2.70 | 52 | 4.91 | -4.47 |
| modelo_k1 | Antioquia | asr_recientes | 364.00 | 6.28 | 6.00 | -0.04 | 2.45 | 5.76 | 2.27 | 52 | 4.33 | -0.19 |
| modelo_k1 | Atlantico | base | 231.00 | 5.52 | 4.72 | -0.28 | 3.01 | 5.08 | 0.77 | 31 | 3.04 | -0.34 |
| modelo_k1 | Atlantico | mesa | 231.00 | 5.12 | 4.60 | -0.23 | 2.54 | 4.72 | 0.69 | 31 | 2.96 | -0.28 |
| modelo_k1 | Atlantico | asr_anual | 231.00 | 8.24 | 6.98 | -0.68 | 4.67 | 7.45 | 1.92 | 31 | 3.99 | -0.53 |
| modelo_k1 | Atlantico | asr_anual_alineado | 231.00 | 8.20 | 7.02 | -0.63 | 4.58 | 7.50 | 1.87 | 31 | 4.21 | -0.62 |
| modelo_k1 | Atlantico | asr_anual_desfase | 231.00 | 8.23 | 6.95 | -0.81 | 4.66 | 7.29 | 1.92 | 31 | 3.96 | -0.68 |
| modelo_k1 | Atlantico | asr_anual_clic_asc | 231.00 | 11.33 | 10.38 | -9.04 | 3.96 | 9.75 | 0.98 | 31 | 9.57 | -9.34 |
| modelo_k1 | Atlantico | asr_recientes | 231.00 | 8.93 | 7.91 | -0.23 | 4.67 | 7.65 | 1.62 | 31 | 5.73 | 0.13 |

## 3. WAPE % por tipo de día

| base | ucp | tipo_dia | wape_asr_anual | wape_asr_anual_alineado | wape_asr_recientes | wape_base | wape_mesa | n_dias |
|---|---|---|---|---|---|---|---|---|
| modelo_k1 | Antioquia | domingo | 4.51 | 4.58 | 4.98 | 2.77 | 2.81 | 51 |
| modelo_k1 | Antioquia | festivo | 17.65 | 9.27 | 19.09 | 3.59 | 3.75 | 18 |
| modelo_k1 | Antioquia | laboral | 4.93 | 5.09 | 6.12 | 2.66 | 2.52 | 243 |
| modelo_k1 | Antioquia | sabado | 3.75 | 3.73 | 4.33 | 2.60 | 2.39 | 52 |
| modelo_k1 | Atlantico | domingo | 8.60 | 8.96 | 8.54 | 5.95 | 4.95 | 30 |
| modelo_k1 | Atlantico | festivo | 10.91 | 9.33 | 12.81 | 6.49 | 5.60 | 10 |
| modelo_k1 | Atlantico | laboral | 8.02 | 8.03 | 8.88 | 5.37 | 5.15 | 157 |
| modelo_k1 | Atlantico | sabado | 8.23 | 8.09 | 8.42 | 5.60 | 5.02 | 34 |

### Comparación día a día (WAPE)

| base | ucp | comparacion | n_dias | gana_asr_anual_pct | dif_wape_media_pp | gana_mesa_pct |
|---|---|---|---|---|---|---|
| modelo_k1 | Antioquia | asr_anual vs base | 364 | 31.59 | 2.63 | — |
| modelo_k1 | Atlantico | asr_anual vs base | 234 | 26.92 | 2.85 | — |
| modelo_k1 | Antioquia | asr_anual vs mesa | 364 | 24.45 | 2.74 | — |
| modelo_k1 | Atlantico | asr_anual vs mesa | 234 | 20.51 | 3.23 | — |
| modelo_k1 | Antioquia | mesa vs base | 364 | — | -0.11 | 56.59 |
| modelo_k1 | Atlantico | mesa vs base | 364 | — | -0.34 | 60.16 |

## 4. Efecto del validador (cluster ±15 %)

| base | ucp | escenario | wape_con_validador | wape_sin_validador | dias_con_clip_pct | periodos_modificados_media | dias_especiales |
|---|---|---|---|---|---|---|---|
| modelo_k1 | Antioquia | anual | 5.26 | 5.26 | 5.49 | 0.21 | 20 |
| modelo_k1 | Atlantico | anual | 8.31 | 8.34 | 13.68 | 0.46 | 11 |
| modelo_k1 | Antioquia | anual_alineado | 5.02 | 5.02 | 4.95 | 0.16 | 20 |
| modelo_k1 | Atlantico | anual_alineado | 8.31 | 8.35 | 14.29 | 0.48 | 13 |
| modelo_k1 | Antioquia | anual_desfase | 5.27 | 5.27 | 5.49 | 0.21 | 20 |
| modelo_k1 | Atlantico | anual_desfase | 8.32 | 8.35 | 14.89 | 0.48 | 11 |
| modelo_k1 | Antioquia | anual_clic_asc | 6.02 | 6.02 | 2.47 | 0.10 | 20 |
| modelo_k1 | Atlantico | anual_clic_asc | 11.34 | 11.35 | 5.98 | 0.16 | 11 |
| modelo_k1 | Antioquia | recientes | 6.28 | 6.26 | 6.87 | 0.34 | 20 |
| modelo_k1 | Atlantico | recientes | 8.79 | 8.83 | 15.38 | 0.53 | 19 |

## 5. Descomposición nivel / forma del ASR `anual`

`solo_nivel`: la base reescalada a la energía diaria del ASR (aporta solo el nivel). `solo_forma`: la forma del ASR con la energía de la base (aporta solo la forma).

| base | ucp | serie | n_dias | wape_pct | mape_energia_dia_pct | sesgo_energia_dia_pct | err_forma_pct | err_pico_pct | dif_hora_pico_h |
|---|---|---|---|---|---|---|---|---|---|
| modelo_k1 | Antioquia | base | 364.00 | 2.70 | 2.37 | -0.85 | 1.29 | 2.42 | 2.00 |
| modelo_k1 | Antioquia | mesa | 364.00 | 2.59 | 2.36 | -0.66 | 1.16 | 2.28 | 1.76 |
| modelo_k1 | Antioquia | asr_anual | 364.00 | 5.26 | 5.02 | -0.11 | 2.15 | 4.80 | 2.20 |
| modelo_k1 | Antioquia | asr_anual_solo_nivel | 364.00 | 5.14 | 5.02 | -0.11 | 1.29 | 4.93 | 2.00 |
| modelo_k1 | Antioquia | asr_anual_solo_forma | 364.00 | 3.24 | 2.37 | -0.85 | 2.15 | 2.45 | 2.20 |
| modelo_k1 | Atlantico | base | 234.00 | 5.50 | 4.69 | -0.29 | 3.01 | 5.08 | 0.77 |
| modelo_k1 | Atlantico | mesa | 234.00 | 5.11 | 4.58 | -0.24 | 2.54 | 4.72 | 0.68 |
| modelo_k1 | Atlantico | asr_anual | 234.00 | 8.31 | 7.07 | -0.60 | 4.69 | 7.54 | 1.91 |
| modelo_k1 | Atlantico | asr_anual_solo_nivel | 234.00 | 7.67 | 7.07 | -0.60 | 3.01 | 7.20 | 0.77 |
| modelo_k1 | Atlantico | asr_anual_solo_forma | 234.00 | 6.43 | 4.69 | -0.29 | 4.69 | 5.71 | 1.91 |

### Corrección de energía aplicada vs necesaria

aplicada = E_serie/E_base − 1; necesaria = E_real/E_base − 1 (por día). Una corrección útil correlaciona con la necesaria y es más chica que ella.

| base | ucp | serie | n_dias | corr | pendiente | std_aplicada_pct | std_necesaria_pct |
|---|---|---|---|---|---|---|---|
| modelo_k1 | Antioquia | mesa | 364 | 0.21 | 0.55 | 1.12 | 2.85 |
| modelo_k1 | Antioquia | asr_anual_sin_validador | 364 | 0.03 | 0.01 | 7.28 | 2.85 |
| modelo_k1 | Antioquia | asr_anual_alineado_sin_validador | 364 | 0.10 | 0.04 | 6.46 | 2.85 |
| modelo_k1 | Antioquia | asr_recientes_sin_validador | 364 | 0.08 | 0.02 | 9.35 | 2.85 |
| modelo_k1 | Atlantico | mesa | 364 | 0.20 | 0.62 | 1.80 | 5.70 |
| modelo_k1 | Atlantico | asr_anual_sin_validador | 234 | 0.11 | 0.08 | 8.02 | 5.84 |
| modelo_k1 | Atlantico | asr_anual_alineado_sin_validador | 252 | 0.15 | 0.11 | 8.26 | 5.78 |
| modelo_k1 | Atlantico | asr_recientes_sin_validador | 364 | 0.14 | 0.09 | 8.84 | 5.70 |

### Históricos de otro tipo de día en h1, h2 o h3 (`asr_anual`)

Día contaminado = alguno de h1, h2, h3 es de distinto tipo (laboral/sábado/domingo/festivo) que el día objetivo (p. ej. h1 festivo para un laboral, o h3 festivo del año anterior).

| base | ucp | historico_de_otro_tipo | n_dias | wape_base | wape_mesa | wape_asr_anual |
|---|---|---|---|---|---|---|
| modelo_k1 | Antioquia | no | 313 | 2.57 | 2.48 | 3.77 |
| modelo_k1 | Antioquia | sí | 51 | 3.55 | 3.28 | 14.78 |
| modelo_k1 | Atlantico | no | 198 | 5.37 | 5.03 | 7.76 |
| modelo_k1 | Atlantico | sí | 36 | 6.19 | 5.58 | 11.29 |

## 6. Antes/después (base `modelo_k1` sin fuga)

Comparación con `asr_comparacion_20261112.md` (bases `ingenuo` + `archivo`, mismas 52 semanas). Las series `asr_*` no dependen de la base (se construyen con históricos reales), así que sus métricas son idénticas; cambian `base` y `mesa`.

| UCP | serie | WAPE antes (ingenuo, 20261112) | WAPE después (modelo_k1, 20261113_modelo) | Δ (pp) |
|---|---|---|---|---|
| Antioquia | base | 4.05 | 2.70 | −1.35 |
| Antioquia | mesa | 3.71 | 2.59 | −1.12 |
| Antioquia | asr_anual | 5.26 | 5.26 | 0.00 |
| Atlantico | base | 6.36 | 5.52 | −0.84 |
| Atlantico | mesa | 6.00 | 5.12 | −0.88 |
| Atlantico | asr_anual | 8.24 | 8.24 | 0.00 |

- La mesa mejora a la base también sobre el modelo real: Antioquia 2.70 → 2.59 (−0.11 pp) y Atlántico 5.52 → 5.12 (−0.40 pp); con `ingenuo` la mejora era −0.34 y −0.36 pp.
- El ASR sigue muy por detrás de base y mesa: 5.26 vs 2.59 (Antioquia) y 8.24 vs 5.12 (Atlántico); gana día a día solo 31.6 % / 26.9 % de los días frente a la base.
- Festivos: `asr_anual` 17.65 (Antioquia) y 10.91 (Atlántico) vs mesa 3.75 / 5.60; `anual_alineado` baja Antioquia a 9.27. Atlántico festivo (n=10) es **no concluyente** (<15 días).
- `archivo` (Antioquia, 21 semanas) sigue siendo la base con menor WAPE (base 3.03, mesa 2.79, asr_anual 5.42), pero es una sola emisión con ajustes post-hoc y no es comparable en ventana.

