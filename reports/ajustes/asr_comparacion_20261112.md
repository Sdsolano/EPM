# Comparación ASR actual vs mesa de ajustes vs real — 20261112

Generado por `scripts/comparar_asr.py --historico`. Detalle en `asr_comparacion_20261112.csv` (no versionado; una fila por UCP × base × semana × día × serie).

## 1. Configuración

- Bases: ingenuo, archivo. Semanas: últimas 52 completas por UCP; corte = sábado anterior (el domingo es brecha). Contexto sin fuga (histórico ≤ corte; los históricos del ASR salen de ese mismo histórico).
- `mesa` = `run_pipeline` con los defaults vigentes sobre la misma base.
- `asr_<escenario>` = ASR de Sphere reproducido (`src/adjustments/asr.py`) + validador del endpoint (models/{ucp}, festivos de librería para `is_special_day`). `_sin_validador` = objetivo h1 + desvi2 sin el validador. `_solo_nivel` = forma de la base con la energía del ASR; `_solo_forma` = forma del ASR con la energía de la base.
- Escenarios (orden de clic): `anual` h1 = último mismo día de semana real, h2 = d−364, h3 = h1−364; `anual_alineado` = con el desplazamiento de la semana de referencia de la mesa (festivo homólogo); `anual_desfase` = h3 = d−371 aunque h1 sea d−14; `anual_clic_asc` = misma selección marcada de abajo arriba (desvi2 congelado = h2 − h1); `recientes` = los 3 últimos mismos días de semana.

### Semanas evaluadas

| base | ucp | primera_semana | ultima_semana | n_semanas |
|---|---|---|---|---|
| archivo | Antioquia | 2026-01-05 | 2026-05-25 | 21 |
| ingenuo | Antioquia | 2025-06-30 | 2026-06-22 | 52 |
| ingenuo | Atlantico | 2025-05-05 | 2026-04-27 | 52 |

### Cobertura

Las secciones 2, 3 y 5 usan solo los días en que todas las series comparadas tienen dato (el ASR necesita h2/h3 de hace un año; Atlántico tiene huecos en 2024).

| base | ucp | dias_total | dias_comunes |
|---|---|---|---|
| archivo | Antioquia | 147 | 147 |
| ingenuo | Antioquia | 364 | 364 |
| ingenuo | Atlantico | 364 | 231 |

## 2. Error por serie (días comunes)

WAPE = Σ|pred−real|/Σreal. Energía día/semana: MAPE y sesgo. err_forma = 100·Σ|p/Σp − r/Σr| (media diaria). Pico: |error %| del máximo y diferencia de hora.

| base | ucp | serie | n_dias | wape_pct | mape_energia_dia_pct | sesgo_energia_dia_pct | err_forma_pct | err_pico_pct | dif_hora_pico_h | n_semanas | mape_energia_semana_pct | sesgo_energia_semana_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ingenuo | Antioquia | base | 364.00 | 4.05 | 3.92 | -0.11 | 1.70 | 3.55 | 1.85 | 52 | 2.69 | -0.25 |
| ingenuo | Antioquia | mesa | 364.00 | 3.71 | 3.59 | -0.11 | 1.16 | 3.47 | 1.76 | 52 | 2.35 | -0.22 |
| ingenuo | Antioquia | asr_anual | 364.00 | 5.26 | 5.02 | -0.10 | 2.15 | 4.80 | 2.19 | 52 | 3.18 | -0.23 |
| ingenuo | Antioquia | asr_anual_alineado | 364.00 | 5.02 | 4.71 | -0.49 | 2.07 | 4.53 | 2.15 | 52 | 3.06 | -0.56 |
| ingenuo | Antioquia | asr_anual_desfase | 364.00 | 5.27 | 5.01 | -0.11 | 2.16 | 4.77 | 2.18 | 52 | 3.07 | -0.24 |
| ingenuo | Antioquia | asr_anual_clic_asc | 364.00 | 6.01 | 5.78 | -4.35 | 1.73 | 5.49 | 2.68 | 52 | 4.91 | -4.46 |
| ingenuo | Antioquia | asr_recientes | 364.00 | 6.28 | 6.00 | -0.04 | 2.45 | 5.76 | 2.29 | 52 | 4.33 | -0.19 |
| ingenuo | Atlantico | base | 231.00 | 6.36 | 5.68 | -1.07 | 3.26 | 5.65 | 0.94 | 31 | 3.64 | -0.91 |
| ingenuo | Atlantico | mesa | 231.00 | 6.00 | 5.47 | -0.82 | 2.54 | 5.59 | 0.69 | 31 | 3.41 | -0.70 |
| ingenuo | Atlantico | asr_anual | 231.00 | 8.24 | 6.98 | -0.67 | 4.67 | 7.44 | 1.87 | 31 | 3.99 | -0.53 |
| ingenuo | Atlantico | asr_anual_alineado | 231.00 | 8.20 | 7.01 | -0.62 | 4.58 | 7.50 | 1.81 | 31 | 4.21 | -0.61 |
| ingenuo | Atlantico | asr_anual_desfase | 231.00 | 8.22 | 6.95 | -0.80 | 4.66 | 7.29 | 1.87 | 31 | 3.96 | -0.67 |
| ingenuo | Atlantico | asr_anual_clic_asc | 231.00 | 11.32 | 10.37 | -9.03 | 3.96 | 9.74 | 0.97 | 31 | 9.56 | -9.34 |
| ingenuo | Atlantico | asr_recientes | 231.00 | 8.93 | 7.91 | -0.22 | 4.67 | 7.65 | 1.62 | 31 | 5.73 | 0.14 |
| archivo | Antioquia | base | 147.00 | 3.03 | 2.75 | -0.73 | 1.27 | 2.56 | 2.31 | 21 | 2.11 | -0.79 |
| archivo | Antioquia | mesa | 147.00 | 2.79 | 2.55 | -0.56 | 1.15 | 2.60 | 2.11 | 21 | 1.90 | -0.60 |
| archivo | Antioquia | asr_anual | 147.00 | 5.42 | 5.16 | 0.08 | 2.20 | 5.07 | 2.56 | 21 | 3.44 | -0.06 |
| archivo | Antioquia | asr_anual_alineado | 147.00 | 5.28 | 4.93 | -0.38 | 2.13 | 4.91 | 2.58 | 21 | 3.34 | -0.43 |
| archivo | Antioquia | asr_anual_desfase | 147.00 | 5.46 | 5.20 | 0.04 | 2.21 | 5.19 | 2.56 | 21 | 3.38 | -0.09 |
| archivo | Antioquia | asr_anual_clic_asc | 147.00 | 6.91 | 6.66 | -5.38 | 1.70 | 6.29 | 2.54 | 21 | 6.12 | -5.46 |
| archivo | Antioquia | asr_recientes | 147.00 | 7.03 | 6.75 | -0.58 | 2.56 | 6.54 | 2.66 | 21 | 5.02 | -0.72 |

## 3. WAPE % por tipo de día

| base | ucp | tipo_dia | wape_asr_anual | wape_asr_anual_alineado | wape_asr_recientes | wape_base | wape_mesa | n_dias |
|---|---|---|---|---|---|---|---|---|
| archivo | Antioquia | domingo | 4.00 | 3.98 | 5.55 | 2.90 | 2.89 | 21 |
| archivo | Antioquia | festivo | 19.23 | 5.88 | 17.06 | 4.31 | 4.34 | 6 |
| archivo | Antioquia | laboral | 5.28 | 5.81 | 7.08 | 3.08 | 2.76 | 99 |
| archivo | Antioquia | sabado | 3.82 | 3.61 | 5.56 | 2.56 | 2.49 | 21 |
| ingenuo | Antioquia | domingo | 4.51 | 4.58 | 4.98 | 3.53 | 3.51 | 51 |
| ingenuo | Antioquia | festivo | 17.65 | 9.27 | 19.09 | 16.37 | 12.57 | 18 |
| ingenuo | Antioquia | laboral | 4.93 | 5.09 | 6.12 | 3.61 | 3.38 | 243 |
| ingenuo | Antioquia | sabado | 3.75 | 3.73 | 4.33 | 2.96 | 2.78 | 52 |
| ingenuo | Atlantico | domingo | 8.60 | 8.96 | 8.54 | 6.24 | 5.71 | 30 |
| ingenuo | Atlantico | festivo | 10.91 | 9.33 | 12.82 | 11.41 | 9.48 | 10 |
| ingenuo | Atlantico | laboral | 8.02 | 8.03 | 8.87 | 6.05 | 5.83 | 157 |
| ingenuo | Atlantico | sabado | 8.23 | 8.09 | 8.42 | 6.55 | 6.07 | 34 |

### Comparación día a día (WAPE)

| base | ucp | comparacion | n_dias | gana_asr_anual_pct | dif_wape_media_pp | gana_mesa_pct |
|---|---|---|---|---|---|---|
| archivo | Antioquia | asr_anual vs base | 147 | 33.33 | 2.45 | — |
| ingenuo | Antioquia | asr_anual vs base | 364 | 31.87 | 1.19 | — |
| ingenuo | Atlantico | asr_anual vs base | 234 | 25.64 | 1.96 | — |
| archivo | Antioquia | asr_anual vs mesa | 147 | 29.25 | 2.67 | — |
| ingenuo | Antioquia | asr_anual vs mesa | 364 | 25.27 | 1.55 | — |
| ingenuo | Atlantico | asr_anual vs mesa | 234 | 17.95 | 2.33 | — |
| archivo | Antioquia | mesa vs base | 147 | — | -0.23 | 61.22 |
| ingenuo | Antioquia | mesa vs base | 364 | — | -0.37 | 54.40 |
| ingenuo | Atlantico | mesa vs base | 364 | — | -0.38 | 55.22 |

## 4. Efecto del validador (cluster ±15 %)

| base | ucp | escenario | wape_con_validador | wape_sin_validador | dias_con_clip_pct | periodos_modificados_media | dias_especiales |
|---|---|---|---|---|---|---|---|
| archivo | Antioquia | anual | 5.42 | 5.43 | 4.08 | 0.14 | 6 |
| ingenuo | Antioquia | anual | 5.26 | 5.26 | 5.49 | 0.21 | 20 |
| ingenuo | Atlantico | anual | 8.31 | 8.34 | 13.68 | 0.46 | 11 |
| archivo | Antioquia | anual_alineado | 5.28 | 5.29 | 4.08 | 0.11 | 6 |
| ingenuo | Antioquia | anual_alineado | 5.02 | 5.02 | 4.95 | 0.16 | 20 |
| ingenuo | Atlantico | anual_alineado | 8.31 | 8.35 | 14.29 | 0.47 | 13 |
| archivo | Antioquia | anual_desfase | 5.46 | 5.47 | 4.08 | 0.14 | 6 |
| ingenuo | Antioquia | anual_desfase | 5.27 | 5.27 | 5.49 | 0.21 | 20 |
| ingenuo | Atlantico | anual_desfase | 8.32 | 8.35 | 14.89 | 0.47 | 11 |
| archivo | Antioquia | anual_clic_asc | 6.91 | 6.91 | 2.04 | 0.04 | 6 |
| ingenuo | Antioquia | anual_clic_asc | 6.01 | 6.02 | 2.47 | 0.10 | 20 |
| ingenuo | Atlantico | anual_clic_asc | 11.34 | 11.34 | 5.98 | 0.16 | 11 |
| archivo | Antioquia | recientes | 7.03 | 7.02 | 8.16 | 0.32 | 6 |
| ingenuo | Antioquia | recientes | 6.28 | 6.26 | 6.87 | 0.34 | 20 |
| ingenuo | Atlantico | recientes | 8.79 | 8.83 | 15.38 | 0.53 | 19 |

## 5. Descomposición nivel / forma del ASR `anual`

`solo_nivel`: la base reescalada a la energía diaria del ASR (aporta solo el nivel). `solo_forma`: la forma del ASR con la energía de la base (aporta solo la forma).

| base | ucp | serie | n_dias | wape_pct | mape_energia_dia_pct | sesgo_energia_dia_pct | err_forma_pct | err_pico_pct | dif_hora_pico_h |
|---|---|---|---|---|---|---|---|---|---|
| ingenuo | Antioquia | base | 364.00 | 4.05 | 3.92 | -0.11 | 1.70 | 3.55 | 1.85 |
| ingenuo | Antioquia | mesa | 364.00 | 3.71 | 3.59 | -0.11 | 1.16 | 3.47 | 1.76 |
| ingenuo | Antioquia | asr_anual | 364.00 | 5.26 | 5.02 | -0.10 | 2.15 | 4.80 | 2.19 |
| ingenuo | Antioquia | asr_anual_solo_nivel | 364.00 | 5.19 | 5.02 | -0.10 | 1.70 | 4.70 | 1.85 |
| ingenuo | Antioquia | asr_anual_solo_forma | 364.00 | 4.37 | 3.92 | -0.11 | 2.15 | 3.90 | 2.19 |
| ingenuo | Atlantico | base | 234.00 | 6.37 | 5.68 | -1.07 | 3.28 | 5.67 | 0.94 |
| ingenuo | Atlantico | mesa | 234.00 | 6.01 | 5.48 | -0.82 | 2.54 | 5.62 | 0.68 |
| ingenuo | Atlantico | asr_anual | 234.00 | 8.31 | 7.07 | -0.59 | 4.69 | 7.54 | 1.86 |
| ingenuo | Atlantico | asr_anual_solo_nivel | 234.00 | 7.68 | 7.07 | -0.59 | 3.28 | 7.22 | 0.94 |
| ingenuo | Atlantico | asr_anual_solo_forma | 234.00 | 7.12 | 5.68 | -1.07 | 4.69 | 6.10 | 1.86 |
| archivo | Antioquia | base | 147.00 | 3.03 | 2.75 | -0.73 | 1.27 | 2.56 | 2.31 |
| archivo | Antioquia | mesa | 147.00 | 2.79 | 2.55 | -0.56 | 1.15 | 2.60 | 2.11 |
| archivo | Antioquia | asr_anual | 147.00 | 5.42 | 5.16 | 0.08 | 2.20 | 5.07 | 2.56 |
| archivo | Antioquia | asr_anual_solo_nivel | 147.00 | 5.29 | 5.16 | 0.08 | 1.27 | 5.13 | 2.31 |
| archivo | Antioquia | asr_anual_solo_forma | 147.00 | 3.55 | 2.75 | -0.73 | 2.20 | 2.69 | 2.56 |

### Corrección de energía aplicada vs necesaria

aplicada = E_serie/E_base − 1; necesaria = E_real/E_base − 1 (por día). Una corrección útil correlaciona con la necesaria y es más chica que ella.

| base | ucp | serie | n_dias | corr | pendiente | std_aplicada_pct | std_necesaria_pct |
|---|---|---|---|---|---|---|---|
| ingenuo | Antioquia | mesa | 364 | 0.74 | 2.98 | 1.65 | 6.61 |
| ingenuo | Antioquia | asr_anual_sin_validador | 364 | 0.25 | 0.27 | 5.93 | 6.61 |
| ingenuo | Antioquia | asr_anual_alineado_sin_validador | 364 | 0.38 | 0.41 | 6.16 | 6.61 |
| ingenuo | Antioquia | asr_recientes_sin_validador | 364 | -0.18 | -0.19 | 6.23 | 6.61 |
| ingenuo | Atlantico | mesa | 364 | 0.23 | 1.02 | 1.62 | 7.29 |
| ingenuo | Atlantico | asr_anual_sin_validador | 234 | -0.02 | -0.03 | 5.72 | 7.54 |
| ingenuo | Atlantico | asr_anual_alineado_sin_validador | 252 | 0.07 | 0.08 | 6.35 | 7.57 |
| ingenuo | Atlantico | asr_recientes_sin_validador | 364 | 0.06 | 0.06 | 7.14 | 7.29 |
| archivo | Antioquia | mesa | 147 | 0.33 | 0.94 | 1.15 | 3.32 |
| archivo | Antioquia | asr_anual_sin_validador | 147 | 0.05 | 0.02 | 7.24 | 3.32 |
| archivo | Antioquia | asr_anual_alineado_sin_validador | 147 | 0.12 | 0.06 | 6.75 | 3.32 |
| archivo | Antioquia | asr_recientes_sin_validador | 147 | 0.24 | 0.08 | 10.21 | 3.32 |

### Históricos de otro tipo de día en h1, h2 o h3 (`asr_anual`)

Día contaminado = alguno de h1, h2, h3 es de distinto tipo (laboral/sábado/domingo/festivo) que el día objetivo (p. ej. h1 festivo para un laboral, o h3 festivo del año anterior).

| base | ucp | historico_de_otro_tipo | n_dias | wape_base | wape_mesa | wape_asr_anual |
|---|---|---|---|---|---|---|
| ingenuo | Antioquia | no | 313 | 2.78 | 2.77 | 3.77 |
| ingenuo | Antioquia | sí | 51 | 12.21 | 9.68 | 14.78 |
| ingenuo | Atlantico | no | 198 | 5.87 | 5.74 | 7.76 |
| ingenuo | Atlantico | sí | 36 | 9.14 | 7.49 | 11.29 |
| archivo | Antioquia | no | 126 | 2.88 | 2.67 | 3.64 |
| archivo | Antioquia | sí | 21 | 3.92 | 3.52 | 16.10 |

## 6. Diagnóstico: por qué falla el ASR

Lectura del código (Sphere `Pronosticos.tsx`, rama `test`, commit 57eb4fb; endpoint
`/validate-hourly-adjustments`) y números de este reporte. Base de referencia en el texto: `ingenuo`
Antioquia salvo que se diga otra cosa; los números de `archivo` (un pronóstico real del modelo) van entre
paréntesis.

**Causa raíz: el ASR descarta el pronóstico y lo reemplaza por un día suelto más el ruido de otros dos.**
Por hora, `delta = (h2 − h3) − (pred − h1)`, y el resultado antes del validador es
`pred + delta = h1 + (h2 − h3)`. El pronóstico `pred` se cancela: lo que diga el modelo no cuenta (salvo
±0,5 MW de redondeo). Con una base buena, el ASR la empeora: con `archivo` el WAPE pasa de 3,03 % a 5,42 %.

**Alcance de la evidencia (leer antes de la tabla).**
- **Lo que no depende de supuestos.** La fórmula `h1 + (h2 − h3)`, que el endpoint ignora `tipo_dia` y
  el bug de orden en el código.
- **Lo que se midió con una selección de históricos supuesta.** El resto se midió con los escenarios
  de la sección 1; el principal es `anual`. No sabemos qué marca el analista en realidad, ni se midió
  el caso en que usa "históricos pronóstico". Los veredictos con números son **a validar con su caso
  real el viernes**.
- **Base `ingenuo`.** Con esta base, base = h1, así que la diferencia ASR − base es exactamente
  `h2 − h3`. La comparación es casi circular.
- **Base `archivo`.** Es la evidencia independiente, pero chica: solo Antioquia, 21 semanas y 6 festivos.
- **Muestras chicas.** Las cifras de festivos (n = 6–18 días) y de días "contaminados" las mueven pocos
  eventos: los 10 peores días concentran el 40–52 % del exceso de error del ASR en Antioquia (25 % en
  Atlántico). Aun así, la mediana por día también es peor que la base: +0,8 a +1,4 pp de WAPE.
- **Periodos y n distintos entre tablas.** Antioquia va de jun-2025 a jun-2026 y Atlántico de
  may-2025 a abr-2026, con huecos en 2024. Por eso el n de Atlántico cambia de tabla a tabla
  (231–364 días).

| # | Hipótesis (REQUERIMIENTOS R1) | Veredicto | Evidencia |
|---|---|---|---|
| 1 | El delta depende del orden en que se seleccionan los históricos | **Bug de código confirmado; impacto condicional** (solo si marca los históricos en un orden distinto al de la lista) | `h1` sale del orden por fecha (el más reciente), pero `desvi2` se congela con el orden de clic. La misma selección marcada de abajo arriba da `objetivo = h2`, es decir, el día del año anterior crudo, sin crecimiento: WAPE 6,01 % frente a 5,26 % (6,91 % frente a 5,42 %), sesgo de energía −4,4 % (−5,4 %) y −9,0 % en Atlántico. Además, el gráfico de áreas del modo ajustado rotula H1, H2 y H3 en orden de clic, distinto del h1 que usa el ASR. `desvi2` tampoco se recalcula si cambia la selección (solo se limpia al salir del modo). |
| 2 | Sin alineación por tipo de día ni festivo | **Confirmada** | En el 14 % de los días (51 de 364) algún histórico es de otro tipo de día. En esos días el WAPE es 14,8 % con ASR, 12,2 % con la base y 9,7 % con la mesa (con `archivo`: 16,1 / 3,9 / 3,5). Alinear con la semana de referencia de la mesa (`anual_alineado`) baja el WAPE de los festivos de 17,7 % a 9,3 % (de 19,2 % a 5,9 %). No sirve cuando el contaminado es h1, por ejemplo después de Semana Santa. |
| 2b | Sin alinear el desfase entre el último dato real y la semana objetivo | **Refutada como causa relevante** | Solo afecta al domingo (h1 = d − 14). Domingos (días comunes): `anual` 4,51 % frente a `anual_desfase` 4,54 %; en Atlántico 8,71 frente a 8,63. Con `archivo`, el desfase sí empeora los domingos en 0,4 pp (4,00 frente a 4,40). Total: 5,26 frente a 5,27. |
| 3 | Compara días sueltos en vez de una semana de referencia | **Confirmada (causa principal)** | Aun sin contaminación de tipo de día, el ASR es peor: 3,77 % frente a 2,78 % de la base (3,64 frente a 2,88; Atlántico 7,76 frente a 5,87). La corrección de energía diaria que aplica, E_ASR/E_base − 1, casi no se relaciona con la que hacía falta: correlación 0,25 en Antioquia, −0,02 en Atlántico y 0,05 con `archivo`. Y es tan grande como el error que intenta corregir: desviación estándar 5,9 % frente a 6,6 % necesaria (7,2 % frente a 3,3 %), así que es ruido de un par de días. La mesa (ratio de energía con el ancla y α = 0,25) corrige poco pero en la dirección correcta: correlación 0,74, desviación estándar 1,65 % (0,33 y 1,15 %). |
| 4 | Hora a hora: mezcla nivel y forma | **Confirmada** | Las dos partes empeoran. Solo el nivel del ASR sobre la forma de la base: WAPE 5,19 % (5,29 %). Solo la forma del ASR con la energía de la base: 4,37 % (3,55 %). Base: 4,05 % (3,03 %). La mayor parte del daño viene del nivel. Error de forma: ASR 2,15, base 1,70, mesa 1,16. |
| 5 | No conserva la energía ni revisa la coherencia semanal | **Confirmada** | MAPE de energía semanal: ASR 3,18 %, base 2,69 %, mesa 2,35 % (3,44 / 2,11 / 1,90). El ASR no tiene ningún control semanal y se aplica día por día. |
| 6 | `tipo_dia: "laboral"` fijo | **Refutada como causa** | El endpoint no lee `tipo_dia`; lo verifica `test_tipo_dia_no_cambia_la_respuesta_del_endpoint`. El cluster se elige con `is_special_day` (festivo, 24-dic o 2-ene) o por día de la semana. |
| 7 | El cluster del validador no representa el escenario | **Irrelevante para el error: el validador casi no actúa** (no prueba si el cluster representa o no el escenario; además, los clusters actuales se entrenaron con datos de 2026, posteriores a varias semanas evaluadas) | Recorta en el 4–15 % de los días, con 0,1–0,5 periodos en promedio. El WAPE con y sin validador difiere ≤ 0,04 pp. La banda de ±15 % por hora sobre la proporción del cluster es demasiado ancha: ni salva ni daña. |

Otras causas encontradas:
- **El escenario "recientes" es el peor** (los 3 últimos días del mismo día de la semana, que es lo que
  ofrece el filtro "Similares"): WAPE 6,28 % (7,03 %). Extrapola el cambio de la semana pasada, y su
  corrección se mueve en contra de la necesaria (correlación −0,18).
- **El ASR gana pocas veces:** `asr_anual` le gana a la base en el 32 % de los días en Antioquia, 26 % en
  Atlántico y 33 % con `archivo`; a la mesa, en el 25 %, 18 % y 29 %. La mesa le gana a la base en el
  54–61 % de los días.
- **Evidencia en contra de la mesa.**
  - En los festivos de Antioquia con base `ingenuo`, el ASR alineado (9,27 %) le gana a la mesa (12,57 %).
  - Después de Semana Santa (2026-04-09 y 2026-04-10) la mesa empeora a la base: 3,2 frente a 2,1 y
    3,8 frente a 2,9. También el 2025-06-15 en Atlántico: 6,0 frente a 2,9.
  - Su ganancia media sobre la base es chica: 0,23–0,38 pp. La mediana por día es −0,06 pp en
    Antioquia `ingenuo`, −0,15 pp en Atlántico y −0,27 pp con `archivo`.
  - Los festivos y los días después de festivos largos son el punto débil de la mesa (backlog Fase 2b/4).
- **Sin medir** (pendiente del caso real): clima, puentes y eventos como causa de error del ASR. El
  ajuste a mano del analista no se pudo comparar porque todavía no tenemos sus curvas.
- Menores (no medidos o despreciables): `Math.round` del delta, ±0,5 MW por hora; los clusters vienen de
  modelos entrenados con los 3 meses previos a 2026-07 (KMeans de sklearn 1.7 cargado en 1.9), pero no
  pesan porque el validador casi no actúa.

## 7. Ejemplos concretos (escenario `anual`)

WAPE % del día: base / mesa / ASR / ASR alineado.

- **2026-01-12, Antioquia (Reyes, lunes festivo).** h1 = 2026-01-05, h2 = 2025-01-13 y h3 = 2025-01-06,
  que fue el festivo de Reyes de 2025. `h2 − h3` resta un festivo a un laboral y sube el festivo de 2026 a
  nivel laboral. WAPE: archivo 10,8 / 8,4 / **39,6** / 3,3; ingenuo 13,5 / 10,3 / **39,6** / 3,3.
- **2026-04-09 y 2026-04-10, Antioquia (jueves y viernes después de Semana Santa).** h1 = 2026-04-02 y
  2026-04-03, el Jueves y el Viernes Santo. El ASR lleva días laborales al nivel de un festivo. WAPE
  (archivo): 2,1 / 3,2 / **22,6** y 2,9 / 3,8 / **27,4**. Alinear no ayuda (22,6 y 27,4), porque el
  contaminado es h1.
- **2026-01-19, Antioquia (lunes laboral).** h1 = 2026-01-12, que fue festivo. WAPE (archivo) 1,3 / 1,7 /
  **19,4**.
- **2025-11-17, Antioquia (lunes festivo).** WAPE (ingenuo) 19,1 / 12,2 / **41,8** / 16,4.
- **2025-06-15, Atlántico (domingo normal, sin contaminación).** h2 − h3 de 2024 trae un cambio de un día
  suelto: 2,9 / 6,0 / **17,2**.

Para ver cualquier día hora a hora: `scripts/comparar_asr.py --ucp <UCP> --semana <lunes> --horas`
(sección 9).

## 8. Recomendación

**Propuesta, a confirmar el viernes con el caso real: reemplazar el ASR por la mesa de ajustes en Sphere
(Fase 3), no corregirlo.** La mesa es claramente mejor que el ASR y levemente mejor que la base.
- El defecto de fondo es de diseño: descarta el pronóstico y corrige con un solo par de días del año
  anterior. Ese par es ruido: no predice la corrección necesaria (correlación entre −0,02 y 0,25) y es tan
  grande como el error. Arreglar el orden o la alineación no cambia eso: en días limpios sigue siendo
  1 pp peor que la base.
- La mesa hace lo que el ASR intentaba: nivel con un ratio de energía contra un ancla comparable, con el
  mismo desfase y encogido con α; forma promedio; festivos homólogos; control semanal. Mejora a la base
  poco: −0,23 a −0,38 pp de WAPE medio, mediana por día cercana a cero (−0,06 a −0,27 pp), y gana el
  54–61 % de los días. Deja
  trazabilidad. Sus puntos débiles son los festivos y los días después de Semana Santa (sección 6).
- **Paliativo mínimo mientras llega la Fase 3** (no requiere backtest nuevo, porque no cambia el
  algoritmo de la mesa):
  1. Marcar el botón ASR como "no recomendado", o retirarlo.
  2. Si se mantiene, corregir el bug de orden: calcular `desvi2` con el mismo orden por fecha que `h1`, y
     recalcularlo al cambiar la selección.
  3. Mejor todavía: que el botón llame a `POST /adjustments/preview` con `base_curvas` = el pronóstico
     de la semana y muestre la curva de la mesa del día.
  No se recomienda "arreglar" la fórmula del ASR, por ejemplo encogiendo el delta: sería reinventar el
  paso `nivel` sin su evidencia.
- Salvedad: las bases evaluadas son la ingenua y un pronóstico archivado (Antioquia, ene–jun 2026), no
  el modelo semanal real. El resultado del ASR no depende de la base, así que su error absoluto (5–8 %)
  es el mismo con cualquier modelo. Lo que puede cambiar es la ventaja de la mesa sobre la base; lo mide
  la Fase 2a.

## 9. Guía para la sesión del viernes 2026-11-13 con el analista (1 página)

**Qué pedirle (para su semana real):**
1. UCP, semana objetivo (lunes) y fecha de corte (último dato real que tenía al pronosticar).
2. El **pronóstico base** tal como salió del modelo, antes de tocarlo: CSV con `fecha,P1..P24`, o
   `fecha,periodo,valor`, desde el día siguiente al corte hasta el domingo.
3. Su **resultado final ajustado a mano**, en el mismo formato.
4. Para cada día en que usó el ASR: **qué históricos marcó y en qué orden** (de arriba abajo o al
   revés), si alguno era un "histórico pronóstico" y si cambió la selección después de entrar al modo
   ajustado.
5. Qué hizo a mano que el ASR no hizo: festivos, puentes, clima, eventos o tendencia de la semana.

**Paso 0: el histórico tiene que estar al día.** El comparador saca h1, h2 y h3, el corte y la mesa de
`data/raw/{ucp}/datos.csv`.
- En el PC de desarrollo ese archivo llega hasta 2026-07-04 en Antioquia y hasta 2026-05-07 en
  Atlántico. Con él, una semana de noviembre queda con el corte en julio, la mesa y el ASR no son los
  que vería el analista, y `base_curvas` tendría que cubrir de julio a noviembre.
- El comparador lo avisa en `advertencias` (más de 6 días entre el corte y el lunes).
- Opciones:
  1. Usar el servidor de la API, si su `datos.csv` está al día. `comparar-asr` no refresca datos: lo hacen
     `/adjustments/run` y `/adjustments/preview` sin `base_curvas`. Además, el endpoint solo existe donde
     se haya desplegado esta rama.
  2. Antes del script (desde esta rama), actualizar los datos con
     `python -c "from src.pipeline.update_csv import full_update_csv; full_update_csv('Antioquia')"`.
     Ojo: usa la red (datos de demanda y API de clima), requiere el `.env` de ese entorno y reescribe
     `data/raw/Antioquia/datos.csv`. Hay que hacerlo en el entorno del analista o del servidor, no en un
     backtest.
  - Para comprobar que los datos están al día, la salida debe venir sin `advertencias` y el corte debe
    ser el sábado anterior a la semana.
- Si nada de eso es posible el viernes, revisen con él la lógica, la sección 7 y la fórmula, y corran el
  caso real cuando haya datos al día.

**Cómo correr el comparador (dos opciones):**
- Script (desde la raíz del repo EPM, con los datos al día; lee CSV):

  ```
  python scripts/comparar_asr.py --ucp Antioquia --semana 2026-11-16 \
      --fecha-emision 2026-11-15 --base-csv base.csv --analista-csv final.csv \
      --seleccion 2026-11-16=2026-11-09,2025-11-17,2025-11-10 --horas --descomponer --out caso_viernes.md
  ```

  `--seleccion OBJETIVO=F1,F2,F3` va en el **orden de clic**; sin ella se usan los escenarios de este
  reporte. Cuando llegue el real, se corre igual: la columna de error contra el real aparece sola.
- Endpoint (API de EPM levantada; misma cabecera `X-Ajustes-Token` que `/adjustments`). Recibe JSON, no
  CSV: `base_curvas` y `analista_curvas` van como `{"YYYY-MM-DD": [24 valores]}`, y `base_curvas` debe
  cubrir desde el día siguiente al corte hasta el domingo. `POST /adjustments/comparar-asr` con
  `{"ucp": "Antioquia", "semana_inicio": "2026-11-16", "fecha_emision": "2026-11-15",
  "base_curvas": {...}, "analista_curvas": {...}, "seleccion": {"2026-11-16": ["2026-11-09", "2025-11-17",
  "2025-11-10"]}}`. Devuelve las series (base, mesa, ASR con y sin validador, analista y real si existe),
  métricas por día y semana, y el detalle del ASR (h1/h2/h3, par de desvi2, cluster, periodos recortados).

**Qué mirar con él, día por día:** energía (ASR frente a mesa frente a la suya), hora y MW del pico, y la
forma (`err_forma`). Revisen en especial los días donde algún histórico es festivo o puente.
Preguntas clave:
- ¿Esperaba que el ASR conservara el nivel del modelo? Hoy lo descarta.
- ¿Marca los históricos de arriba abajo?
- ¿Qué regla usa para elegir la semana del año anterior?
- ¿Qué hace con clima, puentes y eventos? No está medido en este reporte.

**Salida de la sesión:**
- Confirmar el reemplazo del ASR por la mesa en Sphere (Fase 3) y el paliativo mientras tanto.
- Anotar en `REQUERIMIENTOS.md` las decisiones del analista que la mesa todavía no reproduce. Son la
  entrada del backlog de la Fase 4.
