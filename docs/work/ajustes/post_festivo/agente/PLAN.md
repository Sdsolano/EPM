# Sub-plan del agente de implementación — factor post-festivo

Fuente de verdad: `../PLAN.md` (aprobado). Este archivo solo fija el diseño concreto antes de tocar código.

## Diseño
- `src/adjustments/referencia_reciente.py`
  - `DEFAULTS_R1` += `post_festivo_factor` (bool, False) y `post_festivo_fuente` ("homologo"|"recientes"|"combinado",
    default "combinado"); `validar_params_r1` y `kwargs_r1` los validan/pasan (nivel y forma, mismos valores).
  - Constantes fijas (no params): `ANIOS_POST_FESTIVO = 3`, `RECIENTES_POST_FESTIVO = 4`, `MIN_RATIOS_POST_FESTIVO = 3`,
    `MAX_SEMANAS_POST_FESTIVO = 104`.
  - `info_post_festivo(fuente)` (definición única de las claves) y `_factor_post_festivo(...)`:
    - homólogo: festivo previo = d−1 (por definición de `clase_dia`, un post_festivo es lun–vie con d−1 festivo) →
      `find_homologous_holiday_date(d−1, año)` para los 3 años anteriores → pf = homólogo + 1 si es post_festivo, mismo
      día de semana que d, con curva válida (E > 0) y ≤ corte;
    - recientes: desde `ancla_mismo_dow(d, corte)` hacia atrás de 7 en 7 (hasta 104 semanas) los 4 post_festivo más
      recientes con curva válida;
    - combinado: unión deduplicada por fecha (origen "homologo+reciente" si coincide);
    - N de cada pf: −7, −14, +7, +14 (en ese orden), ≤ corte, `_energia_normal_referencia`; sin N → descartado;
    - F_pf = mediana de E(pf)/E(N); con < 3 ratios → sin factor.
  - `seleccionar_recientes(..., post_festivo_factor=False, post_festivo_fuente="combinado")`: objetivo post_festivo con el
    flag y F_pf → R1 = búsqueda de día normal (`_buscar_con_limpieza`, respeta `r1_limpia`); si no → selección de siempre
    + alerta info `POST_FESTIVO_SIN_FACTOR` (`detalle.motivo`: `pocos_ratios` | `sin_normal_reciente`).
  - `RecientesDia.post_festivo`, `factor_post_festivo`, `diag_post_festivo()` (mismo patrón que festivo).
- `steps/nivel.py`: F = factor_festivo o factor_post_festivo; regla `reciente_factor_post_festivo`; diag por día.
- `steps/forma.py`: con F_pf, la forma es la de R1 normal (camino `reciente_clase` sobre las fechas normales); diag.
- `pipeline._ALERTAS_R1_COMPARTIDAS` += `POST_FESTIVO_SIN_FACTOR`.
- Backtest: conjunto de configs `post_festivo` (default; `jamer` = perfil actual con festivo_por_tipo como base, más
  `jamer+pf_homologo|recientes|combinado`) y conjunto `festivos_r1` (reproducción de la fase anterior con la base sin
  festivo_por_tipo) vía `--conjunto`. Segmento `post_festivo`, semanas de la parte 2 con día post_festivo (columna),
  elección de variante por MAPE h de la parte 2 2022–2024, veredictos solo para la elegida.
- Tests: `tests/test_adjustments_post_festivo.py` + asserts de dicts exactos + tests del script.
- Docs: CONTRATO_BACKEND.md, REPORTE.md de este agente, línea "Implementación" del LOG.

## Decisiones propias
- Años excluidos (2020) no se filtran en las fuentes del factor: igual que `festivo_por_tipo` (la mediana amortigua).
- "Los 4 recientes" son los 4 post-festivos más recientes con datos; si alguno no tiene N, se descarta sin reemplazo.
- Forma con F_pf: fuente `reciente_clase` (las fechas son los días normales de R1).
