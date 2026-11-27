# Agente 1 — REPORTE (festivo_por_tipo + r1_limpia)

Sin commit. Flags default off; NO activados en `perfiles.params_perfil` (lo decide el backtest).

## Diseño
- **F viaja en la selección**: `seleccionar_recientes(..., festivo_por_tipo, r1_limpia, umbral_atipico)` devuelve
  `RecientesDia.festivo` = {tipo_festivo, homologo, normal_homologo, E_homologo, E_normal_homologo, factor} y la
  propiedad `factor_festivo`. Con el factor aplicado, `fechas` son días NORMALES recientes (misma búsqueda que un
  objetivo normal, limpiada si `r1_limpia`). nivel: `E_obj = E(R1)·(1+g)·F` (g con lag d − R1 normal; regla
  `reciente_factor_festivo`); forma (`reciente_clase`): forma normalizada de H (fuente `festivo_homologo_tipo`) con
  la mecánica de delta plano de siempre. Una sola definición para ambos pasos.
- H = `find_homologous_holiday_date(d, d.year−1)` con datos y <= corte; N_H = H−7, H+7, H−14, H+14 (normal, válido,
  <= corte). Falta H, N_H o R1 normal → selección de siempre + alerta info `FESTIVO_SIN_HOMOLOGO` (detalle.motivo).
- `r1_limpia`: candidatos de la clase pedida que caen en Semana Santa (`semana_santa`), 24/dic–6/ene (`temporada`)
  o atípicos (`atipico`: |E(c)/mediana(hasta 4 normales previos del mismo dow, buscados hasta 8 semanas atrás)−1|
  > `UMBRAL_ATIPICO`=0.06, param `r1_umbral_atipico` en (0, 0.5]; < 3 previos → no se juzga) se saltan, también los
  extra de n=2. Sin ninguno limpio → búsqueda sin limpieza + alerta info `R1_SIN_LIMPIA` (detalle.descartados).
- Params: `referencia_reciente.DEFAULTS_R1` / `validar_params_r1` / `kwargs_r1` (compartidos), incluidos en
  `nivel.DEFAULTS_RECIENTE` y `forma.DEFAULTS_JAMER` (se persisten en modo reciente / reciente_clase como el resto).
- `pipeline._ALERTAS_R1_COMPARTIDAS` += `FESTIVO_SIN_HOMOLOGO`, `R1_SIN_LIMPIA` (forma no las repite).
- Pascua: `data.domingos_semana_santa` / `data.es_semana_santa` (movido desde `analogos`, que ahora lo reutiliza;
  `analogos.es_semana_santa` sigue existiendo como alias). Nuevo `data.tipo_festivo`.

## Decisiones tomadas
1. El criterio "atípico" solo se aplica a candidatos de clase normal (se mide contra días normales; un post-festivo
   o puente sería "atípico" por construcción). Semana Santa/temporada sí aplican a toda clase no festiva.
2. "Fallback actual" de `r1_limpia` = la selección sin limpieza (no el fallback de otra clase), con alerta.
3. N_H no se filtra por Semana Santa/temporada (el plan solo pide normal válido).
4. Diagnósticos nuevos (`tipo_festivo`, `homologo`, `normal_homologo`, `factor_festivo`) solo aparecen con el flag
   activo y objetivo festivo → con flags off curvas, diagnósticos y alertas son idénticos.

## Archivos
src/adjustments/{data.py, analogos.py, referencia_reciente.py, pipeline.py, steps/nivel.py, steps/forma.py};
tests nuevos `tests/test_adjustments_festivos_tipo.py` (35), `tests/test_adjustments_r1_limpia.py` (27);
`tests/test_adjustments_nivel_jamer.py` (dict exacto actualizado).

## Verificación
- No-regresión: snapshot del código previo (8 semanas × 2 perfiles, curvas+diagnósticos+alertas de todos los pasos)
  idéntico tras el cambio; test `test_flags_off_identico_a_sin_claves` + `test_flags_off_igual_al_snapshot`.
- Suite completa: `1030 passed, 16 skipped, 1 failed` (solo `test_christmas_profile_uses_prior_christmases`,
  preexistente).

## Pendiente / riesgos
- `steps[].params` de nivel/forma en modo réplica ahora incluye 3 claves nuevas (con su default): records viejos sin
  ellas siguen funcionando (se completan con default).
- `docs/work/ajustes/fase3/CONTRATO_BACKEND.md` no lista los params nuevos (no se tocó).
