# Correcciones de la ronda 1 de revisión (festivos_r1)

Sin commit. Flags siguen default off; con flags off la salida es idéntica (tests `test_flags_off_*` pasan).

| # | Sev. | Hallazgo | Corrección | Test |
|---|---|---|---|---|
| 1 | alta | Backtest usaba 48 semanas por defecto (PLAN: 33, 20/13) | `--desde` default `2026-04-06` (`DESDE_PLAN`); `--muestra-informativa` (+ `--desde-informativa`, default 2025-12-01) añade la sección "Parte 1 INFORMATIVO — no decide". Una sola corrida cubre ambas muestras; veredictos solo con la del PLAN. CSV de detalle con columna `muestra_plan` | `test_con_tramo_60_40_plan` (33 -> 20/13), `test_main_sintetico` (PLAN 2 sem vs informativa 4 sem), `test_help_no_falla_en_cp1252` |
| 2 | alta | Homólogo H podía caer en otro tipo de día | `_homologo_con_normal`: H en año-1..-3 (`ANIOS_HOMOLOGO=3`), el más reciente <= corte, curva válida y E>0, del mismo tipo de día (laboral/sábado/domingo). Ninguno -> selección de siempre + `FESTIVO_SIN_HOMOLOGO` con `detalle.motivo="sin_homologo_mismo_tipo_dia"`. `anios_probados` en `festivo`, en la alerta y en el diagnóstico (`homologo_anios_probados`) | `test_homologo_del_mismo_tipo_de_dia_mas_reciente` (8/dic/2020 -> 2017), `test_sin_homologo_del_mismo_tipo_de_dia_fallback_con_alerta` (20/jul/2025 dom), `test_pipeline_alerta_sin_homologo_una_sola_vez` (ajustado) |
| 3 | alta | Mediana de `_es_atipico` incluía Semana Santa / 24-dic–6-ene | `_energia_normal_referencia` (clase normal, fuera de SS y temporada, E>0) para los previos | `test_enero_mediana_sin_temporada` (escenario de enero: R1 = 14/ene) |
| 4 | media | N_H sin filtro temporada/SS; orden -7,+7,-14,+14 | N_H usa `_energia_normal_referencia`; orden `(-7, -14, 7, 14)` | `test_primero_de_enero_normal_homologo_antes_y_fuera_de_temporada`, `test_normal_homologo_orden_antes_primero` |
| 5 | media | Script con copia propia de la regla R1 limpia | Borrados `motivo_r1_limpia`, `_energia`, `UMBRAL_ATIPICO`, `N_PREVIOS_ATIPICO`. Segmento `r1_saltada` = días cuya R1 de nivel (`dias[f]["r1"]`) en la corrida de la config difiere de la de `jamer` (`_r1_distinta`). Columna informativa `r1_saltada`/`motivo_r1` desde la corrida real de `jamer+r1_limpia`. Parte 2: `r1_cambia_con_limpia` llama `seleccionar_recientes` con `r1_limpia` False/True | `test_r1_cambia_con_limpia_usa_la_regla_de_produccion`, `test_filas_dia_r1_saltada_desde_la_corrida_real`, `test_resumen_y_segmentos` |
| 6 | media | VF evaluaba días <= corte | VF solo días > corte; descartes por semana (`descartes_vf`) en el reporte y en consola | `test_parte1_vf_solo_dias_posteriores_al_corte` |
| 7 | media | Parte 2 KeyError sin semanas | `semanas_ampliadas` devuelve vacío con columnas; `md_parte2` -> "sin datos" | `test_semanas_ampliadas_sin_semanas_no_falla` |
| 8 | media | forma caía a la forma de los normales si H sin forma | H exige E(H)>0 y curva válida => factor implica forma de H; sin fall-through en `_forma_reciente_clase` | `test_homologo_con_energia_cero_no_da_factor`, `test_forma_usa_forma_del_homologo` |
| 9 | media | `--help` UnicodeEncodeError (cp1252) | Script sin U+2212 / flechas / unión (todo representable en cp1252) + `_salida_tolerante()` | `test_help_no_falla_en_cp1252`; `--help` con `PYTHONIOENCODING=cp1252` exit 0 |
| 10 | media | CONTRATO_BACKEND sin params nuevos | Documentados `festivo_por_tipo`, `r1_limpia`, `r1_umbral_atipico` (fijo, no calibrar), diagnósticos y alertas `FESTIVO_SIN_HOMOLOGO`, `R1_SIN_LIMPIA` | — |
| 11 | baja | Duplicaciones en el script | `data.tipo_festivo` directo; `es_semana_santa`, `PERIODOS`, `dias_semana` de `data`; `partir_calibracion` de `backtest_jamer` (import sin efectos; >=1 semana por lado); `metricas_dia` -> `errores_dia` | tests del script |
| 12 | baja | Helpers de test duplicados | `curvas_planas`, `seleccionar`, `fechas` en `tests/ajustes_sinteticos.py` | — |
| 13 | baja | Claves de diagnóstico festivo duplicadas | `RecientesDia.diag_festivo()` usado por nivel y forma (se borró `forma._diag_festivo`) | tests nivel/forma |
| 14 | baja | Sin anotaciones de retorno | `domingos_semana_santa -> Tuple[date, date]`, `_buscar`, `_buscar_con_limpieza` | — |

## Archivos
`src/adjustments/{referencia_reciente.py, data.py, steps/nivel.py, steps/forma.py}`, `scripts/backtest_mesa_vs_jamer.py`,
`tests/{ajustes_sinteticos.py, test_adjustments_festivos_tipo.py, test_adjustments_r1_limpia.py, test_backtest_mesa_vs_jamer.py}`,
`docs/work/ajustes/fase3/CONTRATO_BACKEND.md`.

## Verificación
- Suite completa: ver LOG (Hitos).
- `--help` con consola cp1252: exit 0.
- Humo parte 1 (`--desde 2026-11-02 --partes 1 --muestra-informativa --desde-informativa 2026-10-26`, salida al scratchpad):
  3 semanas PLAN + 4 informativas, 9 s, 0 días VF descartados; secciones y veredictos se generan.
  NO se corrió el backtest completo (lo corre el orquestador tras la ronda 2).

## Notas / decisiones
- El filtro temporada/SS de N_H casi nunca se activa con el calendario real (-7 del 1/ene es el 25/dic, festivo); se prueba
  con un set de festivos sin el 25/dic.
- Si el H más reciente del mismo tipo de día no tiene N_H, no se prueba otro año (motivo "día normal junto al homólogo").
- `r1_saltada` y `r1_cambia` comparten máscara (R1 de nivel distinta de `jamer`); se conservan ambos nombres por la descripción.
