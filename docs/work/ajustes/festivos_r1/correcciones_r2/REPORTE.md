# Correcciones de la ronda 2 de revisión (festivos_r1)
Sin commit. Flags default off; con flags off salida idéntica (test_flags_off_* y snapshot pasan). Incluye la Enmienda 2026-11-27.

| # | Hallazgo | Corrección | Test |
|---|---|---|---|
| A | Regla 1 de festivos no evaluable en las 13 sem de prueba; regla 2 ambigua | segmento `festivo_tipo` (fijo/semana_santa); `partes_regla1`: festivos = parte 2 por año de la fecha objetivo 2022–2024 / 2025–2026; R1 = parte 2 + prueba parte 1; `ambas` = ambas mitades (mitad R1 sin festivos fijo/SS). `veredicto_reglas(partes, par, tipos)` con días por parte; regla 2 estricta comentada; `nota_festivos_prueba` calculada | test_veredicto_*, test_partes_regla1_*, test_nota_festivos_prueba_calculada, test_resumen_y_segmentos, test_main_sintetico |
| B | Homólogo lunes-puente para festivo entre semana | descarte `lunes_puente`; `anios_probados=[{anio,homologo,descarte}]`; motivo `homologo_lunes_puente` | 8/dic/2026→2023; 1/may/2024 no toma 2023; solo-lunes-puente |
| C | Temporada/SS excluida aunque el objetivo esté en el periodo | `_motivo_no_limpio(c, d, …)` | 31/dic/2025→24/dic; 14/ene/2026→17/dic; Pascua acepta Domingo de Ramos |
| D | Lunes trasladado en SS salía semana_santa | lunes primero; semana_santa = Pascua −3/−2 | 2024-03-25, 2027-03-22 = lunes |
| E | motivo mezclaba texto | códigos snake_case + TEXTO_MOTIVO_SIN_HOMOLOGO completo; `info_festivo()` único | test_motivos_son_codigos_con_texto_completo |
| F | Contrato desactualizado | ΣR1·(1+g)·F, regla reciente_factor_festivo, R1 normal, motivos saltados, homólogo 3 años/mismo tipo/no lunes-puente, códigos, mismos flags nivel/forma | — |
| G | Docstrings "año anterior" | "hasta 3 años atrás, mismo tipo de día" | — |
| H | .gitignore | regla reports/ajustes/jamer/ + comentario; check-ignore OK; add -n -f *.md solo el md; límite: add -f de la carpeta lo añade igual (agregar .md por ruta explícita) | — |
| I | Rutas personales en LOG | placeholders | — |
| J | sys.path global en test | importlib spec_from_file_location | — |

Verificación: suite 1053 passed / 16 skipped / 1 failed preexistente; --help cp1252 OK; humo parte 1 (3 sem) + parte 2 desde 2026-08-01 OK (12 s); backtest completo NO corrido.
