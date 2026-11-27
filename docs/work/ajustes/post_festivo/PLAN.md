# Mesa (perfil jamer): factor post-festivo con homólogo del año anterior

## Contexto

El 3/nov/2026 (martes después de Todos los Santos) la Mesa erró −5,5 % (Jamer −1,0 %). Causa: para un día
`post_festivo`, `seleccionar_recientes` (src/adjustments/referencia_reciente.py) copia el post-festivo más reciente del
mismo día de semana (13/oct, que fue atípicamente bajo: 0,944 de su martes normal). Ni `festivo_por_tipo` ni
`r1_limpia` lo cubren (`r1_limpia` solo juzga candidatos de clase normal).

El usuario propone mirar el mismo post-festivo del año anterior (homólogo). Datos (post-festivo / martes normal vecino):
- Todos los Santos: 2023 1,028 · 2024 0,948 · 2025 0,998 · **2026 real 0,992** · lo que usó la Mesa (13/oct) 0,944.
- Día de la Raza: 2023 0,988 · 2024 1,021 · 2025 0,965 · 2026 0,944.
Un solo caso es ruidoso (2024 0,948) → usar la **mediana** de varios casos, no uno solo.

## Regla nueva: `post_festivo_factor` (bool, default **off**)

Para un día objetivo de clase `post_festivo`, con el flag:
- **R1** = día **normal** más reciente del mismo día de semana (misma búsqueda que un día normal; respeta `r1_limpia`).
- **Energía** = E(R1)·(1+g)·F_pf, con g por el lag d − R1 (como `festivo_por_tipo`).
- **F_pf** = mediana de ratios E(post-festivo)/E(día normal vecino del mismo día de semana, búsqueda −7, −14, +7, +14,
  fuera de temporada/Semana Santa, ≤ corte), tomados de una fuente fija por variante (param `post_festivo_fuente`):
  - `"homologo"`: el día posterior al **mismo festivo** en los 3 años anteriores (homólogo con
    `find_homologous_holiday_date`, src/prediction/holiday_homology.py, + 1 día; mismo día de semana).
  - `"recientes"`: los 4 post-festivos más recientes del mismo día de semana (≤ corte, hasta 2 años atrás).
  - `"combinado"` (default del param): unión de ambos conjuntos, mediana.
  - Mínimo 3 ratios; si no, comportamiento actual + alerta info `POST_FESTIVO_SIN_FACTOR`.
- **Forma**: la del R1 normal (con delta plano), sin cambio de forma → solo cambia el nivel. (Mantiene el diseño
  simple; la forma de un post-festivo es casi la de un día normal.)
- Diagnóstico por día: fuente, fechas usadas, ratios, F_pf. Constantes 3 años / 4 recientes / mínimo 3: fijas, no se
  calibran.
- Reusar: `seleccionar_recientes`/`_buscar`, `_energia_normal_referencia`, `info_festivo`/`RecientesDia.diag_festivo`
  (patrón de `festivo_por_tipo`), `validar_params_r1`/`kwargs_r1` y `DEFAULTS_R1` (nivel y forma), alertas en
  `pipeline._ALERTAS_R1_COMPARTIDAS`.

## Backtest y criterio (fijado antes de correr)

`scripts/backtest_mesa_vs_jamer.py`: configs nuevas sobre el perfil `jamer` actual (que ya trae `festivo_por_tipo`):
`jamer+pf_homologo`, `jamer+pf_recientes`, `jamer+pf_combinado`. Segmento = días `post_festivo`.
- **Elección de variante**: la de menor MAPE h en la parte ampliada de **calibración** (2022–2024). Solo esa se evalúa
  para activar (las otras dos son informativas) → no se elige mirando la prueba.
- **Regla 1**: mejora (diferencia de MAPE h < 0) en el segmento en ampliada 2022–2024 **y** 2025–2026 (las 13 semanas de
  prueba del PLAN tienen solo 3 post-festivos; se reportan como información).
- **Regla 2** (estricta): pareado global 33 semanas, media ≤ 0 y IC95 no entero > 0.
- **Regla 3**: ningún tipo de día empeora > 0,1 pp.
- Si pasa: se activa en `perfiles.params_perfil("jamer")` (nivel y forma). Si no: queda apagada y se documenta.
- Después: regenerar los CSV del tablero (incluyendo la serie nueva si se activa) y actualizar el tablero y el HTML.

## Dónde y cómo

- Rama `feat/ajustes-festivos-r1` (worktree `…/scratchpad/epm_jamer`), nuevo commit; sin push.
- Docs: `docs/work/ajustes/post_festivo/{PLAN,LOG,REPORTE}.md` (LOG creado al aprobar).
- Metodología: 1 agente de implementación (selector + nivel/forma + tests + configs del backtest; mismos archivos →
  un solo agente) → ronda de revisión completa (ángulos en paralelo con sonnet + regresión); repetir solo si hay
  hallazgos críticos/altos → backtest → decisión → commit.

## Verificación

- Suite completa (`<python> -m pytest tests -q --ignore=tests/test_hourly_disaggregation.py`): solo la falla preexistente.
- Flag off ⇒ salida idéntica (tests `test_flags_off_*` + uno nuevo para el flag).
- Caso 3/nov/2026: con el flag, F_pf y el error del día revisados a mano (homólogo 2025 = 0,998).
- Backtest con veredicto por regla en `REPORTE.md`.
