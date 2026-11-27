# Mesa de ajustes (perfil jamer): festivos por tipo + referencia R1 limpia

## Contexto

Backtest de 33 semanas (abr–nov 2026, Antioquia): Mesa perfil `jamer` vs envíos VF de Jamer = empate
(MAPE h 2,57 % vs 2,50 %; IC95 de la diferencia incluye 0). La Mesa pierde en dos sitios concretos:

1. **Festivos** (MAPE 5,36 % vs 3,57 %), sobre todo los que no caen lunes: 1/may −14,3 %, 7/ago −5,3 %.
   Causa (código): `seleccionar_recientes` (src/adjustments/referencia_reciente.py:191-261) busca el
   festivo más reciente con el **mismo día de semana** en 12 semanas → el 1/may (vie) toma el Viernes Santo;
   el 8/dic (lun) toma un festivo lunes de puente. El homólogo del año anterior solo es fallback.
2. **R1 contaminada**: para días normales la única exclusión es validez de datos (`_curva_valida`, :111).
   Semana Santa, días atípicos y fines de semana de puente entran como R1 (p. ej. 13/4 tomó el Lunes de
   Pascua; 18/4 −10,9 % vs −2,5 % de Jamer).

No se toca `g` (la calibración no fue estable fuera de muestra → sería overfitting) ni clima (poca data).

Objetivo: dos reglas fijas, detrás de params con default **off**, que solo se activan en el perfil `jamer`
si pasan un criterio de aceptación definido **antes** de ver resultados.

## Dónde se trabaja

- Repo EPM, nueva rama `feat/ajustes-festivos-r1` creada desde `origin/feat/ajustes-replica-jamer`, en el
  worktree `…/scratchpad/epm_jamer` (no se toca el checkout `test` de EPM ni el despliegue de Railway).
- Docs de la fase (convención del módulo): `docs/work/ajustes/festivos_r1/PLAN.md`, `LOG.md` (log de
  sesión resumible, creado apenas se aprueba el plan y actualizado en cada hito), `REPORTE.md`.

## Cambios

### A. Festivos por tipo (`festivo_por_tipo`, bool, default False)

- `src/adjustments/data.py`: `tipo_festivo(d, festivos) -> "lunes" | "semana_santa" | "fijo"`
  (Jue/Vie Santo vía cálculo de Pascua ya existente en `analogos.py:304-306`; "fijo" = festivo no lunes
  fuera de Semana Santa: 1/ene, 1/may, 20/jul, 7/ago, 8/dic si no cae lunes, 25/dic…).
- `referencia_reciente.seleccionar_recientes`: con el flag, para target `festivo`:
  - `lunes` → comportamiento actual (festivo lunes reciente).
  - `fijo` / `semana_santa` → **factor festivo**: R1 = día normal más reciente del mismo día de semana
    (energía actual) y la relación festivo/normal sale del año anterior:
    `E = E(R1_normal)·(1+g) · E(homólogo)/E(normal mismo dow junto al homólogo)`. Homólogo con
    `find_homologous_holiday_date` (src/prediction/holiday_homology.py). Forma = forma normalizada del
    homólogo. Si falta el homólogo → comportamiento actual + alerta info `FESTIVO_SIN_HOMOLOGO`.
  - Se registra en `diagnostics["dias"][f]` (tipo, homólogo, factor) y en `saltados`.
- `steps/nivel.py` (DEFAULTS_RECIENTE + `_separar_reciente`) y `steps/forma.py` (DEFAULTS_JAMER +
  `_separar_jamer`, `_forma_reciente_clase`): aceptar y pasar el flag (patrón de `saltar_puente`).

### B. Referencia limpia (`r1_limpia`, bool, default False)

- En `seleccionar_recientes`, con el flag, para targets no festivos se **salta** un candidato a R1 si:
  - cae en Semana Santa (Domingo de Ramos–Domingo de Pascua) o entre 24/dic y 6/ene; o
  - es atípico: |E(c)/mediana(E de los 4 mismos-dow normales anteriores) − 1| > 6 % (umbral fijo,
    elegido por magnitud física, no ajustado con el backtest; se reporta sensibilidad 4/6/8 % solo
    como información).
  - Motivos en `saltados` (`semana_santa`, `temporada`, `atipico`); si no queda candidato → fallback
    actual + alerta `R1_SIN_LIMPIA`.
- Variante sin código nuevo a evaluar también: `saltar_puente=True` (flag existente, forma.py:374).

### C. Backtest reproducible (`scripts/backtest_mesa_vs_jamer.py`)

- Generaliza el backtest de esta sesión: entradas CSV (histórico real y envíos de Jamer con fecha de
  envío), **sin credenciales en el repo**; corte = envío − 1; base `ingenuo`; sin fuga (assert).
- Configs: `jamer` (actual), `+festivo_por_tipo`, `+r1_limpia`, `+saltar_puente`, `+ambas`.
- Métricas: MAPE h, |err día|, sesgo energía, por tipo de día y por lead; pareado por semana con IC
  bootstrap; partición temporal 60/40.
- **Evaluación ampliada de festivos y semanas con Semana Santa contra real (sin Jamer), 2022–2026**, para
  no decidir con solo 12 festivos.
- Salida: `reports/ajustes/backtest_mesa_vs_jamer_<fecha>.md` (solo agregados).
- Export de datos (fuera del repo, en el scratchpad): script que baja de Jano `actualizaciondatos`,
  `festivos` y `sesiones`/`sesiones_periodos` (solo lectura) a CSV.

### Criterio de aceptación (fijado antes de correr)

Una regla se activa en `perfiles.params_perfil("jamer")` solo si:
1. Mejora MAPE h en su segmento objetivo (festivos / días con R1 saltada) en la muestra ampliada 2022–2026
   **y** en las 13 semanas de prueba;
2. MAPE h global de las 33 semanas no empeora (IC pareado no positivo);
3. Ningún tipo de día empeora > 0,1 pp.
Si no pasa, el código queda disponible con default off y se documenta por qué.

## Ejecución (metodología)

1. Crear rama + `docs/work/ajustes/festivos_r1/{PLAN,LOG}.md`.
2. Subagentes en paralelo:
   - **Agente 1 (selector)**: A + B en data.py / referencia_reciente.py / nivel.py / forma.py + tests
     nuevos (`tests/test_adjustments_festivos_tipo.py`, `tests/test_adjustments_r1_limpia.py`, con
     `tests/ajustes_sinteticos.py` y feriados reales como en `test_adjustments_r1.py`); actualizar asserts
     de dict exacto (`test_adjustments_nivel_jamer.py:43-44`, `test_adjustments_perfiles.py:94`).
   - **Agente 2 (backtest)**: C (script + test de métricas puras al estilo `tests/test_backtest_jamer.py`).
3. Ronda de revisión multi-ángulo (agentes separados, model sonnet) sobre el diff + agente de regresión
   (suite completa, comparando contra las 4 fallas preexistentes conocidas). Repetir ronda completa solo
   si hay hallazgos críticos/altos.
4. Correr el backtest, aplicar el criterio de aceptación, fijar defaults del perfil, `REPORTE.md`.
5. Commit en la rama (sin push ni despliegue; se pregunta antes).

## Verificación

- `<EPM>/.venv/Scripts/python.exe -m pytest tests -q --ignore=tests/test_hourly_disaggregation.py`
  desde el worktree: solo las 4 fallas preexistentes.
- Con flags off, la salida del perfil `jamer` es idéntica a hoy (test de no-regresión + reproducción de la
  corrida de Railway del 23/11: energía idéntica).
- Backtest: tablas por config y veredicto del criterio en `REPORTE.md`; casos 1/may, 7/ago, 13/4 y 18/4
  revisados día a día.

## Enmienda 2026-11-27 (decidida por el usuario tras la ronda 2 de revisión)
- Regla 1 para `festivo_por_tipo` (y la parte festiva de `ambas`): las 13 semanas de prueba solo tienen festivos lunes
  (que la regla no cambia) → la condición "en las semanas de prueba" se reemplaza, SOLO para el segmento festivos, por un
  corte temporal de la muestra ampliada: calibración 2022–2024, prueba 2025–2026 (festivos fijos y Semana Santa).
  El segmento de festivos para la regla 1 = festivos tipo `fijo`/`semana_santa` (los lunes no cambian).
- Regla 2: lectura estricta — diferencia media ≤ 0 Y el IC95 pareado no está entero sobre 0.
