# LOG — factor post-festivo (perfil jamer)

Leer primero `PLAN.md` (aprobado por el usuario 2026-11-27). Fase previa: `../festivos_r1/` (commit d19d25f).

## Contexto para retomar
- `<worktree>` = worktree del repo EPM en la rama `feat/ajustes-festivos-r1` (scratchpad de la sesión, carpeta `epm_jamer`).
- `<python>` = `.venv/Scripts/python.exe` del checkout EPM en JM_Datalabs; correr con cwd = `<worktree>`.
- Datos del backtest (fuera del repo): `<scratchpad>/datos_bt/` (historico_antioquia.csv, festivos_antioquia.csv, envios_jamer.csv).
- Backtest: `scripts/backtest_mesa_vs_jamer.py` (ver `../festivos_r1/REPORTE.md`).

## Decisiones confirmadas con el usuario
| # | Decisión | Por qué |
|---|---|---|
| 1 | Mirar el post-festivo del mismo festivo en años anteriores (homólogo) | propuesta del usuario tras el 3/nov |
| 2 | Mediana de varios casos, 3 variantes (homólogo/recientes/combinado), variante elegida solo con calibración 2022–2024 | evitar overfitting |
| 3 | Sin push ni despliegue sin preguntar | reglas del usuario |

## Progreso real
- [x] Plan aprobado, docs creados (2026-11-27)
- [x] Implementación (agente) + tests (2026-11-27): flag `post_festivo_factor` + `post_festivo_fuente` (default off/combinado), alerta `POST_FESTIVO_SIN_FACTOR`, backtest `--conjunto post_festivo` (default) con elección de variante por calibración 2022–2024; sub-plan en `agente/PLAN.md` (el REPORTE del agente quedó en su respuesta final: el harness bloquea escribir reportes). Caso 3/nov/2026: homologo F_pf 0,998 → +0,77 % (jamer −5,51 %); recientes 0,956 → −3,45 %; combinado 0,957 → −3,36 %.
- [ ] Ronda de revisión
- [ ] Backtest + decisión
- [ ] Tablero, REPORTE, commit
| 4 | Revisión: solo 1 agente verificador + 1 agente de regresión (sin multi-ángulo) | pedido explícito del usuario 2026-11-27 |
- 2026-11-27: Backtest completo (171 s). Variante elegida en calibración: pf_homologo (−0,11 pp, 45 d 2022–2024).
  Prueba 2025–2026: +0,43 pp (28 d) → NO PASA regla 1. Regla 2 PASA (−0,025 pp, IC95 [−0,077; +0,017]); regla 3 PASA.
  Informativo: post-festivos de las 13 semanas de prueba (incl. 3/nov) −1,46 pp (3 d). Reporte: reports/ajustes/backtest_post_festivo_20261127.md.
  Pendiente: verificador + regresión; decisión: no activar.
- 2026-11-27: Verificador y regresión sin hallazgos altos (bajos aceptados, ver REPORTE). Decisión: no activar. Commit sin push.
