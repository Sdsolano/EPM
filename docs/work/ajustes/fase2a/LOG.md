# Fase 2a — LOG (progreso real)

## Estado actual (coordinador 2, retomado 2026-11-13) — manda sobre las secciones de estado de abajo
- El coordinador anterior fue detenido; los revisores que él lanzó NO cuentan. Este coordinador lanza una ronda 1 NUEVA.
- Hecho: A (pandas3 + SandboxModelo), C (`alpha_por_tipo`), B (backtest modelo k=1/k=4 + recalibración holdout 60/40;
  DEFAULTS NO cambiados; reporte `reports/ajustes/backtest_modelo_20261113.md`; ver B_REPORTE.md).
- En curso: ronda 1 NUEVA de revisión (8 ángulos + regresión, sonnet) contra el diff sin commitear (lanzándose; IDs abajo
  cuando se conozcan).
- Falta: corregir hallazgos; si hay crítico/alto, ronda completa de nuevo; verificación estándar; REPORTE.md de fase;
  commit local (sin push; main avanzó a c94c5d1, el orquestador rebasea).
- Archivos sin commitear: M `src/adjustments/steps/nivel.py`, `src/pipeline/feature_engineering.py`, `tests/test_pipeline.py`;
  nuevos `scripts/backtest_modelo.py`, `scripts/recalibrar_ajustes.py`, `tests/test_backtest_modelo.py`,
  `tests/test_nivel_alpha_por_tipo.py`, `tests/test_pandas3_compat.py`, `tests/test_recalibrar_ajustes.py`, docs fase2a,
  reporte .md. NO commitear CSV/logs de `reports/ajustes/` (curvas_modelo_*, hash_*, progreso_*, *.out, *.csv.gz).
- Verificar: `<venv python> -m pytest tests -q --ignore=tests/test_hourly_disaggregation.py` (esperado 280 passed,
  5 skipped, 1 failed permitido) y `python -c "import src.api.main"`.
- Subagentes vivos: 9 revisores de la ronda 1 (lanzados por el coordinador; esperan nada, solo reportan).
- Siguiente paso exacto: consolidar hallazgos de los 9 revisores en este LOG y corregir.

## Estado actual — agente B (actualizado 2026-11-13 ~08:00) — TERMINADO
- Hecho: todo lo de la parte B. Cache k=1/k=4 (hash intacto), grid + extra evaluados, reporte
  `reports/ajustes/backtest_modelo_20261113.md` (con bloque manual de resumen/decisiones), `B_REPORTE.md`, tests
  (`tests/test_recalibrar_ajustes.py`, 10), suite 280 passed / 5 skipped / 1 failed permitido, `import src.api.main` OK.
- DEFAULTS no cambiados (ninguna candidata cumple la regla en prueba); banda 2.0 se mantiene; `alpha_por_tipo` no
  recomendado. Ver B_REPORTE.md.
- En curso: nada. Falta (coordinador): ronda de revisión y commit. `/reports` y `/docs` están en .gitignore →
  `git add -f` para el reporte .md y los docs de fase si se versionan (CSV/gz NO).
- Archivos sin commitear de B: `scripts/recalibrar_ajustes.py`, `tests/test_recalibrar_ajustes.py`,
  `docs/work/ajustes/fase2a/{B_PLAN.md,B_REPORTE.md,LOG.md}`, `reports/ajustes/backtest_modelo_20261113.md`.
- Regenerar el reporte sin re-correr: `python scripts/recalibrar_ajustes.py recalibrar --reusar-extra`.
- Subagentes vivos: ninguno. Procesos en background: ninguno.

## Estado
- [x] Lectura de reglas, plan maestro, docs Fase 1, backtest v3 y script (coordinador).
- [x] PLAN.md escrito.
- [x] A. pandas3 + proveedor aislado + entrenamiento con corte — hecho (fix ffill; `scripts/backtest_modelo.py` SandboxModelo entrenar/predecir/proveedor; entreno ~15–19 s, pred 8 d ~3–4 s por UCP → k=1 ≈ 30 min; hashes intactos; `tests/test_pipeline.py` ya no escribe data/features; suite 270 passed, 5 skipped, 1 failed permitido). Ver A_REPORTE.md
- [x] C. `alpha_por_tipo` en nivel — hecho (aditivo en nivel.py; 21 tests nuevos; suite 263 passed, 4 skipped, 1 failed permitido). Ver C_REPORTE.md
- [ ] B. backtest modelo + recalibración + reporte — espera a A y C
- [ ] Ronda de revisión (8 ángulos + regresión)
- [ ] Verificación estándar y commit local (sin push)

## Decisiones tomadas sin el usuario
- D1: el proveedor `modelo` aislado se implementa en un script nuevo `scripts/backtest_modelo.py` (sandbox con `chdir` +
  monkeypatch de red/rutas) en vez de modificar `main.py`, para no cambiar la API ni chocar con fase 2b.

## Notas para retomar
- Python: `C:/Users/samue/OneDrive/Documentos/GitHub/EPM/.venv/Scripts/python.exe`, cwd = worktree.

## Hitos del coordinador
- 2026-11-13: A y C reportaron (ver A_REPORTE.md, C_REPORTE.md). Revisado `git status`: data/features restaurado (tracked, limpio).
  Lanzado agente B (backtest modelo k=1 + sensibilidad k=4, recalibración con holdout 60/40, reporte
  `reports/ajustes/backtest_modelo_20261113.md`). Espero su notificación.
- D2 (sin el usuario): k=1 (reentreno semanal) como corrida principal porque cuesta ~30 min (entreno ~17 s/UCP); k=4
  como sensibilidad (más parecido a la operación, donde el champion se reentrena con poca frecuencia).
- 2026-11-13: el agente B (id a8f9f127063692673) murió por límite de uso de la API con `scripts/recalibrar_ajustes.py`
  parcial (197 líneas) y sin cache generado. Tras reiniciarse el límite se reanudó con SendMessage. Espero su notificación.

## Hitos del agente B
- 2026-11-13: B_PLAN.md escrito. `scripts/recalibrar_ajustes.py` (nuevo): subcomando `generar` (cache
  `reports/ajustes/curvas_modelo_k{k}.csv`, reanudable; progreso en `curvas_modelo_k{k}_progreso.log`; hashes en
  `curvas_modelo_k{k}_hash.json`). Semanas: Antioquia 2025-06-30..2026-06-22, Atlántico 2025-05-05..2026-04-27 (52 c/u,
  todas con corte = sábado). Primera corrida k=1 se cortó (límite de API); relanzada. Para retomar:
  `python scripts/recalibrar_ajustes.py generar --k 1` (salta semanas ya cacheadas).
- 2026-11-13 ~04:45: cache ahora por UCP (`curvas_modelo_k{k}_{ucp}.csv`, progreso `progreso_k{k}_{ucp}.log`, hash
  `hash_k{k}_{ucp}.json`) para correr 4 procesos en paralelo (k=1/k=4 × Antioquia/Atlántico). Entreno ~25–80 s por
  semana bajo carga (fase 2b corre tests en paralelo). Subcomandos `evaluar` (grid 35 configs → `backtest_modelo_grid.csv.gz`)
  y `recalibrar` (holdout + `alpha_por_tipo` corrido de verdad → `backtest_modelo_extra.csv.gz` + reporte) escritos;
  `tests/test_recalibrar_ajustes.py` (10 tests, sintéticos) pasan.
- 2026-11-13 ~05:00: **cache k=4 listo** (52 semanas × 2 UCP; ~11 min por UCP; hash de data/raw, models, data/features,
  data/processed intacto: 54 archivos, 0 cambios). k=1 en curso. Prueba en seco de `evaluar`+`recalibrar` con el cache
  k=4 copiado como k=1 en el scratchpad (no en reports/).
- 2026-11-13 ~05:08: prueba en seco OK (evaluar ~5 min con 4 procesos, 245k filas; recalibrar ~1 min). Ajustes al
  reporte (escape de '|' en tablas, secciones 5 y 7 acotadas). Suite completa corriendo; k=1 en semana ~27/52.
- 2026-11-13 ~05:15: **cache k=1 listo** (52 sem × 2 UCP, ~29 min por UCP en paralelo; hash intacto 54 archivos). Corriendo evaluar + recalibrar.
- 2026-11-13 ~07:51: **evaluar + recalibrar listos**; reporte escrito. Resultado: no se cambian DEFAULTS (detalle en
  B_REPORTE.md). B terminado.

## Retoma coordinador nuevo (2026-11-13)
- Retomo desde este LOG. Estado: A+B+C sin commit. Paso: lanzar ronda 1 (8 angulos + regresion, sonnet) contra el diff real.
- Ronda 1 cerrada: 0 criticos/altos. Medio: tabla_md no escapa | con dtype str (pandas3). Bajos: doble validar_params e import duplicado en nivel.py, imports sin uso, docstring/fecha default en recalibrar. Corrigiendo inline + 1 verificacion estandar.
- Correcciones aplicadas; verificacion: 280 passed, 5 skipped, 1 failed (christmas); import API OK. Commit local.
- Aceptados: eficiencia (reevaluar evidencias por config, hash de arboles), refactors cosmeticos en backtest_modelo.py (helpers de red/loggers, proveedor sin uso), fuga de mkdtemp si falla _preparar_arbol, alpha_por_tipo={} distinto de None, monkeypatch idiomatico en test_pipeline.
