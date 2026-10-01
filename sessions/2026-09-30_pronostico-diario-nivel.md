# Sesión 2026-09-30 — Revisión y mejora de la rama `test` (pronóstico diario)

Snapshot resumible: otro agente debe poder continuar leyendo SOLO este archivo + `PROMPT_CONTINUAR.md`
(misma carpeta) + lo que ya está en el código. Worktree de trabajo: `EPM-test` (rama local `test-mejora`).

## Contexto / tarea
Samuel pidió revisar y mejorar la rama `test` del repo EPM (pronósticos DIARIOS, `/predict-daily`; reentreno con clima
min/media/máx y días después de festivos), sin introducir ninguna regresión, trabajando sin preguntarle (dormía).
Datos: API de producción; solo hay demanda diaria de "Atlantico Norte" (608 días, 2025-01-01 a 2026-08-31).

## Decisiones ya tomadas (no relitigar sin información nueva)
| # | Decisión | Por qué |
|---|---|---|
| D1 | Trabajar en worktree aparte (`EPM-test`), no tocar `main` del usuario | El árbol de `main` tiene cambios sin commitear del usuario |
| D2 | Saltar el modo plan formal | El usuario pidió explícitamente no preguntar; un plan bloquearía esperando aprobación |
| D3 | Modelo de NIVEL (target = TOTAL/ancla 28d) en módulo nuevo, despachado por `model_kind` | Los árboles sobre TOTAL crudo no extrapolan la tendencia (+12% interanual); el +5.3% hardcodeado era de otro mercado |
| D4 | NO tocar `connectors.py`/`feature_engineering.py` ni flujos horarios | Son compartidos con Atlantico/Antioquia; cero regresión |
| D5 | Limpiar la humedad corrupta (p_h==p_t) solo en `load_daily_weather` (camino diario) | El feed está corrupto desde 2026-03 (~2/3 de filas); arreglarlo en el conector cambiaría los modelos horarios |
| D6 | p_i se trata como código de condición (lluvia = 200–599), no mm | Verificado en los datos (500/800/804...) |
| D7 | Clima real solo hasta h=12; después climatología (también en train/validación) | En producción el API trae observado + ~12 días de pronóstico |
| D8 | Festivos ordinarios NO se mezclan con el valor de hace 1 año (flag False) | Empeoraba el MAPE de festivos (6.36 → 7.87) |
| D9 | Mezcla solo en días muy especiales/Navidad/Semana Santa, alineada a Pascua y escalada por crecimiento interanual (clip 0.8–1.3) | Paridad con la regla de negocio legacy, coherente con la premisa de nivel |
| D10 | Features de Pascua (sem_santa, santo) en el modelo | Jueves/Viernes Santo pasaron de 6.5%/8.4% a 3.1%/2.8% de error |
| D11 | Migración automática del champion legacy (respaldo .legacy, sidecar .meta.json, escritura atómica, lock por UCP, fast-path, backoff 10 min) y del ilegible (.corrupt) | Si no, los UCP con champion diario existente nunca verían la mejora |
| D12 | `train_model_if_needed` = envoltorio; cuerpo original a `_train_model_legacy` | Permite tomar el lock en el modo daily sin alterar hourly |
| D13 | Import tolerante en `forecaster.py` | Un revisor confirmó regresión: `import prediction.forecaster` con solo `src` en el path dejó de funcionar |
| D14 | No arreglar la humedad corrupta de Atlantico (~10%) / Antioquia (~72%) en flujos horarios | Fuera de alcance/riesgo; se reporta al usuario |
| D15 | No hacer push ni abrir PR | La rama remota `test` es del equipo; decisión del usuario |
| D16 | Aceptados/documentados: sin reentrenamiento automático, lock solo de proceso, MAPE sin mezcla, backoff en memoria, n_days>30 con ancla congelada, flags DAILY_* | Ver docs/MODELO_DIARIO_NIVEL.md |

## Resultados (backtest real `bt3.py`, 7 cortes × 30 días)
- Código original, clima hasta h=10: MAPE 4.704 (sesgo −3.32, festivos 5.39).
- NUEVO WL=10: **3.053** (sesgo −0.38, festivos 4.23). NUEVO WL=30: **3.062**. Original WL=30: 3.816.
- Sin el ajuste +5.3%, el original daba 6.74% con clima perfecto.
- Tests: 46 passed (3 corridas) con `--ignore=tests/test_hourly_disaggregation.py` (error de colección PREEXISTENTE).

## Progreso real hasta ahora
- [x] Investigación y prototipo (scratchpad: proto.py, p2–p4.py, bt.py/bt3.py).
- [x] Implementación inicial (agente implementador) + backtest.
- [x] Ronda 1 de revisión (8 ángulos + regresión): hallazgos → Semana Santa, migración del champion, import, n_jobs, robustez.
- [x] Ronda 2: hallazgos de robustez (excepciones en migración, escritura atómica/lock, sidecar, cobertura de clima).
- [x] Ronda 3: regresión de import confirmada (corregida), fast-path sin lock, backoff, champion ilegible, métricas.
- [x] Ronda 4: sin defectos CONFIRMED; PLAUSIBLE corregidos (flake, ilegible reentrenable, cobertura por columna,
      lock en legacy diario, temp único, log de mezcla, docs). Regresión 8/8 PASA antes de esas últimas correcciones.
- [x] **Ronda 5 (tandas 1–3)**: 0 CONFIRMED; 2 PLAUSIBLE (eficiencia: payload ExtraTrees 43.8 de 44.1 MB,
      recargado por request; altitud: literal `120` en main.py duplicaba `MIN_DAYS` del modelo) + NITs.
      Sin deadlock; refactor envoltorio/`_train_model_legacy` byte-idéntico al original de EPM-base.
      El agente de regresión quedó INTERRUMPIDO por una pausa de sesión (sin veredicto).
      -> CORREGIDO: ExtraTrees `n_estimators` 300->100 (payload 44.1 -> 14.9 MB; backtest idéntico 3.053/3.064);
         main.py importa `MIN_DAYS as DAILY_LEVEL_MIN_DAYS`; import sin uso eliminado del test; `_finite_or_none`
         cubre np.integer/np.bool_; `clean_demand` ordena antes de deduplicar; condición redundante en la mezcla.
         Smoke: 46 passed; pyflakes sin warnings nuevos; bt3 new10 = 3.053 (recuperado).
- [x] **Ronda 6 (tanda A: 1,2,3,9)**: 1 PLAUSIBLE — mi fix de `clean_demand` ordenaba con quicksort NO estable,
      haciendo no determinista el `keep='last'` para fechas duplicadas (reproducido por el agente). Resto: CERO.
      Regresión PASA (46 passed x2; backtest base10 4.704 / new10 3.053 / new30 3.064; horario idéntico;
      sin deadlock; contrato OK). -> CORREGIDO con `sort_index(kind='stable')` (verificado: conserva la última
      ocurrencia del input de forma determinista).
- [x] **Ronda 7 (completa, 9 ángulos)**: 2 PLAUSIBLE — (a) eficiencia: el clima new.csv se parseaba 2 veces por
      request (~200 ms evitables; `load_daily_weather` + el connector del forecaster); (b) altitud: nombres de
      artefactos del modelo hardcodeados en `main.py` podían desincronizarse de `check_model_exists`.
      -> CORREGIDO: caché de `load_daily_weather` por ruta+mtime+tamaño; `ExtraTrees.set_params(n_jobs=1)` tras
         el fit (predict ~3x más rápido); `_champion_path(ucp, variant)` como única fuente del nombre del champion
         (usado por `check_model_exists`, `_train_daily_level_model` y `.corrupt`). Smoke: 46 passed; pyflakes OK;
         bt3 3.053/3.064; payload 14.92 MB.
- [x] **Ronda 8 (completa, 9 ángulos)**: CERO CONFIRMED/PLAUSIBLE. Regresión PASA (46 passed ×2; base10 4.704 /
      new10 3.053 / new30 3.064; horario idéntico; sin deadlock; contrato OK; caché de clima y `_champion_path`
      verificados). Solo NITs (p.ej. `_train_model_legacy` no limpia el sidecar `.meta.json` al escribir el champion
      diario; nombres aún literales en el camino legacy).
- [x] **CAMBIO DE FOCO (pedido del usuario)**: no más rondas por NITs. Objetivo: subir MAPE/calidad del modelo diario.
      Metodología: backtest extendido de 23 cortes cada 20 días (2025-05..2026-07, 690 obs) como criterio robusto,
      además del 7-corte documentado. Experimentos probados (todos con impacto medido):
        * pesos de ensamble NNLS: mejora el 7-corte (3.053->3.027) pero EMPEORA el extendido (4.241->4.329) => descartado.
        * ponderación temporal (recencia) half-life 60/90/120/180: ídem, sobreajusta el 7-corte y empeora el extendido => descartado.
        * features de día de mes (dom/Fourier), rezagos estacionales lag7/14/28, tendencia 28d, climatología,
          todos los horizontes 1..30, más capacidad LightGBM, ancla mediana, anomalías de clima: todas empeoran => descartadas.
        * **ANCLA: ventana 21 días** (antes 28): MEJORA en AMBOS backtests de forma consistente.
- [x] **Cambio final aplicado**: `ANCHOR_DAYS = 21` (antes 28) en `daily_level_model.py`; variables/docstrings/docs
      actualizadas (`a28` -> "ancla"). Backtest 7-corte: **2.880 (WL10) / 2.867 (WL30)** (antes 3.053/3.064; original
      4.704). Extendido 23 cortes: 4.132 (28d) -> vs 4.241. Tests 46 passed; pyflakes limpio. Se descarta NNLS/recencia
      por no generalizar (sobreajuste al 7-corte).
- [x] Limpieza de artefactos y commit local en `test-mejora`: **172307c** (8 archivos; `git add -f` para
      `src/models/daily_level_model.py` y `docs/MODELO_DIARIO_NIVEL.md`). Sin push, sin amend, sin force.
      Working tree limpio; EPM-base en 705e649; repo principal `EPM` sin tocar.
- [x] Informe final entregado al usuario.

## Resultado final
- Modelo diario de nivel. MAPE backtest 7-corte: **2.880 (WL10) / 2.867 (WL30)** (original 4.704). Sin el hack +5.3%.
- Commit: `test-mejora` @ 172307c. Empujar: `git -C EPM-test push origin test-mejora` (o fusionar en `test`).

## Riesgos abiertos / a informar al usuario
- Humedad corrupta del feed afecta también flujos horarios de Atlantico y Antioquia (no tocado).
- Navidad sin validar (falta 1 año de datos de Atlantico Norte; se valida en dic-2026).
- El primer `/predict-daily` de cada mercado con champion legacy entrena/migra (~5–30 s).
- `comparacion_modelos` cambia de miembros (lightgbm/extratrees/ridge).
- <120 días de historia usa el camino legacy (con el +5.3% hardcodeado).

## Notas operativas (gotchas)
- `tests/test_pipeline.py` modifica `data/features/data_with_features_latest.csv` (trackeado): `git checkout --` después.
- `data/raw/Atlantico Norte/` está gitignored; copiarlo entre worktrees con `cp -r` desde `EPM-base`.
- `.gitignore` ignora `models/` (incluye `src/models/`) y `/docs`: usar `git add -f`.
- Revisores en paralelo se pisan entre sí (models/Atlantico Norte, data/raw): usar copias aisladas para e2e.
- Festivos: la API de producción responde por red; funciona.
