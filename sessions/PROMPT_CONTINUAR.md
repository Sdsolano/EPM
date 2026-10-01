# Prompt completo para continuar el trabajo (pronóstico diario, modelo de nivel)

Este archivo es el prompt de relevo. Para el registro de decisiones y el estado ver
`sessions/2026-09-30_pronostico-diario-nivel.md` (en esta misma carpeta).

---

Eres el agente que RETOMA un trabajo a medias. Nadie te va a hacer preguntas ni puede responderlas: el usuario
(Samuel) está durmiendo y pidió "nonstop, sin preguntarme, hasta que quede muy bien, sin ninguna regresión".
Decide con criterio, documenta y termina. Responde y comenta en español.

## 1. Qué pidió el usuario
Un compañero subió a https://github.com/Sdsolano/EPM la rama `test` (pronósticos DIARIOS, endpoint
POST /predict-daily; el reentreno incorpora clima min/media/máx y días después de festivos; el resto igual a los
otros modelos). Pidió REVISARLA Y MEJORARLA sin introducir NINGUNA regresión. La BD real es la API de producción
(https://pronosticos.jmdatalabs.co). Solo existe demanda diaria para el UCP "Atlantico Norte" (desde 2025-01-01,
~608 días, termina 2026-08-31). Hay un Excel de respaldo en
C:\Users\samue\Downloads\DATOS DIARIOS DE TERRITORIOS PARA PROYECCION DE DEMANDA (1).xlsx (mismos datos; la API manda).

## 2. Dónde está todo (Windows, PowerShell/Git Bash)
- Repo principal (NO TOCAR; árbol sucio del usuario con cambios sin commitear en main):
  C:\Users\samue\OneDrive\Documentos\GitHub\EPM
- Worktree con el TRABAJO: C:\Users\samue\OneDrive\Documentos\GitHub\EPM-test (rama local `test-mejora`, basada en
  origin/test, SIN COMMITEAR todavía).
- Worktree ORIGINAL para comparar: C:\Users\samue\OneDrive\Documentos\GitHub\EPM-base (detached en origin/test;
  debe estar limpio).
- Scratchpad (scripts de backtest y datos):
  C:\Users\samue\AppData\Local\Temp\claude\C--Users-samue-OneDrive-Documentos-GitHub-EPM\31527af0-b25f-4c62-8f57-c0292da5f92a\scratchpad
  - bt3.py = backtest real. Uso desde la raíz de un worktree: `python <scratchpad>\bt3.py <base|new> <WL>`
    (WL = días de clima disponibles tras el corte: 10 = realista, 30 = clima perfecto). Usa feat_base.csv y
    data/raw/Atlantico Norte/clima_new.csv. 7 cortes x 30 días.
  - proto.py/p2..p4.py = prototipos de investigación; e2e_r*.py = simulaciones end-to-end de run_predict_daily_flow.
- Datos gitignored que los scripts necesitan en CADA worktree: data/raw/Atlantico Norte/{datos_diario.csv,clima_new.csv}.
  Si faltan: `cp -r "C:\Users\samue\OneDrive\Documentos\GitHub\EPM-base\data\raw\Atlantico Norte" <worktree>\data\raw\`
  (o bajarlos con `full_update_csv_diario('Atlantico Norte')`).

## 3. Diagnóstico (hallazgos de la investigación)
1. El modelo original entrenaba árboles sobre TOTAL crudo: no extrapolan nivel/tendencia (la demanda crece ~12%
   interanual). El MAPE a 30 días con clima real era 3.90% SOLO gracias a un ajuste hardcodeado "+5.3% desde
   2026-03-25" en ForecastPipeline.predict_next_n_days, pensado para el mercado Atlantico (no Atlantico Norte).
   Sin ese ajuste: 6.74%.
2. Skew train/predict del clima: en train `temp_lag1d` era el clima del día anterior; en predict se pasaba el del
   mismo día (y rain_lag1d usaba rain_mean vs rain_sum).
3. Feed de clima corrupto en la API: desde 2026-03, ~2/3 de las filas tienen p_h == p_t (humedad copiada de la
   temperatura); solo los periodos múltiplos de 3 traen lectura real. Además p_i NO son mm sino códigos de condición
   (200-599 lluvia, 800 despejado, 801-804 nubes). Esto también afecta a Atlantico (~10% de filas) y Antioquia (~72%)
   en sus flujos HORARIOS: NO se arregló (fuera de alcance / riesgo de regresión); hay que REPORTARLO al usuario.
4. Con 600 días no hay historia suficiente para aprender efectos finos (Semana Santa, festivos) con árboles crudos.

## 4. Solución implementada (en EPM-test, sin commitear)
Archivos: src/models/daily_level_model.py (NUEVO), src/api/main.py, src/prediction/forecaster.py,
tests/test_daily_level_model.py (NUEVO), README.md, docs/MODELO_DIARIO_NIVEL.md (nuevo; /docs está en .gitignore,
hay que `git add -f`), y esta carpeta sessions/.
- Modelo de NIVEL: target = TOTAL_t / ancla28(origen) (media de 28 días previos al origen sin festivos ni Navidad
  23-dic..6-ene) + features m7/m14 (momentum) + horizonte h (clip 1..30). Ensamble promedio LightGBM +
  ExtraTrees(n_jobs=min(4,cpu)) + Ridge. Features de calendario (dow, mes, festivo, post_fest, post2, pre_fest,
  puente, navidad, Semana Santa sem_santa/santo, etc.) + clima diario limpio (tmean/tmin/tmax, himean/himax, wind,
  rain, cloud). Clima real solo hasta h=12 (WEATHER_KNOWN_HORIZON); después climatología mensual (igual en train,
  validación y predicción). Cobertura de clima por columna < 0.5 → esa columna se descarta (si cae tmean, todo el clima).
- load_daily_weather limpia la humedad corrupta SOLO en el camino diario (no toca src/pipeline/connectors.py: es
  compartido con los flujos horarios).
- Mezcla de "días muy especiales" (12-24, 12-25, 12-08, 01-01, 01-02), Navidad y Semana Santa con el valor de hace
  1 año alineado a Pascua y escalado por crecimiento interanual (yoy_growth_factor, clip [0.8,1.3]); pesos 0.70 / 0.60;
  festivos ordinarios NO se mezclan (DAILY_BLEND_ORDINARY_FESTIVOS=False: empeoraba). Las constantes DAILY_* son
  perillas de negocio: déjalas.
- Despacho en ForecastPipeline por model_kind == 'daily_level_ratio_v1'; camino legacy y flujos horarios intactos.
  Sin el +5.3%.
- train_model_if_needed(variant='daily') con >=120 filas usa el modelo nuevo; archivos:
  models/{ucp}/trained_diario/daily_level.joblib, registry/champion_model_diario.joblib, .meta.json (sidecar con
  model_kind/size/columnas_clima...), .legacy.joblib (respaldo del champion legacy), .corrupt.joblib (champion
  ilegible). Migra automáticamente champions legacy/ilegibles una vez. Escrituras atómicas (tmp + os.replace), lock
  por UCP (threading, solo de proceso) con fast-path sin lock para champion de nivel vigente, backoff en memoria de
  10 min tras fallo de migración, fallback a legacy (bajo el mismo lock) si <120 filas o excepción; force_retrain
  salta el backoff; si force_retrain falla con champion de nivel vigente se conserva.
- train_model_if_needed ahora es un ENVOLTORIO y el cuerpo original se movió a _train_model_legacy (hourly delega sin
  cambios). ESTO ES LO MÁS IMPORTANTE A VERIFICAR: que hourly y legacy no cambiaron ni una línea de lógica y que NO
  hay deadlock (threading.Lock no reentrante: el fallback legacy corre bajo el lock desde _daily_level_flow;
  comprobar que _train_model_legacy no vuelve a tomarlo).
- run_predict_daily_flow: entrenamiento en run_in_threadpool; ForecastPipeline se construye aparte (error de carga →
  500 "champion ilegible, reintente con force_retrain=true"; ValueError de predict → 400); temp de features único por
  request (uuid) y borrado en finally; metricas_modelo sanitizada (inf/nan → None) y con claves aditivas
  columnas_clima, cobertura_clima, cobertura_clima_por_columna, columnas_clima_descartadas (comparacion_modelos ahora
  lista lightgbm/extratrees/ridge en vez de xgboost/lightgbm/randomforest: documentado).
- forecaster.py: import tolerante del módulo nuevo (try relativo → agrega raíz del repo a sys.path y reimporta como
  src.models → literal 'daily_level_ratio_v1' + RuntimeError claro solo si se usa sin el módulo). Corrigió una
  regresión real: `import prediction.forecaster` con solo `src` en sys.path debe seguir funcionando igual que en HEAD.

## 5. Resultados (backtest real bt3.py, 7 cortes x 30 días, Atlantico Norte)
| | MAPE | sesgo | festivos |
|---|---|---|---|
| código original, clima hasta h=10 | 4.704 | -3.32 | 5.39 |
| NUEVO, WL=10 (realista) | 3.053 | -0.38 | 4.23 |
| original, WL=30 | 3.816 | -1.84 | 5.08 |
| NUEVO, WL=30 | 3.062 | -0.31 | 4.23 |

Semana Santa 2026 (corte 2026-03-31): Jueves/Viernes Santo error 3.1% / 2.8% (6.5% / 8.4% sin tratamiento). Navidad
NO es validable aún (falta 1 año de datos). Estas cifras (WL10 ≈ 3.053, WL30 ≈ 3.062) deben mantenerse tras
cualquier cambio: si se mueven más de ~0.03pp, investiga.

Tests: `python -m pytest tests -q -p no:cacheprovider --ignore=tests/test_hourly_disaggregation.py` → 46 passed
(3 corridas seguidas). tests/test_hourly_disaggregation.py tiene un ERROR DE COLECCIÓN PREEXISTENTE (import
relativo) idéntico en EPM-base: no cuenta como regresión. tests/test_pipeline.py modifica el archivo trackeado
data/features/data_with_features_latest.csv: tras correrlo SIEMPRE
`git checkout -- data/features/data_with_features_latest.csv`.

## 6. Metodología OBLIGATORIA (del CLAUDE.md global del usuario) y estado
La revisión NO es una sola pasada inline. Cada ronda = estos agentes SEPARADOS en paralelo (Agent tool,
general-purpose, SOLO LECTURA, prompts autocontenidos) contra el diff COMPLETO actual (`git diff HEAD` en EPM-test;
los archivos nuevos ya están con intent-to-add: `git add -f -N` si hace falta):
1. línea por línea (correctitud)
2. comportamiento eliminado/cambiado (compara con EPM-base)
3. trazador cross-file (e2e run_predict_daily_flow con asyncio.run + PredictRequest, mockeando
   full_update_csv_diario a no-op; los festivos por red funcionan)
4. reuso
5. simplificación
6. eficiencia
7. altitud/abstracción
8. convenciones (incluye pruebas reales de import en los modos `src.prediction.forecaster` y `prediction.forecaster`
   con sys.path=[src], desde raíz y desde src/, comparando contra `git archive` de HEAD)
9. AGENTE DE REGRESIÓN: suite completa, pyflakes (sin warnings nuevos vs EPM-base en main.py/forecaster.py), camino
   horario idéntico (ForecastPipeline.predict_next_n_days con models/Atlantico/registry/champion_model.joblib +
   data/features/Atlantico, 14 días, assert_frame_equal entre ambos worktrees), train_model_if_needed hourly/legacy
   idéntico, bt3.py base 10 / new 10 / new 30, e2e (force_retrain, reutilizar, n_days 7/30/45, offset_scalar,
   end_date, champion legacy sembrado migra una vez, ilegible se reentrena, dos llamadas concurrentes entrenan una
   vez, fallo forzado → backoff, <120 filas cae a legacy SIN deadlock, contrato de respuesta idéntico a EPM-base
   salvo claves aditivas).

Si CUALQUIER agente devuelve un hallazgo CONFIRMED o PLAUSIBLE: corregirlo (tú o un agente implementador) y REPETIR
LA RONDA COMPLETA (los 9 agentes, diff completo, sin reducir ángulos ni alcance) hasta que una ronda vuelva con CERO
hallazgos de todos los ángulos y regresión confirme que nada se rompió.

Hallazgos ya decididos como aceptados/documentados (no los repitas ni los "arregles"): sin política de reentrenamiento
automático (solo force_retrain), lock solo de proceso, MAPE reportado no incluye la mezcla de días especiales,
backoff en memoria, n_days>30 con ancla congelada (h saturado a 30), flags DAILY_*, constante literal duplicada en
el import tolerante, reuso de la lista de días especiales del legacy (no se toca el legacy).

ESTADO: Rondas 1, 2, 3 y 4 COMPLETADAS con sus correcciones aplicadas. La ronda 5 se lanzó pero los 9 agentes
FALLARON por límite de sesión (HTTP 429; se reinicia a las 6:20pm America/Bogota): NO hay resultados de la ronda 5.
Tu primera tarea es REPETIR LA RONDA 5 COMPLETA. Para no chocar con el rate limit lanza los agentes en tandas más
chicas (3-4 a la vez) y, si ves 429, espera y reintenta (no los des por buenos). Los revisores en paralelo interfieren
entre sí (crean/borran models/Atlantico Norte, data/raw/Atlantico Norte): pídeles trabajar en una COPIA aislada del
worktree cuando ejecuten e2e, y que limpien lo que generen. Pídeles también verificar especialmente el refactor
envoltorio/_train_model_legacy (diff línea a línea contra EPM-base) y la ausencia de deadlock del lock.

## 7. Al terminar (solo cuando una ronda completa salga limpia)
- `git status` de EPM-test debe mostrar SOLO: src/api/main.py, src/models/daily_level_model.py,
  src/prediction/forecaster.py, tests/test_daily_level_model.py, README.md, sessions/ (+ docs/MODELO_DIARIO_NIVEL.md
  con `git add -f`). Nada de datos/modelos/CSV/logs. La regla .gitignore `models/` ignora src/models/: confirma que el
  archivo nuevo entra (`git add -f src/models/daily_level_model.py`). Borra artefactos (models/Atlantico Norte,
  data/features/Atlantico Norte, UCPs temporales tipo zz_bt*/ZZTest) y restaura data_with_features_latest.csv.
- COMMITEA en la rama local `test-mejora` (commit NUEVO, sin --amend, sin force, sin --no-verify). Mensaje en español,
  claro, terminando con la línea: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`
- NO hagas push ni abras PR sin confirmación del usuario (la rama remota `test` es de su equipo). Déjalo listo para
  que el usuario haga `git push origin test-mejora` (o fusionar en `test`) al despertar.
- Actualiza el log `sessions/2026-09-30_pronostico-diario-nivel.md` en cada hito (cada ronda cerrada, cada hallazgo
  corregido, antes y después del commit).
- Informe final AL USUARIO en español (corto, sin jerga): qué estaba mal, qué se cambió, resultados (MAPE 4.70% →
  3.05% con clima realista; sin el hack del +5.3%), cómo se verificó (rondas de revisión multi-ángulo + regresión,
  tests, backtest, e2e), riesgos/decisiones que debe conocer (champions diarios existentes se migran solos una vez al
  primer request y se respaldan; ese primer request entrena ~5-30 s; la humedad corrupta del feed también afecta a
  Atlantico y Antioquia en los flujos horarios y NO se arregló: decisión suya; Navidad sin validar aún; el lock es solo
  de proceso; cambio de claves en comparacion_modelos; historial corto <120 días usa el modelo legacy) y los comandos
  para empujar la rama.

## 8. Reglas de seguridad
No toques el repo principal ni su árbol sucio. No borres ni reescribas historial. No publiques nada externo. No hagas
acciones destructivas sin verificar el objetivo. Informa fielmente: si algo falla, dilo con la salida. Si te quedas sin
presupuesto, deja el estado escrito (qué ronda, qué hallazgos pendientes) en el log de sesión para que otro continúe.
