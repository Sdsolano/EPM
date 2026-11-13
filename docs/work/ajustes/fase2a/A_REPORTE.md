# Fase 2a — Parte A: reporte (pandas 3 + proveedor `modelo` aislado)

Agente A, 2026-11-13. Sin commits. No se tocó `nivel.py`, `main.py` ni `scripts/backtest_adjustments.py`.

## Archivos
- `src/pipeline/feature_engineering.py:493` — `fillna(method='ffill').fillna(0)` → `.ffill().fillna(0)` (mismo resultado).
- `scripts/backtest_modelo.py` (nuevo) — sandbox + proveedor + hash + CLI de medición.
- `tests/test_pandas3_compat.py` (nuevo, 2 tests) — fija el relleno ffill→0 de `FeatureEngineer.get_feature_importance_ready_df`.
- `tests/test_backtest_modelo.py` (nuevo, 5 rápidos + 1 real opt-in con `EPM_TEST_MODELO_REAL=1`, ~40 s).
- `tests/test_pipeline.py` (modificado) — ahora escribe sus salidas en un `TemporaryDirectory` (ver "Hallazgo").

## Incompatibilidades pandas 3 (3.0.6)
Solo una en la ruta `run_predict_flow`: `fillna(method=)` en `feature_engineering.py:493` (TypeError). Con el fix, el
flujo completo (pipeline de features, `train_model_if_needed` con XGBoost/LightGBM/RF, desagregador horario,
`ForecastPipeline`, incluido el bloque de "predicción de prueba" de 30 días de `run_predict_flow`) corre con
`-W default` sin errores ni FutureWarning/DeprecationWarning de pandas (no hay `filterwarnings` en `src`). Las demás
coincidencias de grep (`inplace=True` sobre DataFrames completos, lecturas `iloc[-1][col]`) son válidas en pandas 3.

**Hallazgo**: `tests/test_pipeline.py` corre `run_automated_pipeline` sin `output_dir` y traga excepciones. Con pandas 3
fallaba en silencio (por eso "pasaba"); con el fix pasaba de verdad y **reescribía el archivo versionado
`data/features/data_with_features_latest.csv`** y `data/processed/power_clean_latest.csv`. Se restauró el archivo
(`git checkout`), se borró el procesado (la carpeta estaba vacía) y el test ahora usa un directorio temporal
(+ `orchestrator.PROCESSED_DATA_DIR` parcheado y restaurado). Verificado: hash de `data/raw`, `models`, `data/features`,
`data/processed` idéntico antes/después de la suite completa.

## Sandbox (`SandboxModelo`)
```python
# sys.path.insert(0, "scripts")  (o importlib, como tests/test_backtest_modelo.py)
from backtest_modelo import SandboxModelo, hash_arbol, diferencias_hash
with SandboxModelo(ucps=["Antioquia", "Atlantico"], usar_flujo_api=False, silenciar=True) as sb:
    curvas = sb.entrenar(ucp, corte, n_days=8)     # force_retrain=True; devuelve Curves de corte+1..corte+n
    curvas = sb.predecir(ucp, corte, n_days=8)     # force_retrain=False con los modelos del sandbox
    prov   = sb.proveedor(ucp, modelo_corte=None)  # provider(ucp, end_date, n_days) -> Curves (BaseProvider)
    sb.modelo_corte[ucp]; sb.tiempos               # corte de los modelos vigentes; registro de tiempos
```
- `__enter__`: importa `src.api.main` (y los módulos hourly/monitoring/update_csv), crea `tempfile.mkdtemp`
  (`epm_sandbox_modelo_*`) con `data/raw/{ucp}/{datos,clima_new}.csv` (+ copia `.completo`), `config/`, `models/{ucp}`
  vacío, `data/features`, `data/processed`, `logs`. Todo se deshace al salir (ExitStack, LIFO) y el tempdir se borra.
- `os.chdir` al sandbox (rutas relativas de `main.py`).
- Rutas absolutas: toda constante `BASE_DIR/DATA_DIR/LOGS_DIR/MODELS_DIR/CONFIG_DIR/REPORTS_DIR/RAW_DATA_DIR/
  PROCESSED_DATA_DIR/FEATURES_DATA_DIR` de cualquier módulo cargado del proyecto (`src.*` y los alias `config`,
  `config.settings` que crean los `sys.path.append`) que apunte al repo → misma ruta en el sandbox (43 atributos, p. ej.
  `src.config.settings.*`, `orchestrator.*`, `hourly_disaggregator/special_days/disaggregation_engine.MODELS_DIR`,
  `monitoring.LOGS_DIR`, `adjustments.data.RAW_DATA_DIR`). Los `FileHandler` de logging que escriben fuera del sandbox
  se quitan temporalmente; los creados dentro se cierran al salir (en Windows impedían borrar el tempdir).
- `full_update_csv` / `full_update_csv_diario` → no-op en todo módulo que los tenga (`src.api.main`, `update_csv`).
- Festivos: `FestivosAPIClient.get_festivos` → `festivos_offline` (librería `holidays` vía
  `adjustments.data.festivos_colombia`); cubre orchestrator, forecaster, `hourly/calendar_utils` y router.
- Red: `requests.get/post/put/delete/request` y `Session.request` lanzan `RedBloqueadaError`; `socket.connect`,
  `connect_ex` y `create_connection` también, salvo loopback (asyncio en Windows usa socketpair local). Testeado.
- **No-fuga**: antes de cada llamada `preparar_corte` reescribe `datos.csv` del sandbox con `FECHA <= corte` (filtro sobre
  el texto, celdas intactas) y además se pasa `end_date=corte`. `predecir` rechaza modelos con corte posterior al pedido.
  `_a_curvas` exige que las fechas devueltas sean exactamente corte+1..corte+n_days. Test real: multiplicar ×3 los datos
  posteriores al corte no cambia ninguna curva (tolerancia 1e-6).
- Modos: `usar_flujo_api=False` (default) llama los pasos 1–4 de `run_predict_flow` con los mismos argumentos, sin su
  predicción de prueba de 30 días (que solo calcula `should_retrain`); `True` llama `asyncio.run(run_predict_flow(...))`.
  Verificado: mismas curvas (dif. máx. 0.005 = redondeo a 2 decimales de la API). El reentreno es determinista
  (dos entrenamientos con el mismo corte → mismas curvas).

## Tiempos medidos (datos del worktree, este PC)
| UCP | corte | entrenar (+pred 8 d) directo / API | predecir 8 d directo / API |
|---|---|---|---|
| Antioquia | 2026-05-09 | 16–19 s / 20 s | 3.4–3.8 s / 6.7 s |
| Atlántico | 2026-03-07 | 14 s / 15 s | 3.2–4.3 s / 6.2 s |
| Antioquia | 2025-06-21 | 14 s | 3.2 s |
| Atlántico | 2025-04-26 | 14 s | 2.9 s |
(+ ~5 s una vez por proceso para importar `src.api.main`). WAPE horario de sanidad: Antioquia 1.9–3.4 %, Atlántico 3.4–6.7 %.

## k recomendado
**k = 1 (reentreno semanal)**: `entrenar` ya devuelve las curvas de su propio corte, así que cada semana cuesta ~15–19 s →
52 sem × 2 UCP ≈ 104 × 17 s ≈ **30 min** (muy por debajo del presupuesto de 2–3 h). Es la variante sin "modelo viejo".
Opcional como sensibilidad (más parecido a operación, donde el modelo no se reentrena cada semana): k = 4
(26 entrenos + 78 predicciones ≈ 26×17 + 78×4 ≈ 12 min). No hace falta reducir el algoritmo (RandomForest entrena
rápido: ~3.4 k filas diarias).

## Verificación por hash (sha256, 54 archivos de `data/raw`, `models`, `data/features`, `data/processed` del worktree)
Intacto (0 nuevos/borrados/modificados) tras entrenamiento + predicción reales de Antioquia y Atlántico en 4 corridas
(directo y `--flujo-api`, cortes recientes y de 2025), tras los tests (incluido el real) y tras la suite completa.

## Suite
`pytest tests -q --ignore=tests/test_hourly_disaggregation.py` → **270 passed, 5 skipped, 1 failed**
(`test_christmas_profile_uses_prior_christmases`, permitido). `import src.api.main` OK.

## Limitaciones y riesgos
- Clima: el futuro usa clima OBSERVADO (optimista). `Antioquia/clima_new.csv` solo cubre 2025-12-01..2026-07-20: para
  cortes anteriores el entrenamiento tiene features de clima en 0 y el pronóstico usa promedios históricos (fallback de
  `ForecastPipeline`), igual que haría producción con esos archivos. Atlántico tiene clima desde 2018.
- Festivos: `holidays` (Colombia) en lugar de la API externa; si la API incluye días especiales por UCP, puede diferir.
- El desagregador horario se entrena con los ~87 días recientes (comportamiento del código de producción).
- Monkeypatch global en el proceso: no usar dos sandboxes a la vez ni hilos; módulos importados por primera vez dentro
  del sandbox conservarían rutas del sandbox (se precargan todos los del flujo para evitarlo).
- `silenciar=True` oculta stdout y logging < ERROR del flujo durante las llamadas.

## Decisiones tomadas sin el usuario
- Modo directo por defecto (mismas funciones/argumentos que `run_predict_flow`, sin la predicción de prueba de 30 días):
  verificado equivalente y ~2× más rápido en predicción.
- Arreglar `tests/test_pipeline.py` para que no escriba en el repo (antes solo "pasaba" porque fallaba en silencio).
