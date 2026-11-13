# Fase 2a — Parte A: pandas 3 + proveedor `modelo` aislado (sub-plan)

Dueño: agente A. Worktree `EPM-wt-2a`, rama `feat/ajustes-fase2a`. Sin commits. No tocar `nivel.py`; `main.py` solo
costura mínima sin cambio de comportamiento (objetivo: no tocarlo).

## Pasos
1. Reproducir `run_predict_flow` offline con pandas 3.0.6 y listar cada fallo (fillna(method=), freq 'H', etc.).
   Arreglar solo lo que falle/avise, sin cambiar resultados. Test `tests/test_pandas3_compat.py` (relleno ffill→0).
2. `scripts/backtest_modelo.py` (nuevo):
   - `hash_arbol(dirs)` sha256 por archivo; `SandboxModelo` (context manager): tempdir con copia de
     `data/raw/{ucp}/{datos,clima_new}.csv` y `config/`; `chdir`; monkeypatch de `full_update_csv*` (no-op),
     `FestivosAPIClient.get_festivos` (calendario `holidays` Colombia), `requests` (bloqueo: lanza),
     constantes de ruta (`src.config.settings.*`, `orchestrator.*`, `monitoring.LOGS_DIR`, módulos hourly) → sandbox.
   - Recorte del `datos.csv` del sandbox a `<= corte` antes de cada llamada (no-fuga explícita); clima completo
     (limitación: clima observado como "pronóstico").
   - `entrenar(ucp, corte)` (force_retrain=True), `predecir(ucp, corte, n_days) -> Curves`,
     `proveedor(ucp, modelo_corte)` compatible con `pipeline.BaseProvider`.
3. Medir tiempos reales (entreno con corte y predicción de 8 días por UCP) → k recomendado.
4. Hash del worktree antes/después de entreno+predicción reales (Antioquia y Atlántico).
5. `tests/test_backtest_modelo.py` (rápidos, sin red; el entreno real con `EPM_TEST_MODELO_REAL=1`).
6. Suite completa + `import src.api.main`. `A_REPORTE.md` y línea en `LOG.md`.
