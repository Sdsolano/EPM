# Fase 2a — backtest con el modelo real y recalibración (sub-plan)

Coordinador: agente de fase 2a. Worktree `C:/Users/samue/OneDrive/Documentos/GitHub/EPM-wt-2a`, rama
`feat/ajustes-fase2a` (desde main 74bc100). Reglas: `C:/Users/samue/OneDrive/Documentos/GitHub/EPM/docs/work/ajustes/REGLAS_AGENTES.md`.
Plan maestro: `.../EPM/docs/work/ajustes/PLAN_NOCHE.md`. En paralelo trabaja fase 2b (`feat/ajustes-fase2b`,
análogos: toca `forma.py`, `nivel.py`, `scripts/backtest_adjustments.py`) → aquí cambios ACOTADOS y ADITIVOS en
esos archivos (funciones/params/opciones nuevas, sin reordenar ni reformatear).

## Objetivos (en orden)
1. pandas 3: `src/pipeline/feature_engineering.py:493` `fillna(method='ffill')` → `.ffill().fillna(0)` y cualquier otra
   incompatibilidad con pandas 3.0.6 en la ruta `run_predict_flow` (feature pipeline, entrenamiento, ForecastPipeline,
   desagregación). Test que fije el comportamiento.
2. Proveedor `modelo` aislado: sandbox temporal (copias de `data/raw/{ucp}`, `config/`, modelos si aplica), `chdir` al
   sandbox, `full_update_csv` no-op, festivos offline (`holidays`), constantes de ruta absolutas
   (`src.config.settings`, `orchestrator.PROCESSED_DATA_DIR`, etc.) redirigidas. Verificación por hash de
   `data/raw`, `models/`, `data/features`, `data/processed` del worktree antes/después. Sin tocar `main.py` (si hace falta
   costura: mínima y sin cambio de comportamiento de la API).
3. Sin fuga de entrenamiento: modelos propios entrenados en el sandbox con datos <= corte; rolling origin (reentreno cada
   k semanas, k según tiempo medido; documentar). Limitación: clima observado.
4. Backtest base `modelo` ~52 semanas por UCP (Antioquia hasta 2026-07-04; Atlántico hasta ~2026-05): WAPE horario,
   energía diaria/semanal (MAPE y sesgo), pico (MW y hora), por UCP y tipo de día, por paso (base/nivel/forma) y config.
5. Recalibración con holdout temporal (calibrar en semanas antiguas, reportar en las recientes): `alpha_por_tipo`
   (param opcional retrocompatible del paso nivel: laboral/sabado/domingo/festivo), fuente de forma y `n`,
   `banda_referencia_pp` del control (precision/recall). Cambiar DEFAULTS solo si el holdout mejora consistentemente en
   ambas UCP (o por UCP justificado). Comparar contra bases `ingenuo` y `archivo`.
6. Reporte `reports/ajustes/backtest_modelo_<AAAAMMDD>.md` (CSV no se commitean) + recomendación de qué pasos pasan a
   automático por segmento (R2).

## División en partes (subagentes)
| Parte | Dueño | Archivos | Terminado cuando |
|---|---|---|---|
| A. pandas3 + proveedor aislado + entrenamiento con corte | agente A | `src/pipeline/feature_engineering.py` (y otros del flujo si fallan con pandas 3), `scripts/backtest_modelo.py` (nuevo: sandbox + proveedor), `tests/test_pandas3_compat.py`, `tests/test_backtest_modelo.py` | `run_predict_flow` corre en sandbox para Antioquia y Atlántico con corte arbitrario y modelos entrenados <= corte; hashes intactos; tiempos medidos (entreno y predicción) en `fase2a/A_REPORTE.md`; suite sin fallos nuevos |
| C. `alpha_por_tipo` en nivel | agente C | `src/adjustments/steps/nivel.py` (aditivo), tests | param opcional `alpha_por_tipo: {laboral,sabado,domingo,festivo}` (claves parciales; ausente = `alpha`), validado, sin cambio con default; tests; suite ok |
| B. backtest modelo + recalibración + reporte | agente B (tras A y C) | `scripts/backtest_modelo.py` (extender), opciones aditivas mínimas en `scripts/backtest_adjustments.py` si hace falta, `reports/ajustes/backtest_modelo_<fecha>.md` | backtest 52 sem × 2 UCP con base modelo sin fuga; holdout; tablas; recomendación R2; defaults cambiados solo con evidencia |
| Revisión | coordinador | — | 8 ángulos + regresión (sonnet) limpios |

## Diseño del backtest con modelo (acordado)
- Corte = sábado anterior a la semana objetivo (igual que backtest v3). Predicción `n_days` = 8 (domingo brecha + semana).
- Rolling origin: en cada origen (cada k semanas) se entrena en el sandbox con `force_retrain=True` y `end_date` = corte de
  ese origen (modelos de demanda diaria + desagregador horario); las semanas siguientes del bloque usan esos modelos
  (`force_retrain=False`) con su propio corte (los datos de features sí se recortan a cada corte; solo el modelo queda
  "viejo" hasta k semanas, como en operación).
- Las curvas base por semana se cachean a disco (CSV/parquet en el scratchpad o `reports/ajustes/`, no commiteado) para que
  la recalibración no reentrene.
- Holdout: semanas ordenadas por fecha; calibración = primeras ~60 % (≈31), prueba = últimas ~40 % (≈21) por UCP.

## Decisiones tomadas sin el usuario
(ver LOG.md)
