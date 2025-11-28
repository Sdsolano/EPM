from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
from fastapi import APIRouter
from pathlib import Path
import pandas as pd
from src.models.trainer import ModelTrainer
from src.models.metrics import calculate_all_metrics, evaluate_model_performance
from src.prediction.hourly import HourlyDisaggregationEngine
from src.prediction import ForecastPipeline
# Importa tu pipeline REAL
from src.pipeline.orchestrator import run_automated_pipeline

app = FastAPI()

# Modelo de entrada EXACTO según tu función
class PipelineRequest(BaseModel):
    power_data_path: str
    weather_data_path: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    output_dir: Optional[str] = None  # lo convertimos después a Path

@app.post("/run-pipeline")
async def run_pipeline(req: PipelineRequest):

    # Convertir output_dir a Path si viene
    output_dir = Path(req.output_dir) if req.output_dir else None

    # LLAMADA EXACTA A TU MÉTODO, SIN TOCARLO
    df, report=run_automated_pipeline(
        power_data_path=req.power_data_path,
        weather_data_path=req.weather_data_path,
        start_date=req.start_date,
        end_date=req.end_date,
        output_dir=output_dir
    )
    print(df.columns)
    exclude_cols = ['FECHA', 'TOTAL'] + [f'P{i}' for i in range(1, 25)]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].fillna(0)
    y = df['TOTAL'].copy()

    # Eliminar NaN en y
    mask = ~y.isnull()
    X = X[mask]
    y = y[mask]

    # Split temporal
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n  Train: {len(X_train)} registros")
    print(f"  Test: {len(X_test)} registros")
    print(f"  Features: {len(feature_cols)}")

    # Crear entrenador
    trainer = ModelTrainer(
        optimize_hyperparams=False,  # Deshabilitado para testing rápido
        cv_splits=3
    )

    # Entrenar todos los modelos
    trained_models = trainer.train_all_models(
        X_train, y_train,
        X_test, y_test,
        models=['xgboost', 'lightgbm', 'randomforest']
    )

    # Seleccionar mejor modelo
    best_name, best_model, best_results = trainer.select_best_model(
        criterion='rmape',
        use_validation=True
    )

    # Evaluar en test
    y_test_pred = best_model.predict(X_test)
    test_metrics = calculate_all_metrics(y_test, y_test_pred)

    print(f"\n{'='*70}")
    print(f"EVALUACIÓN EN TEST SET")
    print(f"{'='*70}")
    print(f"  MAPE: {test_metrics['mape']:.4f}%")
    print(f"  rMAPE: {test_metrics['rmape']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.2f}")
    print(f"  R²: {test_metrics['r2']:.4f}")

    # Guardar modelos
    saved_paths = trainer.save_all_models(timestamp='test')



    engine = HourlyDisaggregationEngine(auto_load=True)

    # Verificar estado
    status = engine.get_engine_status()
    print(f"\n✓ Modelos guardados:")
    for name, path in saved_paths.items():
        print(f"    {name}: {path}")
    if not (status['normal_disaggregator']['fitted'] and status['special_disaggregator']['fitted']):
        print("\n🔧 Entrenando sistema...")
        engine.train_all(n_clusters_normal=35, n_clusters_special=15, save=True)
    result = engine.predict_hourly(req.start_date, 12000)
    # Devuelves lo que quieres tú
    print(f"\n{'='*70}")
    print(result)
    return {"message": "Pipeline ejecutado correctamente"}

# {
#   "power_data_path": "/Users/pablo/Documents/GitHub/EPM/data/processed/power_clean_20251114_064158.csv",
#   "weather_data_path": "/Users/pablo/Documents/GitHub/EPM/data/processed/weather_clean_20251114_064158.csv",
#   "start_date": "2023-06-01",
#   "end_date": "2024-01-15",
#   "output_dir": "string"
# }
@app.post("/predict")
def train_models_endpoint():
    """
    Entrena los modelos usando ModelTrainer.
    No devuelve métricas por ahora, solo confirma que corrió.
    """
    try:
        pipeline = ForecastPipeline(
        model_path='models/trained/xgboost_20251125_115900.joblib',
        historical_data_path='data/features/data_with_features_latest.csv',
        festivos_path='data/calendario_festivos.json'
    )

    # Predecir próximos 30 días
        predictions = pipeline.predict_next_n_days(n_days=30)

        # Guardar resultados
        pipeline.save_predictions(predictions)

        print(predictions)

    except Exception as e:
        return {"error": str(e)}
