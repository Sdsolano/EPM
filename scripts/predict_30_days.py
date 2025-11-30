"""
Ejemplo de Uso del Pipeline de Predicción
==========================================

Script simple para ejecutar predicción de 30 días
"""

import sys
from pathlib import Path

# Añadir raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction.forecaster import ForecastPipeline

# Ejecutar predicción
pipeline = ForecastPipeline()
predictions = pipeline.predict_next_n_days(n_days=30)
pipeline.save_predictions(predictions)

print("✅ Predicción completada. Revisa predictions/predictions_next_30_days.csv")
