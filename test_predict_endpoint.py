"""
Script de prueba para el endpoint /predict
Prueba la integración completa: connectors -> feature engineering -> training -> prediction
"""
import sys
from pathlib import Path

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from src.api.main import PredictRequest, predict_demand
import asyncio

async def test_predict():
    """Prueba el endpoint de predicción"""
    print("=" * 80)
    print("PROBANDO ENDPOINT /PREDICT - INTEGRACIÓN COMPLETA")
    print("=" * 80)

    # Crear request
    request = PredictRequest(
        power_data_path="data/raw/datos.csv",
        weather_data_path="data/raw/clima.csv",
        start_date="2017-01-01",  # Alinear con datos disponibles
        n_days=30,
        force_retrain=False  # Usar modelo existente si lo hay
    )

    print(f"\nParametros de prediccion:")
    print(f"   - Datos de potencia: {request.power_data_path}")
    print(f"   - Datos de clima: {request.weather_data_path}")
    print(f"   - Fecha inicio: {request.start_date}")
    print(f"   - Dias a predecir: {request.n_days}")
    print(f"   - Forzar reentrenamiento: {request.force_retrain}")

    print("\n" + "=" * 80)
    print("EJECUTANDO PIPELINE...")
    print("=" * 80)

    try:
        # Llamar al endpoint
        response = await predict_demand(request)

        print("\n" + "=" * 80)
        print("PREDICCION EXITOSA")
        print("=" * 80)

        print(f"\nResultados:")
        print(f"   - Predicciones generadas: {len(response.predictions)} dias")
        if hasattr(response, 'model_used'):
            print(f"   - Modelo usado: {response.model_used}")
        if hasattr(response, 'model_path'):
            print(f"   - Ruta del modelo: {response.model_path}")

        if hasattr(response, 'training_metrics') and response.training_metrics:
            print(f"\nMetricas de entrenamiento:")
            for key, value in response.training_metrics.items():
                if isinstance(value, float):
                    print(f"   - {key}: {value:.4f}")
                else:
                    print(f"   - {key}: {value}")

        print(f"\nMuestra de predicciones (primeros 3 dias):")
        for i, pred in enumerate(response.predictions[:3], 1):
            # Convert Pydantic object to dict
            pred_dict = pred.dict() if hasattr(pred, 'dict') else pred
            print(f"\n   Dia {i} ({pred_dict['fecha']}) - {pred_dict.get('dia_semana', 'N/A')}:")
            print(f"      Total diario: {pred_dict['demanda_total']:.2f} MWh")
            print(f"      Festivo: {pred_dict.get('is_festivo', False)}, Fin de semana: {pred_dict.get('is_weekend', False)}")
            print(f"      Horas (P1-P5): {pred_dict['P1']:.2f}, {pred_dict['P2']:.2f}, {pred_dict['P3']:.2f}, {pred_dict['P4']:.2f}, {pred_dict['P5']:.2f}...")

        # Verificar estructura
        if response.predictions:
            first_pred = response.predictions[0]
            first_pred_dict = first_pred.dict() if hasattr(first_pred, 'dict') else first_pred
            hourly_keys = [f'P{i}' for i in range(1, 25)]
            missing_keys = [k for k in hourly_keys if k not in first_pred_dict]
            if missing_keys:
                print(f"\nADVERTENCIA: Faltan claves horarias: {missing_keys}")
            else:
                print(f"\nTodas las 24 horas presentes en predicciones")

        print("\n" + "=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"ERROR EN PREDICCION")
        print("=" * 80)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_predict())
