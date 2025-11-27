"""
Script de prueba para validar el pipeline de Fase 1
"""
import sys
import pandas as pd
from pathlib import Path

# Añadir raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.orchestrator import run_automated_pipeline

def test_pipeline():
    """Prueba el pipeline completo"""
    print("="*70)
    print("PROBANDO PIPELINE AUTOMATIZADO - FASE 1")
    print("="*70)

    # Rutas
    base_dir = Path(__file__).parent.parent
    power_path = base_dir / "data" / "raw" / "datos.csv"
    weather_path = base_dir / "data" / "raw" / "data_cleaned_weather.csv"

    # Verificar que los archivos existen
    if not power_path.exists():
        print(f"ERROR: No se encuentra {power_path}")
        return False

    if not weather_path.exists():
        print(f"ADVERTENCIA: No se encuentra {weather_path}, continuando sin datos meteorologicos")
        weather_path = None

    # Ejecutar pipeline
    try:
        print("\nEjecutando pipeline...")
        df_final, report = run_automated_pipeline(
            power_data_path=str(power_path),
            weather_data_path=str(weather_path) if weather_path else None,
            start_date='2017-01-01'
        )

        # Validaciones
        print("\n" + "="*70)
        print("VALIDANDO RESULTADOS")
        print("="*70)

        # 1. Verificar que hay datos
        assert len(df_final) > 0, "El DataFrame final esta vacio"
        print(f"OK - Datos procesados: {len(df_final)} registros")

        # 2. Verificar features creadas
        expected_min_features = 50
        actual_features = report['data_summary']['features_created']
        assert actual_features >= expected_min_features, f"Se esperaban al menos {expected_min_features} features, se crearon {actual_features}"
        print(f"OK - Features creadas: {actual_features}")

        # 3. Verificar que no hay demasiados valores faltantes
        missing_pct = (df_final.isnull().sum().sum() / df_final.size) * 100
        assert missing_pct < 5, f"Demasiados valores faltantes: {missing_pct:.2f}%"
        print(f"OK - Valores faltantes: {missing_pct:.2f}%")

        # 4. Verificar que la calidad de datos paso
        assert report['quality_reports']['power_data']['passed'], "Reporte de calidad de power data fallo"
        print("OK - Calidad de datos validada")

        # 5. Verificar que el archivo de salida existe
        output_file = base_dir / "data" / "features" / "data_with_features_latest.csv"
        assert output_file.exists(), f"No se genero el archivo de salida: {output_file}"
        print(f"OK - Archivo de salida generado: {output_file}")

        # 6. Mostrar resumen de features
        print("\nRESUMEN DE FEATURES:")
        print(f"  - Calendar features: {report['feature_summary']['stats']['calendar_features']}")
        print(f"  - Demand features: {report['feature_summary']['stats']['demand_features']}")
        print(f"  - Weather features: {report['feature_summary']['stats']['weather_features']}")
        print(f"  - Interaction features: {report['feature_summary']['stats']['interaction_features']}")

        # 7. Mostrar estadisticas del dataset
        print("\nESTADISTICAS DEL DATASET:")
        print(f"  - Shape: {df_final.shape}")
        print(f"  - Columnas: {len(df_final.columns)}")
        print(f"  - Rango de fechas: {df_final['FECHA'].min()} a {df_final['FECHA'].max()}")

        print("\n" + "="*70)
        print("TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
        print("="*70)

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pipeline()
    exit(0 if success else 1)
