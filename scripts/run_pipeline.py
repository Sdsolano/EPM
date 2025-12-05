"""
Script para ejecutar el pipeline completo de datos
Ejecuta desde la línea de comandos: python scripts/run_pipeline.py
"""
import sys
from pathlib import Path

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.orchestrator import run_automated_pipeline

def main():
    print("="*70)
    print("EJECUTANDO PIPELINE AUTOMATIZADO DE DATOS")
    print("Sistema de Pronóstico de Demanda Energética EPM")
    print("="*70 + "\n")

    # Configurar rutas
    base_dir = Path(__file__).parent.parent
    power_path = base_dir / "data" / "raw" / "datos.csv"
    weather_path = base_dir / "data" / "raw" / "clima_new.csv"  # API EPM format

    # Verificar que los archivos existen
    if not power_path.exists():
        print(f"⚠️  ERROR: No se encuentra el archivo {power_path}")
        print("   Asegúrate de que los datos estén en data/raw/")
        return 1

    # Ejecutar pipeline
    try:
        df_final, report = run_automated_pipeline(
            power_data_path=str(power_path),
            weather_data_path=str(weather_path) if weather_path.exists() else None,
            start_date='2017-01-01'  # Filtrar datos desde 2017
        )

        print("\n📊 RESULTADOS FINALES:")
        print(f"  - Shape del dataset: {df_final.shape}")
        print(f"  - Columnas totales: {len(df_final.columns)}")
        print(f"  - Rango de fechas: {df_final['FECHA'].min()} a {df_final['FECHA'].max()}")
        print(f"\n✓ Pipeline completado exitosamente")
        print(f"✓ Datos guardados en: data/features/data_with_features_latest.csv")
        return 0

    except Exception as e:
        print(f"\n❌ Error en el pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
