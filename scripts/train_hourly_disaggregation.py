"""
Script para Entrenar el Sistema de Desagregación Horaria

Este script entrena y guarda los modelos de clustering para desagregación horaria:
- Modelo para días normales (35 clusters)
- Modelo para días especiales/festivos (15 clusters)

Uso:
    python scripts/train_hourly_disaggregation.py

Los modelos se guardan en:
    - models/hourly_disaggregator.pkl
    - models/special_days_disaggregator.pkl
"""

import sys
from pathlib import Path
import logging

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction.hourly import HourlyDisaggregationEngine
from src.config.settings import FEATURES_DATA_DIR

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Función principal"""
    logger.info("=" * 80)
    logger.info("ENTRENAMIENTO DEL SISTEMA DE DESAGREGACIÓN HORARIA")
    logger.info("=" * 80)

    # Ruta a datos históricos
    data_path = Path(FEATURES_DATA_DIR) / "data_with_features_latest.csv"

    if not data_path.exists():
        logger.error(f"❌ No se encontró el archivo de datos: {data_path}")
        logger.error("   Por favor, ejecute primero el pipeline de datos:")
        logger.error("   python scripts/run_pipeline.py")
        sys.exit(1)

    logger.info(f"📂 Cargando datos desde: {data_path}")

    # Crear motor de desagregación
    engine = HourlyDisaggregationEngine(auto_load=False)

    # Entrenar ambos modelos
    try:
        engine.train_all(
            data_path=data_path,
            n_clusters_normal=35,    # Más clusters para días normales (más variedad)
            n_clusters_special=15,   # Menos clusters para festivos (menos datos)
            save=True                # Guardar modelos
        )

        # Verificar estado
        status = engine.get_engine_status()

        logger.info("\n" + "=" * 80)
        logger.info("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("=" * 80)
        logger.info(f"\n📊 Resumen:")
        logger.info(f"  ✓ Desagregador normal:")
        logger.info(f"      - Clusters: {status['normal_disaggregator']['n_clusters']}")
        logger.info(f"      - Estado: {'Entrenado' if status['normal_disaggregator']['fitted'] else 'No entrenado'}")
        logger.info(f"\n  ✓ Desagregador especial:")
        logger.info(f"      - Clusters: {status['special_disaggregator']['n_clusters']}")
        logger.info(f"      - Días festivos: {status['special_disaggregator']['n_special_days']}")
        logger.info(f"      - Estado: {'Entrenado' if status['special_disaggregator']['fitted'] else 'No entrenado'}")

        # Probar predicción de ejemplo
        logger.info("\n" + "=" * 80)
        logger.info("🧪 PRUEBA DE PREDICCIÓN")
        logger.info("=" * 80)

        import pandas as pd

        test_cases = [
            ("2024-03-15", 1500.0, "Viernes normal"),
            ("2024-03-17", 1300.0, "Domingo"),
            ("2024-12-25", 1100.0, "Navidad"),
        ]

        for date_str, total, description in test_cases:
            result = engine.predict_hourly(date_str, total)

            logger.info(f"\n  📅 {description} ({date_str}):")
            logger.info(f"      Tipo: {result['day_type']} | Método: {result['method']}")
            logger.info(f"      Total predicho: {total:,.2f} MWh")
            logger.info(f"      Suma horaria: {result['validation']['sum']:,.2f} MWh")
            logger.info(f"      Diferencia: {result['validation']['difference']:.6f} MWh")
            logger.info(f"      Validación: {'✓ OK' if result['validation']['is_valid'] else '✗ FALLO'}")

        logger.info("\n" + "=" * 80)
        logger.info("✅ Sistema listo para usar en producción")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
