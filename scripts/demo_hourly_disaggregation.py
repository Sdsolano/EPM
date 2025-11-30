"""
Demo del Sistema de Desagregación Horaria

Script de demostración rápida del sistema de clustering
para desagregación horaria de demanda energética.

Uso:
    python scripts/demo_hourly_disaggregation.py
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction.hourly import HourlyDisaggregationEngine, CalendarClassifier

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_calendar_classifier():
    """Demostración del clasificador de calendario"""
    print("\n" + "=" * 80)
    print("DEMOSTRACIÓN 1: CLASIFICADOR DE CALENDARIO")
    print("=" * 80)

    classifier = CalendarClassifier()

    # Fechas de prueba
    fechas = [
        "2024-01-01",  # Año Nuevo
        "2024-05-01",  # Día del Trabajo
        "2024-12-25",  # Navidad
        "2024-03-15",  # Viernes normal
        "2024-03-16",  # Sábado
        "2024-04-15",  # Temporada lluviosa
    ]

    print("\n📅 Clasificación de días:")
    for fecha_str in fechas:
        fecha = pd.to_datetime(fecha_str)
        info = classifier.get_full_classification(fecha)

        print(f"\n  {fecha_str}:")
        print(f"    Día: {info['dia_semana']}")
        print(f"    Tipo: {info['tipo_dia']}")
        print(f"    Festivo: {info['es_festivo']}")
        if info['es_festivo']:
            print(f"    Nombre: {info['nombre_festivo']}")
        print(f"    Temporada: {info['temporada']}")


def demo_hourly_engine():
    """Demostración del motor de desagregación completo"""
    print("\n" + "=" * 80)
    print("DEMOSTRACIÓN 2: MOTOR DE DESAGREGACIÓN HORARIA")
    print("=" * 80)

    # Cargar motor
    print("\n🔧 Cargando sistema de desagregación...")
    try:
        engine = HourlyDisaggregationEngine(auto_load=True)
        print("✓ Sistema cargado exitosamente")
    except Exception as e:
        print(f"⚠ Error al cargar sistema: {e}")
        print("\nPara entrenar el sistema, ejecute:")
        print("  python scripts/train_hourly_disaggregation.py")
        return

    # Verificar estado
    status = engine.get_engine_status()
    print(f"\n📊 Estado del sistema:")
    print(f"  Normal: {'✓ Entrenado' if status['normal_disaggregator']['fitted'] else '✗ No entrenado'}")
    print(f"  Especial: {'✓ Entrenado' if status['special_disaggregator']['fitted'] else '✗ No entrenado'}")

    if not (status['normal_disaggregator']['fitted'] and status['special_disaggregator']['fitted']):
        print("\n⚠ El sistema no está completamente entrenado")
        return

    # Casos de prueba
    print("\n" + "=" * 80)
    print("PREDICCIONES DE EJEMPLO")
    print("=" * 80)

    test_cases = [
        ("2024-03-15", 1500.0, "Viernes normal"),
        ("2024-03-16", 1300.0, "Sábado"),
        ("2024-03-17", 1200.0, "Domingo"),
        ("2024-12-25", 1100.0, "Navidad"),
        ("2024-01-01", 1050.0, "Año Nuevo"),
    ]

    for date_str, total, description in test_cases:
        result = engine.predict_hourly(date_str, total)

        print(f"\n📅 {description} ({date_str})")
        print(f"   Tipo de día: {result['day_type']}")
        print(f"   Método usado: {result['method']}")
        print(f"   Temporada: {result['season']}")

        if result['is_holiday']:
            print(f"   Festivo: {result['holiday_name']}")

        print(f"\n   Total diario: {total:,.2f} MWh")
        print(f"   Suma horaria: {result['validation']['sum']:,.2f} MWh")
        print(f"   Diferencia: {result['validation']['difference']:.6f} MWh")
        print(f"   Validación: {'✓ OK' if result['validation']['is_valid'] else '✗ FALLO'}")

        # Mostrar primeros 6 períodos como ejemplo
        print(f"\n   Distribución horaria (primeros 6 períodos):")
        for i in range(6):
            hour = i
            value = result['hourly'][i]
            pct = (value / total) * 100
            print(f"      P{i+1} ({hour:02d}:00-{hour+1:02d}:00): {value:6.2f} MW ({pct:5.2f}%)")
        print(f"      ...")


def demo_batch_prediction():
    """Demostración de predicción en batch"""
    print("\n" + "=" * 80)
    print("DEMOSTRACIÓN 3: PREDICCIÓN EN BATCH")
    print("=" * 80)

    try:
        engine = HourlyDisaggregationEngine(auto_load=True)
    except:
        print("⚠ Sistema no entrenado. Saltando demostración.")
        return

    # Crear batch de fechas
    dates = pd.date_range('2024-03-01', periods=7, freq='D')
    totals = pd.Series([1500.0, 1480.0, 1520.0, 1450.0, 1510.0, 1350.0, 1280.0])

    print(f"\n🔄 Prediciendo {len(dates)} días...")

    results_df = engine.predict_batch(dates, totals, return_dataframe=True)

    print(f"\n✓ Predicciones completadas")
    print(f"\n📊 Resumen:")
    print(results_df[['FECHA', 'TOTAL', 'method', 'day_type', 'validation_ok']].to_string(index=False))


def main():
    """Función principal"""
    print("\n" + "=" * 80)
    print("🚀 DEMO - SISTEMA DE DESAGREGACIÓN HORARIA EPM")
    print("=" * 80)

    try:
        # Demo 1: Clasificador
        demo_calendar_classifier()

        # Demo 2: Motor completo
        demo_hourly_engine()

        # Demo 3: Batch
        demo_batch_prediction()

        print("\n" + "=" * 80)
        print("✅ Demo completada exitosamente")
        print("=" * 80)
        print("\n📚 Para más información:")
        print("  - Documentación: docs/DESAGREGACION_HORARIA.md")
        print("  - Tests: pytest tests/test_hourly_disaggregation.py -v")
        print("  - Entrenar: python scripts/train_hourly_disaggregation.py")
        print()

    except Exception as e:
        logger.error(f"\n❌ Error en demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
