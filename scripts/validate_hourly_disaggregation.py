"""
Script Interno de Validación - Sistema de Desagregación Horaria

Valida la precisión del sistema de desagregación horaria comparando
predicciones contra datos históricos reales.

Genera un reporte completo con:
- Métricas globales (MAPE, MAE, RMSE)
- Desglose por tipo de día (laboral/festivo/fin_de_semana)
- Desglose por método (normal/special clustering)
- Validación de suma (P1-P24 = TOTAL)
- Identificación de períodos problemáticos

Uso:
    python scripts/validate_hourly_disaggregation.py [--days 60] [--output validation_report.txt]
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import argparse

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction.hourly import HourlyDisaggregationEngine, CalendarClassifier

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HourlyDisaggregationValidator:
    """Validador del sistema de desagregación horaria"""

    def __init__(self, engine: HourlyDisaggregationEngine):
        self.engine = engine
        self.classifier = CalendarClassifier()
        self.results = []

    def load_historical_data(self, days: int = 60):
        """Carga datos históricos para validación"""
        logger.info(f"Cargando últimos {days} días de datos históricos...")

        try:
            # Cargar datos de demanda
            data_path = Path(__file__).parent.parent / "data" / "raw" / "datos.csv"
            df = pd.read_csv(data_path)

            # Convertir columna de fecha
            df['FECHA'] = pd.to_datetime(df['FECHA'])

            # Filtrar últimos N días
            df = df.sort_values('FECHA').tail(days)

            logger.info(f"✓ Cargados {len(df)} días desde {df['FECHA'].min().date()} hasta {df['FECHA'].max().date()}")

            return df

        except Exception as e:
            logger.error(f"Error cargando datos históricos: {e}")
            raise

    def validate_day(self, date: pd.Timestamp, total_real: float, hourly_real: np.ndarray):
        """Valida la predicción de un día específico"""

        # Obtener predicción
        prediction = self.engine.predict_hourly(date, total_real, validate=True)
        hourly_pred = prediction['hourly']

        # Calcular errores
        errors_abs = np.abs(hourly_pred - hourly_real)
        errors_pct = (errors_abs / hourly_real) * 100

        # Métricas
        mae = np.mean(errors_abs)
        rmse = np.sqrt(np.mean(errors_abs ** 2))
        mape = np.mean(errors_pct)

        # Clasificación del día
        day_info = self.classifier.get_full_classification(date)

        # Validación de suma
        sum_valid = prediction['validation']['is_valid']
        sum_diff = prediction['validation']['difference']

        result = {
            'date': date,
            'day_type': day_info['tipo_dia'],
            'is_holiday': day_info['es_festivo'],
            'holiday_name': day_info.get('nombre_festivo', ''),
            'season': day_info['temporada'],
            'method': prediction['method'],
            'total_real': total_real,
            'total_pred_sum': prediction['validation']['sum'],
            'sum_valid': sum_valid,
            'sum_diff': sum_diff,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'max_error': errors_abs.max(),
            'max_error_hour': errors_abs.argmax() + 1,
        }

        return result

    def validate_batch(self, df: pd.DataFrame):
        """Valida un batch de días"""
        logger.info(f"\n{'='*80}")
        logger.info("INICIANDO VALIDACIÓN POR LOTES")
        logger.info(f"{'='*80}\n")

        total_days = len(df)

        for idx, row in df.iterrows():
            date = row['FECHA']
            total = row['TOTAL']

            # Extraer valores horarios (P1-P24)
            hourly_cols = [f'P{i}' for i in range(1, 25)]
            hourly = row[hourly_cols].values

            # Validar
            result = self.validate_day(date, total, hourly)
            self.results.append(result)

            # Log progreso
            if (len(self.results)) % 10 == 0:
                logger.info(f"Procesados {len(self.results)}/{total_days} días...")

        logger.info(f"✓ Validación completada: {len(self.results)} días procesados\n")

    def generate_report(self):
        """Genera reporte completo de validación"""
        if not self.results:
            logger.warning("No hay resultados para generar reporte")
            return ""

        df_results = pd.DataFrame(self.results)

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("REPORTE DE VALIDACIÓN - SISTEMA DE DESAGREGACIÓN HORARIA EPM")
        report_lines.append("=" * 80)
        report_lines.append(f"\nFecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Período evaluado: {df_results['date'].min().date()} a {df_results['date'].max().date()}")
        report_lines.append(f"Total de días evaluados: {len(df_results)}")

        # SECCIÓN 1: Validación de Suma
        report_lines.append(f"\n{'='*80}")
        report_lines.append("1. VALIDACIÓN DE SUMA (sum(P1-P24) = TOTAL)")
        report_lines.append("="*80)

        sum_valid_count = df_results['sum_valid'].sum()
        sum_valid_pct = (sum_valid_count / len(df_results)) * 100

        report_lines.append(f"\n✓ Días con suma válida: {sum_valid_count}/{len(df_results)} ({sum_valid_pct:.2f}%)")

        if sum_valid_count < len(df_results):
            invalid_days = df_results[~df_results['sum_valid']]
            report_lines.append(f"\n⚠ Días con suma inválida: {len(invalid_days)}")
            report_lines.append("\nDetalle:")
            for _, row in invalid_days.iterrows():
                report_lines.append(f"  - {row['date'].date()}: diferencia = {row['sum_diff']:.6f} MWh")

        max_sum_diff = df_results['sum_diff'].max()
        report_lines.append(f"\nDiferencia máxima encontrada: {max_sum_diff:.6f} MWh")

        # SECCIÓN 2: Métricas Globales
        report_lines.append(f"\n{'='*80}")
        report_lines.append("2. MÉTRICAS GLOBALES DE PRECISIÓN")
        report_lines.append("="*80)

        report_lines.append(f"\nMAE (Error Absoluto Medio):  {df_results['mae'].mean():.4f} MW")
        report_lines.append(f"RMSE (Raíz Error Cuadrático): {df_results['rmse'].mean():.4f} MW")
        report_lines.append(f"MAPE (Error Porcentual Medio): {df_results['mape'].mean():.2f}%")
        report_lines.append(f"\nError máximo encontrado: {df_results['max_error'].max():.4f} MW")

        # SECCIÓN 3: Desglose por Tipo de Día
        report_lines.append(f"\n{'='*80}")
        report_lines.append("3. DESGLOSE POR TIPO DE DÍA")
        report_lines.append("="*80)

        for day_type in df_results['day_type'].unique():
            subset = df_results[df_results['day_type'] == day_type]
            count = len(subset)
            pct = (count / len(df_results)) * 100

            report_lines.append(f"\n{day_type.upper()} ({count} días, {pct:.1f}%):")
            report_lines.append(f"  MAE:  {subset['mae'].mean():.4f} MW")
            report_lines.append(f"  RMSE: {subset['rmse'].mean():.4f} MW")
            report_lines.append(f"  MAPE: {subset['mape'].mean():.2f}%")

        # SECCIÓN 4: Desglose por Método
        report_lines.append(f"\n{'='*80}")
        report_lines.append("4. DESGLOSE POR MÉTODO DE CLUSTERING")
        report_lines.append("="*80)

        for method in df_results['method'].unique():
            subset = df_results[df_results['method'] == method]
            count = len(subset)
            pct = (count / len(df_results)) * 100

            method_name = "Normal (35 clusters)" if method == "normal" else "Especial (15 clusters)"

            report_lines.append(f"\n{method_name} ({count} días, {pct:.1f}%):")
            report_lines.append(f"  MAE:  {subset['mae'].mean():.4f} MW")
            report_lines.append(f"  RMSE: {subset['rmse'].mean():.4f} MW")
            report_lines.append(f"  MAPE: {subset['mape'].mean():.2f}%")

        # SECCIÓN 5: Días Festivos
        if df_results['is_holiday'].sum() > 0:
            report_lines.append(f"\n{'='*80}")
            report_lines.append("5. ANÁLISIS DE DÍAS FESTIVOS")
            report_lines.append("="*80)

            holidays = df_results[df_results['is_holiday']]
            report_lines.append(f"\nTotal festivos evaluados: {len(holidays)}")
            report_lines.append(f"MAE promedio festivos: {holidays['mae'].mean():.4f} MW")
            report_lines.append(f"MAPE promedio festivos: {holidays['mape'].mean():.2f}%")

            report_lines.append("\nFestivos evaluados:")
            for _, row in holidays.iterrows():
                report_lines.append(f"  - {row['date'].date()} ({row['holiday_name']}): MAPE = {row['mape']:.2f}%")

        # SECCIÓN 6: Días con Mayor Error
        report_lines.append(f"\n{'='*80}")
        report_lines.append("6. DÍAS CON MAYOR ERROR (Top 10)")
        report_lines.append("="*80)

        worst_days = df_results.nlargest(10, 'mape')
        report_lines.append("\n")
        for idx, row in worst_days.iterrows():
            holiday_info = f" ({row['holiday_name']})" if row['is_holiday'] else ""
            report_lines.append(f"  {row['date'].date()}{holiday_info}:")
            report_lines.append(f"    Tipo: {row['day_type']}, Método: {row['method']}")
            report_lines.append(f"    MAPE: {row['mape']:.2f}%, MAE: {row['mae']:.2f} MW")
            report_lines.append(f"    Error máximo: {row['max_error']:.2f} MW en hora P{row['max_error_hour']}")
            report_lines.append("")

        # SECCIÓN 7: Desglose por Temporada
        report_lines.append(f"{'='*80}")
        report_lines.append("7. DESGLOSE POR TEMPORADA")
        report_lines.append("="*80)

        for season in df_results['season'].unique():
            subset = df_results[df_results['season'] == season]
            count = len(subset)
            pct = (count / len(df_results)) * 100

            report_lines.append(f"\n{season.upper()} ({count} días, {pct:.1f}%):")
            report_lines.append(f"  MAE:  {subset['mae'].mean():.4f} MW")
            report_lines.append(f"  RMSE: {subset['rmse'].mean():.4f} MW")
            report_lines.append(f"  MAPE: {subset['mape'].mean():.2f}%")

        # CONCLUSIONES
        report_lines.append(f"\n{'='*80}")
        report_lines.append("8. CONCLUSIONES Y RECOMENDACIONES")
        report_lines.append("="*80)

        avg_mape = df_results['mape'].mean()

        report_lines.append(f"\n✓ Precisión Global (MAPE): {avg_mape:.2f}%")

        if avg_mape < 5:
            report_lines.append("  → Excelente precisión. Sistema listo para producción.")
        elif avg_mape < 10:
            report_lines.append("  → Buena precisión. Sistema confiable con margen de mejora.")
        elif avg_mape < 15:
            report_lines.append("  → Precisión aceptable. Considerar ajustes al sistema.")
        else:
            report_lines.append("  → Precisión baja. Revisar configuración de clusters.")

        # Validación de suma
        if sum_valid_pct == 100:
            report_lines.append(f"\n✓ Validación de Suma: PERFECTA (100% de días válidos)")
        elif sum_valid_pct >= 99:
            report_lines.append(f"\n✓ Validación de Suma: EXCELENTE ({sum_valid_pct:.2f}% de días válidos)")
        else:
            report_lines.append(f"\n⚠ Validación de Suma: REVISAR ({sum_valid_pct:.2f}% de días válidos)")

        # Comparación métodos
        if 'normal' in df_results['method'].values and 'special' in df_results['method'].values:
            normal_mape = df_results[df_results['method'] == 'normal']['mape'].mean()
            special_mape = df_results[df_results['method'] == 'special']['mape'].mean()

            report_lines.append(f"\n✓ Comparación de Métodos:")
            report_lines.append(f"  - Normal (días regulares):  MAPE = {normal_mape:.2f}%")
            report_lines.append(f"  - Especial (festivos):      MAPE = {special_mape:.2f}%")

            if special_mape > normal_mape * 1.5:
                report_lines.append("  → Los días especiales tienen mayor error. Considerar más clusters.")

        report_lines.append(f"\n{'='*80}")
        report_lines.append("FIN DEL REPORTE")
        report_lines.append("="*80)

        return "\n".join(report_lines)

    def save_detailed_results(self, output_path: Path):
        """Guarda resultados detallados en CSV"""
        if not self.results:
            return

        df = pd.DataFrame(self.results)
        csv_path = output_path.parent / (output_path.stem + "_detailed.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"✓ Resultados detallados guardados en: {csv_path}")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Validación del Sistema de Desagregación Horaria')
    parser.add_argument('--days', type=int, default=60, help='Número de días históricos a validar (default: 60)')
    parser.add_argument('--output', type=str, default='validation_report.txt', help='Archivo de salida para el reporte')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("VALIDACION INTERNA - SISTEMA DE DESAGREGACION HORARIA EPM")
    print("="*80)

    try:
        # Cargar sistema
        logger.info("\n1. Cargando sistema de desagregacion...")
        engine = HourlyDisaggregationEngine(auto_load=True)
        logger.info("Sistema cargado exitosamente")

        # Verificar estado
        status = engine.get_engine_status()
        if not (status['normal_disaggregator']['fitted'] and status['special_disaggregator']['fitted']):
            logger.error("El sistema no esta completamente entrenado")
            logger.info("\nPara entrenar el sistema, ejecute:")
            logger.info("  python scripts/train_hourly_disaggregation.py")
            return

        # Crear validador
        logger.info("\n2. Inicializando validador...")
        validator = HourlyDisaggregationValidator(engine)

        # Cargar datos históricos
        logger.info(f"\n3. Cargando datos históricos ({args.days} días)...")
        df_historical = validator.load_historical_data(days=args.days)

        # Ejecutar validación
        logger.info("\n4. Ejecutando validación...")
        validator.validate_batch(df_historical)

        # Generar reporte
        logger.info("\n5. Generando reporte...")
        report = validator.generate_report()

        # Guardar reporte
        output_path = Path(__file__).parent.parent / "logs" / "validation" / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"✓ Reporte guardado en: {output_path}")

        # Guardar resultados detallados
        validator.save_detailed_results(output_path)

        # Imprimir reporte en consola
        print("\n" + report)

        logger.info("\nValidacion completada exitosamente")

    except Exception as e:
        logger.error(f"\nError durante validacion: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
