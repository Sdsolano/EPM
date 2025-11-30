"""
Visualización: Predicciones de Desagregación Horaria vs Datos Reales

Este script compara las predicciones del sistema de clustering
con los datos históricos reales para validar visualmente la precisión.

Uso:
    python scripts/visualize_hourly_predictions.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction.hourly import HourlyDisaggregationEngine
from src.config.settings import FEATURES_DATA_DIR

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurar estilo de gráficas
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


def load_historical_data():
    """Carga datos históricos"""
    data_path = Path(FEATURES_DATA_DIR) / "data_with_features_latest.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró {data_path}")

    logger.info(f"Cargando datos desde {data_path}...")
    df = pd.read_csv(data_path)
    df['FECHA'] = pd.to_datetime(df['FECHA'])

    return df


def compare_single_day(engine, df_historico, date_str, fig_num=1):
    """
    Compara predicción vs real para un día específico

    Args:
        engine: Motor de desagregación
        df_historico: DataFrame con datos históricos
        date_str: Fecha en formato 'YYYY-MM-DD'
        fig_num: Número de figura
    """
    date = pd.to_datetime(date_str)

    # Obtener datos reales
    day_data = df_historico[df_historico['FECHA'] == date]

    if len(day_data) == 0:
        logger.warning(f"No hay datos históricos para {date_str}")
        return None

    # Extraer valores reales
    period_cols = [f'P{i}' for i in range(1, 25)]
    real_hourly = day_data[period_cols].values[0]
    real_total = day_data['TOTAL'].values[0] if 'TOTAL' in day_data.columns else real_hourly.sum()

    # Predecir con el sistema
    result = engine.predict_hourly(date, real_total)
    pred_hourly = result['hourly']

    # Crear figura
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Comparación: {date_str} ({result['day_name']} - {result['day_type']})\n"
                 f"Método: {result['method']} | Festivo: {result['holiday_name'] if result['is_holiday'] else 'No'}",
                 fontsize=14, fontweight='bold')

    hours = list(range(24))

    # 1. Comparación directa
    ax1 = axes[0, 0]
    ax1.plot(hours, real_hourly, 'o-', label='Real', linewidth=2, markersize=6, color='#2E86AB')
    ax1.plot(hours, pred_hourly, 's--', label='Predicción', linewidth=2, markersize=5, color='#A23B72')
    ax1.set_xlabel('Hora del día')
    ax1.set_ylabel('Demanda (MW)')
    ax1.set_title('Demanda Horaria: Real vs Predicción')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 24, 2))

    # 2. Error absoluto por hora
    ax2 = axes[0, 1]
    error_abs = np.abs(pred_hourly - real_hourly)
    colors = ['red' if e > 5 else 'green' for e in error_abs]
    ax2.bar(hours, error_abs, color=colors, alpha=0.6)
    ax2.axhline(y=5, color='red', linestyle='--', label='Umbral 5 MW', linewidth=2)
    ax2.set_xlabel('Hora del día')
    ax2.set_ylabel('Error Absoluto (MW)')
    ax2.set_title('Error Absoluto por Período')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(range(0, 24, 2))

    # 3. Error porcentual
    ax3 = axes[1, 0]
    error_pct = (pred_hourly - real_hourly) / real_hourly * 100
    ax3.bar(hours, error_pct, color='steelblue', alpha=0.6)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='±5%')
    ax3.axhline(y=-5, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Hora del día')
    ax3.set_ylabel('Error Porcentual (%)')
    ax3.set_title('Error Porcentual por Período')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(range(0, 24, 2))

    # 4. Métricas resumen
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Calcular métricas
    mae = np.mean(error_abs)
    rmse = np.sqrt(np.mean((pred_hourly - real_hourly)**2))
    mape = np.mean(np.abs(error_pct))
    max_error = np.max(error_abs)
    errors_over_5pct = np.sum(np.abs(error_pct) > 5)

    metrics_text = f"""
    📊 MÉTRICAS DE DESEMPEÑO
    {'='*40}

    Total Diario:
      • Real:          {real_total:,.2f} MW
      • Predicción:    {result['total_daily']:,.2f} MW
      • Diferencia:    {result['validation']['difference']:.4f} MW
      • Validación:    {'✓ OK' if result['validation']['is_valid'] else '✗ FALLO'}

    Errores Horarios:
      • MAE:           {mae:.2f} MW
      • RMSE:          {rmse:.2f} MW
      • MAPE:          {mape:.2f}%
      • Error máximo:  {max_error:.2f} MW
      • Períodos > 5%: {errors_over_5pct}/24

    Clasificación:
      • Tipo de día:   {result['day_type']}
      • Temporada:     {result['season']}
      • Método usado:  {result['method']}
    """

    ax4.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Guardar figura
    output_dir = Path('outputs/hourly_validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"comparison_{date_str}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Gráfica guardada: {output_path}")

    return {
        'date': date_str,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'max_error': max_error,
        'errors_over_5pct': errors_over_5pct,
        'method': result['method'],
        'day_type': result['day_type']
    }


def compare_week(engine, df_historico, start_date_str):
    """
    Compara una semana completa

    Args:
        engine: Motor de desagregación
        df_historico: DataFrame con datos históricos
        start_date_str: Fecha inicial de la semana
    """
    start_date = pd.to_datetime(start_date_str)
    dates = [start_date + timedelta(days=i) for i in range(7)]

    fig, axes = plt.subplots(4, 2, figsize=(18, 16))
    fig.suptitle(f"Comparación Semanal: {start_date_str} (7 días)",
                 fontsize=16, fontweight='bold')

    metrics_summary = []

    for idx, date in enumerate(dates):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        # Obtener datos reales
        day_data = df_historico[df_historico['FECHA'] == date]

        if len(day_data) == 0:
            ax.text(0.5, 0.5, f'Sin datos\n{date.strftime("%Y-%m-%d")}',
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            continue

        period_cols = [f'P{i}' for i in range(1, 25)]
        real_hourly = day_data[period_cols].values[0]
        real_total = day_data['TOTAL'].values[0] if 'TOTAL' in day_data.columns else real_hourly.sum()

        # Predecir
        result = engine.predict_hourly(date, real_total)
        pred_hourly = result['hourly']

        # Graficar
        hours = list(range(24))
        ax.plot(hours, real_hourly, 'o-', label='Real', linewidth=2, markersize=4)
        ax.plot(hours, pred_hourly, 's--', label='Predicción', linewidth=2, markersize=3, alpha=0.7)

        # Calcular error
        mape = np.mean(np.abs((pred_hourly - real_hourly) / real_hourly * 100))

        ax.set_title(f"{date.strftime('%Y-%m-%d')} ({result['day_name']})\n"
                    f"{result['day_type']} - MAPE: {mape:.2f}%",
                    fontsize=10)
        ax.set_xlabel('Hora')
        ax.set_ylabel('Demanda (MW)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 4))

        metrics_summary.append({
            'date': date.strftime('%Y-%m-%d'),
            'day_name': result['day_name'],
            'mape': mape,
            'method': result['method']
        })

    # Última celda: resumen
    ax_summary = axes[3, 1]
    ax_summary.axis('off')

    summary_df = pd.DataFrame(metrics_summary)
    avg_mape = summary_df['mape'].mean()

    summary_text = f"""
    📊 RESUMEN SEMANAL
    {'='*30}

    MAPE Promedio: {avg_mape:.2f}%

    Por Día:
    """

    for _, row in summary_df.iterrows():
        summary_text += f"\n  {row['day_name']:10s}: {row['mape']:5.2f}%"

    ax_summary.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()

    # Guardar
    output_dir = Path('outputs/hourly_validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"week_comparison_{start_date_str}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Gráfica semanal guardada: {output_path}")


def compare_multiple_days(engine, df_historico, test_dates):
    """
    Compara múltiples días y genera estadísticas agregadas

    Args:
        engine: Motor de desagregación
        df_historico: DataFrame con datos históricos
        test_dates: Lista de fechas a probar
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"VALIDACIÓN EN {len(test_dates)} DÍAS")
    logger.info(f"{'='*80}")

    results = []

    for date_str in test_dates:
        try:
            result = compare_single_day(engine, df_historico, date_str)
            if result:
                results.append(result)
                logger.info(f"✓ {date_str}: MAPE={result['mape']:.2f}%, MAE={result['mae']:.2f} MW")
        except Exception as e:
            logger.error(f"✗ Error en {date_str}: {e}")

    # Resumen global
    if results:
        df_results = pd.DataFrame(results)

        print(f"\n{'='*80}")
        print("📊 RESUMEN GLOBAL")
        print(f"{'='*80}")
        print(f"\nMétricas Promedio:")
        print(f"  MAE:  {df_results['mae'].mean():.2f} MW")
        print(f"  RMSE: {df_results['rmse'].mean():.2f} MW")
        print(f"  MAPE: {df_results['mape'].mean():.2f}%")
        print(f"\nPor Tipo de Día:")
        print(df_results.groupby('day_type')['mape'].agg(['mean', 'min', 'max', 'count']))
        print(f"\nPor Método:")
        print(df_results.groupby('method')['mape'].agg(['mean', 'min', 'max', 'count']))

        # Gráfica de distribución de errores
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histograma MAPE
        axes[0].hist(df_results['mape'], bins=20, edgecolor='black', alpha=0.7)
        axes[0].axvline(df_results['mape'].mean(), color='red', linestyle='--',
                       label=f'Media: {df_results["mape"].mean():.2f}%')
        axes[0].set_xlabel('MAPE (%)')
        axes[0].set_ylabel('Frecuencia')
        axes[0].set_title('Distribución de MAPE')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Box plot por tipo de día
        day_types = df_results['day_type'].unique()
        data_by_type = [df_results[df_results['day_type'] == dt]['mape'].values
                       for dt in day_types]
        axes[1].boxplot(data_by_type, labels=day_types)
        axes[1].set_ylabel('MAPE (%)')
        axes[1].set_title('MAPE por Tipo de Día')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = Path('outputs/hourly_validation/summary_statistics.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"\n✓ Estadísticas guardadas: {output_path}")


def main():
    """Función principal"""
    print("="*80)
    print("🔍 VALIDACIÓN: PREDICCIONES HORARIAS vs DATOS REALES")
    print("="*80)

    # Cargar datos históricos
    df_historico = load_historical_data()
    logger.info(f"✓ Datos cargados: {len(df_historico)} días")

    # Cargar motor de desagregación
    logger.info("\n🔧 Cargando sistema de desagregación...")
    try:
        engine = HourlyDisaggregationEngine(auto_load=True)
        logger.info("✓ Sistema cargado exitosamente")
    except Exception as e:
        logger.error(f"❌ Error al cargar sistema: {e}")
        logger.error("\nPara entrenar el sistema, ejecute:")
        logger.error("  python scripts/train_hourly_disaggregation.py")
        return

    # Fechas de prueba (seleccionar fechas disponibles en histórico)
    available_dates = df_historico['FECHA'].dt.strftime('%Y-%m-%d').tolist()

    if len(available_dates) == 0:
        logger.error("No hay datos disponibles")
        return

    # Seleccionar fechas de prueba variadas
    test_dates = []

    # Intentar conseguir: viernes, sábado, domingo, festivos
    for date_str in available_dates[-100:]:  # Últimos 100 días
        date = pd.to_datetime(date_str)
        if date.dayofweek == 4:  # Viernes
            test_dates.append(date_str)
            break

    for date_str in available_dates[-100:]:
        date = pd.to_datetime(date_str)
        if date.dayofweek == 5:  # Sábado
            test_dates.append(date_str)
            break

    for date_str in available_dates[-100:]:
        date = pd.to_datetime(date_str)
        if date.dayofweek == 6:  # Domingo
            test_dates.append(date_str)
            break

    # Agregar más fechas recientes
    test_dates.extend(available_dates[-10:])
    test_dates = list(set(test_dates))[:15]  # Máximo 15 días

    logger.info(f"\n📅 Fechas seleccionadas para prueba: {len(test_dates)}")

    # Comparación múltiple
    compare_multiple_days(engine, df_historico, test_dates)

    # Comparación semanal (usar última semana disponible)
    if len(available_dates) >= 7:
        week_start = available_dates[-7]
        logger.info(f"\n📅 Generando comparación semanal desde {week_start}...")
        compare_week(engine, df_historico, week_start)

    print("\n" + "="*80)
    print("✅ VALIDACIÓN COMPLETADA")
    print("="*80)
    print(f"\n📂 Gráficas guardadas en: outputs/hourly_validation/")
    print()


if __name__ == "__main__":
    main()
