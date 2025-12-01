"""
Script para Generar Reporte de Desempeño del Sistema EPM - Entrega Final
=========================================================================

Genera un reporte completo con:
1. Evaluación del modelo de predicción diaria (train, val, test)
2. Curvas de MAPE y métricas
3. Evaluación de clusters de desagregación horaria
4. Visualizaciones consolidadas
5. Reporte en HTML

Uso:
    python scripts/generate_performance_report.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json
from typing import Dict, Tuple

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.metrics import calculate_all_metrics
from src.prediction.hourly import HourlyDisaggregationEngine

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("GENERADOR DE REPORTE DE DESEMPEÑO - SISTEMA EPM")
print("="*80)


# ============================================================================
# PARTE 1: EVALUACIÓN DEL MODELO DE PREDICCIÓN DIARIA
# ============================================================================

def load_model_and_data():
    """Carga el modelo campeón y datos con features"""
    print("\n1. Cargando modelo campeón y datos...")

    # Cargar modelo
    model_path = Path('models/registry/champion_model.joblib')
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró modelo campeón en {model_path}")

    model_dict = joblib.load(model_path)
    model = model_dict['model'] if isinstance(model_dict, dict) else model_dict
    feature_names = model_dict.get('feature_names', None) if isinstance(model_dict, dict) else None

    print(f"   [OK] Modelo cargado: {model_path.name}")

    # Cargar datos
    data_path = Path('data/features/data_with_features_latest.csv')
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontraron datos en {data_path}")

    df = pd.read_csv(data_path)
    print(f"   [OK] Datos cargados: {len(df)} registros")

    return model, feature_names, df


def prepare_train_val_test_split(df: pd.DataFrame, feature_names=None):
    """Prepara los splits de datos"""
    print("\n2. Preparando splits (train 60% | val 20% | test 20%)...")

    # Definir features a excluir
    FEATURES_TO_EXCLUDE = [
        'FECHA', 'fecha', 'TOTAL', 'demanda_total',
        'total_lag_1d', 'total_lag_7d', 'total_lag_14d',
        'p8_lag_1d', 'p8_lag_7d', 'p12_lag_1d', 'p12_lag_7d',
        'p18_lag_1d', 'p18_lag_7d', 'p20_lag_1d', 'p20_lag_7d',
        'total_day_change', 'total_day_change_pct'
    ] + [f'P{i}' for i in range(1, 25)]

    # Identificar columna target
    target_col = 'TOTAL' if 'TOTAL' in df.columns else 'demanda_total'

    # Seleccionar features
    if feature_names:
        feature_cols = feature_names
    else:
        feature_cols = [col for col in df.columns if col not in FEATURES_TO_EXCLUDE]

    X = df[feature_cols].fillna(0)
    y = df[target_col].copy()

    # Eliminar NaN en target
    mask = ~y.isnull()
    X = X[mask]
    y = y[mask]

    # Extraer fechas para análisis temporal
    if 'FECHA' in df.columns:
        dates = pd.to_datetime(df['FECHA'][mask]).reset_index(drop=True)
    elif 'fecha' in df.columns:
        dates = pd.to_datetime(df['fecha'][mask]).reset_index(drop=True)
    else:
        dates = None

    # Splits temporales
    n = len(X)
    train_idx = int(n * 0.6)
    val_idx = int(n * 0.8)

    X_train = X[:train_idx]
    y_train = y[:train_idx]
    dates_train = dates[:train_idx] if dates is not None else None

    X_val = X[train_idx:val_idx]
    y_val = y[train_idx:val_idx]
    dates_val = dates[train_idx:val_idx] if dates is not None else None

    X_test = X[val_idx:]
    y_test = y[val_idx:]
    dates_test = dates[val_idx:] if dates is not None else None

    print(f"   [OK] Train: {len(X_train)} registros ({dates_train.min().date() if dates_train is not None else 'N/A'} a {dates_train.max().date() if dates_train is not None else 'N/A'})")
    print(f"   [OK] Val:   {len(X_val)} registros ({dates_val.min().date() if dates_val is not None else 'N/A'} a {dates_val.max().date() if dates_val is not None else 'N/A'})")
    print(f"   [OK] Test:  {len(X_test)} registros ({dates_test.min().date() if dates_test is not None else 'N/A'} a {dates_test.max().date() if dates_test is not None else 'N/A'})")

    return (X_train, y_train, dates_train), (X_val, y_val, dates_val), (X_test, y_test, dates_test)


def evaluate_daily_model(model, train_data, val_data, test_data):
    """Evalúa el modelo en train, val y test"""
    print("\n3. Evaluando modelo de predicción diaria...")

    X_train, y_train, dates_train = train_data
    X_val, y_val, dates_val = val_data
    X_test, y_test, dates_test = test_data

    # Predicciones
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Métricas
    train_metrics = calculate_all_metrics(y_train, y_train_pred)
    val_metrics = calculate_all_metrics(y_val, y_val_pred)
    test_metrics = calculate_all_metrics(y_test, y_test_pred)

    print(f"\n   TRAIN - MAPE: {train_metrics['mape']:.4f}% | rMAPE: {train_metrics['rmape']:.4f} | R²: {train_metrics['r2']:.4f}")
    print(f"   VAL   - MAPE: {val_metrics['mape']:.4f}% | rMAPE: {val_metrics['rmape']:.4f} | R²: {val_metrics['r2']:.4f}")
    print(f"   TEST  - MAPE: {test_metrics['mape']:.4f}% | rMAPE: {test_metrics['rmape']:.4f} | R²: {test_metrics['r2']:.4f}")

    results = {
        'train': {'metrics': train_metrics, 'y_true': y_train, 'y_pred': y_train_pred, 'dates': dates_train},
        'val': {'metrics': val_metrics, 'y_true': y_val, 'y_pred': y_val_pred, 'dates': dates_val},
        'test': {'metrics': test_metrics, 'y_true': y_test, 'y_pred': y_test_pred, 'dates': dates_test}
    }

    return results


def plot_daily_model_performance(results: Dict, output_dir: Path):
    """Genera gráficos de desempeño del modelo diario"""
    print("\n4. Generando visualizaciones del modelo diario...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Figura 1: Curvas de MAPE por conjunto
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Gráfico 1: Barras comparativas de métricas
    ax = axes[0, 0]
    datasets = ['Train', 'Val', 'Test']
    mapes = [results['train']['metrics']['mape'],
             results['val']['metrics']['mape'],
             results['test']['metrics']['mape']]

    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = ax.bar(datasets, mapes, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=5, color='red', linestyle='--', linewidth=2, label='Umbral Regulatorio (5%)')
    ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
    ax.set_title('MAPE por Conjunto de Datos', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Agregar valores sobre barras
    for bar, val in zip(bars, mapes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')

    # Gráfico 2: R² por conjunto
    ax = axes[0, 1]
    r2_scores = [results['train']['metrics']['r2'],
                 results['val']['metrics']['r2'],
                 results['test']['metrics']['r2']]
    bars = ax.bar(datasets, r2_scores, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('R² por Conjunto de Datos', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    # Gráfico 3: Predicciones vs Real (Test Set)
    ax = axes[1, 0]
    y_test = results['test']['y_true']
    y_test_pred = results['test']['y_pred']

    ax.scatter(y_test, y_test_pred, alpha=0.5, s=20)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
    ax.set_xlabel('Demanda Real (MWh)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Demanda Predicha (MWh)', fontsize=12, fontweight='bold')
    ax.set_title('Predicciones vs Real (Test Set)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Gráfico 4: Distribución de errores (Test Set)
    ax = axes[1, 1]
    errors = y_test - y_test_pred
    ax.hist(errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Error (Real - Predicho) [MWh]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
    ax.set_title('Distribución de Errores (Test Set)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'daily_model_performance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   [OK] Grafico guardado: {plot_path}")

    # Figura 2: Serie temporal de predicciones (solo Test)
    if results['test']['dates'] is not None:
        fig, ax = plt.subplots(figsize=(16, 6))

        dates = results['test']['dates']
        y_true = results['test']['y_true']
        y_pred = results['test']['y_pred']

        ax.plot(dates, y_true, label='Real', linewidth=2, alpha=0.8)
        ax.plot(dates, y_pred, label='Predicho', linewidth=2, alpha=0.8, linestyle='--')
        ax.fill_between(dates, y_true, y_pred, alpha=0.2, color='gray')

        ax.set_xlabel('Fecha', fontsize=12, fontweight='bold')
        ax.set_ylabel('Demanda Total (MWh)', fontsize=12, fontweight='bold')
        ax.set_title('Serie Temporal: Predicciones vs Real (Test Set)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plot_path = output_dir / 'daily_model_timeseries.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   [OK] Grafico guardado: {plot_path}")


# ============================================================================
# PARTE 2: EVALUACIÓN DE CLUSTERS DE DESAGREGACIÓN HORARIA
# ============================================================================

def evaluate_hourly_disaggregation(output_dir: Path, n_days: int = 90):
    """Evalúa el sistema de desagregación horaria"""
    print("\n5. Evaluando clusters de desagregación horaria...")

    try:
        # Cargar engine
        engine = HourlyDisaggregationEngine(auto_load=True)

        # Verificar estado
        status = engine.get_engine_status()
        if not (status['normal_disaggregator']['fitted'] and status['special_disaggregator']['fitted']):
            print("   ⚠ Sistema de desagregación no está entrenado completamente")
            return None

        print(f"   [OK] Sistema cargado: {status['normal_disaggregator']['n_clusters']} clusters normales, {status['special_disaggregator']['n_clusters']} clusters especiales")

        # Cargar datos históricos
        data_path = Path('data/raw/datos.csv')
        df = pd.read_csv(data_path)
        df['FECHA'] = pd.to_datetime(df['FECHA'])
        df = df.sort_values('FECHA').tail(n_days)

        print(f"   [OK] Evaluando ultimos {len(df)} dias ({df['FECHA'].min().date()} a {df['FECHA'].max().date()})")

        # Evaluar cada día
        results = []
        for _, row in df.iterrows():
            date = row['FECHA']
            total_real = row['TOTAL']
            hourly_real = row[[f'P{i}' for i in range(1, 25)]].values

            # Predecir
            pred = engine.predict_hourly(date, total_real, validate=True)
            hourly_pred = pred['hourly']

            # Errores
            errors_abs = np.abs(hourly_pred - hourly_real)
            errors_pct = (errors_abs / hourly_real) * 100

            results.append({
                'date': date,
                'method': pred['method'],
                'mae': errors_abs.mean(),
                'rmse': np.sqrt((errors_abs ** 2).mean()),
                'mape': errors_pct.mean(),
                'max_error': errors_abs.max(),
                'sum_valid': pred['validation']['is_valid']
            })

        df_results = pd.DataFrame(results)

        # Métricas globales
        metrics = {
            'global': {
                'mae': df_results['mae'].mean(),
                'rmse': df_results['rmse'].mean(),
                'mape': df_results['mape'].mean(),
                'max_error': df_results['max_error'].max(),
                'sum_validation_pct': (df_results['sum_valid'].sum() / len(df_results)) * 100
            },
            'by_method': {}
        }

        # Por método
        for method in df_results['method'].unique():
            subset = df_results[df_results['method'] == method]
            metrics['by_method'][method] = {
                'n_days': len(subset),
                'mae': subset['mae'].mean(),
                'rmse': subset['rmse'].mean(),
                'mape': subset['mape'].mean()
            }

        print(f"\n   Metricas Globales:")
        print(f"   - MAE:  {metrics['global']['mae']:.4f} MW")
        print(f"   - RMSE: {metrics['global']['rmse']:.4f} MW")
        print(f"   - MAPE: {metrics['global']['mape']:.2f}%")
        print(f"   - Validacion de suma: {metrics['global']['sum_validation_pct']:.1f}%")

        # Gráfico
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Gráfico 1: MAPE por método
        ax = axes[0]
        methods = list(metrics['by_method'].keys())
        mapes = [metrics['by_method'][m]['mape'] for m in methods]
        colors = ['#3498db' if m == 'normal' else '#e67e22' for m in methods]

        bars = ax.bar(methods, mapes, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
        ax.set_title('Clusters: MAPE por Método', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        for bar, val, method in zip(bars, mapes, methods):
            height = bar.get_height()
            n_days = metrics['by_method'][method]['n_days']
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}%\n({n_days} días)', ha='center', va='bottom', fontweight='bold')

        # Gráfico 2: Distribución de errores
        ax = axes[1]
        ax.hist(df_results['mape'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=df_results['mape'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Media: {df_results["mape"].mean():.2f}%')
        ax.set_xlabel('MAPE (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frecuencia (días)', fontsize=12, fontweight='bold')
        ax.set_title('Distribución de MAPE en Clusters', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / 'hourly_disaggregation_performance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   [OK] Grafico guardado: {plot_path}")

        return metrics

    except Exception as e:
        print(f"   ⚠ Error evaluando desagregación horaria: {e}")
        return None


# ============================================================================
# PARTE 3: REPORTE HTML CONSOLIDADO
# ============================================================================

def generate_html_report(daily_results: Dict, hourly_metrics: Dict, output_path: Path):
    """Genera reporte HTML consolidado"""
    print("\n6. Generando reporte HTML consolidado...")

    train_metrics = daily_results['train']['metrics']
    val_metrics = daily_results['val']['metrics']
    test_metrics = daily_results['test']['metrics']

    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte de Desempeño - Sistema EPM</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 10px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            border-left: 5px solid #3498db;
            padding-left: 15px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-card.success {{
            background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        }}
        .metric-card.warning {{
            background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        }}
        .metric-card.danger {{
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }}
        .status-badge.success {{
            background-color: #2ecc71;
            color: white;
        }}
        .status-badge.warning {{
            background-color: #f39c12;
            color: white;
        }}
        .img-container {{
            text-align: center;
            margin: 30px 0;
        }}
        .img-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Reporte de Desempeño - Sistema EPM</h1>
        <p><strong>Fecha de generación:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Sistema:</strong> Pronóstico Automatizado de Demanda Energética</p>

        <h2>1. Modelo de Predicción Diaria</h2>

        <h3>Métricas por Conjunto de Datos</h3>
        <table>
            <tr>
                <th>Conjunto</th>
                <th>MAPE (%)</th>
                <th>rMAPE</th>
                <th>MAE (MWh)</th>
                <th>RMSE (MWh)</th>
                <th>R²</th>
                <th>Correlación</th>
            </tr>
            <tr>
                <td><strong>Train</strong></td>
                <td>{train_metrics['mape']:.4f}</td>
                <td>{train_metrics['rmape']:.4f}</td>
                <td>{train_metrics['mae']:.2f}</td>
                <td>{train_metrics['rmse']:.2f}</td>
                <td>{train_metrics['r2']:.4f}</td>
                <td>{train_metrics['correlation']:.4f}</td>
            </tr>
            <tr>
                <td><strong>Validation</strong></td>
                <td>{val_metrics['mape']:.4f}</td>
                <td>{val_metrics['rmape']:.4f}</td>
                <td>{val_metrics['mae']:.2f}</td>
                <td>{val_metrics['rmse']:.2f}</td>
                <td>{val_metrics['r2']:.4f}</td>
                <td>{val_metrics['correlation']:.4f}</td>
            </tr>
            <tr style="background-color: #e8f5e9;">
                <td><strong>Test</strong></td>
                <td><strong>{test_metrics['mape']:.4f}</strong></td>
                <td><strong>{test_metrics['rmape']:.4f}</strong></td>
                <td><strong>{test_metrics['mae']:.2f}</strong></td>
                <td><strong>{test_metrics['rmse']:.2f}</strong></td>
                <td><strong>{test_metrics['r2']:.4f}</strong></td>
                <td><strong>{test_metrics['correlation']:.4f}</strong></td>
            </tr>
        </table>

        <h3>Métricas Clave (Test Set)</h3>
        <div class="metric-grid">
            <div class="metric-card {'success' if test_metrics['mape'] < 5 else 'warning' if test_metrics['mape'] < 10 else 'danger'}">
                <div class="metric-label">MAPE (Test)</div>
                <div class="metric-value">{test_metrics['mape']:.2f}%</div>
                <div class="status-badge {'success' if test_metrics['mape'] < 5 else 'warning'}">
                    {'✓ Cumple (< 5%)' if test_metrics['mape'] < 5 else '⚠ Revisar'}
                </div>
            </div>
            <div class="metric-card success">
                <div class="metric-label">R² Score (Test)</div>
                <div class="metric-value">{test_metrics['r2']:.3f}</div>
            </div>
            <div class="metric-card success">
                <div class="metric-label">Correlación (Test)</div>
                <div class="metric-value">{test_metrics['correlation']:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">MAE (Test)</div>
                <div class="metric-value">{test_metrics['mae']:.0f}</div>
                <div class="metric-label">MWh</div>
            </div>
        </div>

        <h3>Visualizaciones</h3>
        <div class="img-container">
            <img src="daily_model_performance.png" alt="Desempeño del Modelo Diario">
        </div>
        <div class="img-container">
            <img src="daily_model_timeseries.png" alt="Serie Temporal Test Set">
        </div>
"""

    # Agregar sección de desagregación horaria si está disponible
    if hourly_metrics:
        html += f"""
        <h2>2. Sistema de Desagregación Horaria</h2>

        <h3>Métricas Globales</h3>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">MAPE Global</div>
                <div class="metric-value">{hourly_metrics['global']['mape']:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">MAE Global</div>
                <div class="metric-value">{hourly_metrics['global']['mae']:.2f}</div>
                <div class="metric-label">MW</div>
            </div>
            <div class="metric-card success">
                <div class="metric-label">Validación de Suma</div>
                <div class="metric-value">{hourly_metrics['global']['sum_validation_pct']:.1f}%</div>
            </div>
        </div>

        <h3>Desempeño por Método</h3>
        <table>
            <tr>
                <th>Método</th>
                <th>Días Evaluados</th>
                <th>MAPE (%)</th>
                <th>MAE (MW)</th>
                <th>RMSE (MW)</th>
            </tr>
"""

        for method, metrics in hourly_metrics['by_method'].items():
            method_name = "Normal (días regulares)" if method == 'normal' else "Especial (festivos)"
            html += f"""
            <tr>
                <td><strong>{method_name}</strong></td>
                <td>{metrics['n_days']}</td>
                <td>{metrics['mape']:.2f}</td>
                <td>{metrics['mae']:.2f}</td>
                <td>{metrics['rmse']:.2f}</td>
            </tr>
"""

        html += """
        </table>

        <h3>Visualizaciones</h3>
        <div class="img-container">
            <img src="hourly_disaggregation_performance.png" alt="Desempeño Desagregación Horaria">
        </div>
"""

    html += f"""
        <h2>3. Conclusiones</h2>
        <ul>
            <li><strong>Modelo de Predicción Diaria:</strong> {'✓ Excelente desempeño' if test_metrics['mape'] < 5 else '⚠ Requiere ajustes'} con MAPE de {test_metrics['mape']:.2f}% en test set.</li>
            <li><strong>Generalización:</strong> R² de {test_metrics['r2']:.3f} indica {'excelente' if test_metrics['r2'] > 0.9 else 'buena'} capacidad de generalización.</li>
"""

    if hourly_metrics:
        html += f"""
            <li><strong>Desagregación Horaria:</strong> MAPE promedio de {hourly_metrics['global']['mape']:.2f}% en distribución horaria.</li>
            <li><strong>Validación de Suma:</strong> {hourly_metrics['global']['sum_validation_pct']:.1f}% de días con suma válida (P1-P24 = TOTAL).</li>
"""

    html += """
        </ul>

        <div class="footer">
            <p>🤖 Generado automáticamente por el Sistema de Pronóstico EPM</p>
            <p>Empresa de Energía de Antioquia - 2024</p>
        </div>
    </div>
</body>
</html>
"""

    output_path.write_text(html, encoding='utf-8')
    print(f"   [OK] Reporte HTML guardado: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Función principal"""
    try:
        # Crear directorio de salida
        output_dir = Path('reports/performance')
        output_dir.mkdir(parents=True, exist_ok=True)

        # PARTE 1: Modelo diario
        model, feature_names, df = load_model_and_data()
        train_data, val_data, test_data = prepare_train_val_test_split(df, feature_names)
        daily_results = evaluate_daily_model(model, train_data, val_data, test_data)
        plot_daily_model_performance(daily_results, output_dir)

        # PARTE 2: Desagregación horaria
        hourly_metrics = evaluate_hourly_disaggregation(output_dir, n_days=90)

        # PARTE 3: Reporte HTML
        report_path = output_dir / 'reporte_desempeno.html'
        generate_html_report(daily_results, hourly_metrics, report_path)

        print("\n" + "="*80)
        print("[SUCCESS] REPORTE DE DESEMPENO GENERADO EXITOSAMENTE")
        print("="*80)
        print(f"\n[FILES] Archivos generados en: {output_dir.absolute()}")
        print(f"   - reporte_desempeno.html")
        print(f"   - daily_model_performance.png")
        print(f"   - daily_model_timeseries.png")
        if hourly_metrics:
            print(f"   - hourly_disaggregation_performance.png")

        print(f"\n[WEB] Abrir reporte: file:///{report_path.absolute()}")

    except Exception as e:
        print(f"\n[ERROR] Error generando reporte: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
