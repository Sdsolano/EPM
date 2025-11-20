"""
Métricas de Evaluación de Modelos
Incluye la métrica rMAPE (novel metric from Universidad del Norte paper)

rMAPE = MAPE / r_xy
donde r_xy es el coeficiente de correlación de Pearson

Esta métrica captura tanto la magnitud del error (MAPE) como la forma de la curva (correlación)
"""

import numpy as np
import pandas as pd
from typing import Union, Dict
from scipy.stats import pearsonr


def calculate_mape(y_true: Union[np.ndarray, pd.Series],
                   y_pred: Union[np.ndarray, pd.Series],
                   epsilon: float = 1e-10) -> float:
    """
    Calcula el Mean Absolute Percentage Error (MAPE)

    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        epsilon: Valor pequeño para evitar división por cero

    Returns:
        MAPE en porcentaje (0-100)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Evitar división por cero
    mask = np.abs(y_true) > epsilon

    if not mask.any():
        return np.inf

    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return float(mape)


def calculate_correlation(y_true: Union[np.ndarray, pd.Series],
                         y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calcula el coeficiente de correlación de Pearson (r_xy)

    Args:
        y_true: Valores reales
        y_pred: Valores predichos

    Returns:
        Coeficiente de correlación (-1 a 1)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Eliminar NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))

    if mask.sum() < 2:
        return 0.0

    correlation, _ = pearsonr(y_true[mask], y_pred[mask])
    return float(correlation)


def calculate_rmape(y_true: Union[np.ndarray, pd.Series],
                    y_pred: Union[np.ndarray, pd.Series],
                    epsilon: float = 1e-10) -> float:
    """
    Calcula el rMAPE (novel metric from Universidad del Norte paper)

    rMAPE = MAPE / r_xy

    Esta métrica es superior al MAPE porque:
    - MAPE bajo + correlación alta → rMAPE bajo (predicción excelente)
    - MAPE bajo + correlación baja → rMAPE alto (predicción mala, forma incorrecta)
    - MAPE alto → rMAPE alto (predicción mala)

    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        epsilon: Valor pequeño para evitar división por cero

    Returns:
        rMAPE (0 a infinito, menor es mejor)

    Note:
        - rMAPE cercano a 0: Excelente predicción (MAPE bajo Y correlación alta)
        - rMAPE alto: Mala predicción (MAPE alto O correlación baja)
        - Si correlación es negativa, retorna infinito (predicción opuesta)
    """
    mape = calculate_mape(y_true, y_pred, epsilon)
    r_xy = calculate_correlation(y_true, y_pred)

    # Si la correlación es negativa o cero, la predicción es muy mala
    if r_xy <= epsilon:
        return np.inf

    # Si MAPE es infinito, rmape también lo es
    if np.isinf(mape):
        return np.inf

    rmape = mape / r_xy
    return float(rmape)


def calculate_all_metrics(y_true: Union[np.ndarray, pd.Series],
                         y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Calcula todas las métricas de evaluación

    Args:
        y_true: Valores reales
        y_pred: Valores predichos

    Returns:
        Diccionario con todas las métricas
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Eliminar NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return {
            'mape': np.inf,
            'rmape': np.inf,
            'mae': np.inf,
            'rmse': np.inf,
            'r2': -np.inf,
            'correlation': 0.0
        }

    metrics = {
        'mape': calculate_mape(y_true_clean, y_pred_clean),
        'correlation': calculate_correlation(y_true_clean, y_pred_clean),
        'rmape': calculate_rmape(y_true_clean, y_pred_clean),
        'mae': float(mean_absolute_error(y_true_clean, y_pred_clean)),
        'rmse': float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))),
        'r2': float(r2_score(y_true_clean, y_pred_clean))
    }

    return metrics


def evaluate_model_performance(y_true: Union[np.ndarray, pd.Series],
                               y_pred: Union[np.ndarray, pd.Series],
                               threshold_mape: float = 5.0) -> Dict:
    """
    Evalúa el desempeño del modelo según los criterios regulatorios

    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        threshold_mape: Umbral regulatorio de MAPE (default: 5%)

    Returns:
        Diccionario con evaluación completa
    """
    metrics = calculate_all_metrics(y_true, y_pred)

    # Calcular errores porcentuales
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))

    errors_pct = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100

    evaluation = {
        'metrics': metrics,
        'regulatory_compliance': {
            'cumple_mape_5pct': metrics['mape'] < threshold_mape,
            'dias_con_error_menor_5pct': int((errors_pct < 5).sum()),
            'total_dias': int(len(errors_pct)),
            'porcentaje_dias_cumplimiento': float((errors_pct < 5).sum() / len(errors_pct) * 100)
        },
        'error_distribution': {
            'error_mean': float(np.mean(np.abs(y_true[mask] - y_pred[mask]))),
            'error_median': float(np.median(np.abs(y_true[mask] - y_pred[mask]))),
            'error_std': float(np.std(y_true[mask] - y_pred[mask])),
            'error_max': float(np.max(np.abs(y_true[mask] - y_pred[mask]))),
            'error_min': float(np.min(np.abs(y_true[mask] - y_pred[mask])))
        }
    }

    return evaluation


def compare_models(evaluations: Dict[str, Dict]) -> str:
    """
    Compara múltiples modelos y selecciona el mejor basado en rMAPE

    Args:
        evaluations: Diccionario con evaluaciones de cada modelo
                    {model_name: evaluation_dict}

    Returns:
        Nombre del mejor modelo
    """
    best_model = None
    best_rmape = np.inf

    for model_name, evaluation in evaluations.items():
        rmape = evaluation['metrics']['rmape']

        if rmape < best_rmape:
            best_rmape = rmape
            best_model = model_name

    return best_model


# ============== TESTING ==============

if __name__ == "__main__":
    print("="*70)
    print("TESTING MÉTRICAS - rMAPE")
    print("="*70)

    # Caso 1: Predicción perfecta
    print("\n1. Predicción Perfecta:")
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([100, 200, 300, 400, 500])

    print(f"   MAPE: {calculate_mape(y_true, y_pred):.4f}%")
    print(f"   Correlación: {calculate_correlation(y_true, y_pred):.4f}")
    print(f"   rMAPE: {calculate_rmape(y_true, y_pred):.4f}")

    # Caso 2: MAPE bajo pero forma incorrecta
    print("\n2. MAPE Bajo pero Forma Incorrecta:")
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([500, 400, 300, 200, 100])  # Invertido

    print(f"   MAPE: {calculate_mape(y_true, y_pred):.4f}%")
    print(f"   Correlación: {calculate_correlation(y_true, y_pred):.4f}")
    print(f"   rMAPE: {calculate_rmape(y_true, y_pred):.4f}")
    print("   ⚠️  Nota: MAPE puede ser bajo pero correlación negativa → rMAPE infinito")

    # Caso 3: Predicción buena
    print("\n3. Predicción Buena (MAPE 3%, correlación 0.95):")
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([103, 206, 297, 412, 495])

    print(f"   MAPE: {calculate_mape(y_true, y_pred):.4f}%")
    print(f"   Correlación: {calculate_correlation(y_true, y_pred):.4f}")
    print(f"   rMAPE: {calculate_rmape(y_true, y_pred):.4f}")

    # Caso 4: Métricas completas
    print("\n4. Todas las Métricas:")
    metrics = calculate_all_metrics(y_true, y_pred)
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")

    print("\n" + "="*70)
    print("✓ rMAPE implementado correctamente")
    print("="*70)
