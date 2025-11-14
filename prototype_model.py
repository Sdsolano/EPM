"""
Modelo Prototipo Rápido - Validación de Features
Sistema de Pronóstico de Demanda Energética - EPM

Este script entrena un modelo rápido para validar que las features
creadas en Fase 1 son útiles para predicción.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("MODELO PROTOTIPO RAPIDO - VALIDACION DE FEATURES")
print("="*70)

# ============== CARGAR DATOS ==============
print("\n1. Cargando datos con features...")
data_path = Path(__file__).parent / "data" / "features" / "data_with_features_latest.csv"

if not data_path.exists():
    print(f"ERROR: No se encuentra {data_path}")
    print("Ejecuta primero: python pipeline/orchestrator.py")
    exit(1)

df = pd.read_csv(data_path)
print(f"   - Datos cargados: {len(df)} registros")
print(f"   - Columnas: {len(df.columns)}")

# ============== PREPARAR DATOS ==============
print("\n2. Preparando datos para modelado...")

# Columnas de features (excluir targets y metadata)
exclude_cols = ['FECHA', 'TOTAL'] + [f'P{i}' for i in range(1, 25)]
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"   - Features disponibles: {len(feature_cols)}")

# Target: predecir TOTAL de demanda del día
X = df[feature_cols].copy()
y = df['TOTAL'].copy()

# Manejar valores faltantes
X = X.fillna(X.mean())

# Verificar que no hay infinitos
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print(f"   - Shape de X: {X.shape}")
print(f"   - Shape de y: {y.shape}")
print(f"   - Valores faltantes en X: {X.isnull().sum().sum()}")
print(f"   - Valores faltantes en y: {y.isnull().sum()}")

# Eliminar filas donde y es NaN
mask = ~y.isnull()
X = X[mask]
y = y[mask]

print(f"   - Datos finales: {len(X)} registros")

# ============== SPLIT TEMPORAL ==============
print("\n3. Dividiendo datos (temporal split)...")

# Split: 80% train, 20% test (respetando orden temporal)
split_idx = int(len(X) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"   - Train set: {len(X_train)} registros")
print(f"   - Test set: {len(X_test)} registros")

# ============== ENTRENAR MODELOS ==============
print("\n4. Entrenando 3 modelos rapidos...")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n   Entrenando {name}...")

    # Entrenar
    model.fit(X_train, y_train)

    # Predecir
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calcular métricas
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # MAPE (Mean Absolute Percentage Error)
    train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

    results[name] = {
        'model': model,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mape': train_mape,
        'test_mape': test_mape,
        'predictions': y_test_pred
    }

    print(f"      Train MAE: {train_mae:.2f}")
    print(f"      Test MAE: {test_mae:.2f}")
    print(f"      Test MAPE: {test_mape:.2f}%")

# ============== RESULTADOS ==============
print("\n" + "="*70)
print("RESULTADOS COMPARATIVOS")
print("="*70)

# Tabla de resultados
print("\n{:<20} {:>12} {:>12} {:>12} {:>10}".format(
    "Modelo", "Test MAE", "Test RMSE", "Test R²", "Test MAPE"
))
print("-" * 70)

for name, res in results.items():
    print("{:<20} {:>12.2f} {:>12.2f} {:>12.3f} {:>9.2f}%".format(
        name,
        res['test_mae'],
        res['test_rmse'],
        res['test_r2'],
        res['test_mape']
    ))

# ============== MEJOR MODELO ==============
print("\n" + "="*70)
best_model_name = min(results.items(), key=lambda x: x[1]['test_mape'])[0]
best_result = results[best_model_name]

print(f"MEJOR MODELO: {best_model_name}")
print("="*70)
cumple = "CUMPLE" if best_result['test_mape'] < 5 else "NO CUMPLE"
print(f"Test MAPE: {best_result['test_mape']:.2f}% {cumple} (objetivo: <5%)")
print(f"Test MAE: {best_result['test_mae']:.2f}")
print(f"Test RMSE: {best_result['test_rmse']:.2f}")
print(f"Test R²: {best_result['test_r2']:.3f}")

# ============== FEATURE IMPORTANCE ==============
if hasattr(results[best_model_name]['model'], 'feature_importances_'):
    print("\n" + "="*70)
    print("TOP 15 FEATURES MAS IMPORTANTES")
    print("="*70)

    importances = results[best_model_name]['model'].feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.head(15).iterrows():
        print(f"{row['feature']:<40} {row['importance']:.4f}")

    # Guardar feature importance completa
    importance_path = Path(__file__).parent / "data" / "features" / "feature_importance_prototype.csv"
    feature_importance.to_csv(importance_path, index=False)
    print(f"\nFeature importance guardada en: {importance_path}")

# ============== ANALISIS DE ERRORES ==============
print("\n" + "="*70)
print("ANALISIS DE ERRORES")
print("="*70)

y_test_pred = best_result['predictions']
errors = np.abs(y_test.values - y_test_pred)
errors_pct = (errors / y_test.values) * 100

print(f"Error promedio: {errors.mean():.2f}")
print(f"Error mediano: {np.median(errors):.2f}")
print(f"Error máximo: {errors.max():.2f}")
print(f"Error mínimo: {errors.min():.2f}")
print(f"\nDistribución de errores porcentuales:")
print(f"  - Errores < 1%: {(errors_pct < 1).sum()} días ({(errors_pct < 1).sum() / len(errors_pct) * 100:.1f}%)")
print(f"  - Errores < 3%: {(errors_pct < 3).sum()} días ({(errors_pct < 3).sum() / len(errors_pct) * 100:.1f}%)")
print(f"  - Errores < 5%: {(errors_pct < 5).sum()} días ({(errors_pct < 5).sum() / len(errors_pct) * 100:.1f}%)")
print(f"  - Errores > 5%: {(errors_pct > 5).sum()} días ({(errors_pct > 5).sum() / len(errors_pct) * 100:.1f}%)")

# ============== VALIDACION CRUZADA TEMPORAL ==============
print("\n" + "="*70)
print("VALIDACION CRUZADA TEMPORAL (3 Folds)")
print("="*70)

tscv = TimeSeriesSplit(n_splits=3)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_fold_train = X.iloc[train_idx]
    y_fold_train = y.iloc[train_idx]
    X_fold_val = X.iloc[val_idx]
    y_fold_val = y.iloc[val_idx]

    # Entrenar modelo ganador
    model = results[best_model_name]['model'].__class__(**results[best_model_name]['model'].get_params())
    model.fit(X_fold_train, y_fold_train)

    y_fold_pred = model.predict(X_fold_val)
    mape = np.mean(np.abs((y_fold_val - y_fold_pred) / y_fold_val)) * 100

    cv_scores.append(mape)
    print(f"Fold {fold}: MAPE = {mape:.2f}%")

print(f"\nMAPE promedio (CV): {np.mean(cv_scores):.2f}%")
print(f"Desviación estándar: {np.std(cv_scores):.2f}%")

# ============== GUARDAR RESULTADOS ==============
print("\n" + "="*70)
print("GUARDANDO RESULTADOS")
print("="*70)

# Guardar predicciones del mejor modelo
predictions_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_test_pred,
    'error': errors,
    'error_pct': errors_pct
})

predictions_path = Path(__file__).parent / "data" / "features" / "prototype_predictions.csv"
predictions_df.to_csv(predictions_path, index=False)
print(f"Predicciones guardadas en: {predictions_path}")

# Guardar predicciones de TODOS los modelos para comparación
all_predictions = pd.DataFrame({
    'actual': y_test.values
})

for name, res in results.items():
    all_predictions[f'pred_{name.lower().replace(" ", "_")}'] = res['predictions']

all_predictions_path = Path(__file__).parent / "data" / "features" / "prototype_all_models_predictions.csv"
all_predictions.to_csv(all_predictions_path, index=False)
print(f"Predicciones de todos los modelos guardadas en: {all_predictions_path}")

# Resumen
summary = {
    'best_model': best_model_name,
    'test_mape': float(best_result['test_mape']),
    'test_mae': float(best_result['test_mae']),
    'test_rmse': float(best_result['test_rmse']),
    'test_r2': float(best_result['test_r2']),
    'cumple_objetivo_5pct': bool(best_result['test_mape'] < 5),
    'cv_mape_mean': float(np.mean(cv_scores)),
    'cv_mape_std': float(np.std(cv_scores)),
    'total_features': int(len(feature_cols)),
    'train_size': int(len(X_train)),
    'test_size': int(len(X_test))
}

import json
summary_path = Path(__file__).parent / "data" / "features" / "prototype_summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Resumen guardado en: {summary_path}")

# ============== CONCLUSION ==============
print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if best_result['test_mape'] < 5:
    print(">> El modelo CUMPLE con el objetivo regulatorio de MAPE < 5%")
    print(">> Las features creadas en Fase 1 son EFECTIVAS para prediccion")
    print(">> Se puede proceder con confianza a la Fase 2 completa")
else:
    print(">> El modelo NO cumple con MAPE < 5% (pero es solo un prototipo rapido)")
    print(">> Considerar:")
    print("  - Modelos mas sofisticados en Fase 2")
    print("  - Ajuste de hiperparametros")
    print("  - Feature selection mas riguroso")
    print("  - Ingenieria de features adicionales")

print("\n" + "="*70)
print("PROTOTIPO COMPLETADO")
print("="*70)
