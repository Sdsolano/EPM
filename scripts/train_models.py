"""
Script de Entrenamiento Completo - Fase 2
Entrena todos los modelos (XGBoost, LightGBM, RandomForest)
Registra modelos y selecciona el campeón automáticamente
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
import os

# Configurar encoding para Windows
if os.name == 'nt':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

warnings.filterwarnings('ignore')

# Añadir raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.trainer import ModelTrainer
from src.models.registry import ModelRegistry
from src.models.metrics import calculate_all_metrics, evaluate_model_performance

print("="*80)
print("SISTEMA DE ENTRENAMIENTO AUTOMÁTICO DE MODELOS - FASE 2")
print("Sistema de Pronóstico de Demanda Energética EPM")
print("="*80)

# ============== CONFIGURACIÓN ==============

OPTIMIZE_HYPERPARAMS = False  # Cambiar a True para optimización Bayesiana (más lento)
CV_SPLITS = 3
MODELS_TO_TRAIN = ['xgboost', 'lightgbm', 'randomforest']

# ============== CARGAR DATOS ==============

print("\n" + "="*80)
print("1. CARGANDO DATOS")
print("="*80)

data_path = Path(__file__).parent.parent / "data" / "features" / "data_with_features_latest.csv"

if not data_path.exists():
    print(f"\n❌ ERROR: No se encuentra {data_path}")
    print("   Ejecuta primero: python scripts/run_pipeline.py")
    sys.exit(1)

df = pd.read_csv(data_path)
print(f"\n✓ Datos cargados: {len(df)} registros")
print(f"  Rango de fechas: {df['FECHA'].min()} a {df['FECHA'].max()}")

# ============== PREPARAR DATOS ==============

print("\n" + "="*80)
print("2. PREPARANDO DATOS PARA MODELADO")
print("="*80)

# Separar features y target
# IMPORTANTE: Excluir features de lag para evitar train-test mismatch en predicción recursiva
FEATURES_LAG_TO_EXCLUDE = [
    'total_lag_1d', 'total_lag_7d', 'total_lag_14d',
    'p8_lag_1d', 'p8_lag_7d',
    'p12_lag_1d', 'p12_lag_7d',
    'p18_lag_1d', 'p18_lag_7d',
    'p20_lag_1d', 'p20_lag_7d',
    'total_day_change', 'total_day_change_pct'
]

exclude_cols = ['FECHA', 'TOTAL'] + [f'P{i}' for i in range(1, 25)] + FEATURES_LAG_TO_EXCLUDE
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"\n⚠️  NOTA: Excluyendo {len(FEATURES_LAG_TO_EXCLUDE)} features de lag para mejor predicción recursiva")

print(f"\n  Features disponibles: {len(feature_cols)}")
print(f"  Target: TOTAL (demanda diaria)")

X = df[feature_cols].copy()
y = df['TOTAL'].copy()

# Manejar valores faltantes
print(f"\n  Valores faltantes en X: {X.isnull().sum().sum()}")
print(f"  Valores faltantes en y: {y.isnull().sum()}")

X = X.fillna(X.mean()).replace([np.inf, -np.inf], 0)

# Eliminar filas donde y es NaN
mask = ~y.isnull()
X = X[mask]
y = y[mask]
df = df[mask]  # Aplicar la misma máscara al DataFrame

print(f"\n  Datos finales: {len(X)} registros")
print(f"  Shape de X: {X.shape}")

# ============== SPLIT TEMPORAL ==============

print("\n" + "="*80)
print("3. SPLIT TEMPORAL (80% TRAIN, 20% TEST)")
print("="*80)

# Split temporal: 80% train, 20% test
split_idx = int(len(X) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"\n  Train set: {len(X_train)} registros")
print(f"  Test set: {len(X_test)} registros")

# Guardar fechas para referencia
train_dates = df['FECHA'].iloc[:split_idx]
test_dates = df['FECHA'].iloc[split_idx:]

print(f"  Train period: {train_dates.min()} a {train_dates.max()}")
print(f"  Test period: {test_dates.min()} a {test_dates.max()}")

# ============== ENTRENAR MODELOS ==============

print("\n" + "="*80)
print("4. ENTRENAMIENTO DE MODELOS")
print("="*80)

if OPTIMIZE_HYPERPARAMS:
    print("\n⚙️  Optimización Bayesiana HABILITADA (esto tomará más tiempo...)")
else:
    print("\n⚡ Optimización Bayesiana DESHABILITADA (entrenamiento rápido)")

trainer = ModelTrainer(
    optimize_hyperparams=OPTIMIZE_HYPERPARAMS,
    n_optimization_iter=20,
    cv_splits=CV_SPLITS
)

# Entrenar todos los modelos
trained_models = trainer.train_all_models(
    X_train, y_train,
    X_test, y_test,
    models=MODELS_TO_TRAIN
)

# ============== COMPARAR MODELOS ==============

print("\n" + "="*80)
print("5. COMPARACIÓN DE MODELOS")
print("="*80)

# Tabla comparativa
print("\n{:<15} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "Modelo", "Train MAPE", "Val MAPE", "Train rMAPE", "Val rMAPE", "Val R²"
))
print("-" * 80)

for model_name, (model, results) in trained_models.items():
    train_metrics = results['train_metrics']
    val_metrics = results.get('val_metrics', {})

    print("{:<15} {:>9.2f}% {:>9.2f}% {:>10.4f} {:>10.4f} {:>10.4f}".format(
        model_name.upper(),
        train_metrics['mape'],
        val_metrics.get('mape', 0),
        train_metrics['rmape'],
        val_metrics.get('rmape', 0),
        val_metrics.get('r2', 0)
    ))

# ============== SELECCIONAR MEJOR MODELO ==============

print("\n" + "="*80)
print("6. SELECCIÓN DEL MEJOR MODELO")
print("="*80)

best_name, best_model, best_results = trainer.select_best_model(
    criterion='rmape',
    use_validation=True
)

# ============== EVALUACIÓN FINAL EN TEST ==============

print("\n" + "="*80)
print("7. EVALUACIÓN FINAL EN TEST SET")
print("="*80)

y_test_pred = best_model.predict(X_test)

# Métricas completas
test_evaluation = evaluate_model_performance(y_test, y_test_pred, threshold_mape=5.0)

print(f"\n📊 MÉTRICAS DEL MODELO CAMPEÓN: {best_name.upper()}")
print("="*80)

metrics = test_evaluation['metrics']
print(f"\n  MAPE: {metrics['mape']:.4f}%")
print(f"  rMAPE: {metrics['rmape']:.4f}")
print(f"  MAE: {metrics['mae']:.2f}")
print(f"  RMSE: {metrics['rmse']:.2f}")
print(f"  R²: {metrics['r2']:.4f}")
print(f"  Correlación: {metrics['correlation']:.4f}")

# Cumplimiento regulatorio
print(f"\n📋 CUMPLIMIENTO REGULATORIO:")
print("="*80)

compliance = test_evaluation['regulatory_compliance']
cumple = "✅ CUMPLE" if compliance['cumple_mape_5pct'] else "❌ NO CUMPLE"
print(f"  MAPE < 5%: {cumple}")
print(f"  Días con error < 5%: {compliance['dias_con_error_menor_5pct']}/{compliance['total_dias']}")
print(f"  Porcentaje cumplimiento: {compliance['porcentaje_dias_cumplimiento']:.1f}%")

# Distribución de errores
print(f"\n📈 DISTRIBUCIÓN DE ERRORES:")
print("="*80)

error_dist = test_evaluation['error_distribution']
print(f"  Error promedio: {error_dist['error_mean']:.2f}")
print(f"  Error mediano: {error_dist['error_median']:.2f}")
print(f"  Error máximo: {error_dist['error_max']:.2f}")
print(f"  Error mínimo: {error_dist['error_min']:.2f}")
print(f"  Desv. estándar: {error_dist['error_std']:.2f}")

# ============== REGISTRAR MODELOS ==============

print("\n" + "="*80)
print("8. REGISTRANDO MODELOS EN REGISTRY")
print("="*80)

registry = ModelRegistry()

# Registrar todos los modelos entrenados
for model_name, (model, results) in trained_models.items():
    val_metrics = results.get('val_metrics', results['train_metrics'])

    model_id = registry.register_model(
        model=model,
        model_name=model_name,
        metrics=val_metrics,
        metadata={
            'training_time': results['training_time'],
            'n_features': results['n_features'],
            'n_train_samples': results['n_train_samples'],
            'cv_results': results.get('cv_results', {})
        }
    )

# Seleccionar y promocionar campeón
print("\n🏆 Seleccionando modelo campeón...")
champion_id = registry.select_best_and_promote(criterion='rmape')

# ============== GUARDAR MODELOS ==============

print("\n" + "="*80)
print("9. GUARDANDO MODELOS")
print("="*80)

timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
saved_paths = trainer.save_all_models(timestamp=timestamp)

print(f"\n✓ Modelos guardados:")
for name, path in saved_paths.items():
    print(f"    {name}: {path}")

# Guardar predicciones
predictions_df = pd.DataFrame({
    'fecha': test_dates.values,
    'actual': y_test.values,
    'predicted': y_test_pred,
    'error': np.abs(y_test.values - y_test_pred),
    'error_pct': np.abs((y_test.values - y_test_pred) / y_test.values) * 100
})

predictions_path = Path(__file__).parent / "data" / "features" / f"predictions_{timestamp}.csv"
predictions_df.to_csv(predictions_path, index=False)
print(f"\n✓ Predicciones guardadas en: {predictions_path}")

# Guardar feature importance
if best_model.feature_importance is not None:
    importance_path = Path(__file__).parent / "data" / "features" / f"feature_importance_{timestamp}.csv"
    best_model.feature_importance.to_csv(importance_path, index=False)
    print(f"✓ Feature importance guardado en: {importance_path}")

    # Mostrar top 10
    print(f"\n📊 TOP 10 FEATURES MÁS IMPORTANTES:")
    print("="*80)
    for idx, row in best_model.feature_importance.head(10).iterrows():
        print(f"  {row['feature']:<40} {row['importance']:.4f}")

# ============== RESUMEN FINAL ==============

print("\n" + "="*80)
print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
print("="*80)

print(f"\n📝 RESUMEN:")
print(f"  • Modelos entrenados: {len(trained_models)}")
print(f"  • Modelo campeón: {best_name.upper()}")
print(f"  • MAPE en test: {metrics['mape']:.4f}%")
print(f"  • rMAPE en test: {metrics['rmape']:.4f}")
print(f"  • Cumple objetivo regulatorio: {'SÍ' if compliance['cumple_mape_5pct'] else 'NO'}")

print(f"\n📂 ARCHIVOS GENERADOS:")
print(f"  • Modelos: models/trained/")
print(f"  • Registry: models/registry/")
print(f"  • Predicciones: {predictions_path.name}")

print(f"\n🚀 PRÓXIMOS PASOS:")
print(f"  1. Revisar feature importance y predicciones")
print(f"  2. Implementar sistema Auditor (reentrenamiento automático)")
print(f"  3. Crear API Gateway REST")

print("\n" + "="*80)
