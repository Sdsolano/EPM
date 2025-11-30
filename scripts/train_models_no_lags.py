"""
Script de Entrenamiento SIN LAGS - Para mejorar predicción recursiva
Entrena modelos (XGBoost, LightGBM, RandomForest) eliminando features de lag directo
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
import os

# Configurar encoding y logging para Windows
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

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
print("ENTRENAMIENTO SIN FEATURES DE LAG DIRECTO")
print("Sistema de Pronóstico de Demanda Energética EPM")
print("="*80)

# ============== CONFIGURACIÓN ==============

OPTIMIZE_HYPERPARAMS = False
CV_SPLITS = 3
MODELS_TO_TRAIN = ['xgboost', 'lightgbm', 'randomforest']

# Features de LAG a ELIMINAR (causan train-test mismatch en predicción recursiva)
FEATURES_TO_REMOVE = [
    'total_lag_1d',
    'total_lag_7d', 
    'total_lag_14d',
    'p8_lag_1d',
    'p8_lag_7d',
    'p12_lag_1d',
    'p12_lag_7d',
    'p18_lag_1d',
    'p18_lag_7d',
    'p20_lag_1d',
    'p20_lag_7d',
    'total_day_change',      # Depende de lag_1d
    'total_day_change_pct'   # Depende de lag_1d
]

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

# Crear columna de fecha si no existe
if 'FECHA' not in df.columns and 'fecha' not in df.columns:
    if 'year' in df.columns and 'month' in df.columns and 'day' in df.columns:
        df['FECHA'] = pd.to_datetime(df[['year', 'month', 'day']])
        print(f"  ✓ Columna FECHA creada desde year/month/day")
    else:
        print(f"  ⚠️ No se puede determinar rango de fechas")
        
if 'FECHA' in df.columns:
    print(f"  Rango de fechas: {df['FECHA'].min()} a {df['FECHA'].max()}")

# ============== PREPARAR DATOS ==============

print("\n" + "="*80)
print("2. PREPARANDO DATOS (ELIMINANDO LAGS)")
print("="*80)

# Separar features y target
exclude_cols = ['FECHA', 'TOTAL'] + [f'P{i}' for i in range(1, 25)]
exclude_cols.extend(FEATURES_TO_REMOVE)  # ← AGREGAR LAGS A EXCLUSIÓN

feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"\n  Features lag eliminados: {len(FEATURES_TO_REMOVE)}")
for feat in FEATURES_TO_REMOVE:
    if feat in df.columns:
        print(f"    ❌ {feat}")

print(f"\n  Features finales: {len(feature_cols)}")
print(f"  Target: TOTAL (demanda diaria)")

# Mostrar algunos features que SÍ se usarán
print(f"\n  Ejemplos de features a usar:")
for feat in feature_cols[:15]:
    print(f"    ✓ {feat}")

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
df = df[mask]

print(f"\n  Datos finales: {len(X)} registros")
print(f"  Shape de X: {X.shape}")

# ============== SPLIT TEMPORAL ==============

print("\n" + "="*80)
print("3. SPLIT TEMPORAL (80% TRAIN, 20% TEST)")
print("="*80)

split_idx = int(len(X) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"\n  Train set: {len(X_train)} registros")
print(f"  Test set: {len(X_test)} registros")

if 'FECHA' in df.columns:
    train_dates = df['FECHA'].iloc[:split_idx]
    test_dates = df['FECHA'].iloc[split_idx:]
    print(f"  Train period: {train_dates.min()} a {train_dates.max()}")
    print(f"  Test period: {test_dates.min()} a {test_dates.max()}")

# ============== ENTRENAR MODELOS ==============

print("\n" + "="*80)
print("4. ENTRENAMIENTO DE MODELOS")
print("="*80)

if OPTIMIZE_HYPERPARAMS:
    print("\n⚙️  Optimización Bayesiana HABILITADA")
else:
    print("\n⚙️  Usando hiperparámetros por defecto (optimizados previamente)")

trainer = ModelTrainer(
    optimize_hyperparams=OPTIMIZE_HYPERPARAMS,
    cv_splits=CV_SPLITS
)

trained_models = trainer.train_all_models(
    X_train, y_train,
    X_test, y_test,
    models=MODELS_TO_TRAIN
)

# ============== COMPARAR MODELOS ==============

print("\n" + "="*80)
print("5. COMPARACIÓN DE MODELOS")
print("="*80)

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

print(f"\n🏆 MEJOR MODELO: {best_name.upper()}")
print(f"  Val rMAPE: {best_results['val_metrics']['rmape']:.4f}%")
print(f"  Val MAPE:  {best_results['val_metrics']['mape']:.4f}%")
print(f"  Val R²:    {best_results['val_metrics']['r2']:.4f}")

# ============== REGISTRAR MODELOS ==============

print("\n" + "="*80)
print("7. REGISTRANDO MODELOS")
print("="*80)

registry = ModelRegistry()

for model_name, (model, results) in trained_models.items():
    # Preparar metadata adicional
    metadata = {
        'training_time': results.get('training_time', 0),
        'n_features': X_train.shape[1],
        'n_train_samples': len(X_train),
        'feature_names': list(X_train.columns),  # Guardar feature names aquí
        'features_removed': FEATURES_TO_REMOVE,
        'n_features_removed': len(FEATURES_TO_REMOVE)
    }
    
    if 'cv_results' in results:
        metadata['cv_results'] = results['cv_results']
    
    # Registrar modelo (pasa el modelo directamente, no como dict)
    model_id = registry.register_model(
        model=model,
        model_name=model_name,
        metrics=results['val_metrics'],
        metadata=metadata
    )
    
    print(f"✓ {model_name.upper()} registrado: rMAPE = {results['val_metrics']['rmape']:.4f}%")

# ============== PROMOVER CAMPEÓN ==============

print("\n" + "="*80)
print("8. PROMOVIENDO MODELO CAMPEÓN")
print("="*80)

# Obtener el model_id del mejor modelo (que acabamos de registrar)
# El model_id tiene formato: {model_name}_{timestamp}
best_model_ids = [mid for mid in registry.metadata['models'].keys() 
                  if mid.startswith(best_name)]
if best_model_ids:
    # Tomar el más reciente (último en la lista)
    champion_id = sorted(best_model_ids)[-1]
    registry.promote_to_champion(champion_id)
    
    # Acceder directamente a los metadatos
    champion_info = registry.metadata['models'][champion_id]
    
    print(f"\n🏆 MODELO CAMPEÓN ACTUALIZADO: {champion_info['model_name'].upper()}")
    print(f"  Versión: {champion_info['version']}")
    print(f"  rMAPE: {champion_info['metrics']['rmape']:.4f}%")
    print(f"  Features: {champion_info['metadata']['n_features']}")
    print(f"  Features eliminados: {champion_info['metadata']['n_features_removed']}")
else:
    print(f"⚠️  No se pudo encontrar el modelo {best_name} para promover")

print("\n" + "="*80)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*80)
print(f"\n📝 NOTA IMPORTANTE:")
print(f"   Este modelo NO usa features de lag directo (lag_1d, lag_7d, etc.)")
print(f"   Debería tener mejor rendimiento en predicción recursiva (30 días)")
print(f"   porque no depende de predicciones anteriores contaminadas.")
print(f"\n📁 Modelo campeón guardado en: models/registry/champion_model.joblib")
print(f"\n🔄 PRÓXIMO PASO: Probar endpoint /predict con este nuevo modelo")

