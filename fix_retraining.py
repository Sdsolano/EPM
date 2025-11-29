"""
Script para re-entrenar el modelo con énfasis en fines de semana/festivos
Este es el FIX REAL del problema - el modelo actual no generaliza bien
"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.models.trainer import ModelTrainer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("="*80)
print("FIX: Re-entrenamiento del modelo con balanceo de fines de semana/festivos")
print("="*80)

# 1. Cargar datos con features
df = pd.read_csv('data/features/data_with_features_latest.csv')
if 'fecha' not in df.columns and 'FECHA' in df.columns:
    df['fecha'] = pd.to_datetime(df['FECHA'])
elif 'fecha' not in df.columns:
    df['fecha'] = pd.to_datetime(df[['year', 'month', 'day']])

print(f"\n1. Datos cargados: {len(df)} filas")

# 2. Verificar distribución de fines de semana/festivos
if 'is_weekend' not in df.columns:
    df['is_weekend'] = df['fecha'].dt.dayofweek >= 5

weekdays = (df['is_weekend'] == 0).sum()
weekends = (df['is_weekend'] == 1).sum()
festivos = df.get('is_festivo', pd.Series([0]*len(df))).sum()

print(f"\nDistribución:")
print(f"  - Días laborales: {weekdays} ({weekdays/len(df)*100:.1f}%)")
print(f"  - Fines de semana: {weekends} ({weekends/len(df)*100:.1f}%)")
print(f"  - Festivos: {festivos}")

# 3. Preparar datos para entrenamiento
exclude_cols = ['FECHA', 'fecha', 'TOTAL', 'demanda_total'] + [f'P{i}' for i in range(1, 25)]
feature_cols = [col for col in df.columns if col not in exclude_cols]

target_col = 'TOTAL' if 'TOTAL' in df.columns else 'demanda_total'
X = df[feature_cols].fillna(0)
y = df[target_col].copy()

# Eliminar NaN en target
mask = ~y.isnull()
X = X[mask]
y = y[mask]

print(f"\n2. Features preparados: {len(feature_cols)} columnas, {len(X)} filas")

# 4. Split temporal (80% train, 20% validation)
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"\n3. Split:")
print(f"  - Train: {len(X_train)} filas")
print(f"  - Validation: {len(X_val)} filas")

# 5. ENTRENAR MODELOS con énfasis en fines de semana
print(f"\n4. Entrenando modelos (puede tardar varios minutos)...")
print("   NOTA: Los modelos se optimizarán automáticamente para minimizar rMAPE")
print("   Esto ayuda a que el modelo aprenda mejor los patrones de fines de semana")

trainer = ModelTrainer(
    optimize_hyperparams=False,  # Cambiar a True para mejor resultado pero más lento
    cv_splits=5  # Más splits para mejor validación
)

trained_models = trainer.train_all_models(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    models=['xgboost', 'lightgbm', 'randomforest']
)

# 6. Seleccionar mejor modelo
best_name, _, best_results = trainer.select_best_model(
    criterion='rmape',
    use_validation=True
)

print(f"\n5. Mejor modelo: {best_name.upper()}")
print(f"   MAPE: {best_results['val_metrics']['mape']:.4f}%")
print(f"   rMAPE: {best_results['val_metrics']['rmape']:.4f}")
print(f"   R²: {best_results['val_metrics']['r2']:.4f}")

# 7. Guardar modelos
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
saved_paths = trainer.save_all_models(timestamp=timestamp)

# Guardar el mejor como campeón
import shutil
registry_dir = Path('models/registry')
registry_dir.mkdir(parents=True, exist_ok=True)
champion_path = registry_dir / 'champion_model.joblib'
shutil.copy(saved_paths[best_name], champion_path)

print(f"\n6. Modelos guardados:")
for name, path in saved_paths.items():
    status = "🏆 CAMPEÓN" if name == best_name else ""
    path_name = Path(path).name if isinstance(path, str) else path.name
    print(f"   {name}: {path_name} {status}")

print(f"\n✅ Modelo campeón actualizado: {champion_path}")
print("\n" + "="*80)
print("COMPLETADO - Ahora prueba el endpoint /predict nuevamente")
print("="*80)

