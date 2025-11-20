"""
Modelos Base para Pronóstico de Demanda Energética
- XGBoost: Modelo principal (estado del arte)
- LightGBM: Modelo rápido para reentrenamiento frecuente
- RandomForest: Modelo robusto (fallback)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from abc import ABC, abstractmethod
import logging

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Clase base abstracta para todos los modelos"""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.feature_importance = None
        self.training_history = {}
        self.hyperparameters = kwargs

    @abstractmethod
    def build_model(self, **kwargs):
        """Construye el modelo con hiperparámetros"""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Entrena el modelo"""
        if self.model is None:
            self.build_model(**self.hyperparameters)

        logger.info(f"Entrenando {self.name}...")
        self.feature_names = list(X.columns)

        # Entrenar modelo
        self.model.fit(X, y, **kwargs)
        self.is_trained = True

        # Calcular feature importance
        self._calculate_feature_importance()

        logger.info(f"✓ {self.name} entrenado exitosamente")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Genera predicciones"""
        if not self.is_trained:
            raise ValueError(f"{self.name} no ha sido entrenado")

        return self.model.predict(X)

    def _calculate_feature_importance(self):
        """Calcula la importancia de features"""
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Retorna las N features más importantes"""
        if self.feature_importance is None:
            return None
        return self.feature_importance.head(top_n)

    def save(self, path: Path):
        """Guarda el modelo"""
        if not self.is_trained:
            raise ValueError("No se puede guardar un modelo no entrenado")

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'name': self.name,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'hyperparameters': self.hyperparameters
        }, path)

        logger.info(f"✓ {self.name} guardado en {path}")

    def load(self, path: Path):
        """Carga el modelo"""
        data = joblib.load(path)
        self.model = data['model']
        self.name = data['name']
        self.feature_names = data['feature_names']
        self.feature_importance = data.get('feature_importance')
        self.hyperparameters = data.get('hyperparameters', {})
        self.is_trained = True

        logger.info(f"✓ {self.name} cargado desde {path}")

    def get_params(self) -> Dict:
        """Retorna los parámetros del modelo"""
        if self.model is None:
            return self.hyperparameters
        return self.model.get_params()


class XGBoostModel(BaseModel):
    """
    XGBoost - Extreme Gradient Boosting
    Estado del arte para datos tabulares
    """

    def __init__(self, **kwargs):
        # Hiperparámetros por defecto optimizados para series temporales
        default_params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',  # Más rápido
            'objective': 'reg:squarederror'
        }
        default_params.update(kwargs)
        super().__init__("XGBoost", **default_params)

    def build_model(self, **kwargs):
        """Construye el modelo XGBoost"""
        self.model = xgb.XGBRegressor(**kwargs)


class LightGBMModel(BaseModel):
    """
    LightGBM - Light Gradient Boosting Machine
    Más rápido que XGBoost, similar performance
    Ideal para reentrenamiento automático frecuente
    """

    def __init__(self, **kwargs):
        # Hiperparámetros por defecto optimizados
        default_params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,  # Sin warnings
            'force_col_wise': True  # Más rápido
        }
        default_params.update(kwargs)
        super().__init__("LightGBM", **default_params)

    def build_model(self, **kwargs):
        """Construye el modelo LightGBM"""
        self.model = lgb.LGBMRegressor(**kwargs)


class RandomForestModel(BaseModel):
    """
    Random Forest - Modelo robusto y confiable
    Usado como fallback y baseline
    """

    def __init__(self, **kwargs):
        # Hiperparámetros por defecto
        default_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0
        }
        default_params.update(kwargs)
        super().__init__("RandomForest", **default_params)

    def build_model(self, **kwargs):
        """Construye el modelo Random Forest"""
        self.model = RandomForestRegressor(**kwargs)


# ============== UTILIDADES ==============

def create_model(model_type: str, **kwargs) -> BaseModel:
    """
    Factory function para crear modelos

    Args:
        model_type: Tipo de modelo ('xgboost', 'lightgbm', 'randomforest')
        **kwargs: Hiperparámetros del modelo

    Returns:
        Instancia del modelo

    Example:
        >>> model = create_model('xgboost', n_estimators=500)
    """
    model_map = {
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'randomforest': RandomForestModel
    }

    model_type = model_type.lower()

    if model_type not in model_map:
        raise ValueError(f"Modelo '{model_type}' no reconocido. "
                        f"Opciones: {list(model_map.keys())}")

    return model_map[model_type](**kwargs)


# ============== TESTING ==============

if __name__ == "__main__":
    print("="*70)
    print("TESTING MODELOS BASE")
    print("="*70)

    # Crear datos de prueba
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)

    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nDatos de prueba: {len(X_train)} train, {len(X_test)} test")

    # Probar cada modelo
    models = ['xgboost', 'lightgbm', 'randomforest']

    for model_name in models:
        print(f"\n{'='*70}")
        print(f"Probando {model_name.upper()}")
        print('='*70)

        # Crear y entrenar
        model = create_model(model_name)
        model.fit(X_train, y_train)

        # Predecir
        y_pred = model.predict(X_test)

        # Evaluar
        from sklearn.metrics import mean_squared_error, r2_score
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")

        # Feature importance
        print("\nTop 5 Features:")
        importance = model.get_feature_importance(top_n=5)
        if importance is not None:
            for idx, row in importance.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")

        # Guardar y cargar
        save_path = Path(f"test_{model_name}.joblib")
        model.save(save_path)

        model2 = create_model(model_name)
        model2.load(save_path)

        y_pred2 = model2.predict(X_test)
        assert np.allclose(y_pred, y_pred2), "Error: modelo cargado no coincide"

        save_path.unlink()  # Eliminar archivo de prueba

    print("\n" + "="*70)
    print("✓ Todos los modelos funcionan correctamente")
    print("="*70)
