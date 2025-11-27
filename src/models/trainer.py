"""
Sistema de Entrenamiento Automático de Modelos
Incluye optimización Bayesiana de hiperparámetros
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Optimización Bayesiana
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    logging.warning("scikit-optimize no disponible. Optimización Bayesiana deshabilitada.")

from src.models.base_models import create_model, BaseModel
from src.models.metrics import calculate_all_metrics, calculate_rmape

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Entrenador de modelos con optimización automática de hiperparámetros
    """

    def __init__(self,
                 models_dir: Path = None,
                 optimize_hyperparams: bool = False,
                 n_optimization_iter: int = 20,
                 cv_splits: int = 3):
        """
        Args:
            models_dir: Directorio para guardar modelos
            optimize_hyperparams: Si True, usa optimización Bayesiana
            n_optimization_iter: Iteraciones para optimización Bayesiana
            cv_splits: Número de splits para validación cruzada temporal
        """
        self.models_dir = models_dir or Path('models/trained')
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.optimize_hyperparams = optimize_hyperparams and BAYESIAN_AVAILABLE
        self.n_optimization_iter = n_optimization_iter
        self.cv_splits = cv_splits

        self.trained_models = {}
        self.training_results = {}

    def train_single_model(self,
                          model_type: str,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_val: Optional[pd.DataFrame] = None,
                          y_val: Optional[pd.Series] = None,
                          **model_kwargs) -> Tuple[BaseModel, Dict]:
        """
        Entrena un modelo individual

        Args:
            model_type: Tipo de modelo ('xgboost', 'lightgbm', 'randomforest')
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_val: Features de validación (opcional)
            y_val: Target de validación (opcional)
            **model_kwargs: Hiperparámetros adicionales

        Returns:
            Tupla (modelo entrenado, métricas)
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Entrenando modelo: {model_type.upper()}")
        logger.info(f"{'='*70}")

        start_time = datetime.now()

        # Optimizar hiperparámetros si está habilitado
        if self.optimize_hyperparams:
            logger.info("Optimizando hiperparámetros (Bayesian Optimization)...")
            best_params = self._optimize_hyperparameters(
                model_type, X_train, y_train
            )
            model_kwargs.update(best_params)
            logger.info(f"Mejores parámetros encontrados: {best_params}")

        # Crear y entrenar modelo
        model = create_model(model_type, **model_kwargs)
        model.fit(X_train, y_train)

        # Predecir en train
        y_train_pred = model.predict(X_train)
        train_metrics = calculate_all_metrics(y_train, y_train_pred)

        results = {
            'model_type': model_type,
            'train_metrics': train_metrics,
            'training_time': (datetime.now() - start_time).total_seconds(),
            'hyperparameters': model.get_params(),
            'n_features': len(X_train.columns),
            'n_train_samples': len(X_train)
        }

        # Evaluar en validación si está disponible
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            val_metrics = calculate_all_metrics(y_val, y_val_pred)
            results['val_metrics'] = val_metrics

            logger.info(f"\n  Train MAPE: {train_metrics['mape']:.4f}%")
            logger.info(f"  Train rMAPE: {train_metrics['rmape']:.4f}")
            logger.info(f"  Val MAPE: {val_metrics['mape']:.4f}%")
            logger.info(f"  Val rMAPE: {val_metrics['rmape']:.4f}")
        else:
            logger.info(f"\n  Train MAPE: {train_metrics['mape']:.4f}%")
            logger.info(f"  Train rMAPE: {train_metrics['rmape']:.4f}")

        # Validación cruzada temporal
        logger.info(f"\nValidación cruzada temporal ({self.cv_splits} folds)...")
        cv_results = self._cross_validate_temporal(model_type, X_train, y_train, model_kwargs)
        results['cv_results'] = cv_results

        logger.info(f"  CV rMAPE medio: {cv_results['mean_rmape']:.4f} ± {cv_results['std_rmape']:.4f}")

        logger.info(f"\n✓ {model_type.upper()} entrenado en {results['training_time']:.2f}s")

        return model, results

    def train_all_models(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: Optional[pd.DataFrame] = None,
                        y_val: Optional[pd.Series] = None,
                        models: List[str] = None) -> Dict[str, Tuple[BaseModel, Dict]]:
        """
        Entrena todos los modelos especificados

        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_val: Features de validación
            y_val: Target de validación
            models: Lista de modelos a entrenar (default: todos)

        Returns:
            Diccionario {model_name: (model, results)}
        """
        if models is None:
            models = ['xgboost', 'lightgbm', 'randomforest']

        logger.info("\n" + "="*70)
        logger.info(f"ENTRENANDO {len(models)} MODELOS")
        logger.info("="*70)
        logger.info(f"  Train samples: {len(X_train)}")
        logger.info(f"  Val samples: {len(X_val) if X_val is not None else 0}")
        logger.info(f"  Features: {len(X_train.columns)}")

        trained_models = {}

        for model_type in models:
            try:
                model, results = self.train_single_model(
                    model_type, X_train, y_train, X_val, y_val
                )

                trained_models[model_type] = (model, results)
                self.trained_models[model_type] = model
                self.training_results[model_type] = results

            except Exception as e:
                logger.error(f"Error entrenando {model_type}: {str(e)}")
                continue

        return trained_models

    def select_best_model(self,
                         criterion: str = 'rmape',
                         use_validation: bool = True) -> Tuple[str, BaseModel, Dict]:
        """
        Selecciona el mejor modelo basado en un criterio

        Args:
            criterion: Métrica para selección ('rmape', 'mape', 'mae', 'r2')
            use_validation: Si True, usa métricas de validación; si False, usa train

        Returns:
            Tupla (nombre_modelo, modelo, resultados)
        """
        if not self.trained_models:
            raise ValueError("No hay modelos entrenados")

        best_model_name = None
        best_score = np.inf if criterion in ['rmape', 'mape', 'mae', 'rmse'] else -np.inf
        minimize = criterion in ['rmape', 'mape', 'mae', 'rmse']

        for model_name, results in self.training_results.items():
            # Seleccionar métricas
            if use_validation and 'val_metrics' in results:
                metrics = results['val_metrics']
            else:
                metrics = results['train_metrics']

            score = metrics.get(criterion)

            if score is None:
                continue

            # Comparar
            if minimize:
                if score < best_score:
                    best_score = score
                    best_model_name = model_name
            else:
                if score > best_score:
                    best_score = score
                    best_model_name = model_name

        if best_model_name is None:
            raise ValueError(f"No se pudo seleccionar modelo con criterio '{criterion}'")

        logger.info(f"\n{'='*70}")
        logger.info(f"MEJOR MODELO SELECCIONADO: {best_model_name.upper()}")
        logger.info(f"{'='*70}")
        logger.info(f"  Criterio: {criterion} = {best_score:.4f}")

        return (
            best_model_name,
            self.trained_models[best_model_name],
            self.training_results[best_model_name]
        )

    def save_all_models(self, timestamp: str = None):
        """Guarda todos los modelos entrenados"""
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        saved_paths = {}

        for model_name, model in self.trained_models.items():
            save_path = self.models_dir / f"{model_name}_{timestamp}.joblib"
            model.save(save_path)
            saved_paths[model_name] = str(save_path)

        # Guardar resultados de entrenamiento
        results_path = self.models_dir / f"training_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(self.training_results, f, indent=2, default=str)

        logger.info(f"\n✓ Todos los modelos guardados en {self.models_dir}")

        return saved_paths

    def _optimize_hyperparameters(self,
                                  model_type: str,
                                  X: pd.DataFrame,
                                  y: pd.Series) -> Dict:
        """
        Optimiza hiperparámetros usando Bayesian Optimization

        Args:
            model_type: Tipo de modelo
            X: Features
            y: Target

        Returns:
            Mejores hiperparámetros encontrados
        """
        # Definir espacios de búsqueda
        search_spaces = {
            'xgboost': {
                'n_estimators': Integer(100, 500),
                'max_depth': Integer(3, 10),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0),
                'min_child_weight': Integer(1, 7),
                'gamma': Real(0, 0.5)
            },
            'lightgbm': {
                'n_estimators': Integer(100, 500),
                'max_depth': Integer(3, 10),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'num_leaves': Integer(20, 50),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0),
                'min_child_samples': Integer(10, 50)
            },
            'randomforest': {
                'n_estimators': Integer(100, 300),
                'max_depth': Integer(5, 20),
                'min_samples_split': Integer(2, 10),
                'min_samples_leaf': Integer(1, 5)
            }
        }

        if model_type not in search_spaces:
            logger.warning(f"No hay espacio de búsqueda para {model_type}")
            return {}

        # Crear modelo base
        base_model = create_model(model_type)

        # Configurar BayesSearchCV
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)

        opt = BayesSearchCV(
            base_model.model,
            search_spaces[model_type],
            n_iter=self.n_optimization_iter,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=0,
            random_state=42
        )

        # Ejecutar optimización
        opt.fit(X, y)

        logger.info(f"  Mejor score: {-opt.best_score_:.2f}")

        return opt.best_params_

    def _cross_validate_temporal(self,
                                 model_type: str,
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 model_kwargs: Dict) -> Dict:
        """
        Validación cruzada temporal

        Args:
            model_type: Tipo de modelo
            X: Features
            y: Target
            model_kwargs: Hiperparámetros del modelo

        Returns:
            Resultados de CV
        """
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)

        cv_rmapes = []
        cv_mapes = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_val = y.iloc[val_idx]

            # Entrenar modelo
            model = create_model(model_type, **model_kwargs)
            model.fit(X_fold_train, y_fold_train)

            # Predecir
            y_fold_pred = model.predict(X_fold_val)

            # Métricas
            rmape = calculate_rmape(y_fold_val, y_fold_pred)
            metrics = calculate_all_metrics(y_fold_val, y_fold_pred)

            cv_rmapes.append(rmape)
            cv_mapes.append(metrics['mape'])

        return {
            'mean_rmape': float(np.mean(cv_rmapes)),
            'std_rmape': float(np.std(cv_rmapes)),
            'mean_mape': float(np.mean(cv_mapes)),
            'std_mape': float(np.std(cv_mapes)),
            'all_rmapes': cv_rmapes,
            'all_mapes': cv_mapes
        }


# ============== TESTING ==============

if __name__ == "__main__":
    print("="*70)
    print("TESTING SISTEMA DE ENTRENAMIENTO")
    print("="*70)

    # Cargar datos reales
    data_path = Path(__file__).parent.parent / "data" / "features" / "data_with_features_latest.csv"

    if not data_path.exists():
        print(f"ERROR: No se encuentra {data_path}")
        print("Ejecuta primero: python pipeline/orchestrator.py")
        exit(1)

    df = pd.read_csv(data_path)

    # Preparar datos
    exclude_cols = ['FECHA', 'TOTAL'] + [f'P{i}' for i in range(1, 25)]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].fillna(0)
    y = df['TOTAL'].copy()

    # Eliminar NaN en y
    mask = ~y.isnull()
    X = X[mask]
    y = y[mask]

    # Split temporal
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n  Train: {len(X_train)} registros")
    print(f"  Test: {len(X_test)} registros")
    print(f"  Features: {len(feature_cols)}")

    # Crear entrenador
    trainer = ModelTrainer(
        optimize_hyperparams=False,  # Deshabilitado para testing rápido
        cv_splits=3
    )

    # Entrenar todos los modelos
    trained_models = trainer.train_all_models(
        X_train, y_train,
        X_test, y_test,
        models=['xgboost', 'lightgbm', 'randomforest']
    )

    # Seleccionar mejor modelo
    best_name, best_model, best_results = trainer.select_best_model(
        criterion='rmape',
        use_validation=True
    )

    # Evaluar en test
    y_test_pred = best_model.predict(X_test)
    test_metrics = calculate_all_metrics(y_test, y_test_pred)

    print(f"\n{'='*70}")
    print(f"EVALUACIÓN EN TEST SET")
    print(f"{'='*70}")
    print(f"  MAPE: {test_metrics['mape']:.4f}%")
    print(f"  rMAPE: {test_metrics['rmape']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.2f}")
    print(f"  R²: {test_metrics['r2']:.4f}")

    # Guardar modelos
    saved_paths = trainer.save_all_models(timestamp='test')

    print(f"\n✓ Modelos guardados:")
    for name, path in saved_paths.items():
        print(f"    {name}: {path}")

    print("\n" + "="*70)
    print("✓ Sistema de entrenamiento funcionando correctamente")
    print("="*70)
