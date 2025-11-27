"""
Sistema de Versionado y Gestión de Modelos
Mantiene registro de todos los modelos entrenados con sus métricas
Permite selección automática del modelo campeón y rollback
"""

import json
import joblib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registro centralizado de modelos entrenados
    Mantiene historial, versionado y selección del modelo campeón
    """

    def __init__(self, registry_dir: Path = None):
        """
        Args:
            registry_dir: Directorio donde se guardan los modelos
        """
        self.registry_dir = registry_dir or Path('models/registry')
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.registry_dir / 'registry_metadata.json'
        self.champion_link = self.registry_dir / 'champion_model.joblib'

        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Carga el metadata del registro"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'models': {},
            'champion': None,
            'history': []
        }

    def _save_metadata(self):
        """Guarda el metadata del registro"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def register_model(self,
                      model,
                      model_name: str,
                      metrics: Dict,
                      metadata: Optional[Dict] = None,
                      version: Optional[str] = None) -> str:
        """
        Registra un nuevo modelo en el registry

        Args:
            model: Modelo entrenado
            model_name: Nombre del modelo (xgboost, lightgbm, etc.)
            metrics: Métricas del modelo
            metadata: Metadata adicional
            version: Versión (si None, se genera automáticamente)

        Returns:
            Version ID del modelo registrado
        """
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')

        model_id = f"{model_name}_{version}"

        # Guardar modelo
        model_path = self.registry_dir / f"{model_id}.joblib"
        model.save(model_path)

        # Registrar metadata
        entry = {
            'model_id': model_id,
            'model_name': model_name,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'path': str(model_path),
            'metrics': metrics,
            'metadata': metadata or {},
            'is_champion': False
        }

        if model_id not in self.metadata['models']:
            self.metadata['models'][model_id] = entry
            self.metadata['history'].append({
                'action': 'register',
                'model_id': model_id,
                'timestamp': entry['timestamp']
            })

        self._save_metadata()

        logger.info(f"✓ Modelo registrado: {model_id}")
        logger.info(f"    rMAPE: {metrics.get('rmape', 'N/A'):.4f}")
        logger.info(f"    MAPE: {metrics.get('mape', 'N/A'):.4f}%")

        return model_id

    def promote_to_champion(self,
                           model_id: str,
                           reason: str = "Manual promotion") -> bool:
        """
        Promociona un modelo a campeón

        Args:
            model_id: ID del modelo a promocionar
            reason: Razón de la promoción

        Returns:
            True si la promoción fue exitosa
        """
        if model_id not in self.metadata['models']:
            logger.error(f"Modelo {model_id} no encontrado en registry")
            return False

        # Despromocionar campeón actual
        if self.metadata['champion']:
            old_champion = self.metadata['champion']
            if old_champion in self.metadata['models']:
                self.metadata['models'][old_champion]['is_champion'] = False

        # Promocionar nuevo campeón
        self.metadata['models'][model_id]['is_champion'] = True
        self.metadata['champion'] = model_id

        # Crear link simbólico al campeón
        model_path = Path(self.metadata['models'][model_id]['path'])
        if self.champion_link.exists() or self.champion_link.is_symlink():
            self.champion_link.unlink()

        # En Windows usamos copia en lugar de symlink
        try:
            shutil.copy(model_path, self.champion_link)
        except Exception as e:
            logger.warning(f"No se pudo crear link a campeón: {e}")

        # Registrar en historial
        self.metadata['history'].append({
            'action': 'promote_champion',
            'model_id': model_id,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })

        self._save_metadata()

        logger.info(f"\n{'='*70}")
        logger.info(f"🏆 NUEVO MODELO CAMPEÓN: {model_id}")
        logger.info(f"{'='*70}")
        logger.info(f"  Razón: {reason}")
        logger.info(f"  Métricas: {self.metadata['models'][model_id]['metrics']}")

        return True

    def get_champion(self) -> Optional[Dict]:
        """
        Obtiene el modelo campeón actual

        Returns:
            Metadata del modelo campeón o None
        """
        if not self.metadata['champion']:
            return None

        return self.metadata['models'].get(self.metadata['champion'])

    def load_champion_model(self):
        """
        Carga el modelo campeón

        Returns:
            Modelo campeón cargado
        """
        champion = self.get_champion()

        if not champion:
            raise ValueError("No hay modelo campeón registrado")

        from models.base_models import create_model

        model_path = Path(champion['path'])
        model = create_model(champion['model_name'])
        model.load(model_path)

        return model

    def select_best_and_promote(self,
                                criterion: str = 'rmape',
                                model_ids: Optional[List[str]] = None) -> str:
        """
        Selecciona el mejor modelo y lo promociona a campeón

        Args:
            criterion: Criterio de selección ('rmape', 'mape', 'mae', 'r2')
            model_ids: Lista de IDs a considerar (si None, considera todos)

        Returns:
            ID del modelo seleccionado
        """
        if model_ids is None:
            model_ids = list(self.metadata['models'].keys())

        if not model_ids:
            raise ValueError("No hay modelos en el registry")

        best_model_id = None
        best_score = np.inf if criterion in ['rmape', 'mape', 'mae', 'rmse'] else -np.inf
        minimize = criterion in ['rmape', 'mape', 'mae', 'rmse']

        for model_id in model_ids:
            if model_id not in self.metadata['models']:
                continue

            metrics = self.metadata['models'][model_id].get('metrics', {})
            score = metrics.get(criterion)

            if score is None:
                continue

            # Comparar
            if minimize:
                if score < best_score:
                    best_score = score
                    best_model_id = model_id
            else:
                if score > best_score:
                    best_score = score
                    best_model_id = model_id

        if best_model_id is None:
            raise ValueError(f"No se pudo seleccionar modelo con criterio '{criterion}'")

        # Promocionar
        self.promote_to_champion(
            best_model_id,
            reason=f"Mejor {criterion}: {best_score:.4f}"
        )

        return best_model_id

    def get_all_models(self) -> pd.DataFrame:
        """
        Obtiene lista de todos los modelos registrados

        Returns:
            DataFrame con información de todos los modelos
        """
        if not self.metadata['models']:
            return pd.DataFrame()

        records = []
        for model_id, entry in self.metadata['models'].items():
            record = {
                'model_id': model_id,
                'model_name': entry['model_name'],
                'version': entry['version'],
                'timestamp': entry['timestamp'],
                'is_champion': entry['is_champion']
            }
            # Añadir métricas
            for metric, value in entry.get('metrics', {}).items():
                record[metric] = value

            records.append(record)

        df = pd.DataFrame(records)
        df = df.sort_values('timestamp', ascending=False)

        return df

    def get_model_history(self, model_name: Optional[str] = None) -> List[Dict]:
        """
        Obtiene el historial de un modelo específico o de todos

        Args:
            model_name: Nombre del modelo (si None, retorna todos)

        Returns:
            Lista de eventos históricos
        """
        history = self.metadata['history']

        if model_name:
            history = [
                h for h in history
                if h['model_id'].startswith(model_name)
            ]

        return history

    def rollback_to_previous_champion(self) -> bool:
        """
        Rollback al campeón anterior

        Returns:
            True si el rollback fue exitoso
        """
        # Buscar último campeón en el historial
        champion_history = [
            h for h in self.metadata['history']
            if h['action'] == 'promote_champion'
        ]

        if len(champion_history) < 2:
            logger.warning("No hay campeón anterior para hacer rollback")
            return False

        # El penúltimo campeón
        previous_champion_entry = champion_history[-2]
        previous_champion_id = previous_champion_entry['model_id']

        logger.info(f"Haciendo rollback a: {previous_champion_id}")

        return self.promote_to_champion(
            previous_champion_id,
            reason="Rollback al campeón anterior"
        )

    def cleanup_old_models(self, keep_last_n: int = 10):
        """
        Limpia modelos antiguos, manteniendo los últimos N

        Args:
            keep_last_n: Número de modelos a mantener por tipo
        """
        # Agrupar por nombre de modelo
        models_by_name = {}
        for model_id, entry in self.metadata['models'].items():
            model_name = entry['model_name']
            if model_name not in models_by_name:
                models_by_name[model_name] = []
            models_by_name[model_name].append((model_id, entry))

        deleted_count = 0

        for model_name, models in models_by_name.items():
            # Ordenar por timestamp
            models.sort(key=lambda x: x[1]['timestamp'], reverse=True)

            # Eliminar modelos antiguos
            for model_id, entry in models[keep_last_n:]:
                # No eliminar el campeón
                if entry['is_champion']:
                    continue

                # Eliminar archivo
                model_path = Path(entry['path'])
                if model_path.exists():
                    model_path.unlink()

                # Eliminar del registry
                del self.metadata['models'][model_id]
                deleted_count += 1

        if deleted_count > 0:
            self._save_metadata()
            logger.info(f"✓ Limpieza completada: {deleted_count} modelos eliminados")

        return deleted_count


# ============== TESTING ==============

if __name__ == "__main__":
    import numpy as np

    print("="*70)
    print("TESTING MODEL REGISTRY")
    print("="*70)

    # Crear registry de prueba
    test_registry_dir = Path("test_registry")
    registry = ModelRegistry(test_registry_dir)

    # Simular registro de modelos
    from models.base_models import create_model

    print("\n1. Registrando modelos...")

    for model_name in ['xgboost', 'lightgbm', 'randomforest']:
        model = create_model(model_name)

        # Simular métricas
        metrics = {
            'rmape': np.random.uniform(3, 10),
            'mape': np.random.uniform(1, 5),
            'mae': np.random.uniform(50, 200),
            'r2': np.random.uniform(0.85, 0.95)
        }

        model_id = registry.register_model(
            model,
            model_name,
            metrics,
            metadata={'test': True}
        )

    # Ver todos los modelos
    print("\n2. Modelos registrados:")
    df = registry.get_all_models()
    print(df[['model_id', 'rmape', 'mape', 'is_champion']])

    # Seleccionar mejor modelo
    print("\n3. Seleccionando mejor modelo...")
    best_id = registry.select_best_and_promote(criterion='rmape')

    # Ver campeón
    print("\n4. Modelo campeón:")
    champion = registry.get_champion()
    print(f"   ID: {champion['model_id']}")
    print(f"   Métricas: {champion['metrics']}")

    # Historial
    print("\n5. Historial:")
    history = registry.get_model_history()
    for entry in history:
        print(f"   {entry['action']} - {entry['model_id']} - {entry['timestamp']}")

    # Limpiar archivos de prueba
    import shutil
    shutil.rmtree(test_registry_dir)

    print("\n" + "="*70)
    print("✓ Model Registry funcionando correctamente")
    print("="*70)
