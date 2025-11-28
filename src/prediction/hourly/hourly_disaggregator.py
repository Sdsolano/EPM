"""
Sistema de Clustering para Desagregación Horaria - Días Normales

Este módulo implementa el clustering K-Means para identificar patrones
de distribución horaria en días normales (laborales, fines de semana).
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import pickle
from typing import Optional, Tuple
import logging

from ...config.settings import FEATURES_DATA_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


class HourlyDisaggregator:
    """
    Desagregador horario basado en clustering K-Means.

    Entrena un modelo de clustering sobre patrones históricos normalizados
    y luego predice la distribución horaria para nuevos días.
    """

    def __init__(self, n_clusters: int = 35, random_state: int = 42):
        """
        Inicializa el desagregador.

        Args:
            n_clusters: Número de clusters para K-Means
            random_state: Semilla aleatoria para reproducibilidad
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.cluster_profiles = None
        self.cluster_by_dayofweek = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, date_column: str = 'FECHA') -> 'HourlyDisaggregator':
        """
        Entrena el modelo de clustering con datos históricos.

        Args:
            df: DataFrame con columnas FECHA, P1-P24
            date_column: Nombre de la columna de fechas

        Returns:
            self (patrón sklearn)
        """
        logger.info(f"Entrenando desagregador horario con {len(df)} días históricos...")

        # Validar columnas
        period_cols = [f'P{i}' for i in range(1, 25)]
        missing_cols = [col for col in period_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas de períodos: {missing_cols}")

        # Preparar datos
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        # Matriz día × 24 períodos
        X = df[period_cols].values

        # Normalizar cada día (para que sume 1)
        daily_totals = X.sum(axis=1).reshape(-1, 1)
        X_normalized = X / daily_totals

        # Clustering K-Means
        logger.info(f"Ejecutando K-Means con k={self.n_clusters}...")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=20
        )
        cluster_labels = self.kmeans.fit_predict(X_normalized)

        # Guardar labels en DataFrame
        df['cluster'] = cluster_labels
        df['dayofweek'] = df[date_column].dt.dayofweek  # 0=Lun, 6=Dom

        # Calcular perfiles promedio normalizados por cluster
        cluster_profiles_raw = df.groupby('cluster')[period_cols].mean()
        cluster_sums = cluster_profiles_raw.sum(axis=1)
        self.cluster_profiles = cluster_profiles_raw.div(cluster_sums, axis=0)

        # Cluster típico por día de la semana (moda)
        self.cluster_by_dayofweek = (
            df.groupby('dayofweek')['cluster']
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
        )

        self.is_fitted = True
        logger.info(f"✓ Desagregador entrenado con {self.n_clusters} clusters")

        return self

    def predict_hourly_profile(
        self,
        date: pd.Timestamp,
        total_daily: float
    ) -> np.ndarray:
        """
        Predice la distribución horaria para una fecha y total diario.

        Args:
            date: Fecha del día a predecir
            total_daily: Demanda total del día (en MWh o unidad correspondiente)

        Returns:
            Array de 24 elementos con la distribución horaria (P1-P24)
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo no ha sido entrenado. Ejecute .fit() primero.")

        date = pd.to_datetime(date)
        dayofweek = date.dayofweek

        # Obtener cluster típico para este día de la semana
        cluster_id = self.cluster_by_dayofweek.get(dayofweek)
        if cluster_id is None:
            logger.warning(f"No hay cluster para día {dayofweek}, usando cluster 0")
            cluster_id = 0

        # Obtener perfil normalizado del cluster
        normalized_profile = self.cluster_profiles.loc[cluster_id].values

        # Escalar por el total diario
        hourly_prediction = normalized_profile * total_daily

        return hourly_prediction

    def predict_batch(
        self,
        dates: pd.Series,
        totals_daily: pd.Series
    ) -> pd.DataFrame:
        """
        Predice distribuciones horarias para múltiples fechas.

        Args:
            dates: Series de fechas
            totals_daily: Series de totales diarios

        Returns:
            DataFrame con columnas FECHA, P1-P24
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo no ha sido entrenado. Ejecute .fit() primero.")

        results = []
        for date, total in zip(dates, totals_daily):
            hourly = self.predict_hourly_profile(date, total)
            results.append({
                'FECHA': date,
                **{f'P{i}': hourly[i-1] for i in range(1, 25)}
            })

        return pd.DataFrame(results)

    def save(self, filepath: Optional[Path] = None) -> None:
        """
        Guarda el modelo entrenado en disco.

        Args:
            filepath: Ruta del archivo (por defecto en MODELS_DIR)
        """
        if not self.is_fitted:
            raise RuntimeError("No hay modelo entrenado para guardar.")

        if filepath is None:
            filepath = Path(MODELS_DIR) / "hourly_disaggregator.pkl"

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'cluster_profiles': self.cluster_profiles,
                'cluster_by_dayofweek': self.cluster_by_dayofweek,
                'n_clusters': self.n_clusters,
                'random_state': self.random_state,
            }, f)

        logger.info(f"✓ Modelo guardado en {filepath}")

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> 'HourlyDisaggregator':
        """
        Carga un modelo entrenado desde disco.

        Args:
            filepath: Ruta del archivo (por defecto en MODELS_DIR)

        Returns:
            Instancia de HourlyDisaggregator cargada
        """
        if filepath is None:
            filepath = Path(MODELS_DIR) / "hourly_disaggregator.pkl"

        if not filepath.exists():
            raise FileNotFoundError(f"No se encontró modelo en {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        instance = cls(
            n_clusters=data['n_clusters'],
            random_state=data['random_state']
        )
        instance.kmeans = data['kmeans']
        instance.cluster_profiles = data['cluster_profiles']
        instance.cluster_by_dayofweek = data['cluster_by_dayofweek']
        instance.is_fitted = True

        logger.info(f"✓ Modelo cargado desde {filepath}")
        return instance


# ============== FUNCIONES DE UTILIDAD ==============

def train_and_save_disaggregator(
    data_path: Optional[Path] = None,
    n_clusters: int = 35
) -> HourlyDisaggregator:
    """
    Función de conveniencia para entrenar y guardar el modelo.

    Args:
        data_path: Ruta al archivo de features (por defecto data_with_features_latest.csv)
        n_clusters: Número de clusters

    Returns:
        Modelo entrenado
    """
    if data_path is None:
        data_path = Path(FEATURES_DATA_DIR) / "data_with_features_latest.csv"

    logger.info(f"Cargando datos desde {data_path}...")
    df = pd.read_csv(data_path)

    # Entrenar
    disaggregator = HourlyDisaggregator(n_clusters=n_clusters)
    disaggregator.fit(df, date_column='FECHA')

    # Guardar
    disaggregator.save()

    return disaggregator


# ============== EJEMPLO DE USO ==============

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Entrenar modelo
    print("=" * 80)
    print("ENTRENAMIENTO DE DESAGREGADOR HORARIO")
    print("=" * 80)

    disaggregator = train_and_save_disaggregator(n_clusters=35)

    # Probar predicción
    test_date = pd.to_datetime("2024-03-15")  # Viernes
    test_total = 1500.0  # 1500 MWh

    hourly = disaggregator.predict_hourly_profile(test_date, test_total)

    print(f"\n📅 Predicción para {test_date.date()} (total: {test_total} MWh)")
    print(f"   Día de la semana: {test_date.day_name()}")
    print(f"\n   Distribución horaria:")
    for i, value in enumerate(hourly, start=1):
        print(f"      P{i:02d}: {value:6.2f} MW")

    print(f"\n   ✓ Suma total: {hourly.sum():.2f} MWh (esperado: {test_total})")
    print(f"   ✓ Diferencia: {abs(hourly.sum() - test_total):.6f}")
