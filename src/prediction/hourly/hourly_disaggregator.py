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
from datetime import timedelta, datetime

from ...config.settings import FEATURES_DATA_DIR, MODELS_DIR

# Importar CalendarClassifier para excluir festivos
try:
    from .calendar_utils import CalendarClassifier
except ImportError:
    CalendarClassifier = None
    logging.warning("CalendarClassifier no disponible. No se excluirán festivos del entrenamiento.")

logger = logging.getLogger(__name__)


class HourlyDisaggregator:
    """
    Desagregador horario basado en clustering K-Means.

    Entrena un modelo de clustering sobre patrones históricos normalizados
    y luego predice la distribución horaria para nuevos días.
    
    IMPORTANTE: El entrenamiento usa solo los últimos 3 meses de datos históricos
    y excluye solo festivos (mantiene días laborales y fines de semana) para capturar
    patrones de demanda más recientes y relevantes.
"""

    def __init__(self, n_clusters: int = 35, random_state: int = 42, ucp: str = 'Antioquia'):
        """
        Inicializa el desagregador.

        Args:
            n_clusters: Número de clusters para K-Means
            random_state: Semilla aleatoria para reproducibilidad
            ucp: Nombre del UCP para identificar festivos (default: 'Antioquia')
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.ucp = ucp
        self.kmeans = None
        self.cluster_profiles = None
        self.cluster_by_dayofweek = None
        self.is_fitted = False
        self.training_date = None  # Fecha de entrenamiento (para validar antigüedad)
        self.training_data_end_date = None  # Fecha más reciente de datos usados para entrenar
        
        # Inicializar CalendarClassifier para excluir festivos
        if CalendarClassifier is not None:
            self.calendar_classifier = CalendarClassifier(ucp=self.ucp)
        else:
            self.calendar_classifier = None
            logger.warning("CalendarClassifier no disponible. Los festivos no se excluirán.")

    def fit(self, df: pd.DataFrame, date_column: str = 'FECHA') -> 'HourlyDisaggregator':
        """
        Entrena el modelo de clustering con datos históricos.
        
        IMPORTANTE: Filtra automáticamente:
        - Solo últimos 3 meses antes de la fecha más reciente
        - Excluye solo festivos (mantiene días laborales y fines de semana)
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
        df.dropna(inplace=True)
        
        # ========================================
        # FILTRO 1: Solo últimos 3 meses
        # ========================================
        fecha_mas_reciente = df[date_column].max()
        fecha_corte = fecha_mas_reciente - timedelta(days=90)  # 3 meses = ~90 días
        
        logger.info(f"Filtrando datos: últimos 3 meses (desde {fecha_corte.date()} hasta {fecha_mas_reciente.date()})")
        mask_fecha = df[date_column] >= fecha_corte
        df = df[mask_fecha].copy()
        
        logger.info(f"Días después de filtro de fecha: {len(df)}")
        
        # ========================================
        # FILTRO 2: Excluir festivos (mantiene días laborales y fines de semana)
        # ========================================
        if self.calendar_classifier is not None:
            # Pre-cargar festivos para todos los años en el DataFrame
            years = df[date_column].dt.year.unique()
            for year in years:
                self.calendar_classifier._load_festivos_for_year(year)
            
            # Filtrar festivos
            df['es_festivo'] = df[date_column].apply(self.calendar_classifier.is_holiday)
            mask_no_festivos = ~df['es_festivo']
            df = df[mask_no_festivos].copy()
            
            logger.info(f"Días (sin festivos) después de filtro: {len(df)}")
        else:
            logger.warning("CalendarClassifier no disponible. No se excluyen festivos.")
        
        if len(df) < 30:
            logger.warning(f"⚠ Pocos datos después de filtrar ({len(df)} días). Considerar aumentar ventana histórica.")
        
        logger.info(f"✓ Usando {len(df)} días recientes para entrenamiento (días laborales y fines de semana, excluyendo festivos)")

        # Matriz día × 24 períodos
        X = df[period_cols].values
        X = np.asarray(X)

        # Asegurar matriz 2D
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        # Sumas por día
        daily_totals = X.sum(axis=1)

        # Filtrar días inválidos (totales cero o NaN)
        valid_mask = (~np.isnan(daily_totals)) & (daily_totals != 0)

        X = X[valid_mask]
        daily_totals = daily_totals[valid_mask]
        df = df.loc[df.index[valid_mask]]

        # Normalización segura
        X_normalized = X / daily_totals[:, None]

        # Eliminar inf / -inf / NaN por precaución
        X_normalized = np.where(np.isinf(X_normalized), np.nan, X_normalized)
        valid_mask2 = ~np.isnan(X_normalized).any(axis=1)

        X_normalized = X_normalized[valid_mask2]
        df = df.loc[df.index[valid_mask2]]

        # K-Means
        logger.info(f"Ejecutando K-Means con k={self.n_clusters}...")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=20
        )
        cluster_labels = self.kmeans.fit_predict(X_normalized)

        # Guardar labels
        df['cluster'] = cluster_labels
        df['dayofweek'] = df[date_column].dt.dayofweek

        # Perfiles promedio por cluster
        cluster_profiles_raw = df.groupby('cluster')[period_cols].mean()
        cluster_sums = cluster_profiles_raw.sum(axis=1)
        self.cluster_profiles = cluster_profiles_raw.div(cluster_sums, axis=0)

        # Cluster típico por día de la semana (moda)
        self.cluster_by_dayofweek = (
            df.groupby('dayofweek')['cluster']
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
        )

        # Guardar fecha de entrenamiento y rango de datos usados
        self.training_date = datetime.now()
        self.training_data_end_date = fecha_mas_reciente  # Fecha más reciente de datos usados
        self.is_fitted = True
        logger.info(f"✓ Desagregador entrenado con {self.n_clusters} clusters")
        logger.info(f"  Fecha de entrenamiento: {self.training_date.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Datos usados: últimos 3 meses (desde {fecha_corte.date()} hasta {fecha_mas_reciente.date()})")

        return self
    
    def is_relevant_for_date(self, prediction_date: pd.Timestamp) -> bool:
        """
        Verifica si el modelo es relevante para una fecha de predicción.
        
        El modelo es relevante si la fecha de predicción está dentro de los últimos
        3 meses ANTES de la fecha más reciente usada para entrenar.
        
        Args:
            prediction_date: Fecha para la cual se quiere predecir
            
        Returns:
            True si el modelo es relevante, False si necesita re-entrenarse
        """
        if not self.is_fitted or self.training_data_end_date is None:
            return False
        
        prediction_date = pd.to_datetime(prediction_date)
        # El modelo es relevante si la fecha está dentro de 3 meses ANTES de training_data_end_date
        fecha_limite_inferior = self.training_data_end_date - timedelta(days=90)
        
        # Relevante si: fecha_limite_inferior <= prediction_date <= training_data_end_date
        return fecha_limite_inferior <= prediction_date <= self.training_data_end_date


    def predict_hourly_profile(
        self,
        date: pd.Timestamp,
        total_daily: float,
        return_normalized: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[int]]:
        """
        Predice la distribución horaria para una fecha y total diario.

        Args:
            date: Fecha del día a predecir
            total_daily: Demanda total del día (en MWh o unidad correspondiente)
            return_normalized: Si True, retorna también el patrón normalizado (senda)

        Returns:
            Si return_normalized=False:
                Array de 24 elementos con la distribución horaria (P1-P24)
            Si return_normalized=True:
                Tupla (hourly_prediction, normalized_profile, cluster_id)
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

        if return_normalized:
            return hourly_prediction, normalized_profile, cluster_id
        else:
            # Mantener retrocompatibilidad
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
                'ucp': self.ucp,
                'training_date': self.training_date,  # Guardar fecha de entrenamiento
                'training_data_end_date': self.training_data_end_date,  # Fecha más reciente de datos
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
            random_state=data['random_state'],
            ucp=data.get('ucp', 'Antioquia')  # Retrocompatibilidad: default si no existe
        )
        instance.kmeans = data['kmeans']
        instance.cluster_profiles = data['cluster_profiles']
        instance.cluster_by_dayofweek = data['cluster_by_dayofweek']
        instance.training_date = data.get('training_date')  # Retrocompatibilidad: puede no existir
        instance.training_data_end_date = data.get('training_data_end_date')  # Retrocompatibilidad
        instance.is_fitted = True

        # Log información del modelo cargado
        if instance.training_data_end_date:
            logger.info(f"✓ Modelo cargado desde {filepath}")
            logger.info(f"  Datos de entrenamiento: hasta {instance.training_data_end_date.strftime('%Y-%m-%d')}")
            if instance.training_date:
                logger.info(f"  Fecha de entrenamiento: {instance.training_date.strftime('%Y-%m-%d %H:%M:%S')}")
        elif instance.training_date:
            logger.info(f"✓ Modelo cargado desde {filepath} (entrenado: {instance.training_date.strftime('%Y-%m-%d')})")
        else:
            logger.info(f"✓ Modelo cargado desde {filepath} (modelo antiguo sin metadatos)")
        
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
