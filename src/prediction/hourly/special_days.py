"""
Sistema de Clustering para Días Especiales (Festivos)

Este módulo maneja la desagregación horaria para días festivos y especiales,
que tienen patrones de demanda significativamente diferentes a los días normales.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import pickle
from typing import Optional, Dict
import logging

from ...config.settings import FEATURES_DATA_DIR, MODELS_DIR
from .calendar_utils import CalendarClassifier

logger = logging.getLogger(__name__)


class SpecialDaysDisaggregator:
    """
    Desagregador horario especializado para días festivos.

    Usa clustering sobre días festivos históricos para capturar
    patrones únicos de demanda en fechas especiales.
    """

    def __init__(self, n_clusters: int = 15, random_state: int = 42, ucp: str = 'Antioquia'):
        """
        Inicializa el desagregador de días especiales.

        Args:
            n_clusters: Número de clusters para K-Means (menor que días normales)
            random_state: Semilla aleatoria para reproducibilidad
            ucp: Nombre del UCP para obtener festivos (ej: 'Antioquia', 'Atlantico')
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.ucp = ucp
        self.kmeans = None
        self.cluster_profiles = None
        self.cluster_by_date = None  # Mapeo mm-dd -> cluster
        self.average_holiday_profile = None  # Perfil promedio de todos los festivos (para festivos no entrenados)
        self.calendar_classifier = CalendarClassifier(ucp=self.ucp)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, date_column: str = 'FECHA') -> 'SpecialDaysDisaggregator':
        """
        Entrena el modelo con datos históricos de días festivos.

        Args:
            df: DataFrame con datos históricos completos
            date_column: Nombre de la columna de fechas

        Returns:
            self
        """
        logger.info("Entrenando desagregador para días especiales...")

        # Preparar datos
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        # Filtrar festivos oficiales
        df['es_festivo'] = df[date_column].apply(self.calendar_classifier.is_holiday)
        
        # Días muy especiales que deben tratarse como festivos aunque no sean oficiales
        # (ej: 24 de diciembre - Nochebuena)
        very_special_dates = ['12-24']  # Nochebuena
        df['mmdd'] = df[date_column].dt.strftime("%m-%d")
        df['es_dia_muy_especial'] = df['mmdd'].isin(very_special_dates)
        
        # Incluir festivos oficiales Y días muy especiales
        df_festivos = df[(df['es_festivo']) | (df['es_dia_muy_especial'])].copy()

        if len(df_festivos) == 0:
            raise ValueError("No se encontraron días festivos en los datos históricos")

        logger.info(f"Encontrados {len(df_festivos)} días festivos y especiales en históricos")
        if df['es_dia_muy_especial'].sum() > 0:
            logger.info(f"  - Incluyendo {df['es_dia_muy_especial'].sum()} días muy especiales (24 dic)")

        # mmdd ya está creado en df_festivos (heredado del df original)

        # Agrupar por fecha (promedio de múltiples años del mismo festivo)
        period_cols = [f'P{i}' for i in range(1, 25)]
        df_agrupado = df_festivos.groupby(date_column)[period_cols].mean()

        # Normalizar
        X = df_agrupado.values
        daily_totals = X.sum(axis=1).reshape(-1, 1)
        X_normalized = X / daily_totals

        # Clustering
        logger.info(f"Ejecutando K-Means con k={self.n_clusters}...")
        self.kmeans = KMeans(
            n_clusters=min(self.n_clusters, len(df_agrupado)),  # No más clusters que datos
            random_state=self.random_state,
            n_init=20
        )
        cluster_labels = self.kmeans.fit_predict(X_normalized)

        # Guardar labels
        df_agrupado['cluster'] = cluster_labels
        df_agrupado['mmdd'] = df_agrupado.index.strftime("%m-%d")

        # Calcular perfiles promedio por cluster
        cluster_profiles_raw = df_agrupado.groupby('cluster')[period_cols].mean()
        cluster_sums = cluster_profiles_raw.sum(axis=1)
        self.cluster_profiles = cluster_profiles_raw.div(cluster_sums, axis=0)

        # Cluster típico por mm-dd (moda)
        self.cluster_by_date = (
            df_agrupado.groupby('mmdd')['cluster']
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
        )
        
        # Calcular perfil promedio de festivos del último año (para festivos no entrenados)
        # Usar solo festivos del último año disponible para tener patrones más recientes
        if len(df_agrupado) > 0:
            # Obtener el último año disponible
            last_year = df_agrupado.index.max().year
            logger.info(f"  - Calculando perfil promedio usando festivos del último año disponible: {last_year}")
            
            # Filtrar festivos del último año en df_agrupado (ya tiene promedios por fecha)
            df_festivos_last_year = df_agrupado[df_agrupado.index.year == last_year]
            
            if len(df_festivos_last_year) > 0:
                # Normalizar cada perfil y promediarlos
                all_profiles_normalized = []
                for idx, row in df_festivos_last_year.iterrows():
                    profile = row[period_cols].values
                    total = profile.sum()
                    if total > 0:
                        all_profiles_normalized.append(profile / total)
                
                if len(all_profiles_normalized) > 0:
                    self.average_holiday_profile = np.mean(all_profiles_normalized, axis=0)
                    # Normalizar para asegurar que suma = 1
                    self.average_holiday_profile = self.average_holiday_profile / self.average_holiday_profile.sum()
                    logger.info(f"  - Perfil promedio calculado usando {len(all_profiles_normalized)} festivos del año {last_year}")
                else:
                    logger.warning("  - No se pudieron normalizar perfiles del último año")
                    self.average_holiday_profile = None
            else:
                logger.warning(f"  - No se encontraron festivos para el año {last_year}, usando promedio general")
                # Fallback: usar promedio de todos los festivos si no hay del último año
                all_profiles_normalized = []
                for idx, row in df_agrupado.iterrows():
                    profile = row[period_cols].values
                    total = profile.sum()
                    if total > 0:
                        all_profiles_normalized.append(profile / total)
                
                if len(all_profiles_normalized) > 0:
                    self.average_holiday_profile = np.mean(all_profiles_normalized, axis=0)
                    self.average_holiday_profile = self.average_holiday_profile / self.average_holiday_profile.sum()
                    logger.info(f"  - Perfil promedio calculado usando todos los festivos históricos (fallback)")
                else:
                    logger.warning("  - No se pudo calcular perfil promedio de festivos")
                    self.average_holiday_profile = None
        else:
            logger.warning("  - No hay datos agrupados para calcular perfil promedio")
            self.average_holiday_profile = None

        self.is_fitted = True
        logger.info(f"✓ Desagregador de días especiales entrenado")
        logger.info(f"  - {len(self.cluster_by_date)} fechas festivas únicas")
        logger.info(f"  - {len(self.cluster_profiles)} clusters identificados")

        return self

    def is_special_day(self, date: pd.Timestamp) -> bool:
        """
        Verifica si una fecha es un día especial conocido.

        Args:
            date: Fecha a verificar

        Returns:
            True si es festivo (aunque no esté en el histórico), o si es un día muy especial
            como el 24 de diciembre (Nochebuena) aunque no sea festivo oficial.
            
        Nota: Si es festivo pero no está en cluster_by_date, se usará el perfil
        promedio de festivos en predict_hourly_profile().
        """
        mmdd = date.strftime("%m-%d")
        
        # Días muy especiales que deben tratarse como festivos para desagregación
        # aunque no sean festivos oficiales (ej: 24 de diciembre - Nochebuena)
        very_special_dates = ['12-24']  # Nochebuena
        
        # Si es un día muy especial, siempre tratarlo como especial
        if mmdd in very_special_dates:
            return True
        
        # Si es festivo oficial, tratarlo como especial (aunque no esté en histórico)
        # Esto permite usar perfil promedio de festivos para festivos nuevos/atípicos
        if self.calendar_classifier.is_holiday(date):
            return True

        return False

    def predict_hourly_profile(
        self,
        date: pd.Timestamp,
        total_daily: float,
        return_normalized: bool = False
    ) -> Optional[np.ndarray]:
        """
        Predice la distribución horaria para un día especial.

        Args:
            date: Fecha del festivo
            total_daily: Demanda total del día
            return_normalized: Si True, retorna también el patrón normalizado (senda)

        Returns:
            Si return_normalized=False:
                Array de 24 elementos con distribución horaria, o None si no es especial
            Si return_normalized=True:
                Tupla (hourly_prediction, normalized_profile, cluster_id) o None
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo no ha sido entrenado. Ejecute .fit() primero.")

        # Verificar si es día especial
        if not self.is_special_day(date):
            logger.debug(f"{date.date()} no es un día especial conocido")
            return None

        # Obtener mm-dd
        mmdd = date.strftime("%m-%d")
        cluster_id = self.cluster_by_date.get(mmdd) if self.cluster_by_date is not None else None

        # Si está en el histórico, usar su cluster específico
        if cluster_id is not None:
            # Obtener perfil normalizado del cluster
            normalized_profile = self.cluster_profiles.loc[cluster_id].values
        else:
            # Festivo no entrenado: usar perfil promedio de festivos
            if self.average_holiday_profile is None:
                logger.warning(f"No hay cluster para festivo {mmdd} y no hay perfil promedio disponible")
                return None
            
            normalized_profile = self.average_holiday_profile
            cluster_id = -1  # Indicador de que se usó perfil promedio
            logger.info(f"Festivo {mmdd} no está en histórico, usando perfil promedio de festivos")

        # Escalar por total diario
        hourly_prediction = normalized_profile * total_daily

        if return_normalized:
            return hourly_prediction, normalized_profile, cluster_id
        else:
            # Mantener retrocompatibilidad
            return hourly_prediction

    def get_special_days_list(self) -> pd.DataFrame:
        """
        Retorna lista de días especiales conocidos por el modelo.

        Returns:
            DataFrame con columnas: mmdd, cluster, nombre_festivo
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo no ha sido entrenado.")

        results = []
        for mmdd, cluster in self.cluster_by_date.items():
            # Crear fecha de ejemplo para obtener nombre
            year = 2024
            date = pd.to_datetime(f"{year}-{mmdd}")
            nombre = self.calendar_classifier.get_holiday_name(date)

            results.append({
                'mmdd': mmdd,
                'cluster': cluster,
                'nombre_festivo': nombre
            })

        return pd.DataFrame(results).sort_values('mmdd')

    def save(self, filepath: Optional[Path] = None) -> None:
        """Guarda el modelo entrenado."""
        if not self.is_fitted:
            raise RuntimeError("No hay modelo entrenado para guardar.")

        if filepath is None:
            filepath = Path(MODELS_DIR) / "special_days_disaggregator.pkl"

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'cluster_profiles': self.cluster_profiles,
                'cluster_by_date': self.cluster_by_date,
                'average_holiday_profile': self.average_holiday_profile,
                'n_clusters': self.n_clusters,
                'random_state': self.random_state,
            }, f)

        logger.info(f"✓ Modelo de días especiales guardado en {filepath}")

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> 'SpecialDaysDisaggregator':
        """Carga un modelo entrenado desde disco."""
        if filepath is None:
            filepath = Path(MODELS_DIR) / "special_days_disaggregator.pkl"

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
        instance.cluster_by_date = data['cluster_by_date']
        # Cargar perfil promedio (compatibilidad hacia atrás: puede no existir en modelos antiguos)
        instance.average_holiday_profile = data.get('average_holiday_profile', None)
        instance.is_fitted = True

        if instance.average_holiday_profile is None:
            logger.warning("Modelo cargado sin perfil promedio de festivos. Re-entrenar para habilitar soporte de festivos no entrenados.")
        else:
            logger.info("Perfil promedio de festivos cargado (soporte para festivos no entrenados habilitado)")

        logger.info(f"✓ Modelo de días especiales cargado desde {filepath}")
        return instance


# ============== FUNCIONES DE UTILIDAD ==============

def train_and_save_special_days(
    data_path: Optional[Path] = None,
    n_clusters: int = 15
) -> SpecialDaysDisaggregator:
    """
    Entrena y guarda el modelo de días especiales.

    Args:
        data_path: Ruta a datos históricos
        n_clusters: Número de clusters

    Returns:
        Modelo entrenado
    """
    if data_path is None:
        data_path = Path(FEATURES_DATA_DIR) / "data_with_features_latest.csv"

    logger.info(f"Cargando datos desde {data_path}...")
    df = pd.read_csv(data_path)

    # Entrenar
    disaggregator = SpecialDaysDisaggregator(n_clusters=n_clusters)
    disaggregator.fit(df, date_column='FECHA')

    # Guardar
    disaggregator.save()

    return disaggregator


# ============== EJEMPLO DE USO ==============

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("ENTRENAMIENTO DE DESAGREGADOR PARA DÍAS ESPECIALES")
    print("=" * 80)

    # Entrenar
    disaggregator = train_and_save_special_days(n_clusters=15)

    # Mostrar días especiales conocidos
    print("\n📋 Días especiales en el modelo:")
    special_days_df = disaggregator.get_special_days_list()
    print(special_days_df.to_string(index=False))

    # Probar predicción para Navidad
    test_date = pd.to_datetime("2024-12-25")
    test_total = 1200.0  # Demanda menor en festivos

    if disaggregator.is_special_day(test_date):
        hourly = disaggregator.predict_hourly_profile(test_date, test_total)

        print(f"\n📅 Predicción para {test_date.date()} - Navidad (total: {test_total} MWh)")
        print(f"\n   Distribución horaria:")
        for i, value in enumerate(hourly, start=1):
            print(f"      P{i:02d}: {value:6.2f} MW")

        print(f"\n   ✓ Suma total: {hourly.sum():.2f} MWh")
    else:
        print(f"\n⚠ {test_date.date()} no es un día especial conocido")
