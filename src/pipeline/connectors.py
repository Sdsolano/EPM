"""
Conectores automatizados para lectura de datos desde múltiples fuentes
Soporta: CSV, API, Database (extensible)
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Union
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataConnector(ABC):
    """Clase base abstracta para todos los conectores de datos"""

    def __init__(self, config: Dict):
        self.config = config
        self.last_read_timestamp = None

    @abstractmethod
    def read_data(self, **kwargs) -> pd.DataFrame:
        """Método abstracto para leer datos"""
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """Valida que la conexión a la fuente de datos sea válida"""
        pass

    def log_read(self, rows: int, source: str):
        """Registra información sobre la lectura de datos"""
        self.last_read_timestamp = datetime.now()
        logger.info(f"✓ Datos leídos desde {source}: {rows} filas a las {self.last_read_timestamp}")


class CSVConnector(DataConnector):
    """Conector para archivos CSV locales"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.file_path = Path(config.get('path', ''))

    def validate_connection(self) -> bool:
        """Valida que el archivo CSV existe"""
        if self.file_path.exists():
            logger.info(f"✓ Archivo CSV encontrado: {self.file_path}")
            return True
        else:
            logger.error(f"✗ Archivo CSV no encontrado: {self.file_path}")
            return False

    def read_data(self, **kwargs) -> pd.DataFrame:
        """Lee datos desde archivo CSV"""
        try:
            if not self.validate_connection():
                raise FileNotFoundError(f"Archivo no encontrado: {self.file_path}")

            df = pd.read_csv(self.file_path, **kwargs)
            self.log_read(len(df), str(self.file_path))
            return df

        except Exception as e:
            logger.error(f"Error leyendo CSV {self.file_path}: {str(e)}")
            raise


class PowerDataConnector(CSVConnector):
    """Conector especializado para datos de demanda eléctrica"""

    def read_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Lee datos de demanda eléctrica con filtros opcionales de fecha

        Args:
            start_date: Fecha inicial en formato 'YYYY-MM-DD'
            end_date: Fecha final en formato 'YYYY-MM-DD'

        Returns:
            DataFrame con datos de demanda
        """
        df = super().read_data()

        # Convertir fecha a datetime
        df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')

        # Filtrar por rango de fechas si se especifica
        if start_date:
            df = df[df['FECHA'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['FECHA'] <= pd.to_datetime(end_date)]

        logger.info(f"✓ Datos de demanda filtrados: {len(df)} registros")
        logger.info(f"  Rango de fechas: {df['FECHA'].min()} a {df['FECHA'].max()}")

        return df

    def read_latest_data(self, days_back: int = 30) -> pd.DataFrame:
        """
        Lee datos de demanda de los últimos N días

        Args:
            days_back: Número de días hacia atrás desde hoy

        Returns:
            DataFrame con datos recientes
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        return self.read_data(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )


class WeatherDataConnector(CSVConnector):
    """
    Conector especializado para datos meteorológicos

    Acepta datos en dos formatos:
    1. Horario (con dt_iso) → Los convierte automáticamente a promedios diarios
    2. Diario (con FECHA) → Los usa directamente
    """

    def read_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Lee datos meteorológicos con filtros opcionales de fecha

        Si los datos vienen en formato HORARIO (columna dt_iso), los convierte automáticamente
        a promedios DIARIOS con las columnas esperadas por el sistema.

        Args:
            start_date: Fecha inicial en formato 'YYYY-MM-DD'
            end_date: Fecha final en formato 'YYYY-MM-DD'

        Returns:
            DataFrame con datos meteorológicos en formato DIARIO
        """
        df = super().read_data()

        # Detectar formato de datos (horario vs diario)
        is_hourly = 'dt_iso' in df.columns
        is_daily = 'FECHA' in df.columns

        if is_hourly:
            logger.info("📊 Datos meteorológicos en formato HORARIO detectados")
            logger.info("   Convirtiendo a promedios diarios...")
            df = self._convert_hourly_to_daily(df, start_date, end_date)
        elif is_daily:
            logger.info("📊 Datos meteorológicos en formato DIARIO detectados")
            df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')

            # Filtrar por fechas
            if start_date:
                df = df[df['FECHA'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['FECHA'] <= pd.to_datetime(end_date)]
        else:
            raise ValueError(
                "Formato de datos meteorológicos no reconocido. "
                "Se esperaba columna 'dt_iso' (horario) o 'FECHA' (diario)"
            )

        logger.info(f"✓ Datos meteorológicos procesados: {len(df)} días")
        logger.info(f"  Rango de fechas: {df['FECHA'].min().date()} a {df['FECHA'].max().date()}")

        return df

    def _convert_hourly_to_daily(self, df: pd.DataFrame,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Convierte datos meteorológicos horarios a promedios diarios

        Formato de entrada (horario):
        - dt_iso: timestamp
        - temp, temp_min, temp_max
        - feels_like
        - humidity

        Formato de salida (diario):
        - FECHA: fecha
        - temp_mean, temp_min, temp_max, temp_std
        - feels_like_mean, feels_like_min, feels_like_max
        - humidity_mean, humidity_min, humidity_max

        Args:
            df: DataFrame con datos horarios
            start_date: Fecha inicial para filtrar
            end_date: Fecha final para filtrar

        Returns:
            DataFrame con promedios diarios
        """
        # Convertir timestamp a datetime (remover sufijo UTC si existe)
        if df['dt_iso'].dtype == 'object':
            df['dt_iso'] = df['dt_iso'].str.replace(r' \+\d{4} UTC$', '', regex=True)

        df['dt_iso'] = pd.to_datetime(df['dt_iso'], errors='coerce')

        # Verificar conversión
        null_dates = df['dt_iso'].isna().sum()
        if null_dates > 0:
            logger.warning(f"   ⚠ {null_dates} timestamps no pudieron ser parseados")

        # Filtrar por rango de fechas ANTES de agregar (más eficiente)
        if start_date:
            df = df[df['dt_iso'] >= pd.to_datetime(start_date)].copy()
        if end_date:
            df = df[df['dt_iso'] <= pd.to_datetime(end_date)].copy()

        # Extraer fecha (sin hora)
        df['FECHA'] = df['dt_iso'].dt.date

        logger.info(f"   Agregando {len(df)} registros horarios a promedios diarios...")

        # Definir agregaciones por columna
        aggregations = {}

        # Temperatura
        if 'temp' in df.columns:
            aggregations['temp'] = ['mean', 'std']

        # temp_min y temp_max pueden venir en el CSV horario
        # Tomamos el mínimo de temp_min (mínimo del día) y máximo de temp_max (máximo del día)
        if 'temp_min' in df.columns:
            aggregations['temp_min'] = 'min'  # Mínimo de los mínimos
        if 'temp_max' in df.columns:
            aggregations['temp_max'] = 'max'  # Máximo de los máximos

        # Sensación térmica
        if 'feels_like' in df.columns:
            aggregations['feels_like'] = ['mean', 'min', 'max']

        # Humedad
        if 'humidity' in df.columns:
            aggregations['humidity'] = ['mean', 'min', 'max']

        # Verificar que tengamos al menos temperatura
        if 'temp' not in aggregations:
            raise ValueError(
                "El archivo de clima no contiene columna 'temp'. "
                "Columnas disponibles: " + ", ".join(df.columns)
            )

        # Agrupar por fecha y calcular agregaciones
        df_daily = df.groupby('FECHA').agg(aggregations).reset_index()

        # Aplanar columnas multi-nivel (temp, mean) -> temp_mean
        new_columns = []
        for col in df_daily.columns.values:
            if not isinstance(col, tuple):
                # Columna simple (no debería ocurrir después del groupby con agg)
                new_columns.append(col)
            else:
                # Tupla de (nombre, agregación)
                base_name, agg_func = col

                # Caso especial: FECHA sin agregación → ('FECHA', '')
                if base_name == 'FECHA' and agg_func == '':
                    new_columns.append('FECHA')
                # temp_min/max con min/max → no duplicar sufijo
                elif agg_func in ['min', 'max'] and base_name in ['temp_min', 'temp_max']:
                    new_columns.append(base_name)
                # Caso general: nombre_agregacion
                else:
                    new_columns.append(f"{base_name}_{agg_func}")

        df_daily.columns = new_columns

        # Convertir FECHA de date a datetime
        if 'FECHA' not in df_daily.columns:
            raise KeyError("La columna 'FECHA' se perdió durante el procesamiento")

        df_daily['FECHA'] = pd.to_datetime(df_daily['FECHA'])

        # Renombrar columnas al formato esperado por el sistema
        rename_map = {}

        # Mapeo de nombres
        if 'temp_mean' not in df_daily.columns and 'temp' in df_daily.columns:
            rename_map['temp'] = 'temp_mean'
        if 'temp_std' not in df_daily.columns and 'temp' in df_daily.columns:
            # Si no hay std, crear columna con valor por defecto
            df_daily['temp_std'] = 2.5

        if rename_map:
            df_daily.rename(columns=rename_map, inplace=True)

        # Asegurar que existan todas las columnas necesarias (con valores por defecto si faltan)
        required_cols = [
            'temp_mean', 'temp_min', 'temp_max', 'temp_std',
            'feels_like_mean', 'feels_like_min', 'feels_like_max',
            'humidity_mean', 'humidity_min', 'humidity_max'
        ]

        for col in required_cols:
            if col not in df_daily.columns:
                logger.warning(f"   ⚠ Columna '{col}' no disponible, usando valores por defecto")

                # Valores por defecto para Medellín
                if 'temp' in col:
                    df_daily[col] = 22.0  # Temperatura promedio Medellín
                elif 'feels_like' in col:
                    df_daily[col] = 22.0
                elif 'humidity' in col:
                    df_daily[col] = 70.0  # Humedad promedio
                elif col == 'temp_std':
                    df_daily[col] = 2.5

        logger.info(f"   ✓ Conversión completada: {len(df_daily)} días con promedios")

        return df_daily

    def read_latest_data(self, days_back: int = 30) -> pd.DataFrame:
        """
        Lee datos meteorológicos de los últimos N días

        Args:
            days_back: Número de días hacia atrás desde hoy

        Returns:
            DataFrame con datos recientes
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        return self.read_data(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )


class DataConnectorFactory:
    """Factory para crear conectores según tipo de datos"""

    @staticmethod
    def create_connector(data_type: str, config: Dict) -> DataConnector:
        """
        Crea un conector según el tipo de datos

        Args:
            data_type: Tipo de datos ('power', 'weather')
            config: Configuración del conector

        Returns:
            Instancia del conector apropiado
        """
        connector_map = {
            'power': PowerDataConnector,
            'weather': WeatherDataConnector,
            'csv': CSVConnector,
        }

        connector_class = connector_map.get(data_type.lower())

        if not connector_class:
            raise ValueError(f"Tipo de conector no soportado: {data_type}")

        logger.info(f"✓ Creando conector de tipo: {data_type}")
        return connector_class(config)


# ============== FUNCIONES DE UTILIDAD ==============

def load_power_data(file_path: Union[str, Path],
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Función de utilidad para cargar datos de demanda eléctrica

    Args:
        file_path: Ruta al archivo CSV
        start_date: Fecha inicial (opcional)
        end_date: Fecha final (opcional)

    Returns:
        DataFrame con datos de demanda
    """
    connector = PowerDataConnector({'path': str(file_path)})
    return connector.read_data(start_date=start_date, end_date=end_date)


def load_weather_data(file_path: Union[str, Path],
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Función de utilidad para cargar datos meteorológicos

    Args:
        file_path: Ruta al archivo CSV
        start_date: Fecha inicial (opcional)
        end_date: Fecha final (opcional)

    Returns:
        DataFrame con datos meteorológicos
    """
    connector = WeatherDataConnector({'path': str(file_path)})
    return connector.read_data(start_date=start_date, end_date=end_date)


# ============== EJEMPLO DE USO ==============

if __name__ == "__main__":
    # Ejemplo de uso de conectores

    # Crear conector de demanda
    power_config = {'path': '../datos.csv'}
    power_connector = DataConnectorFactory.create_connector('power', power_config)

    # Validar conexión
    if power_connector.validate_connection():
        # Leer últimos 30 días
        df_power = power_connector.read_latest_data(days_back=30)
        print(f"\n📊 Datos de demanda cargados: {len(df_power)} registros")
        print(df_power.head())

    # Crear conector meteorológico
    weather_config = {'path': '../data_cleaned_weather.csv'}
    weather_connector = DataConnectorFactory.create_connector('weather', weather_config)

    if weather_connector.validate_connection():
        df_weather = weather_connector.read_latest_data(days_back=30)
        print(f"\n🌤️  Datos meteorológicos cargados: {len(df_weather)} registros")
        print(df_weather.head())
