"""
Conectores automatizados para lectura de datos desde múltiples fuentes
Soporta: CSV, API, Database (extensible)
"""
import pandas as pd
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Union
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

try:
    from ..utils.weather import calcular_sensacion_termica
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.weather import calcular_sensacion_termica

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
    Conector especializado para datos meteorológicos de la API de EPM

    Formato esperado (clima_new.csv):
    - fecha: YYYY-MM-DD
    - periodo: 1-24 (hora del día)
    - p_t: Temperatura (°C)
    - p_h: Humedad (%)
    - p_v: Velocidad del viento (m/s)
    - p_i: Intensidad de precipitación (mm)

    Convierte automáticamente datos horarios a promedios diarios.
    """

    def read_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Lee datos meteorológicos de la API de EPM y los convierte a formato diario

        Args:
            start_date: Fecha inicial en formato 'YYYY-MM-DD'
            end_date: Fecha final en formato 'YYYY-MM-DD'

        Returns:
            DataFrame con datos meteorológicos en formato DIARIO con columnas:
            - FECHA, temp_mean, temp_min, temp_max, temp_std
            - humidity_mean, humidity_min, humidity_max
            - wind_speed_mean, wind_speed_max
            - rain_mean, rain_sum
        """
        df = super().read_data()

        # Validar formato de datos de la API de EPM
        required_cols = ['fecha', 'periodo', 'p_t', 'p_h', 'p_v', 'p_i']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(
                f"Formato de datos meteorológicos incorrecto. "
                f"Columnas faltantes: {missing_cols}. "
                f"Se esperaba formato API EPM: fecha, periodo, p_t, p_h, p_v, p_i"
            )

        logger.info("📊 Datos meteorológicos API EPM detectados (formato horario)")
        logger.info("   Convirtiendo a promedios diarios...")

        df = self._convert_epm_hourly_to_daily(df, start_date, end_date)

        logger.info(f"✓ Datos meteorológicos procesados: {len(df)} días")
        logger.info(f"  Rango de fechas: {df['FECHA'].min().date()} a {df['FECHA'].max().date()}")

        return df

    def _convert_epm_hourly_to_daily(self, df: pd.DataFrame,
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Convierte datos meteorológicos de API EPM (horarios) a promedios diarios

        Formato de entrada (API EPM):
        - fecha: YYYY-MM-DD
        - periodo: 1-24
        - p_t: Temperatura (°C)
        - p_h: Humedad (%)
        - p_v: Velocidad del viento (m/s)
        - p_i: Intensidad de precipitación (mm)

        Formato de salida (diario):
        - FECHA: fecha
        - temp_mean, temp_min, temp_max, temp_std
        - humidity_mean, humidity_min, humidity_max
        - wind_speed_mean, wind_speed_min, wind_speed_max
        - rain_mean, rain_sum
        - heat_index_mean, heat_index_min, heat_index_max (sensación
          térmica calculada por hora con Rothfusz/NWS antes de agregar)

        Args:
            df: DataFrame con datos horarios de API EPM
            start_date: Fecha inicial para filtrar
            end_date: Fecha final para filtrar

        Returns:
            DataFrame con promedios diarios
        """
        # Convertir fecha a datetime
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')

        # Verificar conversión
        null_dates = df['fecha'].isna().sum()
        if null_dates > 0:
            logger.warning(f"   ⚠ {null_dates} fechas no pudieron ser parseadas")
            df = df.dropna(subset=['fecha'])

        # Filtrar por rango de fechas ANTES de agregar (más eficiente)
        if start_date:
            df = df[df['fecha'] >= pd.to_datetime(start_date)].copy()
        if end_date:
            df = df[df['fecha'] <= pd.to_datetime(end_date)].copy()

        # Sensación térmica por hora (Rothfusz/NWS), a partir de p_t + p_h,
        # ANTES de agregar — así min/mean/max reflejan las 24 lecturas
        # horarias del día, igual que temp/humedad, no un valor derivado
        # de un único promedio diario.
        df['heat_index'] = calcular_sensacion_termica(df['p_t'], df['p_h'])

        logger.info(f"   Agregando {len(df)} registros horarios a promedios diarios...")

        # Agrupar por fecha y calcular agregaciones
        df_daily = df.groupby('fecha').agg({
            'p_t': ['mean', 'min', 'max', 'std'],      # Temperatura
            'p_h': ['mean', 'min', 'max'],              # Humedad
            'p_v': ['mean', 'min', 'max'],               # Velocidad del viento
            'p_i': ['mean', 'sum'],                      # Precipitación
            'heat_index': ['mean', 'min', 'max'],         # Sensación térmica (calculada)
        }).reset_index()

        # Aplanar columnas multi-nivel
        new_columns = ['FECHA']  # Primera columna es fecha
        for col in df_daily.columns.values[1:]:  # Skip fecha
            base_name, agg_func = col
            # Mapeo de nombres de API EPM a nombres del sistema
            name_map = {
                'p_t': 'temp',
                'p_h': 'humidity',
                'p_v': 'wind_speed',
                'p_i': 'rain',
                'heat_index': 'heat_index',
            }
            mapped_name = name_map.get(base_name, base_name)
            new_columns.append(f"{mapped_name}_{agg_func}")

        df_daily.columns = new_columns

        # Convertir FECHA a datetime si no lo es
        df_daily['FECHA'] = pd.to_datetime(df_daily['FECHA'])

        # Rellenar valores NaN en temp_std con valor por defecto
        if 'temp_std' in df_daily.columns:
            df_daily['temp_std'] = df_daily['temp_std'].fillna(2.5)
        else:
            df_daily['temp_std'] = 2.5

        logger.info(f"   ✓ Conversión completada: {len(df_daily)} días con promedios")
        logger.info(f"   Variables disponibles: temperatura, humedad, viento, lluvia, sensación térmica")

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
