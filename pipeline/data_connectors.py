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
    """Conector especializado para datos meteorológicos"""

    def read_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Lee datos meteorológicos con filtros opcionales de fecha

        Args:
            start_date: Fecha inicial en formato 'YYYY-MM-DD'
            end_date: Fecha final en formato 'YYYY-MM-DD'

        Returns:
            DataFrame con datos meteorológicos
        """
        df = super().read_data()

        # Convertir dt_iso a datetime
        df['dt_iso'] = pd.to_datetime(df['dt_iso'], errors='coerce')

        # Filtrar por rango de fechas si se especifica
        if start_date:
            df = df[df['dt_iso'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['dt_iso'] <= pd.to_datetime(end_date)]

        logger.info(f"✓ Datos meteorológicos filtrados: {len(df)} registros")
        logger.info(f"  Rango de fechas: {df['dt_iso'].min()} a {df['dt_iso'].max()}")

        return df

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
