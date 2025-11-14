"""
Configuración central del Sistema de Pronóstico Automatizado de Demanda Energética
"""
from pathlib import Path
from typing import Dict, List
import os

# ============== RUTAS DEL PROYECTO ==============
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Crear directorios si no existen
for dir_path in [DATA_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Subdirectorios de datos
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"

for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DATA_DIR]:
    dir_path.mkdir(exist_ok=True)

# ============== CONFIGURACIÓN DE DATOS ==============

# Columnas esperadas para datos de demanda (power)
POWER_COLUMNS = [
    'UCP', 'VARIABLE', 'FECHA', 'Clasificador interno', 'TIPO DIA',
    'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12',
    'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24',
    'TOTAL'
]

# Periodos horarios
HOUR_PERIODS = [f'P{i}' for i in range(1, 25)]

# Columnas esperadas para datos meteorológicos (weather)
WEATHER_COLUMNS = [
    'dt', 'dt_iso', 'timezone', 'city_name', 'lat', 'lon', 'temp',
    'visibility', 'dew_point', 'feels_like', 'temp_min', 'temp_max',
    'pressure', 'humidity', 'wind_speed', 'wind_deg', 'clouds_all',
    'weather_id', 'weather_main', 'weather_description', 'weather_icon'
]

# Columnas opcionales de weather (pueden tener NaN)
WEATHER_OPTIONAL_COLUMNS = ['wind_gust', 'rain_1h', 'rain_3h']

# ============== CONFIGURACIÓN DE LIMPIEZA ==============

# Días laborales
WORKING_DAYS = ["LUNES", "MARTES", "MIERCOLES", "MIÉRCOLES", "JUEVES", "VIERNES"]

# Clasificación de días
DAY_TYPE_MAPPING = {
    "laboral": "LABORAL",
    "festivo": "FESTIVO"
}

# ============== CONFIGURACIÓN DE FEATURE ENGINEERING ==============

# Ventanas para rolling statistics (en días)
ROLLING_WINDOWS = [7, 14, 28]  # 1 semana, 2 semanas, 4 semanas

# Lags para variables de demanda (en días)
DEMAND_LAGS = [1, 7, 14]  # día anterior, semana anterior, 2 semanas

# Variables meteorológicas clave para features
KEY_WEATHER_VARS = ['temp', 'humidity', 'feels_like', 'clouds_all', 'wind_speed']

# ============== CONFIGURACIÓN DE VALIDACIÓN ==============

# Umbrales de calidad de datos
DATA_QUALITY_THRESHOLDS = {
    'max_missing_percentage': 0.05,  # 5% máximo de datos faltantes
    'min_rows_per_day': 20,  # mínimo de registros por día
    'outlier_std_threshold': 4,  # umbral para detección de outliers (desviaciones estándar)
}

# ============== CONFIGURACIÓN REGULATORIA ==============

# Métricas según Acuerdo CNO 1303 de 2020
REGULATORY_METRICS = {
    'mape_threshold': 5.0,  # MAPE mensual < 5%
    'daily_deviation_threshold': 5.0,  # Desviaciones diarias < 5%
    'hourly_deviation_max_count': 60,  # Máximo 60 conteos/mes con desviación > 5%
}

# Horizontes de pronóstico
FORECAST_HORIZONS = {
    'monthly': {
        'update_frequency': 'monthly',
        'anticipation_days': 30,
        'description': 'Actualización mensual, con un mes de antelación'
    },
    'weekly': {
        'update_frequency': 'weekly',
        'update_day': 'thursday',
        'update_time': '12:00',
        'description': 'Entrega jueves antes de 12pm para semana siguiente (lunes-domingo)'
    },
    'daily': {
        'update_frequency': 'daily',
        'update_time': '06:00',
        'description': 'Entrega diaria 6am para día siguiente'
    },
    'intraday': {
        'update_frequency': 'intraday',
        'updates_per_day': 3,
        'description': 'Actualización 3 veces al día'
    }
}

# Granularidades de pronóstico
FORECAST_GRANULARITIES = ['hourly']  # 15min deshabilitado por ahora

# ============== CONFIGURACIÓN DE LOGGING ==============

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# ============== CONFIGURACIÓN DE FUENTES DE DATOS ==============

# Configuración de conectores (para futuro)
DATA_SOURCES = {
    'power': {
        'type': 'csv',  # puede ser 'api', 'database', etc.
        'path': str(RAW_DATA_DIR / 'power_data.csv'),
        'refresh_schedule': 'daily',
        'description': 'Datos históricos de demanda de SCADA/XM/GCE'
    },
    'weather': {
        'type': 'csv',  # puede cambiar a 'api' para datos meteorológicos en tiempo real
        'path': str(RAW_DATA_DIR / 'weather_data.csv'),
        'refresh_schedule': 'hourly',
        'description': 'Datos meteorológicos históricos y pronósticos'
    }
}

# ============== CONFIGURACIÓN DE MODELOS ==============

# Parámetros base para modelos (se refinará en Fase 2)
MODEL_CONFIG = {
    'random_seed': 42,
    'test_size': 0.2,
    'validation_size': 0.1,
}

# ============== EXPORTAR CONFIGURACIÓN ==============

def get_config() -> Dict:
    """Retorna todas las configuraciones como diccionario"""
    return {
        'paths': {
            'base_dir': str(BASE_DIR),
            'data_dir': str(DATA_DIR),
            'logs_dir': str(LOGS_DIR),
            'models_dir': str(MODELS_DIR),
            'reports_dir': str(REPORTS_DIR),
        },
        'data': {
            'power_columns': POWER_COLUMNS,
            'weather_columns': WEATHER_COLUMNS,
            'hour_periods': HOUR_PERIODS,
        },
        'feature_engineering': {
            'rolling_windows': ROLLING_WINDOWS,
            'demand_lags': DEMAND_LAGS,
            'key_weather_vars': KEY_WEATHER_VARS,
        },
        'quality': DATA_QUALITY_THRESHOLDS,
        'regulatory': REGULATORY_METRICS,
        'forecast': {
            'horizons': FORECAST_HORIZONS,
            'granularities': FORECAST_GRANULARITIES,
        }
    }


if __name__ == "__main__":
    # Test de configuración
    config = get_config()
    print("✓ Configuración cargada exitosamente")
    print(f"✓ Directorio base: {config['paths']['base_dir']}")
    print(f"✓ Horizontes de pronóstico: {list(config['forecast']['horizons'].keys())}")
