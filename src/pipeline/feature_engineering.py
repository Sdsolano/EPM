"""
Feature Engineering Automático
Genera variables predictivas para modelos de pronóstico de demanda:
- Variables de demanda (históricos, lags, rolling statistics)
- Variables de calendario (festivos, día de la semana, estacionalidad)
- Variables climáticas (temperatura, humedad, correlaciones)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
import sys

# Añadir directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config import HOUR_PERIODS, ROLLING_WINDOWS, DEMAND_LAGS, KEY_WEATHER_VARS
except ImportError:
    HOUR_PERIODS = [f'P{i}' for i in range(1, 25)]
    ROLLING_WINDOWS = [7, 14, 28]
    DEMAND_LAGS = [1, 7, 14]
    KEY_WEATHER_VARS = ['temp', 'humidity', 'feels_like', 'clouds_all', 'wind_speed']

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Clase principal para ingeniería de características automática"""

    def __init__(self):
        self.feature_names = []
        self.feature_stats = {}

    def create_all_features(self,
                           power_df: pd.DataFrame,
                           weather_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Crea todas las características automáticamente

        Args:
            power_df: DataFrame con datos de demanda limpio
            weather_df: DataFrame con datos meteorológicos limpio (opcional)

        Returns:
            DataFrame con todas las características generadas
        """
        logger.info("="*60)
        logger.info("INICIANDO FEATURE ENGINEERING AUTOMÁTICO")
        logger.info("="*60)

        df = power_df.copy()

        # 1. Features de calendario
        logger.info("\n1️⃣  Creando features de calendario...")
        df = self._create_calendar_features(df)

        # 2. Features de demanda (lags y rolling statistics)
        logger.info("\n2️⃣  Creando features de demanda histórica...")
        df = self._create_demand_features(df)

        # 3. Features de estacionalidad
        logger.info("\n3️⃣  Creando features de estacionalidad...")
        df = self._create_seasonality_features(df)

        # 4. Features climáticas (si hay datos disponibles)
        if weather_df is not None:
            logger.info("\n4️⃣  Integrando features climáticas...")
            df = self._integrate_weather_features(df, weather_df)

        # 5. Features de interacción
        logger.info("\n5️⃣  Creando features de interacción...")
        df = self._create_interaction_features(df)

        # 6. Calcular estadísticas
        self._calculate_feature_stats(df)

        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Feature engineering completado")
        logger.info(f"✓ Total de características creadas: {len(self.feature_names)}")
        logger.info(f"{'='*60}\n")

        return df

    def _create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características basadas en calendario"""
        # Asegurar que FECHA es datetime
        df['FECHA'] = pd.to_datetime(df['FECHA'])

        # Componentes básicos de fecha
        df['year'] = df['FECHA'].dt.year
        df['month'] = df['FECHA'].dt.month
        df['day'] = df['FECHA'].dt.day
        df['dayofweek'] = df['FECHA'].dt.dayofweek  # 0=Lunes, 6=Domingo
        df['dayofyear'] = df['FECHA'].dt.dayofyear
        df['week'] = df['FECHA'].dt.isocalendar().week
        df['quarter'] = df['FECHA'].dt.quarter

        # Indicadores booleanos
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_month_start'] = df['FECHA'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['FECHA'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['FECHA'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['FECHA'].dt.is_quarter_end.astype(int)

        # Festivo (ya viene en los datos)
        if 'TIPO DIA' in df.columns:
            df['is_festivo'] = (df['TIPO DIA'] == 'FESTIVO').astype(int)

        # Features cíclicas para capturar naturaleza periódica
        # Día de la semana (cíclico)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        # Mes (cíclico)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Día del año (cíclico)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

        calendar_features = [
            'year', 'month', 'day', 'dayofweek', 'dayofyear', 'week', 'quarter',
            'is_weekend', 'is_month_start', 'is_month_end', 'is_quarter_start',
            'is_quarter_end', 'is_festivo', 'dayofweek_sin', 'dayofweek_cos',
            'month_sin', 'month_cos', 'dayofyear_sin', 'dayofyear_cos'
        ]

        self.feature_names.extend(calendar_features)
        logger.info(f"   ✓ {len(calendar_features)} features de calendario creadas")

        return df

    def _create_demand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características basadas en demanda histórica"""
        # Ordenar por fecha para cálculos temporales
        df = df.sort_values('FECHA').reset_index(drop=True)

        demand_features = []

        # Lags de TOTAL
        if 'TOTAL' in df.columns:
            for lag in DEMAND_LAGS:
                col_name = f'total_lag_{lag}d'
                df[col_name] = df['TOTAL'].shift(lag)
                demand_features.append(col_name)

            # Rolling statistics para TOTAL
            for window in ROLLING_WINDOWS:
                # Media móvil
                col_name = f'total_rolling_mean_{window}d'
                df[col_name] = df['TOTAL'].rolling(window=window, min_periods=1).mean()
                demand_features.append(col_name)

                # Desviación estándar móvil
                col_name = f'total_rolling_std_{window}d'
                df[col_name] = df['TOTAL'].rolling(window=window, min_periods=1).std()
                demand_features.append(col_name)

                # Mínimo y máximo móvil
                col_name = f'total_rolling_min_{window}d'
                df[col_name] = df['TOTAL'].rolling(window=window, min_periods=1).min()
                demand_features.append(col_name)

                col_name = f'total_rolling_max_{window}d'
                df[col_name] = df['TOTAL'].rolling(window=window, min_periods=1).max()
                demand_features.append(col_name)

        # Lags de periodos horarios (seleccionar periodos clave: picos de demanda)
        key_periods = ['P8', 'P12', 'P18', 'P20']  # Horas pico típicas
        for period in key_periods:
            if period in df.columns:
                for lag in [1, 7]:  # Solo lags de 1 y 7 días para periodos específicos
                    col_name = f'{period.lower()}_lag_{lag}d'
                    df[col_name] = df[period].shift(lag)
                    demand_features.append(col_name)

        # Tasa de cambio día a día
        if 'TOTAL' in df.columns:
            df['total_day_change'] = df['TOTAL'].diff()
            df['total_day_change_pct'] = df['TOTAL'].pct_change() * 100
            demand_features.extend(['total_day_change', 'total_day_change_pct'])

        self.feature_names.extend(demand_features)
        logger.info(f"   ✓ {len(demand_features)} features de demanda histórica creadas")

        return df

    def _create_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características de estacionalidad"""
        seasonality_features = []

        # Temporadas en Colombia (2 temporadas principales: lluvia y seca)
        # Temporada de lluvia: Abril-Mayo, Octubre-Noviembre
        # Temporada seca: Diciembre-Marzo, Junio-Septiembre
        def get_season(month):
            if month in [4, 5, 10, 11]:
                return 'rainy'
            else:
                return 'dry'

        df['season'] = df['month'].apply(get_season)
        df['is_rainy_season'] = (df['season'] == 'rainy').astype(int)
        seasonality_features.extend(['is_rainy_season'])

        # Periodos especiales del año
        df['is_january'] = (df['month'] == 1).astype(int)  # Inicio de año
        df['is_december'] = (df['month'] == 12).astype(int)  # Fin de año, festividades
        seasonality_features.extend(['is_january', 'is_december'])

        # Semana del mes (1-5)
        df['week_of_month'] = (df['day'] - 1) // 7 + 1
        seasonality_features.append('week_of_month')

        self.feature_names.extend(seasonality_features)
        logger.info(f"   ✓ {len(seasonality_features)} features de estacionalidad creadas")

        return df

    def _integrate_weather_features(self,
                                    power_df: pd.DataFrame,
                                    weather_df: pd.DataFrame) -> pd.DataFrame:
        """Integra características climáticas con datos de demanda"""
        # Preparar datos meteorológicos
        weather = weather_df.copy()

        # Detectar si los datos ya están en formato diario (con FECHA) o horario (con dt_iso)
        if 'FECHA' in weather.columns:
            # Ya tenemos datos diarios del conector
            weather['FECHA_DATE'] = pd.to_datetime(weather['FECHA']).dt.date
            weather_daily = weather.copy()

            # Renombrar para mantener consistencia con el resto del código
            # El conector ya proporciona: temp_mean, temp_min, temp_max, temp_std, etc.
            # No es necesario agregar nuevamente
        elif 'dt_iso' in weather.columns:
            # Datos horarios - necesitan agregación
            weather['FECHA'] = pd.to_datetime(weather['dt_iso']).dt.date

            # Agregar datos meteorológicos por día (promedios, min, max)
            weather_daily = weather.groupby('FECHA').agg({
                'temp': ['mean', 'min', 'max', 'std'],
                'humidity': ['mean', 'min', 'max'],
                'feels_like': ['mean', 'min', 'max'],
                'wind_speed': ['mean', 'max'],
                'clouds_all': 'mean',
                'pressure': 'mean'
            }).reset_index()

            # Aplanar nombres de columnas
            weather_daily.columns = ['_'.join(col).strip('_') for col in weather_daily.columns.values]
            weather_daily.rename(columns={'FECHA': 'FECHA_DATE'}, inplace=True)
        else:
            raise ValueError("Datos meteorológicos deben contener columna 'FECHA' (diario) o 'dt_iso' (horario)")

        power_df['FECHA_DATE'] = power_df['FECHA'].dt.date

        # Merge con datos de demanda
        df = power_df.merge(weather_daily, on='FECHA_DATE', how='left')

        # Lags de variables climáticas (día anterior)
        weather_cols = [col for col in df.columns if any(var in col for var in KEY_WEATHER_VARS)]
        for col in weather_cols[:10]:  # Limitar para no crear demasiadas features
            lag_col = f'{col}_lag1d'
            df[lag_col] = df[col].shift(1)
            self.feature_names.append(lag_col)

        # Features de interacción clima-calendario
        if 'temp_mean' in df.columns:
            df['temp_x_is_weekend'] = df['temp_mean'] * df['is_weekend']
            df['temp_x_is_festivo'] = df['temp_mean'] * df.get('is_festivo', 0)
            self.feature_names.extend(['temp_x_is_weekend', 'temp_x_is_festivo'])

        # Limpiar columna temporal
        df = df.drop(columns=['FECHA_DATE'])

        weather_feature_count = len([col for col in df.columns if any(var in col for var in KEY_WEATHER_VARS)])
        logger.info(f"   ✓ {weather_feature_count} features climáticas integradas")

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características de interacción entre variables"""
        interaction_features = []

        # Interacción: día de la semana x festivo
        if 'dayofweek' in df.columns and 'is_festivo' in df.columns:
            df['dayofweek_x_festivo'] = df['dayofweek'] * df['is_festivo']
            interaction_features.append('dayofweek_x_festivo')

        # Interacción: mes x festivo
        if 'month' in df.columns and 'is_festivo' in df.columns:
            df['month_x_festivo'] = df['month'] * df['is_festivo']
            interaction_features.append('month_x_festivo')

        # Interacción: weekend x mes (captura patrones de fin de semana por mes)
        if 'is_weekend' in df.columns and 'month' in df.columns:
            df['weekend_x_month'] = df['is_weekend'] * df['month']
            interaction_features.append('weekend_x_month')

        self.feature_names.extend(interaction_features)
        logger.info(f"   ✓ {len(interaction_features)} features de interacción creadas")

        return df

    def _calculate_feature_stats(self, df: pd.DataFrame):
        """Calcula estadísticas sobre las características creadas"""
        self.feature_stats = {
            'total_features': len(self.feature_names),
            'calendar_features': len([f for f in self.feature_names if any(
                kw in f for kw in ['year', 'month', 'day', 'week', 'quarter', 'is_'])]),
            'demand_features': len([f for f in self.feature_names if 'total' in f or any(
                f'p{i}' in f for i in range(1, 25))]),
            'weather_features': len([f for f in self.feature_names if any(
                var in f for var in KEY_WEATHER_VARS)]),
            'interaction_features': len([f for f in self.feature_names if '_x_' in f]),
            'missing_values': df[self.feature_names].isnull().sum().sum()
        }

    def get_feature_importance_ready_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara el DataFrame para entrenamiento de modelos
        Elimina columnas no necesarias y maneja valores faltantes finales
        """
        # Columnas a mantener: features + target
        cols_to_keep = self.feature_names + HOUR_PERIODS + ['TOTAL', 'FECHA']

        # Filtrar solo columnas que existen
        cols_to_keep = [col for col in cols_to_keep if col in df.columns]

        df_model = df[cols_to_keep].copy()

        # Rellenar valores faltantes de features (forward fill y luego 0)
        feature_cols = [col for col in self.feature_names if col in df_model.columns]
        df_model[feature_cols] = df_model[feature_cols].fillna(method='ffill').fillna(0)

        logger.info(f"\n✓ DataFrame preparado para modelado:")
        logger.info(f"  - Forma: {df_model.shape}")
        logger.info(f"  - Features: {len(feature_cols)}")
        logger.info(f"  - Valores faltantes: {df_model.isnull().sum().sum()}")

        return df_model

    def get_feature_summary(self) -> Dict:
        """Retorna un resumen de las características creadas"""
        return {
            'feature_names': self.feature_names,
            'stats': self.feature_stats,
            'timestamp': datetime.now().isoformat()
        }


# ============== FUNCIONES DE UTILIDAD ==============

def create_features(power_df: pd.DataFrame,
                   weather_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Función de utilidad para crear todas las características

    Args:
        power_df: DataFrame con datos de demanda
        weather_df: DataFrame con datos meteorológicos (opcional)

    Returns:
        Tuple con (DataFrame con features, resumen de features)
    """
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(power_df, weather_df)
    df_ready = engineer.get_feature_importance_ready_df(df_features)
    summary = engineer.get_feature_summary()

    return df_ready, summary


# ============== EJEMPLO DE USO ==============

if __name__ == "__main__":
    print("Probando Feature Engineering Automático...\n")

    # Cargar datos limpios
    try:
        power_df = pd.read_csv('../data/processed/power_cleaned.csv')
        weather_df = pd.read_csv('../data_cleaned_weather.csv')

        print(f"📊 Datos de demanda: {len(power_df)} registros")
        print(f"🌤️  Datos meteorológicos: {len(weather_df)} registros\n")

        # Crear features
        df_features, summary = create_features(power_df, weather_df)

        print("\n" + "="*60)
        print("RESUMEN DE FEATURES CREADAS")
        print("="*60)
        print(f"Total de features: {summary['stats']['total_features']}")
        print(f"  - Calendar features: {summary['stats']['calendar_features']}")
        print(f"  - Demand features: {summary['stats']['demand_features']}")
        print(f"  - Weather features: {summary['stats']['weather_features']}")
        print(f"  - Interaction features: {summary['stats']['interaction_features']}")
        print("="*60)

        # Guardar
        output_path = Path('../data/features/power_with_features.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(output_path, index=False)
        print(f"\n✓ Datos con features guardados en: {output_path}")

    except FileNotFoundError as e:
        print(f"⚠️  Archivo no encontrado: {e}")
        print("   Ejecuta primero el script de limpieza de datos")
