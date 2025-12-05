"""
Sistema automático de limpieza, validación y transformación de datos
Implementa detección de anomalías, tratamiento de datos faltantes y validaciones
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
import sys

# Añadir directorio padre al path para imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config import (
        POWER_COLUMNS, WEATHER_COLUMNS, WEATHER_OPTIONAL_COLUMNS,
        WORKING_DAYS, HOUR_PERIODS, DATA_QUALITY_THRESHOLDS
    )
except ImportError:
    # Valores por defecto si no se puede importar config
    POWER_COLUMNS = ['UCP', 'VARIABLE', 'FECHA', 'Clasificador interno', 'TIPO DIA'] + \
                    [f'P{i}' for i in range(1, 25)] + ['TOTAL']
    HOUR_PERIODS = [f'P{i}' for i in range(1, 25)]
    WORKING_DAYS = ["LUNES", "MARTES", "MIERCOLES", "MIÉRCOLES", "JUEVES", "VIERNES"]

    # Columnas para datos meteorológicos API EPM (formato nuevo)
    WEATHER_COLUMNS = ['FECHA', 'temp_mean', 'temp_min', 'temp_max', 'temp_std',
                       'humidity_mean', 'humidity_min', 'humidity_max',
                       'wind_speed_mean', 'wind_speed_max', 'rain_mean', 'rain_sum']
    WEATHER_OPTIONAL_COLUMNS = ['rain_sum', 'wind_speed_max', 'temp_std']

    DATA_QUALITY_THRESHOLDS = {
        'max_missing_percentage': 0.05,
        'min_rows_per_day': 20,
        'outlier_std_threshold': 4,
    }

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataQualityReport:
    """Clase para almacenar el reporte de calidad de datos"""

    def __init__(self):
        self.timestamp = datetime.now()
        self.issues = []
        self.warnings = []
        self.stats = {}
        self.passed = True

    def add_issue(self, issue_type: str, description: str, severity: str = 'ERROR'):
        """Añade un problema detectado"""
        self.issues.append({
            'type': issue_type,
            'description': description,
            'severity': severity,
            'timestamp': datetime.now()
        })
        if severity == 'ERROR':
            self.passed = False
        logger.warning(f"[{severity}] {issue_type}: {description}")

    def add_warning(self, warning: str):
        """Añade una advertencia"""
        self.warnings.append({
            'description': warning,
            'timestamp': datetime.now()
        })
        logger.warning(f"[WARNING] {warning}")

    def add_stat(self, key: str, value):
        """Añade una estadística al reporte"""
        self.stats[key] = value

    def summary(self) -> str:
        """Genera un resumen del reporte"""
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        summary = f"\n{'='*60}\n"
        summary += f"DATA QUALITY REPORT - {status}\n"
        summary += f"Timestamp: {self.timestamp}\n"
        summary += f"{'='*60}\n\n"

        if self.stats:
            summary += "STATISTICS:\n"
            for key, value in self.stats.items():
                summary += f"  • {key}: {value}\n"
            summary += "\n"

        if self.issues:
            summary += f"ISSUES FOUND: {len(self.issues)}\n"
            for issue in self.issues:
                summary += f"  [{issue['severity']}] {issue['type']}: {issue['description']}\n"
            summary += "\n"

        if self.warnings:
            summary += f"WARNINGS: {len(self.warnings)}\n"
            for warning in self.warnings:
                summary += f"  • {warning['description']}\n"

        return summary


class PowerDataCleaner:
    """Limpiador especializado para datos de demanda eléctrica"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or DATA_QUALITY_THRESHOLDS
        self.report = DataQualityReport()

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Ejecuta el pipeline completo de limpieza

        Args:
            df: DataFrame crudo con datos de demanda

        Returns:
            Tuple con DataFrame limpio y reporte de calidad
        """
        logger.info("Iniciando limpieza de datos de demanda...")
        self.report = DataQualityReport()

        # 1. Validar esquema
        df = self._validate_schema(df)

        # 2. Validar y convertir tipos de datos
        df = self._convert_datatypes(df)

        # 3. Clasificar tipo de día
        df = self._classify_day_type(df)

        # 4. Limpiar columnas innecesarias
        df = self._remove_unnecessary_columns(df)

        # 5. Detectar y tratar valores faltantes
        df = self._handle_missing_values(df)

        # 6. Detectar outliers
        df = self._detect_outliers(df)

        # 7. Validar consistencia de datos
        df = self._validate_consistency(df)
        
        # 8. Calcular estadísticas finales
        self._calculate_final_stats(df)

        # 9. Validar evento de día
        df = self._validate_event_day(df)
        logger.info(f"✓ Limpieza completada: {len(df)} registros válidos")
        return df, self.report

    def _validate_event_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida que existe evento de día y completa valores vacíos."""
        
        # Si no existe la columna, la crea en 0
        if 'Evento Día' not in df.columns:
            self.report.add_issue(
                'MISSING_EVENT_DAY',
                "La columna 'Evento Día' no está presente en el dataset, agregando en 0 por defecto",
                'WARNING'
            )
            df['Evento Día'] = 0
        else:
            # Si existe, llenar los valores vacíos / NaN con 0
            if df['Evento Día'].isna().any() or (df['Evento Día'] == '').any():
                self.report.add_issue(
                    'EMPTY_EVENT_DAY_VALUES',
                    "La columna 'Evento Día' contiene valores vacíos, rellenando con 0",
                    'WARNING'
                )
                df['Evento Día'] = df['Evento Día'].replace('', 0).fillna(0)

        return df



    def _validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida que las columnas esperadas estén presentes"""
        missing_cols = set(POWER_COLUMNS) - set(df.columns)
        if missing_cols:
            self.report.add_issue(
                'SCHEMA_VALIDATION',
                f"Columnas faltantes: {missing_cols}",
                'ERROR'
            )

        # Mantener solo columnas esperadas (ignorar extras)
        valid_cols = [col for col in POWER_COLUMNS if col in df.columns]
        df = df[valid_cols]

        self.report.add_stat('columnas_validas', len(valid_cols))
        return df

    def _convert_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convierte y valida tipos de datos"""
        # Convertir FECHA a datetime
        df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')

        # Convertir periodos horarios a numérico
        for period in HOUR_PERIODS:
            if period in df.columns:
                df[period] = pd.to_numeric(df[period], errors='coerce')

        # Convertir TOTAL a numérico
        if 'TOTAL' in df.columns:
            df['TOTAL'] = pd.to_numeric(df['TOTAL'], errors='coerce')

        # Verificar fechas inválidas
        invalid_dates = df['FECHA'].isna().sum()
        if invalid_dates > 0:
            self.report.add_warning(f"{invalid_dates} fechas inválidas encontradas")

        return df

    def _classify_day_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clasifica días como LABORAL o FESTIVO"""
        if 'TIPO DIA' in df.columns:
            df['TIPO DIA'] = df['TIPO DIA'].apply(
                lambda x: "LABORAL" if str(x).upper() in WORKING_DAYS else "FESTIVO"
            )
            self.report.add_stat(
                'distribucion_dias',
                df['TIPO DIA'].value_counts().to_dict()
            )

        return df

    def _remove_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina columnas innecesarias (Unnamed, etc.)"""
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
            self.report.add_stat('columnas_eliminadas', len(unnamed_cols))

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta y trata valores faltantes"""
        initial_rows = len(df)

        # Calcular porcentaje de valores faltantes por columna
        missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()

        # Reportar columnas con valores faltantes
        cols_with_missing = {k: v for k, v in missing_pct.items() if v > 0}
        if cols_with_missing:
            self.report.add_stat('valores_faltantes_por_columna', cols_with_missing)

        # Eliminar filas con valores faltantes críticos
        critical_cols = ['FECHA', 'UCP', 'VARIABLE'] + HOUR_PERIODS
        critical_cols = [col for col in critical_cols if col in df.columns]

        df = df.dropna(subset=critical_cols)

        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            pct_removed = (rows_removed / initial_rows) * 100
            self.report.add_stat('filas_eliminadas_missing', rows_removed)
            self.report.add_stat('porcentaje_eliminado', f"{pct_removed:.2f}%")

            if pct_removed > self.config['max_missing_percentage'] * 100:
                self.report.add_issue(
                    'EXCESSIVE_MISSING_DATA',
                    f"Se eliminó {pct_removed:.2f}% de datos por valores faltantes (umbral: {self.config['max_missing_percentage']*100}%)",
                    'WARNING'
                )

        return df

    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta outliers en periodos horarios"""
        outliers_found = []

        for period in HOUR_PERIODS:
            if period in df.columns:
                mean = df[period].mean()
                std = df[period].std()
                threshold = self.config['outlier_std_threshold']

                # Detectar outliers (valores más allá de N desviaciones estándar)
                outliers = df[
                    (df[period] < mean - threshold * std) |
                    (df[period] > mean + threshold * std)
                ]

                if len(outliers) > 0:
                    outliers_found.append({
                        'periodo': period,
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(df)) * 100
                    })

        if outliers_found:
            self.report.add_stat('outliers_detectados', outliers_found)
            total_outliers = sum(o['count'] for o in outliers_found)
            self.report.add_warning(f"{total_outliers} outliers detectados en periodos horarios")

        return df

    def _validate_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida consistencia de datos (ej: TOTAL vs suma de periodos)"""
        if 'TOTAL' in df.columns and all(p in df.columns for p in HOUR_PERIODS):
            # Calcular suma de periodos
            df['TOTAL_CALCULATED'] = df[HOUR_PERIODS].sum(axis=1)

            # Comparar con TOTAL reportado
            df['TOTAL_DIFF'] = abs(df['TOTAL'] - df['TOTAL_CALCULATED'])
            df['TOTAL_DIFF_PCT'] = (df['TOTAL_DIFF'] / df['TOTAL']) * 100

            # Identificar discrepancias significativas (>1%)
            inconsistencies = df[df['TOTAL_DIFF_PCT'] > 1.0]

            if len(inconsistencies) > 0:
                self.report.add_warning(
                    f"{len(inconsistencies)} registros con discrepancia entre TOTAL y suma de periodos (>{1}%)"
                )
                self.report.add_stat('registros_inconsistentes', len(inconsistencies))

            # Limpiar columnas temporales
            df = df.drop(columns=['TOTAL_CALCULATED', 'TOTAL_DIFF', 'TOTAL_DIFF_PCT'])

        return df

    def _calculate_final_stats(self, df: pd.DataFrame):
        """Calcula estadísticas finales del dataset limpio"""
        self.report.add_stat('registros_finales', len(df))
        self.report.add_stat('rango_fechas', f"{df['FECHA'].min()} a {df['FECHA'].max()}")

        if 'UCP' in df.columns:
            self.report.add_stat('ucps_unicas', df['UCP'].nunique())

        if 'VARIABLE' in df.columns:
            self.report.add_stat('variables', df['VARIABLE'].unique().tolist())


class WeatherDataCleaner:
    """Limpiador especializado para datos meteorológicos"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or DATA_QUALITY_THRESHOLDS
        self.report = DataQualityReport()

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Ejecuta el pipeline completo de limpieza para datos meteorológicos

        Args:
            df: DataFrame crudo con datos meteorológicos

        Returns:
            Tuple con DataFrame limpio y reporte de calidad
        """
        logger.info("Iniciando limpieza de datos meteorológicos...")
        self.report = DataQualityReport()

        # 1. Validar esquema
        df = self._validate_schema(df)

        # 2. Convertir tipos de datos
        df = self._convert_datatypes(df)

        # 3. Manejar valores faltantes
        df = self._handle_missing_values(df)

        # 4. Detectar outliers
        df = self._detect_outliers(df)

        # 5. Calcular estadísticas finales
        self._calculate_final_stats(df)

        logger.info(f"✓ Limpieza completada: {len(df)} registros meteorológicos válidos")
        return df, self.report

    def _validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida esquema de datos meteorológicos"""
        required_cols = [col for col in WEATHER_COLUMNS if col not in WEATHER_OPTIONAL_COLUMNS]
        missing_cols = set(required_cols) - set(df.columns)

        if missing_cols:
            self.report.add_issue(
                'SCHEMA_VALIDATION',
                f"Columnas obligatorias faltantes: {missing_cols}",
                'ERROR'
            )

        return df

    def _convert_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convierte tipos de datos"""
        # Convertir FECHA a datetime
        if 'FECHA' in df.columns:
            df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')

        # Convertir variables numéricas del formato EPM
        numeric_cols = ['temp_mean', 'temp_min', 'temp_max', 'temp_std',
                        'humidity_mean', 'humidity_min', 'humidity_max',
                        'wind_speed_mean', 'wind_speed_max', 'rain_mean', 'rain_sum']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maneja valores faltantes en datos meteorológicos"""
        initial_rows = len(df)

        # Columnas críticas para formato EPM (no pueden ser nulas)
        critical_cols = ['FECHA', 'temp_mean', 'humidity_mean']
        critical_cols = [col for col in critical_cols if col in df.columns]

        df = df.dropna(subset=critical_cols)

        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            self.report.add_stat('filas_eliminadas', rows_removed)

        return df

    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta outliers en variables meteorológicas"""
        # Rangos razonables para Medellín/Antioquia (formato EPM)
        ranges = {
            'temp_mean': (5, 40),  # °C
            'humidity_mean': (0, 100),  # %
            'wind_speed_mean': (0, 50),  # m/s
            'rain_sum': (0, 300),  # mm/día
        }

        outliers_found = []

        for col, (min_val, max_val) in ranges.items():
            if col in df.columns:
                outliers = df[(df[col] < min_val) | (df[col] > max_val)]
                if len(outliers) > 0:
                    outliers_found.append({
                        'variable': col,
                        'count': len(outliers),
                        'range': f"[{min_val}, {max_val}]"
                    })

        if outliers_found:
            self.report.add_stat('outliers_detectados', outliers_found)

        return df

    def _calculate_final_stats(self, df: pd.DataFrame):
        """Calcula estadísticas finales"""
        self.report.add_stat('registros_finales', len(df))

        if 'dt_iso' in df.columns:
            self.report.add_stat(
                'rango_fechas',
                f"{df['dt_iso'].min()} a {df['dt_iso'].max()}"
            )

        if 'city_name' in df.columns:
            self.report.add_stat('ciudades', df['city_name'].unique().tolist())


# ============== FUNCIONES DE UTILIDAD ==============

def clean_power_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityReport]:
    """Limpia datos de demanda eléctrica"""
    cleaner = PowerDataCleaner()
    return cleaner.clean(df)


def clean_weather_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityReport]:
    """Limpia datos meteorológicos"""
    cleaner = WeatherDataCleaner()
    return cleaner.clean(df)


# ============== EJEMPLO DE USO ==============

if __name__ == "__main__":
    # Ejemplo de uso
    print("Probando sistema de limpieza de datos...\n")

    # Cargar datos de prueba
    power_df = pd.read_csv('../datos.csv')
    print(f"📊 Datos de demanda cargados: {len(power_df)} registros")

    # Limpiar
    power_clean, power_report = clean_power_data(power_df)
    print(power_report.summary())

    # Guardar datos limpios
    output_path = Path('../data/processed/power_cleaned.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    power_clean.to_csv(output_path, index=False)
    print(f"\n✓ Datos limpios guardados en: {output_path}")
