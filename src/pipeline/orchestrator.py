"""
Orquestador Principal del Pipeline Automatizado de Datos - Fase 1
Integra todos los componentes: conectores, limpieza, feature engineering y monitoreo
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import sys

# Añadir directorio raíz al path para imports absolutos
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.config.settings import DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DATA_DIR, RAW_DATA_DIR
except ImportError:
    # Fallback si no se puede importar config
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / 'data'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    FEATURES_DATA_DIR = DATA_DIR / 'features'
    RAW_DATA_DIR = DATA_DIR / 'raw'

from src.pipeline.connectors import PowerDataConnector, WeatherDataConnector
from src.pipeline.cleaning import PowerDataCleaner, WeatherDataCleaner
from src.pipeline.feature_engineering import FeatureEngineer
from src.pipeline.monitoring import PipelineExecutionTracker, DataQualityMonitor


class DataPipelineOrchestrator:
    """
    Orquestador principal del pipeline de datos
    Ejecuta todo el flujo de forma automática: lectura -> limpieza -> feature engineering
    """

    def __init__(self,
                 power_data_path: str,
                 weather_data_path: Optional[str] = None,
                 output_dir: Optional[Path] = None):
        """
        Args:
            power_data_path: Ruta al archivo de datos de demanda
            weather_data_path: Ruta al archivo de datos meteorológicos (opcional)
            output_dir: Directorio de salida para datos procesados
        """
        self.power_data_path = power_data_path
        self.weather_data_path = weather_data_path
        self.output_dir = output_dir or FEATURES_DATA_DIR

        # Asegurar que el directorio de salida existe
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Inicializar tracker de ejecución
        self.tracker = PipelineExecutionTracker("automated_data_pipeline")

        # Inicializar conectores
        self.power_connector = None
        self.weather_connector = None

        # Datos procesados
        self.power_df_raw = None
        self.power_df_clean = None
        self.weather_df_raw = None
        self.weather_df_clean = None
        self.df_with_features = None

        # Reportes
        self.power_quality_report = None
        self.weather_quality_report = None
        self.feature_summary = None

    def run(self,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            save_intermediate: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Ejecuta el pipeline completo de forma automática

        Args:
            start_date: Fecha inicial para filtrar datos (formato YYYY-MM-DD)
            end_date: Fecha final para filtrar datos (formato YYYY-MM-DD)
            save_intermediate: Si True, guarda resultados intermedios

        Returns:
            Tuple con (DataFrame final con features, reporte de ejecución)
        """
        self.tracker.start_pipeline()

        try:
            # ETAPA 1: Lectura de datos
            self._run_data_loading(start_date, end_date)

            # ETAPA 2: Limpieza de datos
            self._run_data_cleaning()

            # ETAPA 3: Feature Engineering
            self._run_feature_engineering()

            # ETAPA 4: Guardar resultados
            if save_intermediate:
                self._save_outputs()

            # Completar pipeline exitosamente
            self.tracker.complete_pipeline(success=True)

            # Generar reporte final
            report = self._generate_final_report()

            print("\n" + "="*70)
            print("PIPELINE COMPLETADO EXITOSAMENTE")
            print("="*70)
            print(f"  Registros finales: {len(self.df_with_features)}")
            print(f"  Features creadas: {self.feature_summary['stats']['total_features']}")
            print(f"  Tiempo total: {report['execution_summary']['total_duration']:.2f}s")
            print("="*70 + "\n")

            return self.df_with_features, report

        except Exception as e:
            from pipeline.monitoring import LogLevel
            self.tracker.logger.log_event(
                LogLevel.ERROR,
                f"Pipeline failed with error: {str(e)}"
            )
            self.tracker.complete_pipeline(success=False)
            raise

    def _run_data_loading(self, start_date: Optional[str], end_date: Optional[str]):
        """Etapa 1: Carga de datos"""
        self.tracker.start_stage("data_loading")

        try:
            # Cargar datos de demanda
            self.power_connector = PowerDataConnector({'path': self.power_data_path})
            self.power_df_raw = self.power_connector.read_data(
                start_date=start_date,
                end_date=end_date
            )

            metadata = {
                'power_records': len(self.power_df_raw),
                'power_date_range': f"{self.power_df_raw['FECHA'].min()} to {self.power_df_raw['FECHA'].max()}"
            }

            # Cargar datos meteorológicos si están disponibles
            if self.weather_data_path:
                self.weather_connector = WeatherDataConnector({'path': self.weather_data_path})
                self.weather_df_raw = self.weather_connector.read_data(
                    start_date=start_date,
                    end_date=end_date
                )
                metadata['weather_records'] = len(self.weather_df_raw)

            self.tracker.complete_stage("data_loading", success=True, metadata=metadata)

        except Exception as e:
            self.tracker.complete_stage("data_loading", success=False)
            raise

    def _run_data_cleaning(self):
        """Etapa 2: Limpieza de datos"""
        self.tracker.start_stage("data_cleaning")

        try:
            # Limpiar datos de demanda
            power_cleaner = PowerDataCleaner()
            self.power_df_clean, self.power_quality_report = power_cleaner.clean(self.power_df_raw)

            # Log del reporte de calidad
            self.tracker.logger.log_data_quality_report(self.power_quality_report)

            metadata = {
                'power_records_after_cleaning': len(self.power_df_clean),
                'power_quality_passed': self.power_quality_report.passed,
                'power_issues_found': len(self.power_quality_report.issues)
            }

            # Limpiar datos meteorológicos si existen
            if self.weather_df_raw is not None:
                weather_cleaner = WeatherDataCleaner()
                self.weather_df_clean, self.weather_quality_report = weather_cleaner.clean(
                    self.weather_df_raw
                )
                self.tracker.logger.log_data_quality_report(self.weather_quality_report)

                metadata['weather_records_after_cleaning'] = len(self.weather_df_clean)
                metadata['weather_quality_passed'] = self.weather_quality_report.passed

            self.tracker.complete_stage("data_cleaning", success=True, metadata=metadata)

        except Exception as e:
            self.tracker.complete_stage("data_cleaning", success=False)
            raise

    def _run_feature_engineering(self):
        """Etapa 3: Feature Engineering"""
        self.tracker.start_stage("feature_engineering")

        try:
            # Crear features
            engineer = FeatureEngineer()
            self.df_with_features = engineer.create_all_features(
                self.power_df_clean,
                self.weather_df_clean
            )

            # Preparar para modelado
            self.df_with_features = engineer.get_feature_importance_ready_df(
                self.df_with_features
            )

            self.feature_summary = engineer.get_feature_summary()

            metadata = {
                'total_features': self.feature_summary['stats']['total_features'],
                'calendar_features': self.feature_summary['stats']['calendar_features'],
                'demand_features': self.feature_summary['stats']['demand_features'],
                'weather_features': self.feature_summary['stats']['weather_features'],
                'final_records': len(self.df_with_features)
            }

            self.tracker.complete_stage("feature_engineering", success=True, metadata=metadata)

        except Exception as e:
            self.tracker.complete_stage("feature_engineering", success=False)
            raise

    def _save_outputs(self):
        """Etapa 4: Guardar resultados"""
        self.tracker.start_stage("saving_outputs")

        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Guardar datos limpios (nombres fijos, sobrescribe)
            power_clean_path = PROCESSED_DATA_DIR / "power_clean_latest.csv"
            self.power_df_clean.to_csv(power_clean_path, index=False)

            if self.weather_df_clean is not None:
                weather_clean_path = PROCESSED_DATA_DIR / "weather_clean_latest.csv"
                self.weather_df_clean.to_csv(weather_clean_path, index=False)

            # Guardar datos con features (nombre fijo, sobrescribe)
            latest_features_path = self.output_dir / "data_with_features_latest.csv"
            self.df_with_features.to_csv(latest_features_path, index=False)

            metadata = {
                'power_clean_path': str(power_clean_path),
                'features_path': str(latest_features_path),
                'latest_features_path': str(latest_features_path)
            }

            self.tracker.complete_stage("saving_outputs", success=True, metadata=metadata)

        except Exception as e:
            self.tracker.complete_stage("saving_outputs", success=False)
            raise

    def _generate_final_report(self) -> Dict:
        """Genera el reporte final de ejecución"""
        execution_report = self.tracker.get_execution_report()

        report = {
            'pipeline_version': '1.0.0',
            'execution_timestamp': datetime.now().isoformat(),
            'execution_summary': {
                'status': 'SUCCESS',
                'start_time': execution_report['start_time'],
                'end_time': execution_report['end_time'],
                'total_duration': execution_report['total_duration']
            },
            'data_summary': {
                'input_records': len(self.power_df_raw),
                'cleaned_records': len(self.power_df_clean),
                'final_records': len(self.df_with_features),
                'features_created': self.feature_summary['stats']['total_features']
            },
            'quality_reports': {
                'power_data': {
                    'passed': self.power_quality_report.passed,
                    'issues_count': len(self.power_quality_report.issues),
                    'warnings_count': len(self.power_quality_report.warnings),
                    'stats': self.power_quality_report.stats
                }
            },
            'feature_summary': self.feature_summary,
            'stages': execution_report['stages']
        }

        if self.weather_quality_report:
            report['quality_reports']['weather_data'] = {
                'passed': self.weather_quality_report.passed,
                'issues_count': len(self.weather_quality_report.issues),
                'warnings_count': len(self.weather_quality_report.warnings),
                'stats': self.weather_quality_report.stats
            }

        # Guardar reporte (modo producción: sobrescribe)
        report_path = self.tracker.save_report(keep_history=False)

        return report


# ============== FUNCIÓN DE UTILIDAD PRINCIPAL ==============

def run_automated_pipeline(power_data_path: str,
                           weather_data_path: Optional[str] = None,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           output_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Función principal para ejecutar el pipeline automatizado completo

    Args:
        power_data_path: Ruta al archivo CSV de datos de demanda
        weather_data_path: Ruta al archivo CSV de datos meteorológicos (opcional)
        start_date: Fecha inicial en formato YYYY-MM-DD (opcional)
        end_date: Fecha final en formato YYYY-MM-DD (opcional)
        output_dir: Directorio de salida (opcional)

    Returns:
        Tuple con (DataFrame con features, reporte de ejecución)

    Example:
        >>> df, report = run_automated_pipeline(
        ...     power_data_path='datos.csv',
        ...     weather_data_path='data_cleaned_weather.csv',
        ...     start_date='2017-01-01',
        ...     end_date='2017-12-31'
        ... )
    """
    orchestrator = DataPipelineOrchestrator(
        power_data_path=power_data_path,
        weather_data_path=weather_data_path,
        output_dir=output_dir
    )

    return orchestrator.run(
        start_date=start_date,
        end_date=end_date,
        save_intermediate=True
    )


# ============== EJEMPLO DE USO ==============

if __name__ == "__main__":
    print("="*70)
    print("EJECUTANDO PIPELINE AUTOMATIZADO DE DATOS - FASE 1")
    print("Sistema de Pronóstico de Demanda Energética EPM")
    print("="*70 + "\n")

    # Configurar rutas
    base_dir = Path(__file__).parent.parent
    power_path = base_dir / "datos.csv"
    weather_path = base_dir / "data_cleaned_weather.csv"

    # Ejecutar pipeline
    try:
        df_final, report = run_automated_pipeline(
            power_data_path=str(power_path),
            weather_data_path=str(weather_path),
            start_date='2017-01-01'  # Filtrar datos desde 2017
        )

        print("\n📊 RESULTADOS FINALES:")
        print(f"  - Shape del dataset: {df_final.shape}")
        print(f"  - Columnas totales: {len(df_final.columns)}")
        print(f"  - Rango de fechas: {df_final['FECHA'].min()} a {df_final['FECHA'].max()}")
        print(f"\n✓ Datos listos para Fase 2: Desarrollo de Modelos")

    except FileNotFoundError as e:
        print(f"\n⚠️  Error: {e}")
        print("   Asegúrate de que los archivos de datos existen en el directorio correcto")

    except Exception as e:
        print(f"\nError en el pipeline: {e}")
        import traceback
        traceback.print_exc()
