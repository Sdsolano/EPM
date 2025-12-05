"""
Sistema de Logging y Monitoreo de Calidad de Datos
Implementa logging estructurado, alertas y seguimiento de métricas
"""
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import sys

# Añadir directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config import LOGS_DIR, DATA_QUALITY_THRESHOLDS
except ImportError:
    LOGS_DIR = Path('logs')
    DATA_QUALITY_THRESHOLDS = {'max_missing_percentage': 0.05}

# Crear directorio de logs si no existe
LOGS_DIR = Path(LOGS_DIR)
LOGS_DIR.mkdir(exist_ok=True)


class LogLevel(Enum):
    """Niveles de logging"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """Tipos de alertas"""
    DATA_QUALITY = "DATA_QUALITY"
    MISSING_DATA = "MISSING_DATA"
    OUTLIER_DETECTED = "OUTLIER_DETECTED"
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    PERFORMANCE_DEGRADATION = "PERFORMANCE_DEGRADATION"


class PipelineLogger:
    """Logger estructurado para el pipeline de datos"""

    def __init__(self, name: str, log_to_file: bool = True):
        self.name = name
        self.log_to_file = log_to_file
        self.logger = self._setup_logger()
        self.events = []
        self.alerts = []

    def _setup_logger(self) -> logging.Logger:
        """Configura el logger con handlers de consola y archivo"""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)

        # Evitar duplicar handlers
        if logger.handlers:
            return logger

        # Formato detallado
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Handler de consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Handler de archivo (sobrescribe archivo fijo)
        if self.log_to_file:
            log_file = LOGS_DIR / f"{self.name}_latest.log"
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')  # mode='w' sobrescribe
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def log_event(self, level: LogLevel, message: str, metadata: Optional[Dict] = None):
        """
        Registra un evento con metadata estructurada

        Args:
            level: Nivel del log
            message: Mensaje descriptivo
            metadata: Metadata adicional (diccionario)
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'level': level.value,
            'message': message,
            'metadata': metadata or {}
        }

        self.events.append(event)

        # Log en el logger de Python
        log_method = getattr(self.logger, level.value.lower())
        log_message = message
        if metadata:
            log_message += f" | Metadata: {json.dumps(metadata, ensure_ascii=False, default=str)}"
        log_method(log_message)

    def log_alert(self, alert_type: AlertType, description: str, severity: str, metadata: Optional[Dict] = None):
        """
        Registra una alerta

        Args:
            alert_type: Tipo de alerta
            description: Descripción de la alerta
            severity: Severidad (LOW, MEDIUM, HIGH, CRITICAL)
            metadata: Metadata adicional
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': alert_type.value,
            'description': description,
            'severity': severity,
            'metadata': metadata or {}
        }

        self.alerts.append(alert)

        # Log como WARNING o ERROR según severidad
        level = logging.ERROR if severity in ['HIGH', 'CRITICAL'] else logging.WARNING
        self.logger.log(
            level,
            f"[ALERT:{alert_type.value}] {description} | Severity: {severity}"
        )

    def log_data_quality_report(self, report: Any):
        """
        Registra un reporte de calidad de datos

        Args:
            report: Objeto DataQualityReport
        """
        metadata = {
            'total_issues': len(report.issues),
            'total_warnings': len(report.warnings),
            'passed': report.passed,
            'stats': report.stats
        }

        level = LogLevel.INFO if report.passed else LogLevel.ERROR
        self.log_event(
            level,
            f"Data Quality Report - Status: {'PASSED' if report.passed else 'FAILED'}",
            metadata
        )

        # Registrar cada issue como alerta
        for issue in report.issues:
            self.log_alert(
                AlertType.DATA_QUALITY,
                issue['description'],
                issue['severity'],
                {'issue_type': issue['type']}
            )

    def save_events_to_file(self, filename: Optional[str] = None):
        """Guarda todos los eventos en un archivo JSON"""
        if not filename:
            filename = f"{self.name}_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = LOGS_DIR / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'logger_name': self.name,
                'timestamp': datetime.now().isoformat(),
                'events': self.events,
                'alerts': self.alerts
            }, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Events saved to: {filepath}")
        return filepath

    def get_summary(self) -> Dict:
        """Retorna un resumen de los eventos y alertas"""
        return {
            'total_events': len(self.events),
            'events_by_level': {
                level.value: sum(1 for e in self.events if e['level'] == level.value)
                for level in LogLevel
            },
            'total_alerts': len(self.alerts),
            'alerts_by_type': {
                alert_type.value: sum(1 for a in self.alerts if a['alert_type'] == alert_type.value)
                for alert_type in AlertType
            },
            'alerts_by_severity': {
                severity: sum(1 for a in self.alerts if a['severity'] == severity)
                for severity in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            }
        }


class DataQualityMonitor:
    """Monitor especializado para calidad de datos"""

    def __init__(self, logger: PipelineLogger):
        self.logger = logger
        self.thresholds = DATA_QUALITY_THRESHOLDS

    def check_missing_data(self, df, column_name: str, threshold: Optional[float] = None) -> bool:
        """
        Verifica porcentaje de datos faltantes

        Returns:
            True si pasa la validación, False si no
        """
        threshold = threshold or self.thresholds.get('max_missing_percentage', 0.05)
        missing_pct = df[column_name].isnull().sum() / len(df)

        if missing_pct > threshold:
            self.logger.log_alert(
                AlertType.MISSING_DATA,
                f"Column '{column_name}' has {missing_pct*100:.2f}% missing data (threshold: {threshold*100}%)",
                'HIGH',
                {'column': column_name, 'missing_percentage': missing_pct}
            )
            return False

        return True

    def check_outliers(self, values: List[float], column_name: str) -> Dict:
        """
        Detecta outliers usando método IQR

        Returns:
            Dict con información de outliers
        """
        import numpy as np

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outliers = [v for v in values if v < lower_bound or v > upper_bound]
        outlier_pct = len(outliers) / len(values) * 100

        if outlier_pct > 1:  # Si más del 1% son outliers
            self.logger.log_alert(
                AlertType.OUTLIER_DETECTED,
                f"Column '{column_name}' has {outlier_pct:.2f}% outliers",
                'MEDIUM',
                {
                    'column': column_name,
                    'outlier_count': len(outliers),
                    'outlier_percentage': outlier_pct,
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }
            )

        return {
            'outlier_count': len(outliers),
            'outlier_percentage': outlier_pct,
            'bounds': {'lower': lower_bound, 'upper': upper_bound}
        }

    def monitor_processing_time(self, process_name: str, start_time: datetime, end_time: datetime):
        """Monitorea tiempo de procesamiento"""
        duration = (end_time - start_time).total_seconds()

        self.logger.log_event(
            LogLevel.INFO,
            f"Process '{process_name}' completed",
            {
                'process_name': process_name,
                'duration_seconds': duration,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
        )


class PipelineExecutionTracker:
    """Rastreador de ejecución del pipeline completo"""

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.logger = PipelineLogger(f"pipeline_{pipeline_name}")
        self.start_time = None
        self.end_time = None
        self.stages = []
        self.current_stage = None

    def start_pipeline(self):
        """Inicia el rastreo del pipeline"""
        self.start_time = datetime.now()
        self.logger.log_event(
            LogLevel.INFO,
            f"Pipeline '{self.pipeline_name}' started",
            {'start_time': self.start_time.isoformat()}
        )

    def start_stage(self, stage_name: str):
        """Inicia el rastreo de una etapa"""
        self.current_stage = {
            'name': stage_name,
            'start_time': datetime.now(),
            'end_time': None,
            'status': 'IN_PROGRESS',
            'errors': []
        }

        self.logger.log_event(
            LogLevel.INFO,
            f"Stage '{stage_name}' started",
            {'stage_name': stage_name}
        )

    def complete_stage(self, stage_name: str, success: bool = True, metadata: Optional[Dict] = None):
        """Completa el rastreo de una etapa"""
        if self.current_stage and self.current_stage['name'] == stage_name:
            self.current_stage['end_time'] = datetime.now()
            self.current_stage['status'] = 'SUCCESS' if success else 'FAILED'
            self.current_stage['duration'] = (
                self.current_stage['end_time'] - self.current_stage['start_time']
            ).total_seconds()
            self.current_stage['metadata'] = metadata or {}

            self.stages.append(self.current_stage)

            level = LogLevel.INFO if success else LogLevel.ERROR
            self.logger.log_event(
                level,
                f"Stage '{stage_name}' completed - Status: {self.current_stage['status']}",
                {
                    'stage_name': stage_name,
                    'duration': self.current_stage['duration'],
                    'status': self.current_stage['status']
                }
            )

            self.current_stage = None

    def complete_pipeline(self, success: bool = True):
        """Completa el rastreo del pipeline"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        status = 'SUCCESS' if success else 'FAILED'

        self.logger.log_event(
            LogLevel.INFO if success else LogLevel.ERROR,
            f"Pipeline '{self.pipeline_name}' completed - Status: {status}",
            {
                'pipeline_name': self.pipeline_name,
                'status': status,
                'total_duration': duration,
                'total_stages': len(self.stages),
                'successful_stages': sum(1 for s in self.stages if s['status'] == 'SUCCESS'),
                'failed_stages': sum(1 for s in self.stages if s['status'] == 'FAILED')
            }
        )

    def get_execution_report(self) -> Dict:
        """Genera un reporte de ejecución completo"""
        return {
            'pipeline_name': self.pipeline_name,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_duration': (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else None,
            'stages': self.stages,
            'logger_summary': self.logger.get_summary()
        }

    def save_report(self, keep_history: bool = False):
        """
        Guarda el reporte de ejecución

        Args:
            keep_history: Si True, usa timestamp. Si False, sobrescribe archivo fijo
        """
        report = self.get_execution_report()

        if keep_history:
            filename = f"pipeline_execution_{self.pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            filename = f"pipeline_execution_{self.pipeline_name}_latest.json"

        filepath = LOGS_DIR / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        self.logger.logger.info(f"Execution report saved to: {filepath}")
        return filepath


# ============== EJEMPLO DE USO ==============

if __name__ == "__main__":
    print("Probando sistema de logging y monitoreo...\n")

    # Crear tracker de pipeline
    tracker = PipelineExecutionTracker("data_pipeline_phase1")
    tracker.start_pipeline()

    # Simular etapas
    tracker.start_stage("data_loading")
    import time
    time.sleep(1)
    tracker.complete_stage("data_loading", success=True, metadata={'rows_loaded': 1000})

    tracker.start_stage("data_cleaning")
    time.sleep(1)
    tracker.complete_stage("data_cleaning", success=True, metadata={'rows_after_cleaning': 950})

    tracker.start_stage("feature_engineering")
    time.sleep(1)
    tracker.complete_stage("feature_engineering", success=True, metadata={'features_created': 45})

    # Completar pipeline
    tracker.complete_pipeline(success=True)

    # Guardar reporte
    report_path = tracker.save_report()
    print(f"\n✓ Reporte guardado en: {report_path}")

    # Mostrar resumen
    print("\n" + "="*60)
    print("RESUMEN DEL LOGGER")
    print("="*60)
    summary = tracker.logger.get_summary()
    print(json.dumps(summary, indent=2))
