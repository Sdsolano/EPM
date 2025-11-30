"""
Motor de Desagregación Horaria Unificado

Este módulo orquesta todo el sistema de desagregación horaria,
decidiendo automáticamente qué método usar según el tipo de día.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union
import logging

from .hourly_disaggregator import HourlyDisaggregator
from .special_days import SpecialDaysDisaggregator
from .calendar_utils import CalendarClassifier
from ...config.settings import MODELS_DIR

logger = logging.getLogger(__name__)


class HourlyDisaggregationEngine:
    """
    Motor unificado de desagregación horaria.

    Orquesta:
    1. Clasificación de días (normal vs especial)
    2. Selección del desagregador apropiado
    3. Generación de predicciones horarias
    4. Validación de resultados
    """

    def __init__(
        self,
        normal_disaggregator: Optional[HourlyDisaggregator] = None,
        special_disaggregator: Optional[SpecialDaysDisaggregator] = None,
        auto_load: bool = True
    ):
        """
        Inicializa el motor de desagregación.

        Args:
            normal_disaggregator: Desagregador para días normales (opcional)
            special_disaggregator: Desagregador para días especiales (opcional)
            auto_load: Si True, intenta cargar modelos guardados automáticamente
        """
        self.calendar_classifier = CalendarClassifier()

        # Cargar o usar desagregadores proporcionados
        if auto_load:
            self.normal_disaggregator = self._load_or_create(
                normal_disaggregator,
                HourlyDisaggregator,
                "hourly_disaggregator.pkl"
            )
            self.special_disaggregator = self._load_or_create(
                special_disaggregator,
                SpecialDaysDisaggregator,
                "special_days_disaggregator.pkl"
            )
        else:
            self.normal_disaggregator = normal_disaggregator
            self.special_disaggregator = special_disaggregator

    def _load_or_create(self, instance, cls, filename):
        """Intenta cargar modelo guardado o crear instancia nueva."""
        if instance is not None:
            return instance

        try:
            filepath = Path(MODELS_DIR) / filename
            if filepath.exists():
                logger.info(f"Cargando {filename}...")
                return cls.load(filepath)
            else:
                logger.warning(f"No se encontró {filename}, creando instancia nueva (sin entrenar)")
                return cls()
        except Exception as e:
            logger.error(f"Error cargando {filename}: {e}")
            return cls()

    def predict_hourly(
        self,
        date: Union[str, pd.Timestamp],
        total_daily: float,
        validate: bool = True
    ) -> Dict:
        """
        Predice la distribución horaria para una fecha y total diario.

        Args:
            date: Fecha del pronóstico
            total_daily: Demanda total del día (MWh)
            validate: Si True, valida que la suma sea correcta

        Returns:
            Dict con:
                - date: Fecha
                - total_daily: Total predicho
                - hourly: Array con distribución horaria (P1-P24)
                - method: Método usado ("special" o "normal")
                - day_type: Tipo de día
                - validation: Dict con información de validación
        """
        date = pd.to_datetime(date)

        # Clasificar el día
        day_info = self.calendar_classifier.get_full_classification(date)
        is_special = self.special_disaggregator.is_special_day(date) if self.special_disaggregator.is_fitted else False

        # Seleccionar método
        if is_special:
            method = "special"
            hourly = self.special_disaggregator.predict_hourly_profile(date, total_daily)
        else:
            method = "normal"
            if not self.normal_disaggregator.is_fitted:
                raise RuntimeError("El desagregador normal no está entrenado")
            hourly = self.normal_disaggregator.predict_hourly_profile(date, total_daily)

        # Validación
        validation = {
            'sum': hourly.sum(),
            'expected': total_daily,
            'difference': abs(hourly.sum() - total_daily),
            'relative_error': abs(hourly.sum() - total_daily) / total_daily * 100,
            'is_valid': abs(hourly.sum() - total_daily) < 0.01  # Tolerancia de 0.01 MWh
        }

        if validate and not validation['is_valid']:
            logger.warning(
                f"⚠ Validación fallida para {date.date()}: "
                f"suma={validation['sum']:.4f}, esperado={total_daily:.4f}"
            )

        return {
            'date': date,
            'total_daily': total_daily,
            'hourly': hourly,
            'method': method,
            'day_type': day_info['tipo_dia'],
            'day_name': day_info['dia_semana'],
            'is_holiday': day_info['es_festivo'],
            'holiday_name': day_info['nombre_festivo'],
            'season': day_info['temporada'],
            'validation': validation
        }

    def predict_batch(
        self,
        dates: pd.Series,
        totals_daily: pd.Series,
        return_dataframe: bool = True
    ) -> Union[pd.DataFrame, list]:
        """
        Predice distribuciones horarias para múltiples fechas.

        Args:
            dates: Series de fechas
            totals_daily: Series de totales diarios
            return_dataframe: Si True, retorna DataFrame; si False, lista de dicts

        Returns:
            DataFrame o lista con predicciones horarias
        """
        if len(dates) != len(totals_daily):
            raise ValueError("dates y totals_daily deben tener la misma longitud")

        results = []
        for date, total in zip(dates, totals_daily):
            result = self.predict_hourly(date, total, validate=True)
            results.append(result)

        if return_dataframe:
            return self._results_to_dataframe(results)
        else:
            return results

    def _results_to_dataframe(self, results: list) -> pd.DataFrame:
        """Convierte lista de resultados a DataFrame."""
        rows = []
        for r in results:
            row = {
                'FECHA': r['date'],
                'TOTAL': r['total_daily'],
                'method': r['method'],
                'day_type': r['day_type'],
                'day_name': r['day_name'],
                'is_holiday': r['is_holiday'],
                'validation_ok': r['validation']['is_valid'],
                **{f'P{i}': r['hourly'][i-1] for i in range(1, 25)}
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def get_engine_status(self) -> Dict:
        """
        Retorna el estado del motor de desagregación.

        Returns:
            Dict con información de estado
        """
        return {
            'normal_disaggregator': {
                'fitted': self.normal_disaggregator.is_fitted,
                'n_clusters': self.normal_disaggregator.n_clusters if self.normal_disaggregator.is_fitted else None,
            },
            'special_disaggregator': {
                'fitted': self.special_disaggregator.is_fitted,
                'n_clusters': self.special_disaggregator.n_clusters if self.special_disaggregator.is_fitted else None,
                'n_special_days': len(self.special_disaggregator.cluster_by_date) if self.special_disaggregator.is_fitted else 0,
            }
        }

    def train_all(
        self,
        data_path: Optional[Path] = None,
        n_clusters_normal: int = 35,
        n_clusters_special: int = 15,
        save: bool = True
    ) -> None:
        """
        Entrena ambos desagregadores con datos históricos.

        Args:
            data_path: Ruta a datos históricos
            n_clusters_normal: Clusters para días normales
            n_clusters_special: Clusters para días especiales
            save: Si True, guarda modelos entrenados
        """
        logger.info("=" * 80)
        logger.info("ENTRENAMIENTO COMPLETO DEL SISTEMA DE DESAGREGACIÓN")
        logger.info("=" * 80)

        from ...config.settings import FEATURES_DATA_DIR

        if data_path is None:
            data_path = Path(FEATURES_DATA_DIR) / "data_with_features_latest.csv"

        logger.info(f"Cargando datos desde {data_path}...")
        df = pd.read_csv(data_path)

        # Entrenar desagregador normal
        logger.info("\n1. Entrenando desagregador para días normales...")
        self.normal_disaggregator = HourlyDisaggregator(n_clusters=n_clusters_normal)
        self.normal_disaggregator.fit(df, date_column='FECHA')

        if save:
            self.normal_disaggregator.save()

        # Entrenar desagregador especial
        logger.info("\n2. Entrenando desagregador para días especiales...")
        self.special_disaggregator = SpecialDaysDisaggregator(n_clusters=n_clusters_special)
        self.special_disaggregator.fit(df, date_column='FECHA')

        if save:
            self.special_disaggregator.save()

        logger.info("\n" + "=" * 80)
        logger.info("✓ SISTEMA DE DESAGREGACIÓN ENTRENADO COMPLETAMENTE")
        logger.info("=" * 80)

        # Mostrar resumen
        status = self.get_engine_status()
        logger.info(f"\nResumen:")
        logger.info(f"  - Días normales: {status['normal_disaggregator']['n_clusters']} clusters")
        logger.info(f"  - Días especiales: {status['special_disaggregator']['n_clusters']} clusters")
        logger.info(f"  - Festivos conocidos: {status['special_disaggregator']['n_special_days']}")


# ============== EJEMPLO DE USO ==============

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("SISTEMA DE DESAGREGACIÓN HORARIA - EPM")
    print("=" * 80)

    # Crear motor
    engine = HourlyDisaggregationEngine(auto_load=True)

    # Verificar estado
    status = engine.get_engine_status()
    print("\n📊 Estado del sistema:")
    print(f"  Normal: {'✓ Entrenado' if status['normal_disaggregator']['fitted'] else '✗ No entrenado'}")
    print(f"  Especial: {'✓ Entrenado' if status['special_disaggregator']['fitted'] else '✗ No entrenado'}")

    # Si no está entrenado, entrenar
    if not (status['normal_disaggregator']['fitted'] and status['special_disaggregator']['fitted']):
        print("\n🔧 Entrenando sistema...")
        engine.train_all(n_clusters_normal=35, n_clusters_special=15, save=True)

    # Probar predicciones
    print("\n" + "=" * 80)
    print("PREDICCIONES DE PRUEBA")
    print("=" * 80)

    test_cases = [
        ("2024-03-15", 1500.0, "Viernes normal"),
        ("2024-03-17", 1200.0, "Domingo"),
        ("2024-12-25", 1100.0, "Navidad"),
        ("2024-01-01", 1050.0, "Año Nuevo"),
    ]

    for date_str, total, description in test_cases:
        result = engine.predict_hourly(date_str, total)

        print(f"\n📅 {description} ({date_str})")
        print(f"   Tipo: {result['day_type']} | Método: {result['method']}")
        print(f"   Total: {total} MWh")
        print(f"   Suma horaria: {result['validation']['sum']:.2f} MWh")
        print(f"   Diferencia: {result['validation']['difference']:.6f} MWh")
        print(f"   Validación: {'✓ OK' if result['validation']['is_valid'] else '✗ FALLO'}")

    print("\n" + "=" * 80)
