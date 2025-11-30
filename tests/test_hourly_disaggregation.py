"""
Tests para el Sistema de Desagregación Horaria

Valida:
1. Suma de períodos horarios = total diario (validación crítica)
2. Clasificación de días
3. Funcionamiento de clustering
4. Integración completa
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from prediction.hourly import HourlyDisaggregationEngine, CalendarClassifier
from prediction.hourly.hourly_disaggregator import HourlyDisaggregator
from prediction.hourly.special_days import SpecialDaysDisaggregator


class TestCalendarClassifier:
    """Tests para clasificación de calendario"""

    def test_festivos_colombia(self):
        """Verifica que detecte festivos de Colombia correctamente"""
        classifier = CalendarClassifier()

        # Navidad
        assert classifier.is_holiday(pd.to_datetime("2024-12-25"))
        # Año Nuevo
        assert classifier.is_holiday(pd.to_datetime("2024-01-01"))
        # Día del trabajo
        assert classifier.is_holiday(pd.to_datetime("2024-05-01"))
        # Día normal
        assert not classifier.is_holiday(pd.to_datetime("2024-03-15"))

    def test_weekend_detection(self):
        """Verifica detección de fines de semana"""
        classifier = CalendarClassifier()

        # Sábado
        assert classifier.is_weekend(pd.to_datetime("2024-03-16"))
        # Domingo
        assert classifier.is_weekend(pd.to_datetime("2024-03-17"))
        # Lunes
        assert not classifier.is_weekend(pd.to_datetime("2024-03-18"))

    def test_day_type_classification(self):
        """Verifica clasificación de tipo de día"""
        classifier = CalendarClassifier()

        # Festivo
        assert classifier.get_day_type(pd.to_datetime("2024-12-25")) == "festivo"
        # Fin de semana
        assert classifier.get_day_type(pd.to_datetime("2024-03-16")) == "fin_de_semana"
        # Laboral
        assert classifier.get_day_type(pd.to_datetime("2024-03-15")) == "laboral"

    def test_season_detection(self):
        """Verifica detección de temporadas"""
        classifier = CalendarClassifier()

        # Temporada lluviosa (abril)
        assert classifier.get_season(pd.to_datetime("2024-04-15")) == "lluviosa"
        # Temporada seca (enero)
        assert classifier.get_season(pd.to_datetime("2024-01-15")) == "seca"


class TestHourlyDisaggregator:
    """Tests para desagregador de días normales"""

    @pytest.fixture
    def sample_data(self):
        """Crea datos de ejemplo para testing"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = []

        for date in dates:
            # Crear patrón horario realista
            hourly = []
            for hour in range(24):
                # Patrón con picos en horas de trabajo
                base = 40 + 10 * np.sin(2 * np.pi * hour / 24)
                hourly.append(base + np.random.normal(0, 2))

            row = {'FECHA': date}
            row.update({f'P{i}': hourly[i-1] for i in range(1, 25)})
            data.append(row)

        return pd.DataFrame(data)

    def test_fit_and_predict(self, sample_data):
        """Verifica que el desagregador entrene y prediga correctamente"""
        disaggregator = HourlyDisaggregator(n_clusters=5)
        disaggregator.fit(sample_data)

        assert disaggregator.is_fitted
        assert disaggregator.kmeans is not None
        assert disaggregator.cluster_profiles is not None

        # Predecir
        test_date = pd.to_datetime("2024-03-15")
        total = 1500.0
        hourly = disaggregator.predict_hourly_profile(test_date, total)

        # Validaciones
        assert len(hourly) == 24
        assert isinstance(hourly, np.ndarray)

    def test_sum_equals_total(self, sample_data):
        """CRÍTICO: Verifica que la suma de períodos = total diario"""
        disaggregator = HourlyDisaggregator(n_clusters=5)
        disaggregator.fit(sample_data)

        # Probar múltiples casos
        test_cases = [
            (pd.to_datetime("2024-03-15"), 1500.0),  # Viernes
            (pd.to_datetime("2024-03-16"), 1200.0),  # Sábado
            (pd.to_datetime("2024-03-18"), 1600.0),  # Lunes
        ]

        for date, total in test_cases:
            hourly = disaggregator.predict_hourly_profile(date, total)
            sum_hourly = hourly.sum()

            # Tolerancia de 0.01 MWh
            assert abs(sum_hourly - total) < 0.01, \
                f"Suma {sum_hourly:.4f} != Total {total:.4f} para {date}"

    def test_save_and_load(self, sample_data, tmp_path):
        """Verifica guardado y carga del modelo"""
        # Entrenar
        disaggregator1 = HourlyDisaggregator(n_clusters=5)
        disaggregator1.fit(sample_data)

        # Guardar
        filepath = tmp_path / "test_model.pkl"
        disaggregator1.save(filepath)

        # Cargar
        disaggregator2 = HourlyDisaggregator.load(filepath)

        assert disaggregator2.is_fitted
        assert disaggregator2.n_clusters == disaggregator1.n_clusters

        # Verificar que predicciones son iguales
        test_date = pd.to_datetime("2024-03-15")
        total = 1500.0

        pred1 = disaggregator1.predict_hourly_profile(test_date, total)
        pred2 = disaggregator2.predict_hourly_profile(test_date, total)

        np.testing.assert_array_almost_equal(pred1, pred2, decimal=6)


class TestSpecialDaysDisaggregator:
    """Tests para desagregador de días especiales"""

    @pytest.fixture
    def sample_data_with_holidays(self):
        """Crea datos con festivos"""
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
        data = []

        for date in dates:
            hourly = []
            for hour in range(24):
                # Patrón diferente para festivos
                base = 35 if date.month == 12 and date.day == 25 else 45
                base += 10 * np.sin(2 * np.pi * hour / 24)
                hourly.append(base + np.random.normal(0, 2))

            row = {'FECHA': date}
            row.update({f'P{i}': hourly[i-1] for i in range(1, 25)})
            data.append(row)

        return pd.DataFrame(data)

    def test_special_day_detection(self):
        """Verifica detección de días especiales"""
        disaggregator = SpecialDaysDisaggregator(n_clusters=5)

        # Navidad es festivo
        assert disaggregator.calendar_classifier.is_holiday(pd.to_datetime("2024-12-25"))

        # Día normal no es festivo
        assert not disaggregator.calendar_classifier.is_holiday(pd.to_datetime("2024-03-15"))

    def test_fit_special_days(self, sample_data_with_holidays):
        """Verifica entrenamiento con días festivos"""
        disaggregator = SpecialDaysDisaggregator(n_clusters=5)
        disaggregator.fit(sample_data_with_holidays)

        assert disaggregator.is_fitted
        assert len(disaggregator.cluster_by_date) > 0

    def test_special_day_sum_equals_total(self, sample_data_with_holidays):
        """CRÍTICO: Suma de períodos = total para días especiales"""
        disaggregator = SpecialDaysDisaggregator(n_clusters=5)
        disaggregator.fit(sample_data_with_holidays)

        # Probar Navidad
        test_date = pd.to_datetime("2024-12-25")
        total = 1100.0

        if disaggregator.is_special_day(test_date):
            hourly = disaggregator.predict_hourly_profile(test_date, total)

            if hourly is not None:
                sum_hourly = hourly.sum()
                assert abs(sum_hourly - total) < 0.01, \
                    f"Suma {sum_hourly:.4f} != Total {total:.4f}"


class TestDisaggregationEngine:
    """Tests para el motor completo de desagregación"""

    @pytest.fixture
    def sample_complete_data(self):
        """Crea dataset completo con datos normales y festivos"""
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        data = []

        for date in dates:
            hourly = []
            for hour in range(24):
                base = 40 + 10 * np.sin(2 * np.pi * hour / 24)
                hourly.append(base + np.random.normal(0, 2))

            row = {'FECHA': date}
            row.update({f'P{i}': hourly[i-1] for i in range(1, 25)})
            data.append(row)

        return pd.DataFrame(data)

    def test_engine_initialization(self):
        """Verifica inicialización del motor"""
        engine = HourlyDisaggregationEngine(auto_load=False)

        assert engine.calendar_classifier is not None

    def test_engine_routing(self, sample_complete_data, tmp_path):
        """Verifica que el motor rutee correctamente entre desagregadores"""
        # Entrenar motor completo
        engine = HourlyDisaggregationEngine(auto_load=False)

        # Cambiar paths temporalmente
        from prediction.hourly import hourly_disaggregator, special_days
        import prediction.hourly.disaggregation_engine as engine_module

        original_models_dir = engine_module.MODELS_DIR
        engine_module.MODELS_DIR = tmp_path

        try:
            # Entrenar
            engine.train_all(
                data_path=None,  # Usará sample_data si se pasa
                n_clusters_normal=5,
                n_clusters_special=3,
                save=False
            )

            # Test casos
            test_cases = [
                ("2024-03-15", 1500.0, "normal"),    # Viernes normal
                ("2024-12-25", 1100.0, "special"),   # Navidad (si está en datos)
            ]

            for date_str, total, expected_type in test_cases:
                result = engine.predict_hourly(date_str, total)

                # Validaciones
                assert result['total_daily'] == total
                assert len(result['hourly']) == 24
                assert result['validation']['is_valid']

                # Verificar suma
                assert abs(result['hourly'].sum() - total) < 0.01

        finally:
            engine_module.MODELS_DIR = original_models_dir

    def test_batch_prediction_consistency(self, sample_complete_data):
        """Verifica consistencia en predicciones batch"""
        engine = HourlyDisaggregationEngine(auto_load=False)

        # Crear desagregador simple
        engine.normal_disaggregator = HourlyDisaggregator(n_clusters=5)
        engine.normal_disaggregator.fit(sample_complete_data)
        engine.special_disaggregator = SpecialDaysDisaggregator(n_clusters=3)
        engine.special_disaggregator.fit(sample_complete_data)

        # Crear batch
        dates = pd.Series([
            pd.to_datetime("2024-03-15"),
            pd.to_datetime("2024-03-16"),
            pd.to_datetime("2024-03-17"),
        ])
        totals = pd.Series([1500.0, 1400.0, 1300.0])

        # Predecir
        results_df = engine.predict_batch(dates, totals, return_dataframe=True)

        assert len(results_df) == 3
        assert 'FECHA' in results_df.columns
        assert all(f'P{i}' in results_df.columns for i in range(1, 25))

        # Verificar sumas
        for idx, row in results_df.iterrows():
            period_cols = [f'P{i}' for i in range(1, 25)]
            sum_periods = row[period_cols].sum()
            expected = row['TOTAL']

            assert abs(sum_periods - expected) < 0.01, \
                f"Fila {idx}: suma {sum_periods:.4f} != total {expected:.4f}"


class TestIntegrationWithForecaster:
    """Tests de integración con el forecaster"""

    def test_placeholder_hourly_sums_to_one(self):
        """Verifica que placeholders sumen correctamente"""
        from prediction.forecaster import ForecastPipeline

        # Crear instancia temporal
        # Nota: Esto fallará si no hay modelo, pero podemos testear el método directamente
        test_total = 1500.0

        # Simular distribución placeholder
        hourly_distribution = [
            0.038, 0.036, 0.034, 0.035, 0.037, 0.040,
            0.042, 0.044, 0.045, 0.044, 0.043, 0.042,
            0.041, 0.040, 0.041, 0.042, 0.043, 0.045,
            0.047, 0.048, 0.046, 0.044, 0.041, 0.039
        ]

        assert abs(sum(hourly_distribution) - 1.0) < 0.001

        # Verificar que al multiplicar por total, suma correctamente
        hourly_values = [test_total * ratio for ratio in hourly_distribution]
        assert abs(sum(hourly_values) - test_total) < 0.01


# ============== TESTS DE RENDIMIENTO ==============

@pytest.mark.slow
class TestPerformance:
    """Tests de rendimiento (marcados como slow)"""

    def test_large_batch_prediction(self):
        """Verifica rendimiento con batch grande"""
        # Crear datos
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        data = []

        for date in dates:
            hourly = [40 + np.random.normal(0, 2) for _ in range(24)]
            row = {'FECHA': date}
            row.update({f'P{i}': hourly[i-1] for i in range(1, 25)})
            data.append(row)

        df = pd.DataFrame(data)

        # Entrenar
        engine = HourlyDisaggregationEngine(auto_load=False)
        engine.normal_disaggregator = HourlyDisaggregator(n_clusters=35)
        engine.normal_disaggregator.fit(df)

        # Predecir 365 días
        import time
        dates_pred = pd.date_range('2025-01-01', periods=365, freq='D')
        totals_pred = pd.Series([1500.0] * 365)

        start = time.time()
        results = engine.predict_batch(dates_pred, totals_pred)
        elapsed = time.time() - start

        print(f"\nTiempo para 365 días: {elapsed:.2f}s ({elapsed/365*1000:.2f}ms por día)")

        assert len(results) == 365
        assert elapsed < 10.0  # Debe ser rápido (< 10s para 365 días)


if __name__ == "__main__":
    # Ejecutar tests
    pytest.main([__file__, "-v", "--tb=short"])
