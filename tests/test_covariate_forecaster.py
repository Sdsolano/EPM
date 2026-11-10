"""Tests del forecaster aditivo estilo TimesFM-3."""

from __future__ import annotations

import numpy as np
import pytest

from src.prediction.covariate_forecaster import DirectCovariateForecaster
from src.prediction.covariate_forecast_service import (
    CovariateForecastService,
    _validate_matrix_mode,
)


def _synthetic_series(n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    # Semanal + ruido; lunes (t%7==0) más altos salvo cuando festivo
    base = 1000 + 80 * np.sin(2 * np.pi * t / 7) + 40 * np.sin(2 * np.pi * t / 365)
    dow = t % 7
    weekday_boost = np.where(dow < 5, 120.0, -80.0)
    noise = rng.normal(0, 15, size=n)
    return (base + weekday_boost + noise).astype(np.float32)


class TestMatrixShapes:
    def test_validate_ok(self):
        ctx = np.zeros((2, 50), dtype=np.float32)
        past = np.zeros((1, 50), dtype=np.float32)
        pf = np.zeros((2, 50 + 7), dtype=np.float32)
        _validate_matrix_mode(ctx, 7, past, pf)

    def test_validate_pf_wrong_length(self):
        ctx = np.zeros((1, 50), dtype=np.float32)
        pf = np.zeros((1, 50), dtype=np.float32)  # falta horizon
        with pytest.raises(ValueError, match="T\\+horizon"):
            _validate_matrix_mode(ctx, 7, None, pf)

    def test_predict_shapes_match_timesfm_contract(self, tmp_path):
        n, T, H = 400, 200, 14
        series = _synthetic_series(n)
        # context de entrenamiento = serie completa; past_future = weekend proxy
        weekend = ((np.arange(n) % 7) >= 5).astype(np.float32)
        festivo = np.zeros(n, dtype=np.float32)
        # marcar algunos "festivos" en lunes
        for i in range(0, n, 28):
            festivo[i] = 1.0
            series[i] = series[i] * 0.85  # demanda baja tipo festivo

        fc = DirectCovariateForecaster(
            models_dir=tmp_path / "m",
            max_horizon=H,
            min_train_samples=80,
        )
        fc.fit(
            context=series[None, :],
            past_only_covariates=None,
            past_future_covariates=np.vstack([festivo, weekend]),
            target_names=["TOTAL"],
            max_horizon=H,
        )

        ctx = series[None, -T:]
        fest_h = np.zeros(T + H, dtype=np.float32)
        week_h = ((np.arange(T + H) % 7) >= 5).astype(np.float32)
        # día 1 del horizonte (offset 0) es "festivo"
        fest_h[T] = 1.0
        pf = np.vstack([fest_h, week_h])

        out = fc.predict(
            context=ctx,
            horizon=H,
            past_future_covariates=pf,
            return_quantiles=True,
        )
        assert out.forecast.shape == (1, H)
        assert out.quantiles is not None
        assert out.quantiles.shape == (1, H, 9)
        assert out.target_names == ["TOTAL"]


class TestFestivoCovariateEffect:
    def test_holiday_covariate_lowers_forecast_vs_workday_flag(self, tmp_path):
        n, T, H = 500, 180, 7
        series = _synthetic_series(n, seed=1)
        weekend = ((np.arange(n) % 7) >= 5).astype(np.float32)
        festivo = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if (i % 7 == 0) and (i % 28 == 0):
                festivo[i] = 1.0
                series[i] *= 0.82

        fc = DirectCovariateForecaster(
            models_dir=tmp_path / "fest",
            max_horizon=H,
            min_train_samples=80,
        )
        fc.fit(
            context=series[None, :],
            past_future_covariates=np.vstack([festivo, weekend]),
            max_horizon=H,
        )

        ctx = series[None, -T:]
        week_h = np.zeros(T + H, dtype=np.float32)  # horizonte "laboral"
        # Caso A: día 0 del horizonte marcado festivo
        pf_fest = np.vstack([np.zeros(T + H, dtype=np.float32), week_h])
        pf_fest[0, T] = 1.0
        # Caso B: mismo día sin festivo
        pf_work = np.vstack([np.zeros(T + H, dtype=np.float32), week_h])

        out_fest = fc.predict(ctx, H, past_future_covariates=pf_fest, return_quantiles=False)
        out_work = fc.predict(ctx, H, past_future_covariates=pf_work, return_quantiles=False)

        # El festivo conocido a futuro debe bajar (o no subir) el día 0 vs flag laboral
        assert out_fest.forecast[0, 0] <= out_work.forecast[0, 0] * 1.02


class TestServiceMatrices:
    def test_service_roundtrip(self, tmp_path):
        n, H = 350, 5
        series = _synthetic_series(n, seed=2)
        fest = np.zeros(n, dtype=np.float32)
        svc = CovariateForecastService(models_root=tmp_path)
        # Fit+predict via matrices (context = hist; pf length T+H)
        T = 120
        ctx = series[None, -T:].tolist()
        pf = np.zeros((1, T + H), dtype=np.float32)
        pf[0, T] = 1.0  # festivo día 1 horizonte
        # For training, service uses pf[:, :T]; ensure train signal exists by
        # fitting with force_retrain on extended series through forecast_matrices
        # which fits on the provided context only — use longer context for fit quality.
        result = svc.forecast_matrices(
            context=series[None, :].tolist(),
            horizon=H,
            past_future_covariates=np.zeros((1, n + H), dtype=np.float32).tolist(),
            return_quantiles=True,
            target_names=["TOTAL"],
            force_retrain=True,
            model_key="synth",
        )
        assert result.forecast.shape == (1, H)
        assert result.quantiles.shape == (1, H, 9)
