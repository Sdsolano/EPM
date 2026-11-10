"""
DirectCovariateForecaster — réplica local de la lógica TimesFM-3 (mejorada).

Contrato de predict():
  context:                (n_targets, T)
  past_only_covariates:   (n_past_cov, T)           opcional
  past_future_covariates: (n_pf_cov, T + horizon)   opcional
  return_quantiles:       bool

  forecast:  (n_targets, horizon)
  quantiles: (n_targets, horizon, 9)  p10..p90

Mejoras vs v1:
- Un modelo por target (horizon_offset como feature) → más datos, mejor DOW
- Contexto rico: últimos 7 días reales, lag_7/14, rolling, nivel reciente
- LightGBM más capacioso; punto L2 (no forzar mediana quantile)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np

logger = logging.getLogger(__name__)

QUANTILE_LEVELS: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MAX_HORIZON = 60
# Índice de la covariate past-future "same_dow_level" cuando el servicio
# arma el bloque estándar (ver covariate_forecast_service._pf_block).
SAME_DOW_LEVEL_INDEX = 9


@dataclass
class CovariateForecastResult:
    forecast: np.ndarray
    quantiles: Optional[np.ndarray] = None
    target_names: Optional[List[str]] = None
    meta: Dict = field(default_factory=dict)


def _as_2d(arr: Optional[np.ndarray], name: str) -> Optional[np.ndarray]:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError(f"{name} debe ser 2D (n, T); recibido shape={a.shape}")
    return a


def _pad_left(values: np.ndarray, n: int) -> np.ndarray:
    if len(values) >= n:
        return values[-n:].astype(np.float32)
    pad = np.full(n - len(values), values[0] if len(values) else 0.0, dtype=np.float32)
    return np.concatenate([pad, values.astype(np.float32)])


def _context_features(series: np.ndarray) -> np.ndarray:
    """Features solo del histórico real (sin predicciones del horizonte)."""
    s = np.asarray(series, dtype=np.float64)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return np.zeros(18, dtype=np.float32)

    def win(n: int) -> np.ndarray:
        return s[-n:] if len(s) >= n else s

    w7, w14, w28 = win(7), win(14), win(28)
    last7 = _pad_left(s, 7)
    lag7 = float(s[-8]) if len(s) >= 8 else float(s[0])
    lag14 = float(s[-15]) if len(s) >= 15 else float(s[0])
    mean7 = float(np.mean(w7))
    mean28 = float(np.mean(w28))

    feats = [
        float(s[-1]),
        lag7,
        lag14,
        mean7,
        float(np.std(w7) if len(w7) > 1 else 0.0),
        float(np.mean(w14)),
        float(np.std(w14) if len(w14) > 1 else 0.0),
        mean28,
        float(np.std(w28) if len(w28) > 1 else 0.0),
        float(np.min(w28)),
        float(np.max(w28)),
        float(s[-1] / mean7) if mean7 else 1.0,
        float(mean7 / mean28) if mean28 else 1.0,
        float(np.mean(s)),
    ]
    return np.concatenate([np.asarray(feats, dtype=np.float32), last7])


def _build_row(
    context_series: np.ndarray,
    past_only_t: Optional[np.ndarray],
    past_future_day: Optional[np.ndarray],
    horizon_offset: int,
) -> np.ndarray:
    parts: List[np.ndarray] = [
        _context_features(context_series),
        np.asarray([float(horizon_offset), float(np.log1p(horizon_offset))], dtype=np.float32),
    ]
    if past_only_t is not None:
        parts.append(np.asarray(past_only_t, dtype=np.float32).ravel())
    if past_future_day is not None:
        parts.append(np.asarray(past_future_day, dtype=np.float32).ravel())
    return np.concatenate(parts)


def _lgbm_point(random_state: int):
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=random_state,
        verbose=-1,
    )


def _lgbm_quantile(alpha: float, random_state: int):
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=47,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=random_state,
        verbose=-1,
    )


class DirectCovariateForecaster:
    """Forecaster aditivo estilo TimesFM-3 (horizonte directo + covariates)."""

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        max_horizon: int = 30,
        min_train_samples: int = 120,
        random_state: int = 42,
    ):
        self.models_dir = Path(models_dir) if models_dir else None
        self.max_horizon = min(max_horizon, MAX_HORIZON)
        self.min_train_samples = min_train_samples
        self.random_state = random_state
        # models[target_idx] = {'point': model, 'q0.1': ..., 'fallback_mean': bool}
        self._models: Dict[int, Dict[str, object]] = {}
        self._is_fitted = False
        self.target_names: List[str] = []

    def fit(
        self,
        context: np.ndarray,
        past_only_covariates: Optional[np.ndarray] = None,
        past_future_covariates: Optional[np.ndarray] = None,
        target_names: Optional[Sequence[str]] = None,
        max_horizon: Optional[int] = None,
    ) -> "DirectCovariateForecaster":
        try:
            import lightgbm as lgb  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("lightgbm es requerido para DirectCovariateForecaster") from exc

        ctx = _as_2d(context, "context")
        assert ctx is not None
        past = _as_2d(past_only_covariates, "past_only_covariates")
        pf = _as_2d(past_future_covariates, "past_future_covariates")

        n_targets, T = ctx.shape
        if past is not None and past.shape[1] != T:
            raise ValueError(f"past_only_covariates T={past.shape[1]} != context T={T}")
        if pf is not None and pf.shape[1] < T:
            raise ValueError(f"past_future_covariates T={pf.shape[1]} < context T={T}")

        H = min(max_horizon or self.max_horizon, MAX_HORIZON, max(1, T // 3))
        self.max_horizon = H
        self.target_names = list(target_names) if target_names else [
            f"target_{i}" for i in range(n_targets)
        ]

        self._models = {}
        min_ctx = 28

        for ti in range(n_targets):
            series = ctx[ti]
            X_rows: List[np.ndarray] = []
            y_rows: List[float] = []

            # Un solo pool: todos los (t, h) → modelo único por target (y absoluto).
            for t in range(min_ctx - 1, T - 1):
                max_h = min(H, T - t - 1)
                for h in range(1, max_h + 1):
                    y = float(series[t + h])
                    if not np.isfinite(y):
                        continue
                    past_t = past[:, t] if past is not None else None
                    pf_day = pf[:, t + h] if pf is not None else None
                    X_rows.append(_build_row(series[: t + 1], past_t, pf_day, h))
                    y_rows.append(y)

            if len(y_rows) < self.min_train_samples:
                logger.warning(
                    f"target={ti}: solo {len(y_rows)} samples; fallback media"
                )
                self._models[ti] = {"fallback_mean": True}
                continue

            X = np.vstack(X_rows)
            y = np.asarray(y_rows, dtype=np.float32)
            bucket: Dict[str, object] = {
                "fallback_mean": False,
                "anchor_blend": 0.55,  # mezcla con same_dow_level en predict
            }

            point = _lgbm_point(self.random_state)
            point.fit(X, y)
            bucket["point"] = point

            for q in QUANTILE_LEVELS:
                qmodel = _lgbm_quantile(q, self.random_state)
                qmodel.fit(X, y)
                bucket[f"q{q}"] = qmodel

            self._models[ti] = bucket
            logger.info(f"✓ target={ti} entrenado con {len(y_rows)} filas (H<={H})")

        self._is_fitted = True
        if self.models_dir is not None:
            self.save(self.models_dir)
        return self

    def predict(
        self,
        context: np.ndarray,
        horizon: int,
        past_only_covariates: Optional[np.ndarray] = None,
        past_future_covariates: Optional[np.ndarray] = None,
        return_quantiles: bool = False,
        target_names: Optional[Sequence[str]] = None,
    ) -> CovariateForecastResult:
        if not self._is_fitted and not self._try_load():
            raise RuntimeError(
                "DirectCovariateForecaster no está entrenado. Llama fit() primero."
            )

        ctx = _as_2d(context, "context")
        assert ctx is not None
        past = _as_2d(past_only_covariates, "past_only_covariates")
        pf = _as_2d(past_future_covariates, "past_future_covariates")

        n_targets, T = ctx.shape
        if horizon < 1:
            raise ValueError("horizon debe ser >= 1")
        if past is not None and past.shape[1] != T:
            raise ValueError(f"past_only_covariates T={past.shape[1]} != context T={T}")
        if pf is not None and pf.shape[1] != T + horizon:
            raise ValueError(
                f"past_future_covariates debe ser (n, T+horizon={T + horizon}); "
                f"recibido {pf.shape}"
            )

        names = list(target_names) if target_names else (
            self.target_names if self.target_names else [f"target_{i}" for i in range(n_targets)]
        )

        forecast = np.zeros((n_targets, horizon), dtype=np.float32)
        quantiles = (
            np.zeros((n_targets, horizon, len(QUANTILE_LEVELS)), dtype=np.float32)
            if return_quantiles
            else None
        )

        for ti in range(n_targets):
            series = ctx[ti]
            bucket = self._models.get(ti, {"fallback_mean": True})
            for step in range(horizon):
                h = step + 1
                model_h = min(h, self.max_horizon)
                past_t = past[:, -1] if past is not None else None
                pf_day = pf[:, T + step] if pf is not None else None
                x = _build_row(series, past_t, pf_day, model_h).reshape(1, -1)

                anchor = 0.0
                if (
                    pf_day is not None
                    and len(pf_day) > SAME_DOW_LEVEL_INDEX
                    and float(pf_day[SAME_DOW_LEVEL_INDEX]) > 1.0
                ):
                    anchor = float(pf_day[SAME_DOW_LEVEL_INDEX])
                is_fest = (
                    float(pf_day[0]) > 0.5
                    if pf_day is not None and len(pf_day) > 0
                    else False
                )
                blend = float(bucket.get("anchor_blend", 0.55))
                # En festivo confiar más en el modelo (aprendió el flag); en hábil anclar nivel
                if is_fest:
                    blend = min(blend, 0.25)

                if bucket.get("fallback_mean"):
                    pred = anchor if anchor > 0 else (
                        float(np.nanmean(series[-28:])) if len(series) else 0.0
                    )
                else:
                    raw = float(bucket["point"].predict(x)[0])  # type: ignore[index]
                    pred = (1.0 - blend) * raw + blend * anchor if anchor > 0 else raw

                forecast[ti, step] = pred

                if quantiles is not None:
                    for qi, q in enumerate(QUANTILE_LEVELS):
                        qmodel = bucket.get(f"q{q}")
                        if qmodel is None:
                            quantiles[ti, step, qi] = pred
                        else:
                            qraw = float(qmodel.predict(x)[0])
                            quantiles[ti, step, qi] = (
                                (1.0 - blend) * qraw + blend * anchor
                                if anchor > 0
                                else qraw
                            )

        return CovariateForecastResult(
            forecast=forecast,
            quantiles=quantiles,
            target_names=names,
            meta={
                "n_targets": n_targets,
                "horizon": horizon,
                "context_length": T,
                "max_horizon_models": self.max_horizon,
                "engine": "direct_covariate_v2",
            },
        )

    def save(self, models_dir: Path) -> None:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "models": self._models,
            "max_horizon": self.max_horizon,
            "target_names": self.target_names,
            "is_fitted": self._is_fitted,
            "random_state": self.random_state,
        }
        path = models_dir / "covariate_forecaster.joblib"
        joblib.dump(payload, path)
        logger.info(f"✓ DirectCovariateForecaster v2 guardado en {path}")

    def load(self, models_dir: Path) -> "DirectCovariateForecaster":
        path = Path(models_dir) / "covariate_forecaster.joblib"
        payload = joblib.load(path)
        self._models = payload["models"]
        self.max_horizon = payload["max_horizon"]
        self.target_names = payload.get("target_names", [])
        self._is_fitted = payload.get("is_fitted", True)
        self.models_dir = Path(models_dir)
        return self

    def _try_load(self) -> bool:
        if self.models_dir is None:
            return False
        path = Path(self.models_dir) / "covariate_forecaster.joblib"
        if not path.exists():
            return False
        self.load(self.models_dir)
        return True
