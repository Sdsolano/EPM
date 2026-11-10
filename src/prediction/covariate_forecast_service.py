"""
Servicio aditivo para /forecast-covariate.

- Valida shapes estilo TimesFM-3
- Modo matrices crudas
- Modo conveniencia UCP: arma context + past/past-future desde raw data,
  clima y festivos (API + homology)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config.settings import MODELS_DIR, RAW_DATA_DIR
from src.prediction.covariate_forecaster import (
    CovariateForecastResult,
    DirectCovariateForecaster,
)
from src.prediction.holiday_homology import holiday_name_for_date

logger = logging.getLogger(__name__)


def _validate_matrix_mode(
    context: np.ndarray,
    horizon: int,
    past_only: Optional[np.ndarray],
    past_future: Optional[np.ndarray],
) -> None:
    if context.ndim != 2:
        raise ValueError(f"context debe ser (n_targets, T); shape={context.shape}")
    n_targets, T = context.shape
    if n_targets < 1 or T < 28:
        raise ValueError("context requiere al menos 1 target y T>=28")
    if horizon < 1:
        raise ValueError("horizon debe ser >= 1")
    if past_only is not None:
        if past_only.ndim != 2 or past_only.shape[1] != T:
            raise ValueError(
                f"past_only_covariates debe ser (n, T={T}); shape={past_only.shape}"
            )
    if past_future is not None:
        expected = T + horizon
        if past_future.ndim != 2 or past_future.shape[1] != expected:
            raise ValueError(
                f"past_future_covariates debe ser (n, T+horizon={expected}); "
                f"shape={past_future.shape}"
            )


def _load_festivos_set(ucp: str, start: str, end: str) -> set:
    try:
        from src.prediction.festivos_api import FestivosAPIClient

        return FestivosAPIClient().get_festivos_set(start, end, ucp)
    except Exception as e:
        logger.warning(f"API festivos no disponible ({e}); usando calendario CO local")
        festivos: set = set()
        y0 = int(start[:4])
        y1 = int(end[:4])
        for y in range(y0, y1 + 1):
            try:
                from src.prediction.holiday_homology import colombia_holiday_map

                festivos.update(colombia_holiday_map(y).keys())
            except Exception:
                continue
        return festivos


def _daily_temp_from_clima(path: Path) -> pd.DataFrame:
    """Agrega clima horario o diario a temp_mean por FECHA."""
    clima = pd.read_csv(path)
    # Formato clima_new: fecha, periodo, p_t, ...
    cols = {c.lower(): c for c in clima.columns}
    if "fecha" in cols and "p_t" in cols:
        fecha_col = cols["fecha"]
        temp_col = cols["p_t"]
        clima[fecha_col] = pd.to_datetime(clima[fecha_col])
        daily = (
            clima.groupby(clima[fecha_col].dt.normalize())[temp_col]
            .mean()
            .rename("temp_mean")
            .reset_index()
        )
        daily.columns = ["fecha", "temp_mean"]
        return daily
    if "fecha" in cols and "temp_mean" in cols:
        out = clima[[cols["fecha"], cols["temp_mean"]]].copy()
        out.columns = ["fecha", "temp_mean"]
        out["fecha"] = pd.to_datetime(out["fecha"]).dt.normalize()
        return out
    raise ValueError(f"Formato de clima no reconocido en {path}")


def build_ucp_matrices(
    ucp: str,
    fecha_inicio: str,
    fecha_fin: str,
    context_days: int = 365,
) -> Dict[str, Any]:
    """
    Arma matrices TimesFM-like desde datos locales del UCP.
    """
    raw_dir = RAW_DATA_DIR / ucp
    datos_path = raw_dir / "datos.csv"
    if not datos_path.exists():
        # Fallback incidente (desarrollo)
        alt = Path("incident/extracted/csvs") / ucp / "datos.csv"
        if alt.exists():
            datos_path = alt
            raw_dir = alt.parent
        else:
            raise FileNotFoundError(
                f"No se encontró datos.csv para UCP={ucp} en {RAW_DATA_DIR / ucp}"
            )

    df = pd.read_csv(datos_path)
    df["fecha"] = pd.to_datetime(df["FECHA"]).dt.normalize()
    df = df.sort_values("fecha").drop_duplicates("fecha")

    start = pd.Timestamp(fecha_inicio)
    end = pd.Timestamp(fecha_fin)
    if end < start:
        raise ValueError("fecha_fin debe ser >= fecha_inicio")
    horizon = (end - start).days + 1

    # Contexto: días previos a fecha_inicio
    hist = df[df["fecha"] < start].tail(context_days).copy()
    if len(hist) < 28:
        raise ValueError(
            f"Histórico insuficiente antes de {fecha_inicio}: {len(hist)} días"
        )

    context = hist[["TOTAL"]].to_numpy(dtype=np.float32).T  # (1, T)
    T = context.shape[1]
    hist_dates = hist["fecha"].tolist()

    # Clima
    clima_path = raw_dir / "clima_new.csv"
    temp_by_date: Dict[pd.Timestamp, float] = {}
    if clima_path.exists():
        try:
            daily_temp = _daily_temp_from_clima(clima_path)
            temp_by_date = {
                pd.Timestamp(r.fecha): float(r.temp_mean)
                for r in daily_temp.itertuples()
            }
        except Exception as e:
            logger.warning(f"No se pudo cargar clima: {e}")

    def temp_for(ts: pd.Timestamp) -> float:
        if ts in temp_by_date:
            return temp_by_date[ts]
        # fallback: media histórica
        if temp_by_date:
            return float(np.mean(list(temp_by_date.values())))
        return 0.0

    past_only = np.array(
        [[temp_for(ts) for ts in hist_dates]], dtype=np.float32
    )  # (1, T)

    # Festivos para hist + horizonte
    fest_start = (hist["fecha"].min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    fest_end = (end + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    festivos = _load_festivos_set(ucp, fest_start, fest_end)
    # Completar con calendario CO (homology) por si API incompleta
    try:
        from src.prediction.holiday_homology import colombia_holiday_map

        for y in range(start.year - 1, end.year + 1):
            festivos.update(colombia_holiday_map(y).keys())
    except Exception:
        pass

    all_dates = hist_dates + [
        start + pd.Timedelta(days=i) for i in range(horizon)
    ]
    def _is_fest(ts: pd.Timestamp) -> float:
        return (
            1.0
            if ts.strftime("%Y-%m-%d") in festivos or holiday_name_for_date(ts)
            else 0.0
        )

    # Índice fecha -> TOTAL para anclar nivel al mismo DOW reciente
    total_by_date = {
        pd.Timestamp(r.fecha): float(r.TOTAL) for r in df.itertuples()
    }

    def _same_dow_level(ts: pd.Timestamp, lookback: int = 90) -> float:
        """Ancla de nivel: mismo DOW hábil, o domingo/festivo si el día es festivo."""
        vals: List[float] = []
        target_is_fest = _is_fest(ts) > 0
        for back in range(1, lookback + 1):
            prev = ts - pd.Timedelta(days=back)
            if prev not in total_by_date:
                continue
            if target_is_fest:
                # Ancla tipo festivo: domingos o festivos previos
                if prev.dayofweek == 6 or _is_fest(prev) > 0:
                    vals.append(total_by_date[prev])
            else:
                if prev.dayofweek != ts.dayofweek:
                    continue
                if _is_fest(prev) > 0:
                    continue
                vals.append(total_by_date[prev])
            if len(vals) >= 4:
                break
        if vals:
            return float(np.mean(vals[:4]))
        return float(df["TOTAL"].tail(28).mean())

    def _pf_block(dates: List[pd.Timestamp]) -> np.ndarray:
        """Past-future covariates ricas (calendario + clima + ancla DOW)."""
        dow = np.array([float(ts.dayofweek) for ts in dates], dtype=np.float32)
        dow_sin = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
        dow_cos = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
        month = np.array([float(ts.month) for ts in dates], dtype=np.float32)
        month_sin = np.sin(2 * np.pi * month / 12.0).astype(np.float32)
        month_cos = np.cos(2 * np.pi * month / 12.0).astype(np.float32)
        is_festivo = np.array([_is_fest(ts) for ts in dates], dtype=np.float32)
        is_weekend = np.array(
            [1.0 if ts.dayofweek >= 5 else 0.0 for ts in dates], dtype=np.float32
        )
        is_monday = np.array(
            [1.0 if ts.dayofweek == 0 else 0.0 for ts in dates], dtype=np.float32
        )
        temp_all = np.array([temp_for(ts) for ts in dates], dtype=np.float32)
        same_dow = np.array(
            [_same_dow_level(ts) for ts in dates], dtype=np.float32
        )
        return np.vstack(
            [
                is_festivo,
                is_weekend,
                is_monday,
                dow,
                dow_sin,
                dow_cos,
                month_sin,
                month_cos,
                temp_all,
                same_dow,
            ]
        )

    past_future = _pf_block(all_dates)  # (9, T+H)

    # Histórico completo para fit (más largo)
    fit_hist = df.tail(max(context_days * 2, 500)).copy()
    fit_context = fit_hist[["TOTAL"]].to_numpy(dtype=np.float32).T
    fit_dates = fit_hist["fecha"].tolist()
    fit_temp = np.array([temp_for(ts) for ts in fit_dates], dtype=np.float32)
    fit_past = fit_temp[None, :]
    fit_pf = _pf_block(fit_dates)

    return {
        "context": context,
        "horizon": horizon,
        "past_only_covariates": past_only,
        "past_future_covariates": past_future,
        "target_names": ["TOTAL"],
        "dates": [ts.strftime("%Y-%m-%d") for ts in all_dates[T:]],
        "fit_context": fit_context,
        "fit_past_only": fit_past,
        "fit_past_future": fit_pf,
        "ucp": ucp,
        "T": T,
    }


class CovariateForecastService:
    """Orquesta fit/predict para el endpoint aditivo."""

    def __init__(self, models_root: Optional[Path] = None):
        self.models_root = Path(models_root) if models_root else MODELS_DIR

    def _forecaster_for(self, key: str) -> DirectCovariateForecaster:
        models_dir = self.models_root / key / "covariate_forecaster"
        return DirectCovariateForecaster(
            models_dir=models_dir, max_horizon=30, min_train_samples=120
        )

    def forecast_matrices(
        self,
        context: Sequence[Sequence[float]],
        horizon: int,
        past_only_covariates: Optional[Sequence[Sequence[float]]] = None,
        past_future_covariates: Optional[Sequence[Sequence[float]]] = None,
        return_quantiles: bool = True,
        target_names: Optional[List[str]] = None,
        force_retrain: bool = False,
        model_key: str = "_matrices",
    ) -> CovariateForecastResult:
        ctx = np.asarray(context, dtype=np.float32)
        past = (
            np.asarray(past_only_covariates, dtype=np.float32)
            if past_only_covariates is not None
            else None
        )
        pf = (
            np.asarray(past_future_covariates, dtype=np.float32)
            if past_future_covariates is not None
            else None
        )
        _validate_matrix_mode(ctx, horizon, past, pf)

        fc = self._forecaster_for(model_key)
        need_fit = force_retrain or not (fc.models_dir and (fc.models_dir / "covariate_forecaster.joblib").exists())
        if need_fit:
            # En modo matrices, el context enviado es el histórico de entrenamiento
            # y past_future debe cubrir al menos T (sin horizonte futuro de train).
            # Para fit usamos solo las columnas de contexto.
            pf_fit = pf[:, : ctx.shape[1]] if pf is not None else None
            fc.fit(
                context=ctx,
                past_only_covariates=past,
                past_future_covariates=pf_fit,
                target_names=target_names,
                max_horizon=min(horizon, 30),
            )
        else:
            fc.load(fc.models_dir)

        return fc.predict(
            context=ctx,
            horizon=horizon,
            past_only_covariates=past,
            past_future_covariates=pf,
            return_quantiles=return_quantiles,
            target_names=target_names,
        )

    def forecast_ucp(
        self,
        ucp: str,
        fecha_inicio: str,
        fecha_fin: str,
        return_quantiles: bool = True,
        force_retrain: bool = False,
        context_days: int = 365,
    ) -> Dict[str, Any]:
        payload = build_ucp_matrices(ucp, fecha_inicio, fecha_fin, context_days)
        fc = self._forecaster_for(ucp)
        model_path = fc.models_dir / "covariate_forecaster.joblib" if fc.models_dir else None
        need_fit = force_retrain or not (model_path and model_path.exists())

        if need_fit:
            logger.info(f"Entrenando DirectCovariateForecaster para UCP={ucp}")
            fc.fit(
                context=payload["fit_context"],
                past_only_covariates=payload["fit_past_only"],
                past_future_covariates=payload["fit_past_future"],
                target_names=payload["target_names"],
                max_horizon=min(max(payload["horizon"], 14), 30),
            )
        else:
            fc.load(fc.models_dir)

        result = fc.predict(
            context=payload["context"],
            horizon=payload["horizon"],
            past_only_covariates=payload["past_only_covariates"],
            past_future_covariates=payload["past_future_covariates"],
            return_quantiles=return_quantiles,
            target_names=payload["target_names"],
        )

        return {
            "ucp": ucp,
            "fecha_inicio": fecha_inicio,
            "fecha_fin": fecha_fin,
            "dates": payload["dates"],
            "target_names": result.target_names,
            "forecast": result.forecast.tolist(),
            "quantiles": result.quantiles.tolist() if result.quantiles is not None else None,
            "meta": result.meta,
        }
