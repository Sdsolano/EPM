"""
Emparejamiento de festivos homólogos entre años.

Problema: el Modelo Base promediaba el mismo mes-día histórico. Eso falla
para festivos móviles de la Ley Emiliani (p.ej. Todos los Santos:
2026-11-02 ↔ 2025-11-03 ↔ 2024-11-04).

Solución:
1. Identificar el festivo por nombre (calendario oficial Colombia).
2. Buscar la misma fiesta en años anteriores.
3. Fallbacks: proximidad en el mismo mes usando el set de festivos del UCP,
   y por último el mismo mes-día (legacy).
"""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

DateLike = Union[date, datetime, pd.Timestamp]


def _as_date(value: DateLike) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    return date.fromisoformat(str(value)[:10])


def _normalize_holiday_name(name: str) -> str:
    return name.replace(" (observed)", "").strip()


@lru_cache(maxsize=32)
def colombia_holiday_map(year: int) -> Dict[str, str]:
    """
    Mapa fecha ISO -> nombre normalizado de festivo oficial Colombia.

    Usa la librería `holidays` (Ley Emiliani + Semana Santa). Independiente
    del UCP, así el matching no se rompe si la API de un mercado está incompleta.
    """
    try:
        import holidays
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Se requiere el paquete 'holidays' para emparejar festivos homólogos"
        ) from exc

    co = holidays.country_holidays("CO", years=year)
    return {
        d.isoformat(): _normalize_holiday_name(name)
        for d, name in co.items()
    }


def holiday_name_for_date(target: DateLike) -> Optional[str]:
    """Nombre normalizado del festivo en `target`, o None si no es festivo."""
    target_d = _as_date(target)
    return colombia_holiday_map(target_d.year).get(target_d.isoformat())


def find_homologous_holiday_date(
    target: DateLike,
    year: int,
    festivos: Iterable[str],
    max_doy_distance: int = 40,
) -> Optional[str]:
    """
    Festivo de `year` homólogo a `target`.

    Preferencia:
    1. Misma fiesta por nombre (calendario Colombia).
    2. Festivo del set UCP en el mismo mes, más cercano en día-del-año
       (con preferencia por el mismo DOW).
    """
    target_d = _as_date(target)
    if year >= target_d.year:
        return None

    target_name = holiday_name_for_date(target_d)
    year_map = colombia_holiday_map(year)

    # 1) Match por identidad de fiesta (robusto a API incompleta del UCP)
    if target_name:
        for fecha_iso, name in year_map.items():
            if name == target_name:
                return fecha_iso

    # 2) Fallback: proximidad restringida al mismo mes sobre festivos del UCP
    festivos_set: Set[str] = {str(f)[:10] for f in festivos}
    anchor_doy = date(year, target_d.month, min(target_d.day, 28)).timetuple().tm_yday
    # Ajuste fino del ancla al día real cuando existe
    try:
        anchor_doy = date(year, target_d.month, target_d.day).timetuple().tm_yday
    except ValueError:
        pass

    best: Optional[str] = None
    best_key: Optional[tuple] = None
    target_dow = target_d.weekday()

    for festivo in festivos_set:
        try:
            candidate = date.fromisoformat(festivo)
        except ValueError:
            continue
        if candidate.year != year or candidate.month != target_d.month:
            continue
        dist = abs(candidate.timetuple().tm_yday - anchor_doy)
        if dist > max_doy_distance:
            continue
        dow_mismatch = 0 if candidate.weekday() == target_dow else 1
        key = (dist, dow_mismatch)
        if best_key is None or key < best_key:
            best = candidate.isoformat()
            best_key = key

    return best


def collect_homologous_holiday_dates(
    target: DateLike,
    festivos: Iterable[str],
    n_years: int = 5,
    max_doy_distance: int = 40,
) -> List[str]:
    """Festivos homólogos en los `n_years` años previos (más reciente primero)."""
    target_d = _as_date(target)
    festivos_set: Set[str] = {str(f)[:10] for f in festivos}
    found: List[str] = []

    for year in range(target_d.year - 1, target_d.year - n_years - 1, -1):
        match = find_homologous_holiday_date(
            target_d, year, festivos_set, max_doy_distance=max_doy_distance
        )
        if match is not None:
            found.append(match)

    return found


def select_holiday_historical_frame(
    df_historico: pd.DataFrame,
    target: DateLike,
    festivos: Iterable[str],
    date_column: str = "fecha",
    n_years: int = 5,
    max_doy_distance: int = 40,
) -> pd.DataFrame:
    """
    Filtra filas históricas de festivos homólogos.

    Si no hay homólogos con datos, cae al promedio del mismo mes-día (legacy).
    """
    target_d = _as_date(target)
    df = df_historico.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    homologous = collect_homologous_holiday_dates(
        target_d, festivos, n_years=n_years, max_doy_distance=max_doy_distance
    )

    if homologous:
        homologous_ts = pd.to_datetime(homologous)
        selected = df[
            df[date_column].dt.normalize().isin(homologous_ts.normalize())
        ].copy()
        if len(selected) > 0:
            selected = selected.sort_values(date_column, ascending=False)
            used = [
                ts.strftime("%Y-%m-%d")
                for ts in selected[date_column].dt.normalize().unique()
            ]
            selected.attrs["holiday_match_mode"] = "homologous"
            selected.attrs["holiday_match_dates"] = used
            selected.attrs["holiday_match_candidates"] = homologous
            selected.attrs["holiday_name"] = holiday_name_for_date(target_d)
            return selected

    legacy = df[
        (df[date_column].dt.month == target_d.month)
        & (df[date_column].dt.day == target_d.day)
        & (df[date_column] < pd.Timestamp(target_d))
    ].copy()
    legacy = legacy.sort_values(date_column, ascending=False).head(n_years)
    legacy.attrs["holiday_match_mode"] = "same_month_day"
    legacy.attrs["holiday_match_dates"] = [
        ts.strftime("%Y-%m-%d") for ts in legacy[date_column]
    ]
    legacy.attrs["holiday_name"] = holiday_name_for_date(target_d)
    return legacy


def _series_total(df: pd.DataFrame, period_cols: Sequence[str]) -> pd.Series:
    if "TOTAL" in df.columns:
        return df["TOTAL"].astype(float)
    return df[list(period_cols)].sum(axis=1).astype(float)


def _recent_sunday_level(
    df: pd.DataFrame,
    before: date,
    date_column: str,
    period_cols: Sequence[str],
    lookback_days: int = 28,
) -> Optional[float]:
    start = pd.Timestamp(before) - pd.Timedelta(days=lookback_days)
    end = pd.Timestamp(before)
    window = df[
        (df[date_column] >= start)
        & (df[date_column] < end)
        & (df[date_column].dt.dayofweek == 6)
    ]
    if len(window) == 0:
        return None
    return float(_series_total(window, period_cols).mean())


def average_holiday_hourly_profile(
    df_historico: pd.DataFrame,
    target: DateLike,
    festivos: Iterable[str],
    period_cols: Optional[Sequence[str]] = None,
    date_column: str = "fecha",
    n_years: int = 5,
    max_doy_distance: int = 40,
    scale_with_recent_sundays: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Perfil horario para un día festivo.

    - Forma: promedio de perfiles homólogos normalizados.
    - Nivel: promedio ponderado (más peso a años recientes) de los totales
      homólogos, ajustado por crecimiento vs domingos recientes.
    """
    if period_cols is None:
        period_cols = [f"P{i}" for i in range(1, 25)]
    period_cols = list(period_cols)

    selected = select_holiday_historical_frame(
        df_historico,
        target,
        festivos,
        date_column=date_column,
        n_years=n_years,
        max_doy_distance=max_doy_distance,
    )

    meta = {
        "match_mode": selected.attrs.get("holiday_match_mode", "none"),
        "match_dates": list(selected.attrs.get("holiday_match_dates", [])),
        "holiday_name": selected.attrs.get("holiday_name"),
        "n_rows": int(len(selected)),
        "growth_factor": 1.0,
        "level": 0.0,
    }

    if len(selected) == 0:
        return np.zeros(24, dtype=float), meta

    totals = _series_total(selected, period_cols).replace(0, np.nan)
    normalized = selected[period_cols].div(totals, axis=0)
    shape = normalized.mean().to_numpy(dtype=float)

    # Pesos exponenciales: más peso al homólogo más reciente
    n = len(selected)
    weights = np.array([0.5 ** i for i in range(n)], dtype=float)
    weights /= weights.sum()
    hist_level = float(np.nansum(totals.to_numpy(dtype=float) * weights))

    growth = 1.0
    level = hist_level
    if scale_with_recent_sundays:
        target_d = _as_date(target)
        df = df_historico.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        recent_sun = _recent_sunday_level(
            df, target_d, date_column, period_cols, lookback_days=28
        )
        # Ratio festivo/domingo emparejado por año (evita sesgo de promedios)
        ratios: List[float] = []
        ratio_weights: List[float] = []
        for weight, (_, row) in zip(weights, selected.iterrows()):
            sun = _recent_sunday_level(
                df, _as_date(row[date_column]), date_column, period_cols, lookback_days=14
            )
            row_total = float(row["TOTAL"]) if "TOTAL" in selected.columns else float(
                row[period_cols].sum()
            )
            if sun is not None and sun > 0 and row_total > 0:
                ratios.append(row_total / sun)
                ratio_weights.append(float(weight))
        if recent_sun and ratios:
            rw = np.array(ratio_weights, dtype=float)
            rw /= rw.sum()
            holiday_to_sunday = float(np.dot(np.array(ratios, dtype=float), rw))
            level = float(recent_sun * holiday_to_sunday)
            growth = float(level / hist_level) if hist_level else 1.0

    meta["growth_factor"] = growth
    meta["level"] = level

    profile = shape * level
    if profile.shape[0] < 24:
        profile = np.pad(profile, (0, 24 - profile.shape[0]))
    return profile[:24], meta
