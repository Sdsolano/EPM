"""Tests del emparejamiento de festivos homólogos (Modelo Base)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.prediction.holiday_homology import (
    average_holiday_hourly_profile,
    collect_homologous_holiday_dates,
    find_homologous_holiday_date,
    select_holiday_historical_frame,
)

# Festivos reales (API) relevantes para los casos de prueba
FESTIVOS = {
    # Todos los Santos (Emiliani — primer lunes de noviembre)
    "2020-11-02",
    "2021-11-01",
    "2022-11-07",
    "2023-11-06",
    "2024-11-04",
    "2025-11-03",
    "2026-11-02",
    # Independencia de Cartagena (también Emiliani, noviembre)
    "2024-11-11",
    "2025-11-17",
    "2026-11-16",
    # Fijos
    "2024-12-25",
    "2025-12-25",
    "2026-12-25",
    "2024-05-01",
    "2025-05-01",
    "2026-05-01",
    "2024-01-01",
    "2025-01-01",
    "2026-01-01",
    # Semana Santa (móvil)
    "2025-04-17",  # Jueves Santo 2025
    "2025-04-18",  # Viernes Santo 2025
    "2026-04-02",  # Jueves Santo 2026
    "2026-04-03",  # Viernes Santo 2026
}


class TestHomologousMatching:
    def test_todos_santos_2026_maps_to_prior_mondays(self):
        dates = collect_homologous_holiday_dates("2026-11-02", FESTIVOS, n_years=5)
        assert dates[0] == "2025-11-03"
        assert dates[1] == "2024-11-04"
        assert dates[2] == "2023-11-06"
        assert "2025-11-02" not in dates  # domingo, no era el festivo
        assert "2024-11-02" not in dates  # sábado

    def test_incomplete_ucp_festivos_still_finds_todos_santos(self):
        """API incompleta (caso Bolívar): sin 2024/2025 Nov, no confundir con Raza."""
        sparse = {
            "2020-11-02",
            "2023-11-06",
            "2022-10-17",  # Día de la Raza — debe ignorarse
            "2021-10-18",
            "2026-11-02",
        }
        dates = collect_homologous_holiday_dates("2026-11-02", sparse, n_years=5)
        assert dates[0] == "2025-11-03"
        assert dates[1] == "2024-11-04"
        assert "2022-10-17" not in dates
        assert "2021-10-18" not in dates

    def test_fixed_christmas_keeps_same_month_day(self):
        dates = collect_homologous_holiday_dates("2026-12-25", FESTIVOS, n_years=3)
        assert dates[:2] == ["2025-12-25", "2024-12-25"]
        assert all(d.endswith("-12-25") for d in dates)

    def test_fixed_labour_day(self):
        match = find_homologous_holiday_date("2026-05-01", 2025, FESTIVOS)
        assert match == "2025-05-01"

    def test_easter_thursday_maps_across_years(self):
        match = find_homologous_holiday_date("2026-04-02", 2025, FESTIVOS)
        assert match == "2025-04-17"

    def test_cartagena_not_confused_with_todos_santos(self):
        match = find_homologous_holiday_date("2026-11-16", 2025, FESTIVOS)
        assert match == "2025-11-17"
        match_ts = find_homologous_holiday_date("2026-11-02", 2025, FESTIVOS)
        assert match_ts == "2025-11-03"

    def test_ignores_future_years(self):
        assert find_homologous_holiday_date("2026-11-02", 2026, FESTIVOS) is None
        assert find_homologous_holiday_date("2026-11-02", 2027, FESTIVOS) is None


def _synthetic_history() -> pd.DataFrame:
    """Histórico sintético: festivos bajos, lunes hábiles altos, 02-nov mixtos."""
    rows = []
    period_cols = [f"P{i}" for i in range(1, 25)]

    def add(fecha: str, total: float):
        hourly = [total / 24.0] * 24
        row = {"fecha": fecha, "TOTAL": total}
        row.update({c: v for c, v in zip(period_cols, hourly)})
        rows.append(row)

    # Todos los Santos históricos (demanda baja tipo domingo)
    for d, total in [
        ("2022-11-07", 7500),
        ("2023-11-06", 8800),
        ("2024-11-04", 9300),
        ("2025-11-03", 11200),
    ]:
        add(d, total)

    # Mismo mes-día 02-nov en años previos (NO son Todos los Santos)
    for d, total in [
        ("2022-11-02", 10800),  # miércoles hábil
        ("2023-11-02", 8400),   # jueves
        ("2024-11-02", 9300),   # sábado
        ("2025-11-02", 11000),  # domingo
    ]:
        add(d, total)

    # Lunes hábiles recientes (altos) — lo que usaría el método "día normal"
    for d, total in [
        ("2026-10-05", 13800),
        ("2026-10-19", 13200),
        ("2026-10-26", 12200),
    ]:
        add(d, total)

    # Navidad
    add("2024-12-25", 7000)
    add("2025-12-25", 7200)

    # Día hábil normal en diciembre
    add("2025-12-18", 12000)

    return pd.DataFrame(rows)


class TestHolidayProfileSelection:
    def test_incident_prefers_homologous_not_same_month_day(self):
        df = _synthetic_history()
        selected = select_holiday_historical_frame(df, "2026-11-02", FESTIVOS)
        dates = set(selected["fecha"].dt.strftime("%Y-%m-%d"))
        assert "2025-11-03" in dates
        assert "2024-11-04" in dates
        assert "2025-11-02" not in dates
        assert selected.attrs["holiday_match_mode"] == "homologous"

        profile, meta = average_holiday_hourly_profile(
            df, "2026-11-02", FESTIVOS, scale_with_recent_sundays=False
        )
        assert meta["match_mode"] == "homologous"
        total = float(profile.sum())
        # Sin escalado: promedio ponderado de homólogos (más peso a 11200)
        assert 9000 < total < 11500

    def test_legacy_same_month_day_is_worse_for_incident(self):
        """El bug original promediaba 02-nov, no los Todos los Santos."""
        df = _synthetic_history()
        legacy = df[
            (df["fecha"].astype(str).str.endswith("-11-02"))
            & (df["fecha"].astype(str) < "2026-11-02")
        ]
        legacy_total = legacy["TOTAL"].mean()
        _, meta = average_holiday_hourly_profile(df, "2026-11-02", FESTIVOS)
        homologous_dates = meta["match_dates"]
        homologous_total = df[
            df["fecha"].astype(str).isin(homologous_dates)
        ]["TOTAL"].mean()
        # Homólogos deben estar más cerca del nivel festivo bajo que el promedio 02-nov
        assert homologous_total < legacy_total or set(homologous_dates) != set(
            legacy["fecha"].astype(str)
        )

    def test_christmas_profile_uses_prior_christmases(self):
        df = _synthetic_history()
        profile, meta = average_holiday_hourly_profile(
            df, "2026-12-25", FESTIVOS, scale_with_recent_sundays=False
        )
        assert meta["match_mode"] == "homologous"
        assert set(meta["match_dates"]) == {"2025-12-25", "2024-12-25"}
        # Pesos 2/3 y 1/3 sobre 7200 y 7000
        assert abs(float(profile.sum()) - (7200 * 2 / 3 + 7000 / 3)) < 1.0
        assert meta["holiday_name"] == "Christmas Day"

    def test_empty_history_returns_zeros(self):
        df = _synthetic_history().iloc[0:0]
        profile, meta = average_holiday_hourly_profile(df, "2026-11-02", FESTIVOS)
        assert meta["n_rows"] == 0
        assert np.allclose(profile, 0.0)
