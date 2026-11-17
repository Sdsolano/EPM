#!/usr/bin/env python3
"""
Valida el fix del Modelo Base para el incidente 2026-11-02 (Todos los Santos)
y el comportamiento en días normales / festivos fijos.

Uso:
  .venv/bin/python scripts/validate_holiday_base_curve_fix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prediction.holiday_homology import (  # noqa: E402
    average_holiday_hourly_profile,
    collect_homologous_holiday_dates,
)

API = (
    "https://pronosticos.jmdatalabs.co/api/v1/admin/configuracion-interna"
    "/listarFestivos/{start}/{end}/{ucp}"
)


def fetch_festivos(ucp: str, start: str, end: str) -> set[str]:
    url = API.format(start=start, end=end, ucp=ucp)
    with urlopen(url, timeout=15) as resp:
        payload = json.load(resp)
    if not payload.get("success"):
        raise RuntimeError(payload)
    return {item["fecha"] for item in payload.get("data", [])}


def load_datos(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["fecha"] = pd.to_datetime(df["FECHA"])
    return df


def pct_err(pred: float, real: float) -> float:
    return (pred - real) / real * 100.0


def main() -> int:
    bolivar = ROOT / "incident/extracted/csvs/Bolivar/datos.csv"
    if not bolivar.exists():
        # fallback: unzip on the fly
        import zipfile

        zip_path = ROOT / "incident/csvs.zip"
        out = ROOT / "incident/extracted"
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out)
        bolivar = out / "csvs/Bolivar/datos.csv"

    df = load_datos(bolivar)
    festivos = fetch_festivos("Bolivar", "2018-01-01", "2026-12-31")

    print("=" * 72)
    print("VALIDACIÓN Modelo Base — festivos homólogos")
    print("=" * 72)
    print(f"UCP datos: Bolivar | festivos API: {len(festivos)}")

    # --- Incidente ---
    target = "2026-11-02"
    assert target in festivos, f"{target} debe ser festivo en API"
    homologous = collect_homologous_holiday_dates(target, festivos, n_years=5)
    print(f"\n[INCIDENTE] {target} Todos los Santos")
    print(f"  Homólogos: {homologous}")

    profile, meta = average_holiday_hourly_profile(df, target, festivos)
    pred_total = float(profile.sum())
    real_total = float(df.loc[df["fecha"] == target, "TOTAL"].iloc[0])

    # Bug legacy: promedio del 02-nov calendario
    legacy = df[
        (df["fecha"].dt.month == 11)
        & (df["fecha"].dt.day == 2)
        & (df["fecha"] < pd.Timestamp(target))
    ].sort_values("fecha", ascending=False).head(5)
    legacy_total = float(legacy["TOTAL"].mean())

    # Si se tratara como lunes normal (recientes)
    mondays = df[
        (df["fecha"] >= "2026-09-01")
        & (df["fecha"] < target)
        & (df["fecha"].dt.dayofweek == 0)
        & (~df["fecha"].dt.strftime("%Y-%m-%d").isin(festivos))
    ].sort_values("fecha", ascending=False).head(5)
    monday_total = float(mondays["TOTAL"].mean()) if len(mondays) else float("nan")

    print(f"  Real TOTAL:           {real_total:,.1f}")
    print(f"  Fix homólogo TOTAL:   {pred_total:,.1f}  ({pct_err(pred_total, real_total):+.1f}%)")
    print(f"  Legacy mes-día TOTAL: {legacy_total:,.1f}  ({pct_err(legacy_total, real_total):+.1f}%)")
    print(f"  Lunes hábil TOTAL:    {monday_total:,.1f}  ({pct_err(monday_total, real_total):+.1f}%)")
    print(f"  match_mode={meta['match_mode']} dates={meta['match_dates']}")

    # El fix debe: (1) usar homólogos Todos Santos, (2) no usar Raza/oct,
    # (3) error abs menor que tratarlo como lunes hábil, (4) error abs < 10%.
    bad_months = [d for d in meta["match_dates"] if not d[5:7] == "11"]
    err_fix = abs(pct_err(pred_total, real_total))
    err_monday = abs(pct_err(monday_total, real_total))
    incident_ok = (
        meta["match_mode"] == "homologous"
        and len(bad_months) == 0
        and err_fix < 10.0
        and err_fix <= err_monday
    )
    print(f"  growth={meta.get('growth_factor')} level={meta.get('level')}")
    print(f"  → incidente OK: {incident_ok}")

    # --- Día normal (martes 2026-11-03) ---
    normal = "2026-11-03"
    assert normal not in festivos
    normal_real = float(df.loc[df["fecha"] == normal, "TOTAL"].iloc[0])
    recent_tue = df[
        (df["fecha"] >= "2026-09-01")
        & (df["fecha"] < normal)
        & (df["fecha"].dt.dayofweek == 1)
        & (~df["fecha"].dt.strftime("%Y-%m-%d").isin(festivos))
    ].sort_values("fecha", ascending=False).head(5)
    tue_pred = float(recent_tue["TOTAL"].mean())
    print(f"\n[REGULAR] {normal} martes hábil (Método 2 — mismo DOW reciente)")
    print(f"  Real: {normal_real:,.1f} | Pred DOW: {tue_pred:,.1f} ({pct_err(tue_pred, normal_real):+.1f}%)")
    regular_ok = abs(pct_err(tue_pred, normal_real)) < 15.0
    print(f"  → regular OK (<15%): {regular_ok}")

    # --- Festivo fijo Navidad (si hay histórico; proyección) ---
    xmas = "2025-12-25"
    if xmas in festivos and (df["fecha"] == xmas).any():
        # Validar matching hacia 2025 usando años previos
        profile_x, meta_x = average_holiday_hourly_profile(df, xmas, festivos)
        real_x = float(df.loc[df["fecha"] == xmas, "TOTAL"].iloc[0])
        print(f"\n[FIJO] {xmas} Navidad")
        print(f"  Homólogos usados: {meta_x['match_dates']}")
        print(f"  Real: {real_x:,.1f} | Pred: {float(profile_x.sum()):,.1f} "
              f"({pct_err(float(profile_x.sum()), real_x):+.1f}%)")
        fixed_ok = meta_x["match_mode"] == "homologous" and all(
            d.endswith("-12-25") for d in meta_x["match_dates"]
        )
        print(f"  → festivo fijo OK: {fixed_ok}")
    else:
        # Matching estructural sin exigir fila 2025
        dates = collect_homologous_holiday_dates("2026-12-25", festivos, n_years=5)
        fixed_ok = all(d.endswith("-12-25") for d in dates) and len(dates) >= 2
        print(f"\n[FIJO] 2026-12-25 homólogos={dates} → OK: {fixed_ok}")

    # Domingo 2026-11-01 no es festivo oficial → no debe entrar a Método 1
    sunday = "2026-11-01"
    sunday_is_festivo = sunday in festivos
    print(f"\n[REGULAR] {sunday} domingo is_festivo={sunday_is_festivo} (esperado False)")
    sunday_ok = not sunday_is_festivo
    print(f"  → OK: {sunday_ok}")

    all_ok = incident_ok and regular_ok and fixed_ok and sunday_ok
    print("\n" + "=" * 72)
    print("RESULTADO:", "PASS" if all_ok else "FAIL")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
