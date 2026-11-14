"""
Genera `src/adjustments/precision_data/{ucp}_k1.csv`: historial compacto de pronósticos emitidos
(reentreno semanal del modelo, corte = sábado previo) que usa `insights.py` para precisión y banda.

Entrada (solo lectura): `reports/ajustes/curvas_modelo_k1_{ucp}.csv` con columnas
ucp,k,semana_inicio,corte,origen,fecha,P1..P24. Salida: fecha,semana_inicio,P1..P24 (1 decimal),
solo los días de la semana objetivo (lunes..domingo; se descarta el domingo de brecha).

Uso: python scripts/generar_historial_pronosticos.py [--origen DIR] [--ucps Antioquia Atlantico]
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "src" / "adjustments" / "precision_data"
PERIODOS = [f"P{i}" for i in range(1, 25)]


def convertir(origen_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(origen_csv)
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
    df["semana_inicio"] = pd.to_datetime(df["semana_inicio"]).dt.normalize()
    dentro = (df["fecha"] >= df["semana_inicio"]) & (df["fecha"] <= df["semana_inicio"] + timedelta(days=6))
    df = df[dentro].drop_duplicates(subset="fecha", keep="last").sort_values("fecha")
    out = df[["fecha", "semana_inicio", *PERIODOS]].copy()
    out["fecha"] = out["fecha"].dt.strftime("%Y-%m-%d")
    out["semana_inicio"] = out["semana_inicio"].dt.strftime("%Y-%m-%d")
    out[PERIODOS] = out[PERIODOS].round(1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", type=Path, default=RAIZ / "reports" / "ajustes")
    ap.add_argument("--ucps", nargs="+", default=["Antioquia", "Atlantico"])
    args = ap.parse_args()
    SALIDA.mkdir(parents=True, exist_ok=True)
    for ucp in args.ucps:
        out = convertir(args.origen / f"curvas_modelo_k1_{ucp}.csv")
        destino = SALIDA / f"{ucp}_k1.csv"
        out.to_csv(destino, index=False)
        print(f"{ucp}: {len(out)} días, {out['semana_inicio'].nunique()} semanas -> {destino}")


if __name__ == "__main__":
    main()
