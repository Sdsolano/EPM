"""Exporta emisiones ex-ante (v1 de cada semana) y real de una UCP para el panel P1 'Lo que aprendio'.

SOLO LECTURA de la BD Jano. Credenciales: variables PGHOST/PGDATABASE/PGUSER/PGPASSWORD/PGPORT del entorno
(nunca en el codigo). Salidas (ignoradas por git, son datos internos reales):
  <feedback>/emisiones_exante.csv, <feedback>/real_hasta.csv, <retro>/<UCP>.json

REGLA EX-ANTE CANONICA (la que debe igualar Sphere_Backend `emisionesRetro`): SOLO filas tipo 'P' de
sesiones_periodos; la decision es POR FECHA (no por semana): para cada fecha, la sesion de menor `codsesion`
guardada >= 3 dias antes de esa fecha (MIN_LEAD). Las filas 'PM' no cuentan.
Uso: python scripts/feedback_import_emisiones.py [--ucp Antioquia] [--desde 2026-01-12] [--hasta 2026-11-19]
`--hasta` solo acota la validacion de cifras (el export incluye todo lo disponible).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adjustments.data import festivos_colombia  # noqa: E402

P = [f"p{i}" for i in range(1, 25)]
FB = ROOT / "reports/ajustes/jamer/feedback"
RETRO = ROOT / "reports/ajustes/jamer/demo_datos/retro"
MIN_LEAD = 3


def _conectar():
    import psycopg2

    con = psycopg2.connect(host=os.environ["PGHOST"], dbname=os.environ["PGDATABASE"], user=os.environ["PGUSER"],
                           password=os.environ["PGPASSWORD"], port=os.environ.get("PGPORT", "5432"),
                           options="-c default_transaction_read_only=on")
    con.set_session(readonly=True)
    return con


def extraer(ucp: str, desde: str):
    con = _conectar()
    cols = ", ".join(f"sp.{p}" for p in P)
    emi = pd.read_sql(
        f"select sp.codsesion, s.version, s.fecha as guardado, to_char(sp.fecha,'YYYY-MM-DD') fecha, {cols} "
        f"from sesiones_periodos sp join sesiones s on s.codigo=sp.codsesion "
        f"where lower(s.ucp)=lower(%(u)s) and sp.tipo='P' and sp.fecha>=%(d)s", con, params={"u": ucp, "d": desde})
    real = pd.read_sql(
        f"select codigo, to_char(fecha,'YYYY-MM-DD') fecha, {', '.join(P)} from actualizaciondatos "
        f"where lower(ucp)=lower(%(u)s) and fecha>=%(d)s order by fecha, codigo", con, params={"u": ucp, "d": desde})
    con.close()
    return emi, real


def exante(emi: pd.DataFrame) -> pd.DataFrame:
    emi = emi.copy()
    emi["fecha"] = pd.to_datetime(emi["fecha"])
    emi["guardado"] = pd.to_datetime(emi["guardado"])
    emi["lead"] = (emi["fecha"] - emi["guardado"].dt.normalize()).dt.days
    ok = emi[emi["lead"] >= MIN_LEAD].sort_values(["fecha", "codsesion"])
    return ok.groupby("fecha").first().reset_index()


def construir(ucp: str, desde: str):
    emi, real = extraer(ucp, desde)
    ex = exante(emi)
    real = real.drop_duplicates("fecha", keep="last").dropna(subset=P)
    real["fecha"] = pd.to_datetime(real["fecha"])
    return ex, real


def validar(ex: pd.DataFrame, real: pd.DataFrame, desde: str, hasta: str) -> dict:
    """Reproduce las cifras de FEEDBACK (sesgo global, MAPE diario, WAPE horario)."""
    r = real.set_index("fecha")
    e = ex.set_index("fecha")
    idx = e.index.intersection(r.index)
    idx = idx[(idx >= pd.Timestamp(desde)) & (idx <= pd.Timestamp(hasta))]
    E, R = e.loc[idx, P].to_numpy(float), r.loc[idx, P].to_numpy(float)
    de, dr = E.sum(1), R.sum(1)
    return {"n_dias": int(len(idx)), "semanas": int(len({d - pd.Timedelta(days=d.dayofweek) for d in idx})),
            "sesgo_pct": float(np.mean(de / dr - 1) * 100), "mape_dia_pct": float(np.mean(np.abs(de / dr - 1)) * 100),
            "wape_h_pct": float(np.abs(E - R).sum() / R.sum() * 100),
            "sesgo_energia_pct": float((de.sum() / dr.sum() - 1) * 100)}


def _curvas(df: pd.DataFrame):
    return {d.strftime("%Y-%m-%d"): [round(float(x), 4) for x in row] for d, row in zip(df["fecha"], df[P].to_numpy())}


def exportar(ucp: str, ex: pd.DataFrame, real: pd.DataFrame) -> Path:
    FB.mkdir(parents=True, exist_ok=True)
    RETRO.mkdir(parents=True, exist_ok=True)
    ex[["fecha", "codsesion", "version", "guardado", "lead"] + P].to_csv(FB / "emisiones_exante.csv", index=False)
    real[["fecha"] + P].to_csv(FB / "real_hasta.csv", index=False)
    semanas: dict = {}
    for d, row in zip(ex["fecha"], ex[P].to_numpy()):
        lunes = (d - pd.Timedelta(days=d.dayofweek)).strftime("%Y-%m-%d")
        semanas.setdefault(lunes, {})[d.strftime("%Y-%m-%d")] = [round(float(x), 4) for x in row]
    emisiones = [{"semana_inicio": k, "version": 1, "curvas": v} for k, v in sorted(semanas.items())]
    anios = range(int(ex["fecha"].min().year), int(real["fecha"].max().year) + 2)
    festivos = [f for f in festivos_colombia(anios) if ex["fecha"].min().strftime("%Y-%m-%d") <= f
                <= (real["fecha"].max() + pd.Timedelta(days=60)).strftime("%Y-%m-%d")]
    out = RETRO / f"{ucp}.json"
    out.write_text(json.dumps({"emisiones": emisiones, "reales": _curvas(real), "festivos": festivos},
                              ensure_ascii=False, separators=(",", ":")))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucp", default="Antioquia")
    ap.add_argument("--desde", default="2026-01-12")
    ap.add_argument("--hasta", default=None, help="fin de la validacion (default: ultimo real disponible)")
    a = ap.parse_args()
    ex, real = construir(a.ucp, a.desde)
    hasta = a.hasta or real["fecha"].max().strftime("%Y-%m-%d")
    print(json.dumps(validar(ex, real, a.desde, hasta), indent=1))
    print(exportar(a.ucp, ex, real), "| dias emitidos:", len(ex), "| real hasta:", real["fecha"].max().date())
