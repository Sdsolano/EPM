"""
B2 (noche 2): arma el cache de componentes por semana para los experimentos de mejora de pronóstico.

Por UCP y semana objetivo (52 semanas de `curvas_modelo_k1_<ucp>.csv`), con corte = sábado anterior y sin fuga
(`construir_contexto` verifica histórico <= corte), guarda para los 7 días objetivo:
  base (k1), nivel, final (nivel+forma default), ana (nivel ancla análogos α .25 + forma default),
  ref (nivel default + forma fuente referencia), real, tipo de día, fecha de corte.

Salida: <out>/b2_<ucp>.npz (solo lectura de data/raw y reports; no escribe nada fuera de <out>).

    python scripts/noche2_b2_datos.py --out <dir> [--ucp Antioquia Atlantico]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import backtest_adjustments as ba  # noqa: E402
from src.adjustments import data, pipeline  # noqa: E402

REPO_PRINCIPAL = Path("C:/Users/samue/OneDrive/Documentos/GitHub/EPM")
COMPONENTES = {
    "default": {},
    "ana": {"nivel": {"ancla": "analogos", "alpha": 0.25}},
    "ref": {"forma": {"fuente": "referencia"}},
}
TIPOS = ["laboral", "sabado", "domingo", "festivo"]


def _curvas_por_semana(ucp: str, k1_dir: Path):
    import pandas as pd
    df = pd.read_csv(Path(k1_dir) / f"curvas_modelo_k1_{ucp}.csv", dtype={"semana_inicio": str, "fecha": str})
    ps = [f"P{i}" for i in range(1, 25)]
    out = {}
    for sem, g in df.groupby("semana_inicio", sort=True):
        out[sem] = {str(f)[:10]: [float(x) for x in v] for f, v in zip(g["fecha"], g[ps].to_numpy())}
    return out


def construir(ucp: str, k1_dir: Path, raw_dir: Path):
    curvas, semanas = ba.cargar_curvas_modelo_k1(ucp, k1_dir)
    hist = data.cargar_historico(ucp, data_dir=raw_dir)
    reales = data.curvas_reales(hist)
    anios = range(int(hist["fecha"].dt.year.min()) - 1, int(hist["fecha"].dt.year.max()) + 2)
    festivos = data.festivos_colombia(anios)
    fset = set(festivos)
    lunes_lista = ba.semanas_modelo_k1(hist, 60, semanas, curvas)
    # IMPORTANTE: cada semana usa SU bloque de curvas (el domingo de brecha de la semana W+1 repite el domingo de W
    # con un pronóstico más informado: usarlo sería fuga).
    por_semana = _curvas_por_semana(ucp, k1_dir)
    n = len(lunes_lista)
    out = {k: np.full((n, 7, 24), np.nan) for k in ("base", "nivel", "final", "ana", "ref", "real")}
    tipo = np.empty((n, 7), dtype="U8")
    fechas = np.empty((n, 7), dtype="U10")
    cortes = np.empty(n, dtype="U10")
    for i, lunes in enumerate(lunes_lista):
        ctx = ba.construir_contexto(ucp, lunes, hist, festivos)
        cortes[i] = ctx.fecha_corte.isoformat()
        dias = [d.isoformat() for d in data.dias_semana(lunes)]
        for nombre, params in COMPONENTES.items():
            prov = ba.proveedor_desde_curvas(por_semana[lunes.isoformat()])
            recs = {r.paso: r for r in pipeline.run_pipeline(ctx, ba.con_cache(prov), params=params)}
            if nombre == "default":
                for j, f in enumerate(dias):
                    out["base"][i, j] = recs["base"].result.curves[f]
                    out["nivel"][i, j] = recs["nivel"].result.curves[f]
                    out["final"][i, j] = recs["forma"].result.curves[f]
            else:
                for j, f in enumerate(dias):
                    out[nombre][i, j] = recs["forma"].result.curves[f]
        for j, f in enumerate(dias):
            out["real"][i, j] = reales[f]
            tipo[i, j] = data.tipo_dia(date.fromisoformat(f), fset)
            fechas[i, j] = f
        print(ucp, lunes, flush=True)
    return out, tipo, fechas, cortes, np.array([l.isoformat() for l in lunes_lista]), festivos


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ucp", nargs="+", default=["Antioquia", "Atlantico"])
    ap.add_argument("--k1-dir", default=str(REPO_PRINCIPAL / "reports" / "ajustes"))
    ap.add_argument("--raw-dir", default=str(REPO_PRINCIPAL / "data" / "raw"))
    a = ap.parse_args(argv)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ucp in a.ucp:
        comp, tipo, fechas, cortes, lunes, festivos = construir(ucp, Path(a.k1_dir), Path(a.raw_dir))
        np.savez_compressed(out_dir / f"b2_{ucp}.npz", tipo=tipo, fechas=fechas, cortes=cortes, lunes=lunes,
                            festivos=np.array(festivos), **comp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
