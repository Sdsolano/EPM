"""
Backtest rolling SIN fuga de las recomendaciones de sesgo de insights (B5, noche 2).

Para cada semana del historial emitido (`precision_data/{ucp}_k1.csv`) usa solo datos hasta el domingo previo, aplica
`sesgo_por_segmento` / `_hay_evidencia` / `validar_rolling` / `_corregir_curva` a cada día de la semana y compara el WAPE del
pronóstico emitido contra el corregido (solo en los días donde la regla se habría emitido). Se evalúa con dos filtros:
- "anterior": ROLLING_MIN_DIAS 4, mejora >= 60 % de días, WAPE agregado menor (lógica previa a B5).
- "actual": `val["pasa"]` con las constantes vigentes (>= 8 días, >= 75 %, mejora >= 0.3 pp).

Uso: [EPM_RAW_DIR=ruta/data/raw] python scripts/noche2_b5_backtest.py [Antioquia Atlantico]
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adjustments import data, insights  # noqa: E402
from src.adjustments.types import parse_date  # noqa: E402


def _pasa_anterior(v: dict) -> bool:
    n = v["dias_evaluados"]
    return bool(n >= 4 and v["pct_dias_mejora"] is not None and v["pct_dias_mejora"] >= 0.60
                and v["wape_antes_pct"] is not None and v["wape_despues_pct"] < v["wape_antes_pct"])


def backtest(ucp: str) -> dict:
    emitidos, semana_de = insights.cargar_emitidos(ucp)
    reales = data.curvas_reales(data.cargar_historico(ucp, os.environ.get("EPM_RAW_DIR")))
    fechas = list(reales)[-1:] + list(emitidos)[:1] + list(emitidos)[-1:]
    d = insights.Datos(emitidos=emitidos, semana_de=semana_de, reales=reales,
                       festivos=insights.festivos_por_defecto(fechas))
    pares = insights._pares(d)
    semanas = sorted({parse_date(f) - timedelta(days=parse_date(f).weekday()) for f in pares})
    todos_f, todos_r = [], []
    res = {"anterior": {"F0": [], "F1": [], "R": []}, "actual": {"F0": [], "F1": [], "R": []}}
    cache = {}
    for lunes in semanas:
        hasta = (lunes - timedelta(days=1)).isoformat()
        sesgos = insights.sesgo_por_segmento(d, hasta=hasta)
        for f in (data.iso(x) for x in data.dias_semana(lunes)):
            if f not in pares:
                continue
            F, R = np.asarray(emitidos[f], float), np.asarray(reales[f], float)
            todos_f.append(F)
            todos_r.append(R)
            seg = insights._segmento(d.tipo(f), f)
            sg = sesgos.get(seg)
            if not sg or not insights._hay_evidencia(sg):
                continue
            if parse_date(sg["fechas"][-1]) < lunes - timedelta(days=insights.MAX_ANTIGUEDAD_DIAS):
                continue
            k = (seg, hasta)
            if k not in cache:
                cache[k] = insights.validar_rolling(d, seg, hasta)
            v = cache[k]
            corr = insights._corregir_curva(sg, F, insights._categoria_regla(sg))
            for nombre, ok in (("anterior", _pasa_anterior(v)), ("actual", bool(v["pasa"]))):
                if ok:
                    res[nombre]["F0"].append(F)
                    res[nombre]["F1"].append(corr)
                    res[nombre]["R"].append(R)
    out = {"ucp": ucp, "dias_total": len(todos_f),
           "wape_global_sin_corregir": insights.wape(np.vstack(todos_f), np.vstack(todos_r))}
    for nombre, x in res.items():
        n = len(x["R"])
        if n == 0:
            out[nombre] = {"dias": 0}
            continue
        F0, F1, R = np.vstack(x["F0"]), np.vstack(x["F1"]), np.vstack(x["R"])
        mej = sum(1 for a, b, r in zip(F0, F1, R) if insights.wape(b, r) < insights.wape(a, r))
        out[nombre] = {"dias": n, "pct_mejora": 100 * mej / n, "wape_antes": insights.wape(F0, R),
                       "wape_despues": insights.wape(F1, R)}
    return out


def main() -> None:
    for ucp in sys.argv[1:] or ["Antioquia", "Atlantico"]:
        r = backtest(ucp)
        print(f"\n{ucp}: {r['dias_total']} días con emitido y real; WAPE global sin corregir {r['wape_global_sin_corregir']:.2f}")
        for nombre in ("anterior", "actual"):
            x = r[nombre]
            if x["dias"] == 0:
                print(f"  filtro {nombre}: 0 días emitidos")
            else:
                print(f"  filtro {nombre}: {x['dias']} días emitidos, mejora en {x['pct_mejora']:.0f} % de días, "
                      f"WAPE {x['wape_antes']:.2f} -> {x['wape_despues']:.2f}")


if __name__ == "__main__":
    main()
