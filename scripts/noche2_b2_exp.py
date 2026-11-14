"""
B2 (noche 2): experimentos consolidados de mejora de pronóstico (sin fuga) -> resultados.json.

Requiere el cache de `scripts/noche2_b2_datos.py`. Solo lectura de data/raw (clima). Escribe únicamente en --out.

    python scripts/noche2_b2_exp.py --cache <dir> --out docs/work/ajustes/noche2/B2
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noche2_b2_lib as L  # noqa: E402

RAW = Path("C:/Users/samue/OneDrive/Documentos/GitHub/EPM/data/raw")
UCPS = ["Antioquia", "Atlantico"]
UMBRAL = 0.05  # pp: por debajo es ruido (prueba = 21 semanas)


def semanal_wape(pred, ds, w):
    r = ds.real[w]
    return np.abs(pred[w] - r).sum() / r.sum()


def evaluar(pred_por_ucp, ref_por_ucp, dss):
    """{ucp: {part: {m, ref, d_wape, d_sem, d_sesgo_sem, d_pico, pct_semanas_mejor}}}."""
    out = {}
    for u, ds in dss.items():
        out[u] = {}
        for part in ("cal", "test"):
            m = L.metricas(pred_por_ucp[u], ds, part)
            r = L.metricas(ref_por_ucp[u], ds, part)
            idx = np.where(ds.mask(part))[0]
            mejor = np.mean([semanal_wape(pred_por_ucp[u], ds, w) < semanal_wape(ref_por_ucp[u], ds, w) - 1e-12
                             for w in idx])
            out[u][part] = dict(m=m, ref=r, d_wape=m["wape"] - r["wape"], d_sem=m["mape_sem"] - r["mape_sem"],
                                d_sesgo_sem=abs(m["sesgo_sem"]) - abs(r["sesgo_sem"]), d_pico=m["err_pico"] - r["err_pico"],
                                pct_semanas_mejor=100 * mejor)
    return out


def veredicto(ev):
    """PASA = las 4 celdas (UCP x particion) mejoran el WAPE en >= UMBRAL y el MAPE semanal no empeora > UMBRAL."""
    dw = [ev[u][p]["d_wape"] for u in ev for p in ("cal", "test")]
    ds_ = [ev[u][p]["d_sem"] for u in ev for p in ("cal", "test")]
    if max(dw) <= -UMBRAL and max(ds_) <= UMBRAL:
        return "PASA"
    if max(dw) < 0 and max(ds_) <= UMBRAL:
        return "MARGINAL (mejora en las 4 celdas pero alguna < 0.05 pp = ruido)"
    if max(dw) < 0:
        return "NO (mejora WAPE pero empeora MAPE semanal > 0.05 pp)"
    return "NO (no mejora en calibracion Y prueba en ambas UCP)"


def elegir(grid, dss, ref, restr_sem=True):
    """Elige en CALIBRACION el hiperparametro con menor suma de delta-WAPE relativo (ambas UCP).

    Restriccion: MAPE semanal de calibracion no empeora mas de UMBRAL.
    """
    mejor, mejor_s = None, np.inf
    sin_restr, sin_restr_s = None, np.inf
    for etiqueta, fn in grid:
        preds = {u: fn(ds) for u, ds in dss.items()}
        s, ok = 0.0, True
        for u, ds in dss.items():
            m = L.metricas(preds[u], ds, "cal")
            r = L.metricas(ref[u], ds, "cal")
            s += (m["wape"] - r["wape"]) / r["wape"]
            if restr_sem and m["mape_sem"] - r["mape_sem"] > UMBRAL:
                ok = False
        if s < sin_restr_s:
            sin_restr, sin_restr_s = (etiqueta + " [sin restriccion semanal]", preds), s
        if ok and s < mejor_s:
            mejor, mejor_s = (etiqueta, preds), s
    return mejor or sin_restr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dss = {u: L.cargar(u, Path(a.cache)) for u in UCPS}
    res = {"candidatas": {},
           "particion": {u: {"n_semanas": ds.W, "calibracion": ds.ncal, "prueba": ds.W - ds.ncal,
                             "cal": [str(ds.lunes[0]), str(ds.lunes[ds.ncal - 1])],
                             "test": [str(ds.lunes[ds.ncal]), str(ds.lunes[-1])]} for u, ds in dss.items()}}
    final = {u: ds.comp["final"] for u, ds in dss.items()}
    base = {u: ds.comp["base"] for u, ds in dss.items()}

    # ---- linea base
    lb = {}
    for u, ds in dss.items():
        lb[u] = {}
        for k in ("base", "nivel", "final", "ana", "ref"):
            lb[u][k] = {part: {"total": L.metricas(ds.comp[k], ds, part),
                               **{t: L.metricas(ds.comp[k], ds, part, t) for t in L.TIPOS}} for part in ("cal", "test")}
    res["linea_base"] = lb

    def reg(nombre, etiqueta, preds, ref):
        ev = evaluar(preds, ref, dss)
        res["candidatas"][nombre] = {
            "param": etiqueta, "veredicto": veredicto(ev), "eval": ev,
            "por_tipo": {u: {part: {t: L.metricas(preds[u], dss[u], part, t) for t in L.TIPOS}
                             for part in ("cal", "test")} for u in dss}}
        print(f"{nombre:34s} {etiqueta}  {veredicto(ev)}", flush=True)
        return preds

    # ---- (a) sesgo multiplicativo segmento x hora, ventana movil
    for comp_nombre, ref in (("final", final), ("base", base)):
        for seg in ("global", "tipo", "dow8"):
            grid = []
            for N, k0, g, suav, modo in itertools.product((4, 8, 13, 26), (0, 3, 8), (0.5, 1.0), (0, 1),
                                                           ("hora", "energia", "forma")):
                if modo != "hora" and suav:
                    continue
                grid.append((f"N={N},k0={k0},g={g},suav={suav},modo={modo}",
                             lambda ds, N=N, k0=k0, g=g, suav=suav, modo=modo, comp=comp_nombre, seg=seg: L.corregir(
                                 ds.comp[comp], ds, seg=seg, N=N, k0=k0, g=g, suav=suav,
                                 solo_energia=(modo == "energia"), conservar_energia=(modo == "forma"))))
            et, preds = elegir(grid, dss, ref)
            reg(f"a_sesgo_{comp_nombre}_{seg}", et, preds, ref)

    # ---- (b) mezcla ponderada
    grid = []
    for en in ("mezcla", "final"):
        for comps in (("final", "ana"), ("base", "final", "ana"), ("nivel", "final", "ana", "ref"),
                      ("base", "final", "ana", "ref")):
            for seg, N, p in itertools.product(("global", "tipo"), (8, 13, 26), (0, 1, 2, 4)):
                def fn(ds, comps=comps, seg=seg, N=N, p=p, en=en):
                    m = L.mezclar(ds, list(comps), seg, N, p)
                    return L.fijar_energia(m, ds.comp["final"]) if en == "final" else m
                grid.append((f"energia={en},comps={'+'.join(comps)},seg={seg},N={N},p={p}", fn))
    et, preds_mezcla = elegir(grid, dss, final)
    reg("b_mezcla_ponderada", et, preds_mezcla, final)
    simple = {u: (ds.comp["final"] + ds.comp["ana"]) / 2 for u, ds in dss.items()}
    reg("b_promedio_final_ana", "50/50 fijo", simple, final)

    # ---- (e) seleccion por tipo de dia
    comps_sel = ("base", "nivel", "final", "ana", "ref")
    elegido = {}
    for t in L.TIPOS:
        mejor, mejor_s = None, np.inf
        for c in comps_sel:
            s = sum((L.metricas(dss[u].comp[c], dss[u], "cal", t)["wape"]
                     / L.metricas(final[u], dss[u], "cal", t)["wape"] - 1) for u in dss)
            if s < mejor_s:
                mejor, mejor_s = c, s
        elegido[t] = mejor
    sel = {}
    for u, ds in dss.items():
        pr = ds.comp["final"].copy()
        for t, c in elegido.items():
            pr[ds.tipo == t] = ds.comp[c][ds.tipo == t]
        sel[u] = pr
    reg("e_seleccion_estatica_por_tipo", json.dumps(elegido), sel, final)
    grid = [(f"N={N},comps={'+'.join(c)}", lambda ds, N=N, c=c: L.mejor_en_ventana(ds, list(c), "tipo", N))
            for N, c in itertools.product((8, 13, 26), (("final", "ana"), ("base", "final", "ana"),
                                                         ("base", "nivel", "final", "ana", "ref")))]
    et, preds = elegir(grid, dss, final)
    reg("e_seleccion_rodante_por_tipo", et, preds, final)

    # ---- (c) festivo / puente
    grid = [(f"N={N},k0={k0},g={g}",
             lambda ds, N=N, k0=k0, g=g: L.corregir(ds.comp["final"], ds, seg="tipo_puente", N=N, k0=k0, g=g,
                                                    N_fest=52, minimo=3))
            for N, k0, g in itertools.product((13, 26), (0, 3, 8), (0.5, 1.0))]
    et, preds = elegir(grid, dss, final)
    reg("c_festivo_puente_sesgo", et, preds, final)
    puente = {}
    for u, ds in dss.items():
        puente[u] = {}
        for nm, mk in (("puente", ds.puente), ("festivo", ds.tipo == "festivo")):
            for part in ("cal", "test"):
                m = ds.mask(part)[:, None] & mk
                r = ds.real[m]
                puente[u][f"{nm}_{part}"] = {
                    "n_dias": int(m.sum()),
                    **{c: 100 * float(np.abs(ds.comp[c][m] - r).sum() / r.sum()) for c in ("base", "final")}}
        e = np.log(ds.real.sum(2) / ds.comp["final"].sum(2))
        puente[u]["sesgo_energia_pct"] = {
            "puente": 100 * float(e[ds.puente].mean()),
            "laboral_normal": 100 * float(e[(ds.tipo == "laboral") & ~ds.puente].mean()),
            "festivo": 100 * float(e[ds.tipo == "festivo"].mean())}
    res["puente_diagnostico"] = puente

    # ---- (c2) clima observado (optimista)
    clima = {}
    for u, ds in dss.items():
        dT = L.anomalia_temperatura(ds, RAW / u / "clima_new.csv")
        clima[u] = dict(dias_con_clima=int(np.isfinite(dT).sum()), dias_cal=int(np.isfinite(dT[: ds.ncal]).sum()),
                        dias_test=int(np.isfinite(dT[ds.ncal:]).sum()))
        ds.dT = dT
    grid = [(f"N={N},k0={k0},minimo={mn},g={g}",
             lambda ds, N=N, k0=k0, mn=mn, g=g: L.corregir_clima(ds.comp["final"], ds, ds.dT, N=N, k0=k0, minimo=mn, g=g))
            for N, k0, mn, g in itertools.product((13, 26, 52), (5.0,), (10, 20), (0.5, 1.0))]
    mejor, mejor_s = None, np.inf  # seleccion solo con Atlantico-calibracion (Antioquia no tiene clima en calibracion)
    for et, fn in grid:
        pr = fn(dss["Atlantico"])
        m, r = L.metricas(pr, dss["Atlantico"], "cal"), L.metricas(final["Atlantico"], dss["Atlantico"], "cal")
        if m["wape"] - r["wape"] < mejor_s:
            mejor, mejor_s = et, m["wape"] - r["wape"]
    fn = dict(grid)[mejor]
    preds = {u: fn(ds) for u, ds in dss.items()}
    reg("c_clima_observado", mejor + " (elegido solo con Atlantico-calibracion)", preds, final)
    res["clima"] = clima

    # ---- (d) intervalos empiricos
    bd = {}
    for esc in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
        bd[esc] = float(np.mean([L.cobertura(*L.bandas(final[u], dss[u], N=26, pool=3, escala=esc), dss[u], "cal")["cobertura"]
                                 for u in dss]))
    esc_sel = min(bd, key=lambda e: abs(bd[e] - 80))
    res["bandas"] = {"escala_elegida_en_calibracion": esc_sel,
                     "cobertura_cal_media_por_escala": {str(k): v for k, v in bd.items()}}
    for nombre, esc in (("cruda", 1.0), ("calibrada", esc_sel)):
        res["bandas"][nombre] = {}
        for u, ds in dss.items():
            lo, hi = L.bandas(final[u], ds, N=26, pool=3, escala=esc)
            res["bandas"][nombre][u] = {
                part: {"total": L.cobertura(lo, hi, ds, part), **{t: L.cobertura(lo, hi, ds, part, t) for t in L.TIPOS}}
                for part in ("cal", "test")}
            m = ds.mask("test")
            m[:8] = False
            inside = (ds.real[m] >= lo[m]) & (ds.real[m] <= hi[m])
            res["bandas"][nombre][u]["test_por_bloque_horario"] = {
                bloque: 100 * float(inside[:, :, list(hs)].mean())
                for bloque, hs in (("madrugada_1-6", range(0, 6)), ("manana_7-12", range(6, 12)),
                                   ("tarde_13-18", range(12, 18)), ("noche_19-24", range(18, 24)))}
            ld, hd, ls, hs_ = L.bandas_energia(final[u], ds, escala=esc)
            m = ds.mask("test")
            ed, es = ds.real.sum(2), ds.real.sum((1, 2))
            okd = m[:, None] & np.isfinite(ld)
            ok_s = m & np.isfinite(ls)
            res["bandas"][nombre][u]["energia_test"] = {
                "cobertura_dia": 100 * float(((ed >= ld) & (ed <= hd))[okd].mean()),
                "ancho_dia_pct": 100 * float(((hd - ld) / ed)[okd].mean()),
                "cobertura_semana": 100 * float(((es >= ls) & (es <= hs_))[ok_s].mean()) if ok_s.any() else None,
                "ancho_semana_pct": 100 * float(((hs_ - ls) / es)[ok_s].mean()) if ok_s.any() else None,
                "n_semanas": int(ok_s.sum())}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "resultados.json").write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False),
                                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
