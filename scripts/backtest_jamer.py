"""
Backtest de la réplica del pronóstico manual de Jamer en la Mesa de ajustes (fase E6).

Parte 1 — Réplica (semana 23–29/nov-2026, Antioquia). La "verdad" es la sesión que Jamer guardó
(`reports/ajustes/jamer/verdad_jamer.csv`, tipo P; ruta configurable con `--verdad`). Se comparan cuatro columnas sobre la misma semana:
  * `crudo`        = paso `base` (OJO: base SINTÉTICA = forma de R1 reescalada a los totales del modelo
                     de la reunión; no es la salida real del champion, ver fase0/REPORTE.md);
  * `mesa_hoy`     = pipeline sin `perfil` (comportamiento de hoy);
  * `jamer`        = perfil `jamer` (réplica pura, w=0);
  * `jamer_modelo` = perfil `jamer_modelo` (w=0.3).
Más tres variantes experimentales rotuladas (g fijo de la cota, tramos automáticos, g=0).
Veredictos pasa/falla por criterio A–G de `docs/work/ajustes/replica_jamer/PLAN_IMPL.md` §7.

Parte 2 — Generalización (semanas completas ene–jun 2026, base `ingenuo`, sin red, sin refrescar datos).
Calibración con el 60 % más antiguo de las semanas y reporte en el 40 % más reciente. Con base ingenua
el modelo no existe, así que `w` (peso del modelo) NO es evaluable y se dice.

Uso (desde la raíz del repo):
    .venv/bin/python scripts/backtest_jamer.py                 # ambas partes
    .venv/bin/python scripts/backtest_jamer.py --parte replica
    .venv/bin/python scripts/backtest_jamer.py --parte generalizacion --ucp Antioquia
    .venv/bin/python scripts/backtest_jamer.py --parte replica --directorio <dir> --verdad <csv> --fixture <json>

Salida: reports/ajustes/backtest_jamer_<YYYYMMDD>.md (solo agregados, versionable) y
reports/ajustes/jamer/backtest_jamer_<YYYYMMDD>_detalle.csv (detalle con demanda real; ignorado por git).
NO llama `full_update_csv`, no escribe `data/raw` ni `models/`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import backtest_adjustments as bta  # noqa: E402  (reutiliza bases, contexto sin fuga, tabla_md)
from src.adjustments import acumulado, data, perfiles, pipeline  # noqa: E402
from src.adjustments.referencia_reciente import seleccionar_recientes  # noqa: E402
from src.adjustments.steps import control as paso_control  # noqa: E402
from src.adjustments.types import Curves  # noqa: E402

logger = logging.getLogger("backtest_jamer")

JAMER_DIR = ROOT / "reports" / "ajustes" / "jamer"
REPORTS_DIR = ROOT / "reports" / "ajustes"
UCP_REPLICA = "Antioquia"
LUNES_REPLICA = date(2026, 11, 23)
FECHAS_REPLICA = [(LUNES_REPLICA + timedelta(days=i)).isoformat() for i in range(7)]
DIAS_ES = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]

# Valores esperados REALES de la réplica (R1 por día, vectores G1, checksums F1): viven en un fixture local
# ignorado por git (`reports/ajustes/jamer/fixture_replica.json`), nunca en este archivo versionable.
FIXTURE_NOMBRE = "fixture_replica.json"
VERDAD_NOMBRE = "verdad_jamer.csv"
BANDA_SALTO_REPLICA_MW = perfiles.BANDA_SALTO_MW


def cargar_fixture(path: Path) -> Dict[str, Any]:
    """Lee el fixture local de valores esperados (claves: r1_esperado, g1_esperado_p1, f1_checksums, d_tramos_esperados)."""
    with open(path, encoding="utf-8") as fh:
        crudo = json.load(fh)
    return {
        "r1_esperado": {str(k): str(v) for k, v in crudo["r1_esperado"].items()},
        "g1_esperado_p1": {str(k): (float(v[0]), float(v[1])) for k, v in crudo["g1_esperado_p1"].items()},
        "f1_checksums": {str(k): (float(v[0]), float(v[1]), float(v[2])) for k, v in crudo["f1_checksums"].items()},
        "d_tramos_esperados": {str(k): (int(v[0]), int(v[1])) for k, v in crudo["d_tramos_esperados"].items()},
    }


# límites del criterio C (tramos=off)
C_MAE, C_P95, C_MAX, C_SESGO, C_RMSE_PCT, C_CORR = 10.0, 25.0, 45.0, 10.0, 1.0, 0.995
CAL_FRACCION = 0.60


# ---------------------------------------------------------------------------
# Carga de insumos (todo local, sin red)
# ---------------------------------------------------------------------------

def _cols_p() -> List[str]:
    return [f"p{i}" for i in range(1, 25)]


def cargar_sesion(path: Path) -> Dict[str, Curves]:
    """CSV de sesión de Sphere -> {tipo: {fecha: [24 valores]}} (tipos P, PM, D)."""
    df = pd.read_csv(path)
    out: Dict[str, Curves] = {}
    for fila in df.itertuples(index=False):
        row = fila._asdict()
        valores = [float(row[c]) for c in _cols_p()]
        out.setdefault(str(row["tipo"]), {})[str(row["fecha"])[:10]] = valores
    return out


def cargar_insumos_replica(
    directorio: Path = JAMER_DIR, verdad: Optional[Path] = None, fixture: Optional[Path] = None
) -> Dict[str, Any]:
    """Histórico de la réplica, base sintética, verdad de Jamer (P) y PM, y fixture de esperados.
    `verdad`/`fixture` default: `<directorio>/verdad_jamer.csv` y `<directorio>/fixture_replica.json`.
    FileNotFoundError si faltan."""
    directorio = Path(directorio)
    hist_path = directorio / "historico_replica.csv"
    base_path = directorio / "base_sintetica_23_29_nov.csv"
    ses_path = Path(verdad) if verdad is not None else directorio / VERDAD_NOMBRE
    fix_path = Path(fixture) if fixture is not None else directorio / FIXTURE_NOMBRE
    for p in (hist_path, base_path, ses_path, fix_path):
        if not p.exists():
            raise FileNotFoundError(f"Falta {p} (ver docs/work/ajustes/replica_jamer/fase0/REPORTE.md)")
    hist = data.normalizar_historico(pd.read_csv(hist_path), ucp=UCP_REPLICA)
    sesion = cargar_sesion(ses_path)
    return {
        "historico": hist,
        "base": bta.cargar_archivo(base_path),
        "verdad": sesion["P"],
        "pm": sesion.get("PM", {}),
        "fixture": cargar_fixture(fix_path),
    }


# ---------------------------------------------------------------------------
# Métricas puras
# ---------------------------------------------------------------------------

def _matriz(curvas: Curves, fechas: Sequence[str]) -> np.ndarray:
    return np.asarray([curvas[f] for f in fechas], dtype=float)


def metricas_periodo(pred: Curves, verdad: Curves, fechas: Sequence[str]) -> Dict[str, Any]:
    """
    Métricas por periodo de `pred` contra `verdad` en `fechas` (MW por periodo = MWh/h).
    - `mae`, `p95`, `max`: sobre |pred − verdad| de todos los periodos de todos los días.
    - `sesgo_dia`: media por día de (pred − verdad); `sesgo_max_abs` = máximo de |sesgo_dia|.
    - Forma: se reescala cada día a la energía de la verdad y se mide el RMSE como % de la media del día
      (`rmse_forma_pct`, agrupado sobre los días) y la correlación de Pearson (`corr_min`, `corr_media`).
    """
    p, v = _matriz(pred, fechas), _matriz(verdad, fechas)
    err = p - v
    abs_err = np.abs(err)
    esc = v.sum(axis=1, keepdims=True) / p.sum(axis=1, keepdims=True)
    err_forma = p * esc - v
    media_dia = v.mean(axis=1)
    rmse_dia = np.sqrt((err_forma ** 2).mean(axis=1))
    corr = np.array([np.corrcoef(p[i], v[i])[0, 1] for i in range(len(fechas))])
    return {
        "mae": float(abs_err.mean()),
        "p95": float(np.percentile(abs_err, 95)),
        "max": float(abs_err.max()),
        "sesgo_dia": [float(x) for x in err.mean(axis=1)],
        "sesgo_max_abs": float(np.abs(err.mean(axis=1)).max()),
        "mae_dia": [float(x) for x in abs_err.mean(axis=1)],
        "rmse_forma_pct": float(100 * np.sqrt((err_forma ** 2).mean()) / media_dia.mean()),
        "rmse_forma_pct_dia": [float(100 * a / b) for a, b in zip(rmse_dia, media_dia)],
        "corr_min": float(corr.min()),
        "corr_media": float(corr.mean()),
        "n_dias": len(fechas),
    }


def error_energia(pred: Curves, verdad: Curves, fechas: Sequence[str]) -> Dict[str, Any]:
    """Error relativo (%) de la energía diaria y semanal de `pred` frente a `verdad`."""
    e_p = np.asarray([sum(pred[f]) for f in fechas], dtype=float)
    e_v = np.asarray([sum(verdad[f]) for f in fechas], dtype=float)
    err = 100 * (e_p / e_v - 1)
    return {
        "dia_pct": [float(x) for x in err],
        "semana_pct": float(100 * (e_p.sum() / e_v.sum() - 1)),
        "mape_pct": float(np.abs(err).mean()),
        "n_le_1": int((np.abs(err) <= 1.0).sum()),
        "n_le_05": int((np.abs(err) <= 0.5).sum()),
    }


def crecimiento_pct(final: float, ref: float) -> Optional[float]:
    """(P/R − 1)·100; None si la referencia no es válida."""
    if ref is None or final is None or not math.isfinite(ref) or ref <= 0:
        return None
    return (final / ref - 1) * 100


def veredicto(ok: Optional[bool]) -> str:
    return "no evaluable" if ok is None else ("pasa" if ok else "falla")


def criterio(cid: str, que: str, valor: str, umbral: str, ok: Optional[bool], nota: str = "") -> Dict[str, str]:
    return {"id": cid, "criterio": que, "valor": valor, "umbral": umbral, "veredicto": veredicto(ok), "nota": nota}


# ---------------------------------------------------------------------------
# Parte 1: corridas de la réplica
# ---------------------------------------------------------------------------

def construir_contexto_replica(historico: pd.DataFrame, festivos: Optional[List[str]] = None):
    """Contexto del lunes 23/11 con corte 14/11 y verificación de no fuga (reutiliza backtest_adjustments)."""
    if festivos is None:
        festivos = data.festivos_colombia(range(2017, 2028))
    ctx = pipeline.build_context(
        UCP_REPLICA, LUNES_REPLICA, fecha_emision=LUNES_REPLICA - timedelta(days=1),
        historico=historico, festivos=festivos,
    )
    max_hist = ctx.historico["fecha"].max().date()
    if max_hist > ctx.fecha_corte or ctx.fecha_corte >= LUNES_REPLICA:
        raise AssertionError(f"Fuga de datos: histórico hasta {max_hist}, corte {ctx.fecha_corte}")
    return ctx


COLUMNAS_REPLICA: Dict[str, Tuple[Optional[str], Dict[str, Dict[str, Any]]]] = {
    "mesa_hoy": (None, {}),
    "jamer": ("jamer", {}),
    "jamer_modelo": ("jamer_modelo", {}),
    # variantes experimentales (rotuladas en el reporte)
    "jamer_g_fijo": ("jamer", {"nivel": {"g_fuente": "fijo", "g_fijo": 0.00494}}),
    "jamer_g0": ("jamer", {"nivel": {"g_fuente": "fijo", "g_fijo": 0.0}}),
    "jamer_tramos": ("jamer", {"forma": {"tramos": "modelo_anual"}}),
}


def correr_replica(insumos: Dict[str, Any]) -> Dict[str, Any]:
    """Corre todas las columnas; devuelve {nombre: {curves, records}} (+ ctx, provider)."""
    ctx = construir_contexto_replica(insumos["historico"])
    provider = bta.proveedor_desde_curvas(insumos["base"])
    out: Dict[str, Any] = {"ctx": ctx, "columnas": {}}
    for nombre, (perfil, extra) in COLUMNAS_REPLICA.items():
        params = perfiles.expandir(perfil, extra or None)
        records = pipeline.run_pipeline(ctx, provider, params=params)
        por_paso = {r.paso: r for r in records}
        out["columnas"][nombre] = {
            "records": records,
            "por_paso": por_paso,
            "final": {f: list(v) for f, v in records[-1].result.curves.items()},
            "alertas": sorted({a.codigo for r in records for a in r.result.alerts}),
        }
    # el crudo es el paso base de cualquier corrida
    base_curves = out["columnas"]["mesa_hoy"]["por_paso"]["base"].result.curves
    out["columnas"] = {"crudo": {"final": {f: list(v) for f, v in base_curves.items()}, "alertas": [],
                                 "por_paso": out["columnas"]["mesa_hoy"]["por_paso"]},
                       **out["columnas"]}
    return out


# ---------------------------------------------------------------------------
# Parte 1: criterios A–G
# ---------------------------------------------------------------------------

def _r1_energia(col: Dict[str, Any], f: str) -> Optional[float]:
    dias = col["por_paso"]["nivel"].result.diagnostics.get("dias", {})
    info = (dias.get(f) or {}).get("r1") or {}
    return info.get("energia")


def criterios_a(ctx, corridas: Dict[str, Any], r1_esperado: Dict[str, str]) -> List[Dict[str, str]]:
    festivos = set(ctx.festivos)
    reales = data.curvas_reales(ctx.historico)
    out: List[Dict[str, str]] = []
    sel = {f: seleccionar_recientes(f, reales, ctx.fecha_corte, festivos, n=1) for f in FECHAS_REPLICA}
    obtenido = {f: s.fechas[0].isoformat() if s.fechas else None for f, s in sel.items()}
    ok = obtenido == r1_esperado
    out.append(criterio("A1", "R1 por día (23–29/11)", f"{sum(obtenido[f] == r1_esperado.get(f) for f in FECHAS_REPLICA)}/7 exactos",
                        "7/7 exactos", ok, "" if ok else f"obtenido {obtenido}"))
    s2 = seleccionar_recientes("2026-11-24", reales, ctx.fecha_corte, festivos, n=2)
    f2 = [x.isoformat() for x in s2.fechas]
    ok2 = f2 == ["2026-11-10", "2026-10-27"] and any(
        x["fecha"] == "2026-11-03" and x["motivo"] == "post_festivo" for x in s2.saltados)
    out.append(criterio("A2", "Martes 24 con n=2 y 03/11 saltado como post_festivo", ", ".join(f2),
                        "10/11, 27/10 + 03/11 post_festivo", ok2))
    # R2 = D−364 y R3 = R1−364 en `referencias` del perfil jamer
    refs = corridas["columnas"]["jamer"]["por_paso"]["nivel"].result.diagnostics.get("referencias", {})
    bien = 0
    for f in FECHAS_REPLICA:
        r = refs.get(f) or {}
        r1 = (r.get("r1") or {}).get("fecha")
        r2 = (r.get("r2") or {}).get("fecha")
        r3 = (r.get("r3") or {}).get("fecha")
        e2 = (date.fromisoformat(f) - timedelta(days=364)).isoformat()
        e3 = (date.fromisoformat(r1) - timedelta(days=364)).isoformat() if r1 else None
        bien += int(r2 == e2 and r3 == e3)
    out.append(criterio("A2b", "R2 = D−364 y R3 = R1−364 en `referencias`", f"{bien}/7", "7/7", bien == 7))
    f_fest = seleccionar_recientes("2026-11-16", reales, ctx.fecha_corte, festivos, n=1)
    f_post = seleccionar_recientes("2026-11-17", reales, ctx.fecha_corte, festivos, n=1)
    a3 = (f_fest.fechas[0].isoformat() if f_fest.fechas else None, f_post.fechas[0].isoformat() if f_post.fechas else None)
    out.append(criterio("A3", "Objetivo festivo 16/11 -> 02/11; post-festivo 17/11 -> 03/11", f"{a3[0]}, {a3[1]}",
                        "02/11, 03/11", a3 == ("2026-11-02", "2026-11-03")))
    return out


def criterios_a_reproducibilidad(insumos: Dict[str, Any], corridas: Dict[str, Any]) -> Dict[str, str]:
    """Sin `perfil` el resultado es idéntico al de pasar params por defecto / vacíos y es determinista."""
    ctx = corridas["ctx"]
    provider = bta.proveedor_desde_curvas(insumos["base"])
    r_none = pipeline.run_pipeline(ctx, provider, params=None)[-1].result.curves
    r_vacio = pipeline.run_pipeline(ctx, provider, params=perfiles.expandir(None, {}))[-1].result.curves
    ref = corridas["columnas"]["mesa_hoy"]["final"]
    iguales = all(np.array_equal(np.asarray(r_none[f]), np.asarray(ref[f])) and
                  np.array_equal(np.asarray(r_vacio[f]), np.asarray(ref[f])) for f in ref)
    return criterio("A4", "Sin `perfil` = comportamiento de hoy (igualdad exacta de `final_curves`, 3 corridas)",
                    "idéntico" if iguales else "difiere", "idéntico", iguales,
                    "No compara con corridas históricas guardadas (no existen para esta semana); "
                    "la regresión byte a byte contra el comportamiento previo vive en los tests de router/steps.")


def criterios_b(corridas: Dict[str, Any], verdad: Curves) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, Any]]]:
    cols = corridas["columnas"]
    err = {n: error_energia(c["final"], verdad, FECHAS_REPLICA) for n, c in cols.items()}
    j = err["jamer"]
    out: List[Dict[str, str]] = []
    ok1 = abs(j["semana_pct"]) <= 0.5 and j["n_le_1"] == 7 and j["n_le_05"] >= 5
    out.append(criterio("B1", "Nivel `jamer`: Σ semana ≤0.5 %, 7/7 días ≤1.0 %, ≥5/7 ≤0.5 %",
                        f"Σsem {j['semana_pct']:+.2f} %, {j['n_le_1']}/7 ≤1 %, {j['n_le_05']}/7 ≤0.5 %",
                        "|Σ|≤0.5 %; 7/7; ≥5/7", ok1))
    # B2: cada día entre ΣR1·(1−0.5 %) y ΣR1·(1+1.5 %)
    fuera = []
    for f in FECHAS_REPLICA:
        e_r1 = _r1_energia(cols["jamer"], f)
        e = sum(cols["jamer"]["final"][f])
        if e_r1 is None or not (e_r1 * 0.995 <= e <= e_r1 * 1.015):
            fuera.append(f)
    out.append(criterio("B2", "Cada día `jamer` en [ΣR1·0.995, ΣR1·1.015]", f"{7 - len(fuera)}/7 dentro", "7/7",
                        not fuera, f"fuera: {fuera}" if fuera else ""))
    mj, mm, mc = err["jamer"]["mape_pct"], err["mesa_hoy"]["mape_pct"], err["crudo"]["mape_pct"]
    out.append(criterio("B3", "MAPE de energía diaria vs Jamer: `jamer` < Mesa hoy < crudo",
                        f"jamer {mj:.2f} %; Mesa hoy {mm:.2f} %; crudo {mc:.2f} %", "jamer < mesa_hoy < crudo",
                        mj < mm < mc,
                        "El crudo es sintético: su error de nivel es el de una base ingenua, no el del champion."))
    dias = cols["jamer"]["por_paso"]["nivel"].result.diagnostics.get("dias", {})
    gs = [dias[f].get("g_usado") for f in FECHAS_REPLICA if f in dias]
    e_r1 = [_r1_energia(cols["jamer"], f) for f in FECHAS_REPLICA]
    e_obj = [dias[f].get("E_obj") for f in FECHAS_REPLICA if f in dias]
    g_sem = (sum(e_obj) / sum(e_r1) - 1) * 100 if all(x is not None for x in e_obj + e_r1) and e_r1 else float("nan")
    finitos = all(g is not None and math.isfinite(g) for g in gs) and len(gs) == 7
    ok4 = finitos and 0.2 <= g_sem <= 0.9 and all(-0.01 <= g <= 0.02 for g in gs)
    out.append(criterio("B4", "`g_usado` semanal en [0.2, 0.9] % y cada día en [g_min, g_max], sin NaN",
                        f"g semanal {g_sem:+.3f} %; por día {min(gs) * 100:+.2f}…{max(gs) * 100:+.2f} %" if finitos else "no finito",
                        "[0.2, 0.9] %", ok4))
    return out, err


def criterios_c(corridas: Dict[str, Any], verdad: Curves) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, Any]]]:
    met = {n: metricas_periodo(c["final"], verdad, FECHAS_REPLICA) for n, c in corridas["columnas"].items()}
    m = met["jamer"]
    out = [
        criterio("C1", "MAE por periodo `jamer` (tramos=off)", f"{m['mae']:.2f} MW", f"≤ {C_MAE:g} MW", m["mae"] <= C_MAE),
        criterio("C2", "P95 |error| por periodo", f"{m['p95']:.2f} MW", f"≤ {C_P95:g} MW", m["p95"] <= C_P95),
        criterio("C3", "Máx |error| por periodo", f"{m['max']:.2f} MW", f"≤ {C_MAX:g} MW", m["max"] <= C_MAX),
        criterio("C4", "|Sesgo medio por día| (máx de los 7)", f"{m['sesgo_max_abs']:.2f} MW", f"≤ {C_SESGO:g} MW",
                 m["sesgo_max_abs"] <= C_SESGO),
        criterio("C5", "RMSE de forma (Σ igualada a Jamer), agrupado, % de la media del día",
                 f"{m['rmse_forma_pct']:.2f} %", f"≤ {C_RMSE_PCT:g} %", m["rmse_forma_pct"] <= C_RMSE_PCT,
                 "Definición fijada antes de medir: RMSE agrupado sobre los 168 periodos."),
        criterio("C6", "Correlación por día (mínima de los 7)", f"{m['corr_min']:.4f}", f"≥ {C_CORR}", m["corr_min"] >= C_CORR),
    ]
    mt = met["jamer_tramos"]
    mejor = mt["mae"] < m["mae"]
    out.append(criterio("C7", "`tramos=modelo_anual` solo se recomienda si MAE < MAE `off`",
                        f"tramos {mt['mae']:.2f} vs off {m['mae']:.2f} MW", "MAE tramos < MAE off", mejor,
                        "NO concluyente: con base sintética es solo la mecánica; y el real de la semana aún no existe, "
                        "así que la 2ª condición (no ser peor contra el real) queda para la fase REAL."))
    return out, met


def _tramos_diag(col: Dict[str, Any]) -> Dict[str, List[Tuple[int, int]]]:
    dias = col["por_paso"]["forma"].result.diagnostics.get("dias", {})
    return {f: [(int(t["desde"]), int(t["hasta"])) for t in (dias.get(f) or {}).get("tramos_retenidos") or []]
            for f in FECHAS_REPLICA}


def criterios_d(corridas: Dict[str, Any], esperados: Dict[str, Tuple[int, int]]) -> List[Dict[str, str]]:
    """`esperados` {fecha: (desde, hasta)}: tramos del manual, del fixture local ignorado por git."""
    ret = _tramos_diag(corridas["columnas"]["jamer_tramos"])
    out: List[Dict[str, str]] = []
    hallados, falsos = 0, []
    for f, (a, b) in esperados.items():
        ok = any(abs(x - a) <= 1 and abs(y - b) <= 1 for x, y in ret.get(f, []))
        hallados += int(ok)
    for f in FECHAS_REPLICA:
        if f not in esperados and ret.get(f):
            falsos.append(f"{DIAS_ES[FECHAS_REPLICA.index(f)]} " + ",".join(f"P{x}–P{y}" for x, y in ret[f]))
    etiqueta = ", ".join(f"{DIAS_ES[FECHAS_REPLICA.index(f)]} P{a}–P{b}" for f, (a, b) in esperados.items())
    out.append(criterio("D1", f"`tramos_retenidos` incluye {etiqueta} (±1)",
                        f"{hallados}/3; falsos tramos: {'; '.join(falsos) if falsos else 'ninguno'}", "3/3",
                        hallados == 3,
                        "Base SINTÉTICA: solo prueba la mecánica de la regla, no la hipótesis (PLAN_IMPL §10 R4). "
                        "`tramos` queda `off`."))
    # valle P1–P5 de lun/mar ≤ 15 MW sobre R1
    peor = 0.0
    for f in ("2026-11-23", "2026-11-24"):
        for nombre in ("jamer", "jamer_tramos"):
            col = corridas["columnas"][nombre]
            refs = col["por_paso"]["nivel"].result.diagnostics.get("referencias", {}).get(f) or {}
            r1 = (refs.get("r1") or {}).get("curva")
            if r1 is None:
                continue
            exceso = max(col["final"][f][h] - r1[h] for h in range(5))
            peor = max(peor, exceso)
    out.append(criterio("D2", "Valle P1–P5 de lun/mar ≤ 15 MW sobre R1 (jamer y jamer_tramos)",
                        f"máx exceso {peor:.1f} MW", "≤ 15 MW", peor <= 15.0))
    return out


def saltos_finales(ctx, curvas: Curves, banda: float = BANDA_SALTO_REPLICA_MW) -> Dict[str, Any]:
    """Alerta SALTO_ENTRE_DIAS sobre curvas dadas (reutiliza el paso `control`, sin duplicar la regla)."""
    prev = {f: list(curvas[f]) for f in FECHAS_REPLICA}
    res = paso_control.step(ctx, prev, {"banda_salto_mw": banda})
    pares = (res.diagnostics.get("saltos") or {}).get("pares", [])
    emitidos = set()
    for a in res.alerts:
        if a.codigo == "SALTO_ENTRE_DIAS":
            for q in (a.detalle or {}).get("pares", []):
                emitidos.add((q["de"], q["a"]))
    return {"pares": pares, "emitidos": emitidos, "alertas": [a.codigo for a in res.alerts]}


def criterios_e(ctx, corridas: Dict[str, Any], verdad: Curves) -> List[Dict[str, str]]:
    """E con las curvas finales de Jamer (el criterio de PLAN_IMPL) y, aparte, con las de la Mesa-perfil."""
    out: List[Dict[str, str]] = []
    pares_esp = {
        ("2026-11-23", "2026-11-24"): True, ("2026-11-24", "2026-11-25"): False,
        ("2026-11-25", "2026-11-26"): False, ("2026-11-26", "2026-11-27"): False,
    }
    for etiqueta, curvas, cid in (("curvas finales de Jamer", verdad, "E1"),
                                  ("curvas de la Mesa `jamer`", corridas["columnas"]["jamer"]["final"], "E2")):
        r = saltos_finales(ctx, curvas, BANDA_SALTO_REPLICA_MW)
        ok = all((par in r["emitidos"]) == esperado for par, esperado in pares_esp.items())
        det = "; ".join(f"{DIAS_ES[FECHAS_REPLICA.index(q['a'])]}: dif {q['dif_mw']:+.1f}" for q in r["pares"]
                        if (q["de"], q["a"]) in pares_esp)
        out.append(criterio(cid, f"SALTO_ENTRE_DIAS (banda {BANDA_SALTO_REPLICA_MW:g}): emite lun→mar y no mar→mié, mié→jue, jue→vie — {etiqueta}",
                            det, "lun→mar sí; resto no", ok,
                            f"La banda {BANDA_SALTO_REPLICA_MW:g} MW es provisional (CAL)." if cid == "E1" else
                            "Informativo: jue→vie queda cerca de la banda; depende del nivel base sintético."))
    return out


def indicador_f1(insumos: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Checksums F1: (P/R − 1)·100 de las semanas de Jamer contra D−364 del histórico real."""
    verdad = {**insumos["pm"], **insumos["verdad"]}
    e_real = data.energia_diaria(insumos["historico"])

    def serie(desde: str, hasta: str) -> Tuple[float, float]:
        d0, d1 = date.fromisoformat(desde), date.fromisoformat(hasta)
        fin = ref = 0.0
        d = d0
        while d <= d1:
            f = d.isoformat()
            r = (d - timedelta(days=364)).isoformat()
            fin += sum(verdad[f])
            ref += e_real[r]
            d += timedelta(days=1)
        return fin, ref

    tramos = {
        "sem 15–21/11": ("2026-11-15", "2026-11-21"), "sem 22–28/11": ("2026-11-22", "2026-11-28"),
        "dom 29/11": ("2026-11-29", "2026-11-29"), "sem 23–29/11": ("2026-11-23", "2026-11-29"),
    }
    filas = []
    for k, (desde, hasta) in tramos.items():
        fin, ref = serie(desde, hasta)
        esp_fin, esp_ref, esp_pct = insumos["fixture"]["f1_checksums"][k]
        pct = crecimiento_pct(fin, ref)
        filas.append({"tramo": k, "pct": pct, "pct_esperado": esp_pct,
                      "dif_fin_pct": 100 * (fin / esp_fin - 1), "dif_ref_pct": 100 * (ref / esp_ref - 1)})
    return filas


def criterios_f(ctx, insumos: Dict[str, Any], corridas: Dict[str, Any], verdad: Curves) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    filas = indicador_f1(insumos)
    # tolerancia 0.05 pp sobre el % (0.1 pp cuando el esperado viene redondeado a 1 decimal) y Σ al 0.01 %
    def pasa(f: Dict[str, Any]) -> bool:
        tol = 0.1 if f["pct_esperado"] * 10 == round(f["pct_esperado"] * 10) else 0.05
        return abs(f["pct"] - f["pct_esperado"]) <= tol and abs(f["dif_fin_pct"]) < 0.01 and abs(f["dif_ref_pct"]) < 0.01

    sem = [f for f in filas if f["tramo"] == "sem 23–29/11"][0]
    resto = [f for f in filas if f["tramo"] != "sem 23–29/11"]
    det = "; ".join(f"{f['tramo']} {f['pct']:.2f} % (esp {f['pct_esperado']})" for f in resto)
    out.append(criterio("F1a", "Indicador (P/R−1)·100, R = D−364, reproduce los checksums de sem 15–21, sem 22–28 y dom 29", det,
                        "±0.05 pp (±0.1 en los de 1 decimal) y Σ al 0.01 %", all(pasa(f) for f in resto),
                        "Se evalúa sobre el histórico `D` real + la verdad de Jamer (P/PM), no sobre la salida de la Mesa."))
    esp_fin, esp_ref, esp_pct = insumos["fixture"]["f1_checksums"]["sem 23–29/11"]
    alt = 100 * (esp_fin / esp_ref - 1)
    out.append(criterio("F1b", f"Indicador de la semana 23–29/11 = +{esp_pct:.2f} % (checksum del fixture {esp_fin}/{esp_ref})",
                        f"{sem['pct']:.2f} % (Σ ref D−364 difiere {sem['dif_ref_pct']:+.3f} % del Σ del fixture)",
                        f"{esp_pct:.2f} % ±0.05 pp", pasa(sem),
                        f"Checksum del plan mal rotulado: el Σ de referencia del fixture es el del 23–29/11/2025 (−365 d), la "
                        f"referencia de la sem 22–28; con R2 = D−364 (24–30/11/2025) la semana da {sem['pct']:.2f} %. "
                        f"Cociente del fixture = {alt:.2f} %."))
    ctx_jamer = corridas["columnas"]["jamer"]
    # control con banda 3 pp sobre las curvas de Jamer y sobre las de la Mesa-perfil; y sin banda
    def correr(curvas: Curves, banda: Optional[float]):
        res = paso_control.step(ctx, {f: list(curvas[f]) for f in FECHAS_REPLICA}, {"banda_crecimiento_pp": banda})
        return res, [a.codigo for a in res.alerts]
    res_off, cod_off = correr(verdad, None)
    res_on, cod_on = correr(verdad, 3.0)
    res_mesa, cod_mesa = correr(ctx_jamer["final"], 3.0)
    publica = (res_off.diagnostics.get("crecimiento") or {}).get("semana_pct") is not None
    ok2 = publica and "CRECIMIENTO_FUERA_DE_TENDENCIA" not in cod_off and "CRECIMIENTO_FUERA_DE_TENDENCIA" not in cod_on
    crec = res_on.diagnostics.get("crecimiento") or {}
    out.append(criterio("F2", "`CRECIMIENTO_FUERA_DE_TENDENCIA`: apagada sin banda (indicador sí se publica) y con 3 pp no alerta en la semana de Jamer",
                        f"semana {crec.get('semana_pct', float('nan')):.2f} % vs Y {crec.get('Y_pct', float('nan')):.2f} %; "
                        f"alerta con 3 pp: {'sí' if 'CRECIMIENTO_FUERA_DE_TENDENCIA' in cod_on else 'no'}; "
                        f"sobre la Mesa `jamer`: {'sí' if 'CRECIMIENTO_FUERA_DE_TENDENCIA' in cod_mesa else 'no'}",
                        "no alerta", ok2))
    # F3: acumulado-anual sobre fixture sintético
    aprobada = {f"2026-03-{d:02d}": 1000.0 * (1.0 + e) for d, e in zip(range(1, 9), [0.0, 0.02, 0.06, -0.07, 0.049, 0.051, 0.0, 0.10])}
    real = {f: 1000.0 for f in aprobada}
    sin_cupo = acumulado.calcular_acumulado_anual(aprobada, real, 2026, banda_pct=5.0, cupo_dias=None)
    con_cupo = acumulado.calcular_acumulado_anual(aprobada, real, 2026, banda_pct=5.0, cupo_dias=4)
    ok3 = sin_cupo["dias_fuera"] == 4 and sin_cupo["alerta"] is None and con_cupo["alerta"] is not None
    out.append(criterio("F3", "Acumulado anual: cuenta días con |error|>banda en fixture sintético y no emite sin cupo",
                        f"dias_fuera={sin_cupo['dias_fuera']} (esperado 4); alerta sin cupo: "
                        f"{'sí' if sin_cupo['alerta'] else 'no'}; con cupo 4: {'sí' if con_cupo['alerta'] else 'no'}",
                        "4; no; sí", ok3, "El cupo real (25 días) no está confirmado; apagado por defecto."))
    return out


def criterios_g(corridas: Dict[str, Any], g1_esperado: Dict[str, Tuple[float, float]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    col = corridas["columnas"]["jamer"]
    forma = col["por_paso"]["forma"].result.diagnostics.get("dias", {})
    nivel = col["por_paso"]["nivel"].result.diagnostics.get("dias", {})
    fallos, detalle = [], []
    for f, (e_esp, p1_esp) in g1_esperado.items():
        info = forma.get(f) or {}
        delta = info.get("delta_plano_mw")
        eb = info.get("energia_base")
        refs = col["por_paso"]["nivel"].result.diagnostics["referencias"][f]["r1"]["curva"]
        # S_P1 reconstruido del pipeline = final_P1 − δ; solo comparable si la base sintética trae el mismo Σ
        if delta is None or eb is None:
            fallos.append(f)
            continue
        s_p1 = col["final"][f][0] - delta
        puro = refs[0] * e_esp / sum(refs)  # fórmula "Ajustado" con el Σ del fixture
        coincide_base = abs(eb - e_esp) < 0.15
        detalle.append(f"{DIAS_ES[FECHAS_REPLICA.index(f)]} S_P1 {s_p1:.1f}" +
                       ("" if coincide_base else f" (base Σ difiere: fórmula {puro:.1f})") + f" vs {p1_esp}")
        objetivo = s_p1 if coincide_base else puro
        consistente = abs(s_p1 - refs[0] * eb / sum(refs)) < 1e-6  # el pipeline aplica la fórmula con su propio E_base
        if abs(objetivo - p1_esp) > 0.1 or not consistente:
            fallos.append(f)
    out.append(criterio("G1", "`Ajustado` (forma R1 al total del modelo) reproduce P1 lun/vie/sáb/dom del fixture (0.1 MW)",
                        "; ".join(detalle), "±0.1 MW", not fallos,
                        "El domingo: la base sintética trae otro Σ; se compara la fórmula con el Σ del fixture y se verifica que el pipeline aplique la misma fórmula con su E_base."))
    max_dif, max_delta = 0.0, 0.0
    for f in FECHAS_REPLICA:
        e_d = (nivel.get(f) or {}).get("E")
        suma = sum(col["final"][f])
        info = forma.get(f) or {}
        if e_d is None:
            continue
        max_dif = max(max_dif, abs(suma - e_d))
        if info.get("delta_plano_mw") is not None and info.get("energia_base") is not None:
            esperado = info["energia_base"] + info["delta_plano_mw"] * len(info.get("periodos_con_gancho", []))
            max_delta = max(max_delta, abs(suma - esperado))
    out.append(criterio("G2", "Delta plano: Σ final = E(d) (1e-6) y Σ = E_base + δ·n_gancho",
                        f"máx |Σ−E| {max_dif:.2e}; máx |Σ−(E_base+δ·n)| {max_delta:.2e}", "≤ 1e-6", max_dif <= 1e-6 and max_delta <= 1e-6))
    return out


def evaluar_replica(insumos: Dict[str, Any]) -> Dict[str, Any]:
    corridas = correr_replica(insumos)
    ctx, verdad = corridas["ctx"], insumos["verdad"]
    cb, err = criterios_b(corridas, verdad)
    cc, met = criterios_c(corridas, verdad)
    criterios = (criterios_a(ctx, corridas, insumos["fixture"]["r1_esperado"]) + [criterios_a_reproducibilidad(insumos, corridas)] + cb + cc
                 + criterios_d(corridas, insumos["fixture"]["d_tramos_esperados"]) + criterios_e(ctx, corridas, verdad)
                 + criterios_f(ctx, insumos, corridas, verdad) + criterios_g(corridas, insumos["fixture"]["g1_esperado_p1"]))
    return {"corridas": corridas, "criterios": criterios, "energia": err, "periodo": met,
            "f1": indicador_f1(insumos)}


# ---------------------------------------------------------------------------
# Parte 2: generalización ene–jun 2026
# ---------------------------------------------------------------------------

def configs_generalizacion() -> Dict[str, Tuple[Optional[str], Dict[str, Dict[str, Any]]]]:
    """Variantes: la Mesa de hoy, el perfil `jamer` (g interanual) y su barrido de calibración."""
    return {
        "mesa_hoy": (None, {}),
        "jamer": ("jamer", {}),
        "jamer_k05": ("jamer", {"nivel": {"g_k": 0.5}}),
        "jamer_k15": ("jamer", {"nivel": {"g_k": 1.5}}),
        "jamer_gfijo05": ("jamer", {"nivel": {"g_fuente": "fijo", "g_fijo": 0.005}}),
        "jamer_g0": ("jamer", {"nivel": {"g_fuente": "fijo", "g_fijo": 0.0}}),
    }


CANDIDATAS_CALIBRACION = ("jamer", "jamer_k05", "jamer_k15", "jamer_gfijo05", "jamer_g0")


def semanas_ene_jun(historico: pd.DataFrame, anio: int = 2026, desde_mes: int = 1, hasta_mes: int = 6) -> List[date]:
    """Lunes de semanas completas (7 días reales válidos) con el lunes entre ene y jun del año; exige el sábado de corte."""
    validas = set(data.curvas_reales(historico))
    out: List[date] = []
    d = date(anio, desde_mes, 1)
    while d.weekday() != 0:
        d += timedelta(days=1)
    while d.month <= hasta_mes and d.year == anio:
        dias = [(d + timedelta(days=i)).isoformat() for i in range(7)]
        corte = (d - timedelta(days=2)).isoformat()
        if all(x in validas for x in dias) and corte in validas:
            out.append(d)
        d += timedelta(days=7)
    return out


def partir_calibracion(semanas: Sequence[date], fraccion: float = CAL_FRACCION) -> Tuple[List[date], List[date]]:
    """60 % más antiguo para calibrar y 40 % más reciente para reportar (orden temporal, sin mezclar)."""
    s = sorted(semanas)
    n_cal = int(round(len(s) * fraccion))
    n_cal = min(max(n_cal, 1), len(s) - 1) if len(s) >= 2 else len(s)
    return s[:n_cal], s[n_cal:]


def evaluar_semana_gen(
    ucp: str, lunes: date, historico: pd.DataFrame, reales: Dict[str, np.ndarray],
    festivos: List[str], configs: Dict[str, Tuple[Optional[str], Dict[str, Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    """Filas (config × día) de una semana con base ingenua; corte = sábado anterior, sin fuga."""
    ctx = bta.construir_contexto(ucp, lunes, historico, festivos)
    provider = bta.con_cache(bta.proveedor_ingenuo(ctx.historico))
    fechas = [d.isoformat() for d in data.dias_semana(lunes)]
    festivos_set = set(ctx.festivos)
    filas: List[Dict[str, Any]] = []
    cache_base: Optional[Curves] = None
    for nombre, (perfil, extra) in configs.items():
        records = pipeline.run_pipeline(ctx, provider, params=perfiles.expandir(perfil, extra or None))
        por = {r.paso: r for r in records}
        base_curves = por["base"].result.curves
        if any(f not in base_curves for f in fechas):
            return []
        if cache_base is None:
            cache_base = base_curves
        finales = {"ingenuo": cache_base, nombre: records[-1].result.curves} if nombre == "mesa_hoy" else {nombre: records[-1].result.curves}
        for etiqueta, curvas in finales.items():
            for f in fechas:
                d = date.fromisoformat(f)
                filas.append({
                    "ucp": ucp, "semana_inicio": lunes.isoformat(), "config": etiqueta, "fecha": f,
                    "dia_semana": d.weekday(), "tipo_dia": data.tipo_dia(d, festivos_set),
                    "clase_dia": data.clase_dia(d, festivos_set),
                    **bta.metricas_dia(curvas[f], reales[f]),
                })
    return filas


def correr_generalizacion(
    ucps: Sequence[str], historicos: Optional[Dict[str, pd.DataFrame]] = None, anio: int = 2026,
    progreso: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, List[date]]]:
    filas: List[Dict[str, Any]] = []
    semanas_por_ucp: Dict[str, List[date]] = {}
    configs = configs_generalizacion()
    for ucp in ucps:
        hist = (historicos or {}).get(ucp)
        hist = data.cargar_historico(ucp) if hist is None else data.normalizar_historico(hist, ucp=ucp)
        reales = data.curvas_reales(hist)
        festivos = data.festivos_colombia(range(int(hist["fecha"].dt.year.min()) - 1, int(hist["fecha"].dt.year.max()) + 2))
        semanas = semanas_ene_jun(hist, anio)
        semanas_por_ucp[ucp] = semanas
        for lunes in semanas:
            try:
                filas.extend(evaluar_semana_gen(ucp, lunes, hist, reales, festivos, configs))
            except AssertionError:
                raise
            except Exception as exc:  # una semana rota no tumba el backtest
                logger.warning("%s %s: %s", ucp, lunes, exc)
            if progreso:
                print(f"  {ucp} {lunes} listo", file=sys.stderr)
    return pd.DataFrame(filas), semanas_por_ucp


def wape(df: pd.DataFrame) -> float:
    return float(100 * df["abs_err_horario"].sum() / df["energia_real"].sum()) if len(df) else float("nan")


def sesgo_energia(df: pd.DataFrame) -> float:
    return float(100 * (df["energia_pred"].sum() / df["energia_real"].sum() - 1)) if len(df) else float("nan")


def tabla_wape(df: pd.DataFrame, claves: List[str]) -> pd.DataFrame:
    """WAPE (%) y sesgo de energía (%) por config y claves adicionales."""
    if df.empty:
        return pd.DataFrame()
    filas = []
    for k, g in df.groupby(["config"] + claves, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        filas.append({**dict(zip(["config"] + claves, k)), "n_dias": len(g), "wape_pct": wape(g), "sesgo_pct": sesgo_energia(g)})
    return pd.DataFrame(filas)


def elegir_calibracion(df_cal: pd.DataFrame, candidatas: Sequence[str] = CANDIDATAS_CALIBRACION) -> Optional[str]:
    """Candidata con menor WAPE en el tramo de calibración (None si no hay datos)."""
    sub = df_cal[df_cal["config"].isin(candidatas)]
    if sub.empty:
        return None
    w = {c: wape(g) for c, g in sub.groupby("config")}
    return min(w, key=w.get)


def wape_semanal(df: pd.DataFrame) -> pd.DataFrame:
    """WAPE por (ucp, semana, config) — base para las comparaciones pareadas."""
    filas = []
    for (u, s, c), g in df.groupby(["ucp", "semana_inicio", "config"], sort=False):
        filas.append({"ucp": u, "semana_inicio": s, "config": c, "wape_pct": wape(g)})
    return pd.DataFrame(filas)


def comparacion_pareada(df: pd.DataFrame, a: str, b: str, semilla: int = 7, n_boot: int = 2000) -> Dict[str, Any]:
    """Diferencia de WAPE semanal a − b (negativa = a mejora a b), semanas a favor de a y IC bootstrap 90 %."""
    ws = wape_semanal(df)
    pa = ws[ws["config"] == a].set_index(["ucp", "semana_inicio"])["wape_pct"]
    pb = ws[ws["config"] == b].set_index(["ucp", "semana_inicio"])["wape_pct"]
    comunes = pa.index.intersection(pb.index)
    dif = (pa.loc[comunes] - pb.loc[comunes]).to_numpy()
    if len(dif) == 0:
        return {"n": 0, "dif_media_pp": float("nan"), "a_favor": 0, "ic90": (float("nan"), float("nan"))}
    rng = np.random.default_rng(semilla)
    boots = np.array([rng.choice(dif, size=len(dif), replace=True).mean() for _ in range(n_boot)])
    return {"n": int(len(dif)), "dif_media_pp": float(dif.mean()), "a_favor": int((dif < 0).sum()),
            "ic90": (float(np.percentile(boots, 5)), float(np.percentile(boots, 95)))}


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def _tabla(filas: List[Dict[str, Any]], cols: Sequence[str], decimales: int = 2) -> str:
    if not filas:
        return "_(sin datos)_\n"
    df = pd.DataFrame(filas)[list(cols)]
    df = df.apply(lambda c: c.map(lambda v: v.replace("|", "/") if isinstance(v, str) else v))  # `|` rompe la tabla md
    return bta.tabla_md(df, decimales)


def _md_replica(res: Dict[str, Any]) -> List[str]:
    corridas = res["corridas"]
    cols = corridas["columnas"]
    out: List[str] = ["## Parte 1 — Réplica de la semana 23–29/11/2026 (Antioquia)\n"]
    out.append(
        "Verdad = sesión guardada por Jamer (tipo P). **La columna `crudo` y la base de todas las corridas son SINTÉTICAS** "
        "(forma de R1 reescalada a los totales del modelo de la reunión; el champion del 18/11 no es recuperable, "
        "`fase0/REPORTE.md`). Por eso: `w>0` (jamer_modelo) y `tramos` NO son evaluables de verdad y el nivel del crudo "
        "es el de la reunión, no el que daría el modelo hoy. Corte 14/11, sin fuga verificada.\n"
    )
    out.append("### Veredicto por criterio (PLAN_IMPL §7 A–G)\n")
    out.append(_tabla(res["criterios"], ["id", "criterio", "valor", "umbral", "veredicto", "nota"]))
    n_falla = sum(c["veredicto"] == "falla" for c in res["criterios"])
    n_pasa = sum(c["veredicto"] == "pasa" for c in res["criterios"])
    out.append(f"\nResumen: {n_pasa} pasan, {n_falla} fallan, "
               f"{sum(c['veredicto'] == 'no evaluable' for c in res['criterios'])} no evaluables.\n")
    out.append("### Columnas: error vs Jamer (modelo crudo sintético / Mesa hoy / perfil jamer y jamer_modelo)\n")
    filas = []
    for n in ("crudo", "mesa_hoy", "jamer", "jamer_modelo"):
        e, m = res["energia"][n], res["periodo"][n]
        filas.append({"columna": n, "Σsem %": e["semana_pct"], "MAPE energía día %": e["mape_pct"],
                      "días ≤1 %": e["n_le_1"], "MAE MW": m["mae"], "P95 MW": m["p95"], "máx MW": m["max"],
                      "sesgo máx día MW": m["sesgo_max_abs"], "RMSE forma %": m["rmse_forma_pct"], "corr mín": m["corr_min"]})
    out.append(_tabla(filas, list(filas[0].keys())))
    out.append("\n### Variantes experimentales (rotuladas; base sintética)\n")
    filas = []
    for n in ("jamer", "jamer_g_fijo", "jamer_g0", "jamer_tramos"):
        e, m = res["energia"][n], res["periodo"][n]
        filas.append({"variante": n, "Σsem %": e["semana_pct"], "MAPE energía día %": e["mape_pct"],
                      "MAE MW": m["mae"], "P95 MW": m["p95"], "máx MW": m["max"], "RMSE forma %": m["rmse_forma_pct"]})
    out.append(_tabla(filas, list(filas[0].keys())))
    out.append("\n### Error de energía diaria vs Jamer (%) por día\n")
    filas = []
    for n in ("crudo", "mesa_hoy", "jamer", "jamer_modelo"):
        filas.append({"columna": n, **{DIAS_ES[i]: res["energia"][n]["dia_pct"][i] for i in range(7)}})
    out.append(_tabla(filas, ["columna"] + DIAS_ES))
    out.append("\n### MAE por periodo vs Jamer (MW) por día\n")
    filas = []
    for n in ("mesa_hoy", "jamer", "jamer_modelo"):
        filas.append({"columna": n, **{DIAS_ES[i]: res["periodo"][n]["mae_dia"][i] for i in range(7)}})
    out.append(_tabla(filas, ["columna"] + DIAS_ES))
    out.append("\n### Indicador de crecimiento (P/R−1)·100 vs checksums (F1)\n")
    out.append(_tabla(res["f1"], ["tramo", "pct", "pct_esperado", "dif_fin_pct", "dif_ref_pct"], 3))
    alertas = {n: ", ".join(c["alertas"]) or "—" for n, c in cols.items()}
    out.append("\n### Alertas emitidas por columna\n")
    out.append("`BASE_INCOMPLETA`/`BRECHA_INVALIDA` salen porque la base sintética solo trae 23–29/11 (no los días de brecha 15–22/11); no son un fallo del perfil.\n")
    out.append(_tabla([{"columna": n, "alertas": a} for n, a in alertas.items()], ["columna", "alertas"]))
    return out


def _md_generalizacion(df: pd.DataFrame, semanas_por_ucp: Dict[str, List[date]]) -> List[str]:
    out: List[str] = ["## Parte 2 — Generalización (semanas completas ene–jun 2026, base `ingenuo`)\n"]
    if df.empty:
        return out + ["_(sin datos: no se pudo correr)_\n"]
    out.append(
        "Base `ingenuo` (mismo día de semana de la última semana real): **no hay modelo, así que `w` (peso del modelo) "
        "NO es evaluable** y `jamer_modelo` no se corre. Con base ingenua `nivel` reciente parte de E_base = ΣR1, por lo "
        "que el perfil solo se diferencia de `ingenuo` por la selección R1 por clase de día, g y la forma/delta plano. "
        "Sin red, sin refrescar datos (se lee `data/raw` solo para lectura). Corte = sábado anterior a cada semana.\n"
    )
    for ucp, semanas in semanas_por_ucp.items():
        if not semanas:
            continue
        cal, rep = partir_calibracion(semanas)
        out.append(f"- {ucp}: {len(semanas)} semanas ({semanas[0]} … {semanas[-1]}); calibración {len(cal)} "
                   f"({cal[0]} … {cal[-1]}), reporte {len(rep)} ({rep[0]} … {rep[-1]}).\n")
    cal_s = {u: set(c.isoformat() for c in partir_calibracion(s)[0]) for u, s in semanas_por_ucp.items() if s}
    es_cal = df.apply(lambda r: r["semana_inicio"] in cal_s.get(r["ucp"], set()), axis=1)
    df_cal, df_rep = df[es_cal], df[~es_cal]

    out.append("### Calibración (60 % antiguo): WAPE por config\n")
    t = tabla_wape(df_cal, [])
    out.append(bta.tabla_md(t.sort_values("wape_pct")))
    elegida = elegir_calibracion(df_cal)
    out.append(f"\nConfig del perfil elegida en calibración (menor WAPE entre {list(CANDIDATAS_CALIBRACION)}): **`{elegida}`**.\n")
    if elegida is None:
        return out
    comp = ["ingenuo", "mesa_hoy", "jamer", elegida]
    comp = list(dict.fromkeys(comp))

    def sec(titulo: str, d: pd.DataFrame, claves: List[str]) -> None:
        out.append(f"\n### {titulo}\n")
        tt = tabla_wape(d[d["config"].isin(comp)], claves)
        piv = tt.pivot_table(index=claves, columns="config", values="wape_pct", sort=False) if claves else \
            tt.set_index("config")[["wape_pct"]].T
        piv = piv[[c for c in comp if c in piv.columns]].reset_index() if claves else piv.reset_index(drop=True)
        out.append(bta.tabla_md(piv))

    sec("Reporte (40 % reciente): WAPE % por mercado", df_rep, ["ucp"])
    sec("Reporte (40 % reciente): WAPE % por tipo de día (todos los mercados)", df_rep, ["tipo_dia"])
    out.append("\n### Reporte: WAPE % por mercado × tipo de día\n")
    tt = tabla_wape(df_rep[df_rep["config"].isin(comp)], ["ucp", "tipo_dia"])
    piv = tt.pivot_table(index=["ucp", "tipo_dia"], columns="config", values="wape_pct", sort=False)
    out.append(bta.tabla_md(piv[[c for c in comp if c in piv.columns]].reset_index()))
    out.append("\n### Todas las semanas (cal + rep): WAPE % por config\n")
    out.append(bta.tabla_md(tabla_wape(df, []).sort_values("wape_pct")))
    out.append("\n### Comparaciones pareadas por semana (WAPE semanal a − b; negativo = a mejora a b)\n")
    filas = []
    for etiqueta, d in (("reporte", df_rep), ("todas", df)):
        for a, b in ((elegida, "mesa_hoy"), (elegida, "ingenuo"), ("jamer", "mesa_hoy"), ("jamer", "ingenuo"),
                     ("mesa_hoy", "ingenuo")):
            r = comparacion_pareada(d, a, b)
            filas.append({"tramo": etiqueta, "a": a, "b": b, "n_sem": r["n"], "dif_media_pp": r["dif_media_pp"],
                          "semanas_a_favor": r["a_favor"], "IC90_inf": r["ic90"][0], "IC90_sup": r["ic90"][1]})
    out.append(_tabla(filas, ["tramo", "a", "b", "n_sem", "dif_media_pp", "semanas_a_favor", "IC90_inf", "IC90_sup"], 3))
    # veredicto automático
    out.append("\n### Veredicto honesto (riesgo #1: replicar a Jamer ≠ bajar el error)\n")
    for a_nombre, a in (("perfil `jamer`", "jamer"), (f"perfil calibrado `{elegida}`", elegida)):
        for b_nombre, b in (("Mesa hoy", "mesa_hoy"), ("ingenuo", "ingenuo")):
            r = comparacion_pareada(df_rep, a, b)
            w_a, w_b = wape(df_rep[df_rep["config"] == a]), wape(df_rep[df_rep["config"] == b])
            mejora = w_a < w_b
            signif = r["n"] > 0 and (r["ic90"][1] < 0 or r["ic90"][0] > 0)
            out.append(f"- {a_nombre} vs {b_nombre} (reporte): WAPE {w_a:.2f} % vs {w_b:.2f} % -> "
                       f"{'MEJORA' if mejora else 'NO MEJORA'}"
                       f"{' (IC90 de la diferencia semanal no cruza 0)' if signif else ' (no distinguible de cero con IC90)'}; "
                       f"a favor en {r['a_favor']}/{r['n']} semanas.\n")
    # celdas donde el perfil no mejora a la Mesa de hoy (honestidad por mercado y tipo de día)
    peores = []
    for (ucp, tipo), g in df_rep[df_rep["config"].isin(["mesa_hoy", "jamer"])].groupby(["ucp", "tipo_dia"]):
        gm, gj = g[g["config"] == "mesa_hoy"], g[g["config"] == "jamer"]
        if len(gm) and wape(gj) > wape(gm):
            peores.append(f"{ucp}/{tipo} ({len(gj)} días: jamer {wape(gj):.2f} % vs Mesa hoy {wape(gm):.2f} %)")
    out.append("\nCeldas del reporte (mercado/tipo de día) donde `jamer` NO mejora a Mesa hoy: "
               f"{'; '.join(peores) if peores else 'ninguna'}.\n")
    out.append("\nLectura: las diferencias de WAPE semanal entre el perfil y la Mesa de hoy son de unas décimas de punto "
               "con 10–17 semanas por tramo; el IC90 decide si hay evidencia. La base ingenua no es el modelo: que el perfil "
               "gane aquí no prueba que gane sobre el champion (ahí R1 sería peor que el modelo, WAPE vigente de la Mesa "
               "2.33 % vs 3.56 % ingenuo en `backtest_adjustments`; riesgo #1 de PLAN_IMPL).\n")
    return out


def escribir_reporte(path: Path, meta: Dict[str, Any], replica: Optional[Dict[str, Any]],
                     gen: Optional[Tuple[pd.DataFrame, Dict[str, List[date]]]]) -> None:
    out: List[str] = [
        f"# Backtest de la réplica de Jamer — {meta['fecha']}\n",
        "Generado por `scripts/backtest_jamer.py` (fase E6 de `docs/work/ajustes/replica_jamer/`). "
        "Este archivo solo trae agregados (errores, no curvas reales por periodo). "
        f"Detalle con demanda real: `reports/ajustes/jamer/{meta['detalle']}` (ignorado por git, no versionar).\n",
        "Lectura obligatoria: la parte 1 usa una base SINTÉTICA y una sola semana de un solo analista; la parte 2 usa base "
        "ingenua (no hay modelo) y semanas ene–jun 2026. Ninguna de las dos mide el valor real de `w` ni de `tramos`.\n",
    ]
    if replica is not None:
        out += _md_replica(replica)
    if gen is not None:
        out += _md_generalizacion(*gen)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def escribir_detalle(path: Path, replica: Optional[Dict[str, Any]], gen: Optional[pd.DataFrame]) -> None:
    partes: List[pd.DataFrame] = []
    if gen is not None and not gen.empty:
        partes.append(gen.assign(parte="generalizacion"))
    if replica is not None:
        filas = []
        for n, c in replica["corridas"]["columnas"].items():
            for f in FECHAS_REPLICA:
                filas.append({"parte": "replica", "config": n, "fecha": f, "energia_pred": float(sum(c["final"][f]))})
        partes.append(pd.DataFrame(filas))
    if partes:
        pd.concat(partes, ignore_index=True).to_csv(path, index=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parte", choices=("replica", "generalizacion", "todo"), default="todo")
    ap.add_argument("--ucp", nargs="+", default=["Antioquia", "Atlantico"])
    ap.add_argument("--fecha", default=None, help="sufijo YYYYMMDD del reporte (default: hoy)")
    ap.add_argument("--salida", default=str(REPORTS_DIR))
    ap.add_argument("--directorio", default=str(JAMER_DIR),
                    help="carpeta con historico_replica.csv, base_sintetica_23_29_nov.csv, verdad y fixture (ignorada por git)")
    ap.add_argument("--verdad", default=None, help=f"CSV de la sesión de Jamer (default: <directorio>/{VERDAD_NOMBRE})")
    ap.add_argument("--fixture", default=None, help=f"JSON de esperados reales (default: <directorio>/{FIXTURE_NOMBRE})")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    stamp = args.fecha or date.today().strftime("%Y%m%d")
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    replica = gen = None
    if args.parte in ("replica", "todo"):
        replica = evaluar_replica(cargar_insumos_replica(Path(args.directorio), args.verdad, args.fixture))
    if args.parte in ("generalizacion", "todo"):
        gen = correr_generalizacion(args.ucp, progreso=True)
    detalle_nombre = f"backtest_jamer_{stamp}_detalle.csv"
    JAMER_DIR.mkdir(parents=True, exist_ok=True)
    escribir_detalle(JAMER_DIR / detalle_nombre, replica, gen[0] if gen else None)
    md = salida / f"backtest_jamer_{stamp}.md"
    escribir_reporte(md, {"fecha": stamp, "detalle": detalle_nombre}, replica, gen)
    print(f"Reporte: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
