"""
Backtest de los ajustes con el modelo real y recalibración con holdout temporal (Fase 2a, parte B).

Tres subcomandos (desde la raíz del repo / worktree):

    python scripts/recalibrar_ajustes.py generar --k 1          # curvas base del modelo (sandbox, sin fuga)
    python scripts/recalibrar_ajustes.py generar --k 4          # sensibilidad: reentreno cada 4 semanas
    python scripts/recalibrar_ajustes.py evaluar                # backtest sobre el cache + ingenuo/archivo
    python scripts/recalibrar_ajustes.py recalibrar             # holdout 60/40 + reporte markdown

- `generar` usa `scripts/backtest_modelo.SandboxModelo`: por UCP, las últimas N semanas completas
  (`backtest_adjustments.semanas_objetivo`); corte = el mismo que `construir_contexto` (sábado anterior,
  o el último día completo antes). Rolling origin: con k=1 se reentrena en cada semana
  (`entrenar(ucp, corte)`); con k>1 se reentrena en la semana 0, k, 2k… y las demás usan `predecir`
  con los modelos del último origen. Las curvas se cachean en
  `reports/ajustes/curvas_modelo_k{k}_{ucp}.csv` (reanudable) y los hashes de `data/raw`, `models`,
  `data/features`, `data/processed` antes/después en `hash_k{k}_{etiqueta}.json`.
- `evaluar` corre el pipeline de ajustes (`run_pipeline`) sobre el cache con configs arbitrarias
  (α, fuente, n, alpha_por_tipo, banda) y sobre las bases `ingenuo` y `archivo` en las MISMAS semanas.
- `recalibrar` elige parámetros SOLO en las semanas de calibración (primeras ~60 %) y reporta en las de
  prueba (últimas ~40 %) y en el total.

Los CSV quedan en `reports/ajustes/` (ignorado por git).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adjustments import data, pipeline  # noqa: E402
from src.adjustments.steps import control as paso_control  # noqa: E402
from src.adjustments.steps import forma as paso_forma  # noqa: E402
from src.adjustments.steps import nivel as paso_nivel  # noqa: E402
from src.adjustments.types import Curves  # noqa: E402


def _cargar_script(nombre: str):
    """Importa un script hermano de `scripts/` como módulo (no son paquete)."""
    if nombre in sys.modules:
        return sys.modules[nombre]
    spec = importlib.util.spec_from_file_location(nombre, Path(__file__).resolve().parent / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


ba = _cargar_script("backtest_adjustments")

logger = logging.getLogger("recalibrar_ajustes")

UCPS_DEFAULT = ("Antioquia", "Atlantico")
OUT_DIR = ROOT / "reports" / "ajustes"
PERIODOS = [f"P{i}" for i in range(1, 25)]
TIPOS = ("laboral", "sabado", "domingo", "festivo")


# ---------------------------------------------------------------------------
# Cache de curvas base
# ---------------------------------------------------------------------------

COLS_CACHE = ["ucp", "k", "semana_inicio", "corte", "origen", "fecha"] + PERIODOS


def ruta_cache(k: int, ucp: str, out_dir: Path = OUT_DIR) -> Path:
    """Un archivo por (k, UCP): permite generar UCP/k en procesos paralelos sin escrituras cruzadas."""
    return Path(out_dir) / f"curvas_modelo_k{k}_{ucp}.csv"


def leer_cache(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=COLS_CACHE)
    return pd.read_csv(path, dtype={"ucp": str, "semana_inicio": str, "corte": str, "origen": str, "fecha": str})


def leer_cache_k(k: int, out_dir: Path = OUT_DIR) -> pd.DataFrame:
    """Cache completo de un k (todas las UCP)."""
    partes = [leer_cache(p) for p in sorted(Path(out_dir).glob(f"curvas_modelo_k{k}_*.csv"))]
    partes = [p for p in partes if not p.empty]
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=COLS_CACHE)


def filas_cache(ucp: str, k: int, lunes: date, corte: date, origen: date, curvas: Curves) -> List[Dict]:
    filas = []
    for f in sorted(curvas):
        fila = {"ucp": ucp, "k": k, "semana_inicio": lunes.isoformat(), "corte": corte.isoformat(),
                "origen": origen.isoformat(), "fecha": f}
        fila.update({p: float(v) for p, v in zip(PERIODOS, curvas[f])})
        filas.append(fila)
    return filas


def curvas_por_semana(cache: pd.DataFrame, ucp: str) -> Dict[str, Curves]:
    """{semana_inicio: {fecha: [P1..P24]}} de una UCP del cache."""
    out: Dict[str, Curves] = {}
    g = cache[cache["ucp"] == ucp]
    for sem, gs in g.groupby("semana_inicio", sort=True):
        out[sem] = {r.fecha: [float(getattr(r, p)) for p in PERIODOS] for r in gs.itertuples(index=False)}
    return out


def corte_de_semana(historico: pd.DataFrame, lunes: date) -> date:
    """Mismo corte que `construir_contexto`: último día completo <= sábado anterior."""
    return data.calcular_fecha_corte(historico, lunes - timedelta(days=2))


def origenes_rolling(n: int, k: int) -> List[int]:
    """Índice de la semana de origen (reentreno) de cada semana i con reentreno cada k semanas."""
    if k < 1:
        raise ValueError("k debe ser >= 1")
    return [i - i % k for i in range(n)]


def generar(ucps: Sequence[str], k: int, n_semanas: int, out_dir: Path = OUT_DIR, n_days: int = 8) -> None:
    """Genera (o completa) el cache de curvas base del modelo con rolling origin cada k semanas."""
    bm = _cargar_script("backtest_modelo")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hechas_set = set()
    for ucp in ucps:
        h = leer_cache(ruta_cache(k, ucp, out_dir))
        hechas_set |= set(zip(h["ucp"], h["semana_inicio"]))

    antes = bm.hash_arbol(ROOT)
    t_ini = time.perf_counter()
    plan: Dict[str, Tuple[pd.DataFrame, List[date]]] = {}
    for ucp in ucps:
        hist = data.cargar_historico(ucp, data_dir=ROOT / "data" / "raw")
        plan[ucp] = (hist, ba.semanas_objetivo(hist, n_semanas))

    etiqueta = "_".join(ucps)
    progreso = out_dir / f"progreso_k{k}_{etiqueta}.log"
    with bm.SandboxModelo(list(ucps)) as sb, open(progreso, "a", encoding="utf-8") as plog:
        for ucp in ucps:
            hist, semanas = plan[ucp]
            cortes = [corte_de_semana(hist, l) for l in semanas]
            origen_idx = origenes_rolling(len(semanas), k)
            for i, lunes in enumerate(semanas):
                if (ucp, lunes.isoformat()) in hechas_set:
                    continue
                corte = cortes[i]
                corte_origen = cortes[origen_idx[i]]
                dias = (lunes + timedelta(days=6) - corte).days
                t0 = time.perf_counter()
                if corte_origen == corte:
                    curvas = sb.entrenar(ucp, corte, dias)
                else:
                    if sb.modelo_corte.get(ucp) != corte_origen:
                        sb.entrenar(ucp, corte_origen, n_days)
                    curvas = sb.predecir(ucp, corte, dias)
                filas = filas_cache(ucp, k, lunes, corte, corte_origen, curvas)
                path = ruta_cache(k, ucp, out_dir)
                pd.DataFrame(filas, columns=COLS_CACHE).to_csv(path, mode="a", header=not path.exists(), index=False)
                msg = (f"{time.strftime('%H:%M:%S')} k={k} {ucp} {i + 1}/{len(semanas)} semana {lunes} "
                       f"corte {corte} origen {corte_origen} ({time.perf_counter() - t0:.1f} s)")
                plog.write(msg + "\n")
                plog.flush()
                logger.info(msg)
    despues = bm.hash_arbol(ROOT)
    dif = bm.diferencias_hash(antes, despues)
    info = {
        "k": k, "ucps": list(ucps), "n_semanas": n_semanas, "archivos_hash": len(antes),
        "hash_intacto": antes == despues, "diferencias": dif,
        "segundos": round(time.perf_counter() - t_ini, 1),
        "semanas": {u: [plan[u][1][0].isoformat(), plan[u][1][-1].isoformat(), len(plan[u][1])]
                    for u in ucps if plan[u][1]},
    }
    (out_dir / f"hash_k{k}_{etiqueta}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    logger.info("hash intacto: %s (%d archivos)", info["hash_intacto"], len(antes))
    if not info["hash_intacto"]:
        raise RuntimeError(f"Se modificaron archivos del repo: {dif}")


# ---------------------------------------------------------------------------
# Configuraciones y evaluación
# ---------------------------------------------------------------------------

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
NS_PROMEDIO = (2, 3, 4, 6, 8)
DEFAULT_ALPHA = float(paso_nivel.DEFAULTS["alpha"])
DEFAULT_FUENTE = paso_forma.DEFAULTS["fuente"]
DEFAULT_N = int(paso_forma.DEFAULTS["n"])


def nombre_config(alpha: float, fuente: str, n: Optional[int] = None,
                  alpha_por_tipo: Optional[Dict[str, float]] = None) -> str:
    """Etiqueta estable: 'a0.25|promedio_n4', 'a0.5|referencia', 'a0.25|reciente|apt(laboral=0.5,...)'."""
    f = f"{fuente}{n}" if fuente == "promedio_n" else fuente
    txt = f"a{alpha:g}|{f}"
    if alpha_por_tipo:
        txt += "|apt(" + ",".join(f"{t}={alpha_por_tipo[t]:g}" for t in TIPOS if t in alpha_por_tipo) + ")"
    return txt


def config(alpha: float, fuente: str, n: Optional[int] = None,
           alpha_por_tipo: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Config de evaluación: nombre + params de `run_pipeline` + columnas descriptivas."""
    n_eff = int(n) if n is not None else DEFAULT_N
    nivel: Dict[str, Any] = {"alpha": float(alpha)}
    if alpha_por_tipo:
        nivel["alpha_por_tipo"] = {t: float(v) for t, v in alpha_por_tipo.items()}
    return {
        "config": nombre_config(alpha, fuente, n_eff if fuente == "promedio_n" else None, alpha_por_tipo),
        "params": {"nivel": nivel, "forma": {"fuente": fuente, "n": n_eff}},
        "alpha": float(alpha), "fuente": fuente,
        "n": n_eff if fuente == "promedio_n" else None,
        "alpha_por_tipo": json.dumps(alpha_por_tipo, sort_keys=True) if alpha_por_tipo else "",
        "es_default": (float(alpha) == DEFAULT_ALPHA and fuente == DEFAULT_FUENTE
                       and (fuente != "promedio_n" or n_eff == DEFAULT_N) and not alpha_por_tipo),
    }


def config_default() -> Dict[str, Any]:
    return config(DEFAULT_ALPHA, DEFAULT_FUENTE, DEFAULT_N)


def grid_configs(alphas: Iterable[float] = ALPHAS, ns: Iterable[int] = NS_PROMEDIO) -> List[Dict[str, Any]]:
    """α × {reciente, referencia, promedio_n(n)}; siempre incluye el default."""
    out: List[Dict[str, Any]] = []
    for a in alphas:
        out.append(config(a, "reciente"))
        out.append(config(a, "referencia"))
        out.extend(config(a, "promedio_n", n) for n in ns)
    if not any(c["es_default"] for c in out):
        out.append(config_default())
    return out


def evaluar_semana_cfg(
    ucp: str,
    lunes: date,
    historico: pd.DataFrame,
    reales: Dict[str, np.ndarray],
    provider: Optional[pipeline.BaseProvider],
    base_nombre: str,
    configs: Sequence[Dict[str, Any]],
    festivos: Sequence[str],
) -> List[Dict]:
    """
    Como `backtest_adjustments.evaluar_semana` pero con configs arbitrarias (α, fuente, n,
    alpha_por_tipo) y el proveedor de ESA semana. Columnas compatibles con los resúmenes de
    `backtest_adjustments` (+ `config`, `n`, `alpha_por_tipo`).
    """
    if provider is None:
        return []
    ctx = ba.construir_contexto(ucp, lunes, historico, list(festivos))
    provider = ba.con_cache(provider)
    fechas = [d.isoformat() for d in data.dias_semana(lunes)]
    festivos_set = set(ctx.festivos)
    previas = [(d - timedelta(days=7)).isoformat() for d in data.dias_semana(lunes)]
    e_prev_real = sum(float(reales[f].sum()) for f in previas) if all(f in reales for f in previas) else None
    e_sem_real = sum(float(reales[f].sum()) for f in fechas)
    tipos = {f: data.tipo_dia(date.fromisoformat(f), festivos_set) for f in fechas}

    filas: List[Dict] = []
    for cfg in configs:
        records = pipeline.run_pipeline(ctx, provider, params=cfg["params"])
        por_paso = {r.paso: r for r in records}
        if any(f not in por_paso["base"].result.curves for f in fechas):
            logger.warning("%s %s %s base incompleta; semana omitida", base_nombre, ucp, lunes)
            return []
        ctrl = por_paso["control"].result.diagnostics
        codigos = {a.codigo for r in records for a in r.result.alerts}
        meta = {
            "ucp": ucp, "base": base_nombre, "semana_inicio": lunes.isoformat(),
            "fecha_corte": ctx.fecha_corte.isoformat(),
            "semana_ref_inicio": ctx.semana_ref_inicio.isoformat() if ctx.semana_ref_inicio else None,
            "config": cfg["config"], "alpha": cfg["alpha"], "fuente": cfg["fuente"], "n": cfg["n"],
            "alpha_por_tipo": cfg["alpha_por_tipo"], "es_default": cfg["es_default"],
            "diferencia_tendencia_pp": ctrl.get("diferencia_tendencia_pp"),
            "tendencia_resultante_pct": ctrl.get("tendencia_resultante_pct"),
            "tendencia_esperada_pct": ctrl.get("tendencia_esperada_pct"),
            "banda_referencia_pp": ctrl.get("banda_referencia_pp"),
            "alerta_tendencia_referencia_info": "TENDENCIA_REFERENCIA_FUERA_BANDA" in codigos,
            "tendencia_real_pct": (e_sem_real / e_prev_real - 1) * 100 if e_prev_real else None,
        }
        for paso in ba.PASOS_EVALUADOS:
            curves = por_paso[paso].result.curves
            for f in fechas:
                filas.append({**meta, "paso": paso, "fecha": f, "dia_semana": date.fromisoformat(f).weekday(),
                              "tipo_dia": tipos[f], **ba.metricas_dia(curves[f], reales[f])})
    return filas


def _festivos_de(historico: pd.DataFrame) -> List[str]:
    anios = range(int(historico["fecha"].dt.year.min()) - 1, int(historico["fecha"].dt.year.max()) + 2)
    return data.festivos_colombia(anios)


def proveedor_de_semana(base_nombre: str, ucp: str, lunes: date, hist_cortado: pd.DataFrame,
                        cache_ucp: Optional[Dict[str, Curves]] = None,
                        curvas_archivo: Optional[Curves] = None) -> Optional[pipeline.BaseProvider]:
    """
    Proveedor de la base para la semana `lunes`:
    - `modelo_k*`: curvas cacheadas de ESA semana (cada semana tiene su propio corte/modelo).
    - `ingenuo`: `proveedor_ingenuo` sobre el histórico ya cortado.
    - `archivo`: pronóstico guardado (solo Antioquia).
    None si la base no cubre la semana.
    """
    if base_nombre.startswith("modelo"):
        c = (cache_ucp or {}).get(lunes.isoformat())
        return ba.proveedor_desde_curvas(c) if c else None
    if base_nombre == "ingenuo":
        return ba.proveedor_ingenuo(hist_cortado)
    if base_nombre == "archivo":
        return ba.proveedor_desde_curvas(curvas_archivo) if ucp == ba.ARCHIVO_UCP and curvas_archivo else None
    raise ValueError(f"base desconocida {base_nombre!r}")


def correr_base(base_nombre: str, ucp: str, semanas: Sequence[date], configs: Sequence[Dict[str, Any]],
                historico: pd.DataFrame, cache: Optional[pd.DataFrame] = None,
                curvas_archivo: Optional[Curves] = None) -> pd.DataFrame:
    """Backtest de una base/UCP sobre `semanas` con `configs` (`historico` = histórico completo)."""
    historico = data.normalizar_historico(historico, ucp=ucp)
    reales = data.curvas_reales(historico)
    festivos = _festivos_de(historico)
    cache_ucp = curvas_por_semana(cache, ucp) if cache is not None and not cache.empty else None
    filas: List[Dict] = []
    for lunes in semanas:
        hist_cortado = historico[historico["fecha"] <= pd.Timestamp(lunes - timedelta(days=2))]
        prov = proveedor_de_semana(base_nombre, ucp, lunes, hist_cortado, cache_ucp, curvas_archivo)
        if prov is None:
            continue
        try:
            filas.extend(evaluar_semana_cfg(ucp, lunes, historico, reales, prov, base_nombre, configs, festivos))
        except AssertionError:
            raise
        except Exception as exc:  # una semana rota no tumba el backtest
            logger.warning("%s %s %s: error %s", base_nombre, ucp, lunes, exc)
    return pd.DataFrame(filas)


def semanas_archivo(semanas: Sequence[date], curvas_archivo: Curves) -> List[date]:
    """Semanas cuya brecha (domingo) + semana estén cubiertas por el archivo."""
    return [l for l in semanas
            if all((l - timedelta(days=1) + timedelta(days=i)).isoformat() in curvas_archivo for i in range(8))]


def plan_semanas(ucps: Sequence[str], n_semanas: int) -> Dict[str, List[date]]:
    return {u: ba.semanas_objetivo(data.cargar_historico(u, data_dir=ROOT / "data" / "raw"), n_semanas)
            for u in ucps}


def _tarea_evaluar(args) -> pd.DataFrame:
    base_nombre, ucp, semanas_iso, configs, out_dir = args
    hist = data.cargar_historico(ucp, data_dir=ROOT / "data" / "raw")
    cache = leer_cache_k(int(base_nombre.split("_k")[1]), out_dir) if base_nombre.startswith("modelo_k") else None
    archivo = ba.cargar_archivo(ba.ARCHIVO_DEFAULT) if base_nombre == "archivo" else None
    t0 = time.perf_counter()
    df = correr_base(base_nombre, ucp, [date.fromisoformat(s) for s in semanas_iso], configs, hist, cache, archivo)
    print(f"evaluado {base_nombre} {ucp}: {df['semana_inicio'].nunique() if not df.empty else 0} semanas, "
          f"{len(df)} filas, {time.perf_counter() - t0:.0f} s", flush=True)
    return df


def evaluar(bases: Sequence[str], ucps: Sequence[str], configs: Sequence[Dict[str, Any]], n_semanas: int,
            out_dir: Path = OUT_DIR, procesos: int = 4) -> pd.DataFrame:
    """Corre cada (base, UCP) en un proceso, todas sobre las MISMAS semanas objetivo."""
    from concurrent.futures import ProcessPoolExecutor

    plan = plan_semanas(ucps, n_semanas)
    tareas = []
    for b in bases:
        for u in ucps:
            sem = plan[u]
            if b == "archivo":
                if u != ba.ARCHIVO_UCP:
                    continue
                sem = semanas_archivo(sem, ba.cargar_archivo(ba.ARCHIVO_DEFAULT))
            tareas.append((b, u, [s.isoformat() for s in sem], list(configs), Path(out_dir)))
    if procesos <= 1:
        frames = [_tarea_evaluar(t) for t in tareas]
    else:
        with ProcessPoolExecutor(max_workers=procesos) as ex:
            frames = list(ex.map(_tarea_evaluar, tareas))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def leer_detalle(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"semana_inicio": str, "fecha": str, "alpha_por_tipo": str, "config": str},
                     keep_default_na=True)
    df["alpha_por_tipo"] = df["alpha_por_tipo"].fillna("")
    df["es_default"] = df["es_default"].astype(bool)
    return df


# ---------------------------------------------------------------------------
# Holdout temporal y selección (solo calibración)
# ---------------------------------------------------------------------------

FRAC_CALIBRACION = 0.6
MIN_FESTIVOS_CAL = 10       # menos festivos en calibración → festivo usa el α global (sin clave propia)
BANDAS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
TOL_MAPE_SEMANAL_PP = 0.05  # empeoramiento material del MAPE semanal
MAX_SEMANAS_PEORES = 0.35   # R2: un paso es "automático" si empeora en <= 35 % de las semanas (prueba)


def particion_semanas(semanas: Sequence[str], frac: float = FRAC_CALIBRACION) -> Dict[str, str]:
    """{semana: 'calibracion' | 'prueba'}: primeras round(frac·n) semanas (ordenadas) = calibración."""
    orden = sorted(set(semanas))
    n_cal = int(round(frac * len(orden)))
    return {s: ("calibracion" if i < n_cal else "prueba") for i, s in enumerate(orden)}


def agregar_particion(df: pd.DataFrame, plan: Dict[str, Sequence[str]], frac: float = FRAC_CALIBRACION) -> pd.DataFrame:
    """Columna `particion` por UCP según su plan de semanas (el mismo para todas las bases)."""
    out = df.copy()
    mapa = {(u, s): p for u, sems in plan.items() for s, p in particion_semanas(sems, frac).items()}
    out["particion"] = [mapa.get((u, s)) for u, s in zip(out["ucp"], out["semana_inicio"])]
    return out


def wape(df: pd.DataFrame) -> float:
    den = float(df["energia_real"].sum())
    return float(100 * df["abs_err_horario"].sum() / den) if den else float("nan")


def mape_semanal(df: pd.DataFrame) -> float:
    sem = df.groupby("semana_inicio")[["energia_pred", "energia_real"]].sum()
    if sem.empty:
        return float("nan")
    return float(100 * (sem["energia_pred"] / sem["energia_real"] - 1).abs().mean())


def tabla_configs(df: pd.DataFrame, paso: str = "forma") -> pd.DataFrame:
    """WAPE / MAPE semanal por (ucp, config) del paso dado."""
    d = df[df["paso"] == paso]
    filas = []
    for (ucp, cfg), g in d.groupby(["ucp", "config"], sort=False):
        filas.append({"ucp": ucp, "config": cfg, "alpha": g["alpha"].iloc[0], "fuente": g["fuente"].iloc[0],
                      "n": g["n"].iloc[0], "es_default": bool(g["es_default"].iloc[0]),
                      "wape_pct": wape(g), "mape_sem_pct": mape_semanal(g), "n_dias": len(g)})
    return pd.DataFrame(filas)


def elegir_config(cal: pd.DataFrame, solo_grid: bool = True) -> Dict[str, Any]:
    """
    Mejor config (min WAPE del paso final) en calibración: por UCP y "conjunta" (min del promedio
    de WAPE entre UCP, la única candidata a default global).
    """
    d = cal[cal["alpha_por_tipo"] == ""] if solo_grid else cal
    t = tabla_configs(d)
    por_ucp = {u: g.loc[g["wape_pct"].idxmin(), "config"] for u, g in t.groupby("ucp")}
    conj = t.groupby("config")["wape_pct"].mean()
    return {"por_ucp": por_ucp, "conjunta": conj.idxmin(), "tabla": t}


def params_de_config(df: pd.DataFrame, nombre: str) -> Dict[str, Any]:
    fila = df[df["config"] == nombre].iloc[0]
    n = None if pd.isna(fila["n"]) else int(fila["n"])
    return {"alpha": float(fila["alpha"]), "fuente": fila["fuente"], "n": n}


def elegir_alpha_por_tipo(cal: pd.DataFrame, fuente: str, n: Optional[int], alpha_global: float,
                          ucps: Optional[Sequence[str]] = None,
                          min_festivos: int = MIN_FESTIVOS_CAL) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Mejor α por tipo de día en calibración (paso final, fuente/n fijas). Nivel y forma actúan día a día,
    así que filtrar días por tipo es legítimo. Con `ucps` de más de una UCP se minimiza el promedio de
    WAPE entre UCP. Festivo: si hay < `min_festivos` días festivos (por UCP) no se fija α propio
    (usa el global). Devuelve (alpha_por_tipo, tabla).
    """
    d = cal[(cal["paso"] == "forma") & (cal["alpha_por_tipo"] == "") & (cal["fuente"] == fuente)]
    if fuente == "promedio_n":
        d = d[d["n"] == n]
    if ucps is not None:
        d = d[d["ucp"].isin(list(ucps))]
    filas = []
    for (ucp, tipo, a), g in d.groupby(["ucp", "tipo_dia", "alpha"]):
        filas.append({"ucp": ucp, "tipo_dia": tipo, "alpha": a, "wape_pct": wape(g), "n_dias": len(g)})
    t = pd.DataFrame(filas)
    out: Dict[str, float] = {}
    if t.empty:
        return out, t
    for tipo in TIPOS:
        g = t[t["tipo_dia"] == tipo]
        if g.empty:
            continue
        if tipo == "festivo" and g.groupby("ucp")["n_dias"].first().min() < min_festivos:
            continue
        prom = g.groupby("alpha")["wape_pct"].mean()
        out[tipo] = float(prom.idxmin())
    # claves iguales al α global no aportan: se omiten (mismo resultado, params más limpios)
    out = {k: v for k, v in out.items() if v != alpha_global}
    return out, t


def evaluar_banda(df: pd.DataFrame, bandas: Sequence[float] = BANDAS) -> pd.DataFrame:
    """
    Alerta TENDENCIA_REFERENCIA_FUERA_BANDA hipotética por banda: alerta = |diferencia_tendencia_pp| > b,
    evento = |error de energía semanal del final| > b. `df` = filas de UNA config (paso forma) con
    columnas ucp, semana_inicio, particion. Se reporta por (ucp, particion, banda).
    """
    d = df[df["paso"] == "forma"]
    sem = d.groupby(["ucp", "particion", "semana_inicio"], sort=True).agg(
        e_pred=("energia_pred", "sum"), e_real=("energia_real", "sum"),
        dif=("diferencia_tendencia_pp", "first")).reset_index()
    sem["err"] = 100 * (sem["e_pred"] / sem["e_real"] - 1)
    filas = []
    for (ucp, part), g in sem.groupby(["ucp", "particion"]):
        for b in bandas:
            alerta = g["dif"].abs() > b  # NaN → False
            evento = g["err"].abs() > b
            tp = int((alerta & evento).sum()); fp = int((alerta & ~evento).sum())
            fn = int((~alerta & evento).sum())
            prec = tp / (tp + fp) if tp + fp else float("nan")
            rec = tp / (tp + fn) if tp + fn else float("nan")
            base = (tp + fn) / len(g) if len(g) else float("nan")
            f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
            filas.append({"ucp": ucp, "particion": part, "banda_pp": b, "semanas": len(g), "alertas": tp + fp,
                          "eventos": tp + fn, "TP": tp, "precision": prec, "recall": rec, "tasa_base": base,
                          "lift": prec - base if tp + fp else float("nan"), "F1": f1})
    return pd.DataFrame(filas)


def elegir_banda(tabla: pd.DataFrame) -> float:
    """En calibración (todas las UCP juntas promediando): max lift (precision − tasa_base); desempate F1."""
    c = tabla[tabla["particion"] == "calibracion"]
    prom = c.groupby("banda_pp")[["lift", "F1"]].mean().dropna(subset=["lift"])
    if prom.empty:
        return float(paso_control.DEFAULTS["banda_referencia_pp"])
    prom = prom.sort_values(["lift", "F1"], ascending=False)
    return float(prom.index[0])


def resumen_pasos(df: pd.DataFrame, claves: Sequence[str]) -> pd.DataFrame:
    """WAPE, energía diaria (MAPE/sesgo), pico y energía semanal (MAPE/sesgo) por claves (+paso)."""
    claves = list(claves)
    r = ba.resumen_por(df, claves + ["paso"])
    s = ba.resumen_semanal(df, claves + ["paso"])
    if r.empty:
        return r
    return r.merge(s, on=claves + ["paso"], how="left")


def r2_segmentos(df: pd.DataFrame, max_peores: float = MAX_SEMANAS_PEORES) -> pd.DataFrame:
    """
    Por UCP × tipo de día y paso (nivel vs base, forma vs nivel), con UNA config: ΔWAPE en calibración y
    prueba, y % de semanas (con días de ese tipo) en que el paso empeora el WAPE del segmento (prueba).
    Criterio "automático": ΔWAPE < 0 en calibración y en prueba, % semanas peores ≤ `max_peores` y
    ≥ 10 días de prueba; "no aplicar": ΔWAPE > 0 en ambas; resto "sugerido".
    """
    filas = []
    pares = (("nivel", "base"), ("forma", "nivel"))
    for (ucp, tipo), g in df.groupby(["ucp", "tipo_dia"]):
        for paso, previo in pares:
            fila: Dict[str, Any] = {"ucp": ucp, "tipo_dia": tipo, "paso": paso}
            for part in ("calibracion", "prueba"):
                gp = g[g["particion"] == part]
                w1, w0 = wape(gp[gp["paso"] == paso]), wape(gp[gp["paso"] == previo])
                fila[f"dwape_{part[:3]}"] = w1 - w0
                fila[f"n_dias_{part[:3]}"] = int((gp["paso"] == paso).sum())
            gp = g[g["particion"] == "prueba"]
            peores = tot = 0
            for _s, gs in gp.groupby("semana_inicio"):
                tot += 1
                peores += int(wape(gs[gs["paso"] == paso]) > wape(gs[gs["paso"] == previo]) + 1e-9)
            fila["pct_semanas_peores_pru"] = 100 * peores / tot if tot else float("nan")
            if (fila["dwape_cal"] < 0 and fila["dwape_pru"] < 0 and tot
                    and peores / tot <= max_peores and fila["n_dias_pru"] >= 10):
                fila["recomendacion"] = "automático"
            elif fila["dwape_cal"] > 0 and fila["dwape_pru"] > 0:
                fila["recomendacion"] = "no aplicar"
            else:
                fila["recomendacion"] = "sugerido"
            filas.append(fila)
    return pd.DataFrame(filas)


def comparar_configs(df: pd.DataFrame, candidatas: Dict[str, str], base: str = "modelo_k1") -> pd.DataFrame:
    """
    WAPE y MAPE semanal del paso final por candidata (etiqueta → nombre de config), UCP y partición
    (calibración, prueba, total), con la diferencia contra el default.
    """
    d = df[(df["base"] == base) & (df["paso"] == "forma")]
    dflt = config_default()["config"]
    filas = []
    for ucp, gu in d.groupby("ucp"):
        for part in ("calibracion", "prueba", "total"):
            gp = gu if part == "total" else gu[gu["particion"] == part]
            g0 = gp[gp["config"] == dflt]
            w0, m0 = wape(g0), mape_semanal(g0)
            for etiqueta, nombre in candidatas.items():
                g = gp[gp["config"] == nombre]
                if g.empty:
                    continue
                w, m = wape(g), mape_semanal(g)
                filas.append({"ucp": ucp, "particion": part, "candidata": etiqueta, "config": nombre,
                              "wape_pct": w, "dif_wape_vs_default_pp": w - w0,
                              "mape_sem_pct": m, "dif_mape_sem_vs_default_pp": m - m0})
    return pd.DataFrame(filas)


def regla_default(comp: pd.DataFrame, etiqueta: str, tol: float = TOL_MAPE_SEMANAL_PP) -> Dict[str, Any]:
    """
    Regla para cambiar DEFAULTS (en PRUEBA): mejora WAPE del final en TODAS las UCP y no empeora el
    MAPE semanal > `tol` pp en ninguna.
    """
    p = comp[(comp["particion"] == "prueba") & (comp["candidata"] == etiqueta)]
    if p.empty:
        return {"cumple": False, "motivo": "sin datos"}
    mejora = bool((p["dif_wape_vs_default_pp"] < 0).all())
    sin_empeorar = bool((p["dif_mape_sem_vs_default_pp"] <= tol).all())
    return {"cumple": mejora and sin_empeorar and p["config"].iloc[0] != config_default()["config"],
            "mejora_wape_todas": mejora, "mape_sem_ok": sin_empeorar,
            "detalle": {r.ucp: (round(r.dif_wape_vs_default_pp, 3), round(r.dif_mape_sem_vs_default_pp, 3))
                        for r in p.itertuples()}}


# ---------------------------------------------------------------------------
# Orquestación de la recalibración + reporte
# ---------------------------------------------------------------------------

NOMBRE_GRID = "backtest_modelo_grid.csv.gz"
NOMBRE_EXTRA = "backtest_modelo_extra.csv.gz"
MARCA_MANUAL = ("<!-- manual:inicio -->", "<!-- manual:fin -->")


def _plan_iso(ucps: Sequence[str], n_semanas: int) -> Dict[str, List[str]]:
    return {u: [s.isoformat() for s in sems] for u, sems in plan_semanas(ucps, n_semanas).items()}


def seleccionar(grid: pd.DataFrame, ucps: Sequence[str]) -> Dict[str, Any]:
    """Toda la selección con las semanas de CALIBRACIÓN de la base modelo_k1."""
    m1 = grid[grid["base"] == "modelo_k1"]
    cal = m1[m1["particion"] == "calibracion"]
    sel = elegir_config(cal)
    candidatas: Dict[str, str] = {"default": config_default()["config"], "mejor_conjunta": sel["conjunta"]}
    for u in ucps:
        candidatas[f"mejor_{u}"] = sel["por_ucp"][u]
    apt: Dict[str, Any] = {}
    nuevas: List[Dict[str, Any]] = []
    grupos = [("default", list(ucps), candidatas["default"]), ("conjunta", list(ucps), sel["conjunta"])]
    grupos += [(u, [u], sel["por_ucp"][u]) for u in ucps]
    for clave, us, nombre in grupos:
        p = params_de_config(m1, nombre)
        por_tipo, tabla = elegir_alpha_por_tipo(cal, p["fuente"], p["n"], p["alpha"], us)
        cfg = config(p["alpha"], p["fuente"], p["n"], por_tipo) if por_tipo else None
        apt[clave] = {"sobre": nombre, "alpha_por_tipo": por_tipo, "config": cfg["config"] if cfg else nombre,
                      "tabla": tabla, "ucps": us}
        if cfg is not None:
            candidatas[f"apt_{clave}"] = cfg["config"]
            if cfg["config"] not in [c["config"] for c in nuevas]:
                nuevas.append(cfg)
    banda_tabla = evaluar_banda(m1[m1["es_default"]])
    return {"sel": sel, "candidatas": candidatas, "apt": apt, "nuevas": nuevas,
            "banda_tabla": banda_tabla, "banda": elegir_banda(banda_tabla)}


def recalibrar(ucps: Sequence[str], n_semanas: int, out_dir: Path = OUT_DIR, fecha: Optional[str] = None,
               procesos: int = 4, reusar_extra: bool = False) -> Path:
    out_dir = Path(out_dir)
    plan = _plan_iso(ucps, n_semanas)
    grid = agregar_particion(leer_detalle(out_dir / NOMBRE_GRID), plan)
    s = seleccionar(grid, ucps)
    path_extra = out_dir / NOMBRE_EXTRA
    if reusar_extra and path_extra.exists():
        extra = leer_detalle(path_extra)
    else:
        extra = evaluar(["modelo_k1", "modelo_k4", "ingenuo"], ucps, s["nuevas"], n_semanas, out_dir, procesos) \
            if s["nuevas"] else pd.DataFrame()
        if not extra.empty:
            extra.to_csv(path_extra, index=False)
            extra = leer_detalle(path_extra)
    full = grid if extra.empty else pd.concat([grid, agregar_particion(extra, plan)], ignore_index=True)
    fecha = fecha or date.today().strftime("%Y%m%d")
    md = out_dir / f"backtest_modelo_{fecha}.md"
    escribir_reporte(full, s, plan, md, out_dir)
    return md


def _hashes(out_dir: Path) -> pd.DataFrame:
    filas = []
    for p in sorted(Path(out_dir).glob("hash_k*.json")):
        info = json.loads(p.read_text(encoding="utf-8"))
        filas.append({"archivo": p.name, "k": info["k"], "ucps": ",".join(info["ucps"]),
                      "archivos_hasheados": info["archivos_hash"], "intacto": info["hash_intacto"],
                      "segundos": info["segundos"]})
    return pd.DataFrame(filas)


def _bloque_manual(md: Path) -> str:
    if not md.exists():
        return ""
    t = md.read_text(encoding="utf-8")
    i, j = t.find(MARCA_MANUAL[0]), t.find(MARCA_MANUAL[1])
    return t[i:j + len(MARCA_MANUAL[1])] + "\n" if i >= 0 and j > i else ""


def tabla_md(df: Optional[pd.DataFrame], decimales: int = 2) -> str:
    """`backtest_adjustments.tabla_md` escapando '|' en las celdas de texto (nombres de config)."""
    if df is not None and not df.empty:
        df = df.copy()
        for c in df.columns:
            if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
                df[c] = df[c].map(lambda v: v.replace("|", "\\|") if isinstance(v, str) else v)
    return ba.tabla_md(df, decimales)


def escribir_reporte(full: pd.DataFrame, s: Dict[str, Any], plan: Dict[str, List[str]], md: Path,
                     out_dir: Path) -> None:
    T = tabla_md
    dflt = config_default()["config"]
    sec: List[str] = [f"# Backtest de ajustes con el modelo real y recalibración — {md.stem.split('_')[-1]}\n",
                      "Generado por `scripts/recalibrar_ajustes.py` (`generar` → `evaluar` → `recalibrar`). "
                      "Detalle en `reports/ajustes/backtest_modelo_grid.csv.gz` y `backtest_modelo_extra.csv.gz` "
                      "(no versionados).\n"]
    manual = _bloque_manual(md)
    sec.append(manual or f"{MARCA_MANUAL[0]}\n_(resumen y decisiones: escritos a mano)_\n{MARCA_MANUAL[1]}\n")

    # 1. Configuración
    filas = []
    for u, sems in plan.items():
        part = particion_semanas(sems)
        cal = [x for x in sems if part[x] == "calibracion"]
        pru = [x for x in sems if part[x] == "prueba"]
        filas.append({"ucp": u, "semanas": len(sems), "calibracion": f"{cal[0]} .. {cal[-1]} ({len(cal)})",
                      "prueba": f"{pru[0]} .. {pru[-1]} ({len(pru)})"})
    cob = full[(full["config"] == dflt) & (full["paso"] == "base")].groupby(["base", "ucp"])["semana_inicio"] \
        .agg(["min", "max", "nunique"]).reset_index()
    cob.columns = ["base", "ucp", "primera", "ultima", "n_semanas"]
    sec += ["## 1. Configuración\n",
            "- Base `modelo`: flujo real (`scripts/backtest_modelo.SandboxModelo`), entrenado SOLO con datos ≤ corte "
            "en un sandbox temporal sin red. `modelo_k1` = reentreno cada semana (corrida principal); `modelo_k4` = "
            "reentreno cada 4 semanas (las intermedias predicen con el modelo del último origen; sensibilidad).\n"
            "- Corte = sábado anterior a la semana objetivo (como `construir_contexto`); `n_days` = 8 (domingo de "
            "brecha + semana).\n"
            f"- Holdout temporal por UCP: primeras {int(FRAC_CALIBRACION * 100)} % de semanas = calibración (TODO se "
            "elige ahí), últimas = prueba (solo se reporta). Bases `ingenuo` y `archivo` en las mismas semanas.\n"
            f"- Grid: α ∈ {list(ALPHAS)} × fuente ∈ {{reciente, referencia, promedio_n(n ∈ {list(NS_PROMEDIO)})}}; "
            f"default actual = `{dflt}` (α {DEFAULT_ALPHA}, `{DEFAULT_FUENTE}`, n {DEFAULT_N}).\n",
            T(pd.DataFrame(filas)), "### Cobertura por base\n", T(cob),
            "### No-fuga y hashes\n",
            "Cada semana: `datos.csv` del sandbox recortado a FECHA ≤ corte + `end_date=corte`; el contexto del pipeline "
            "verifica `max(historico) ≤ fecha_corte ≤ sábado`. Hash sha256 de `data/raw`, `models`, `data/features`, "
            "`data/processed` del worktree antes/después de cada proceso de generación:\n",
            T(_hashes(out_dir))]

    # 2. Por paso, default
    d = full[full["config"] == dflt]
    tot = d.assign(particion="total")
    both = pd.concat([d[d["particion"] == "prueba"], tot], ignore_index=True)
    sec += ["## 2. Resultado por paso con el default actual (modelo vs ingenuo vs archivo)\n",
            "WAPE horario = Σ|pred−real|/Σreal. Energía diaria y semanal: MAPE y sesgo %. Pico: |error %| del máximo "
            "horario y diferencia de hora (circular). `archivo` solo Antioquia (semanas 2026-01..05, mayormente prueba).\n",
            T(resumen_pasos(both, ["particion", "base", "ucp"])),
            "### Calibración (referencia)\n", T(resumen_pasos(d[d["particion"] == "calibracion"], ["base", "ucp"]))]
    m1d = d[d["base"] == "modelo_k1"]
    sec += ["### Por tipo de día — modelo_k1, default (prueba y total)\n",
            T(ba.resumen_por(pd.concat([m1d[m1d["particion"] == "prueba"], m1d.assign(particion="total")]),
                             ["particion", "ucp", "tipo_dia", "paso"]))]

    # 3. Barrido
    m1 = full[(full["base"] == "modelo_k1") & (full["paso"] == "forma") & (full["alpha_por_tipo"] == "")]
    sec += ["## 3. Barrido α × fuente (WAPE % del paso final, modelo_k1)\n"]
    for ucp in plan:
        for part in ("calibracion", "prueba"):
            g = m1[(m1["ucp"] == ucp) & (m1["particion"] == part)]
            t = tabla_configs(g)
            if t.empty:
                continue
            t["fuente_n"] = [f"{f}{'' if pd.isna(n) else int(n)}" for f, n in zip(t["fuente"], t["n"])]
            piv = t.pivot(index="alpha", columns="fuente_n", values="wape_pct").reset_index()
            sec += [f"### {ucp} — {part}\n", T(piv)]
    niv = full[(full["base"] == "modelo_k1") & (full["paso"] == "nivel") & (full["alpha_por_tipo"] == "")
               & (full["config"].str.endswith(f"|{DEFAULT_FUENTE}{DEFAULT_N}"))]
    sec += ["### Paso nivel por α (energía, no depende de la forma) — prueba\n",
            T(resumen_pasos(niv[niv["particion"] == "prueba"], ["ucp", "alpha"])
              .drop(columns=["paso"], errors="ignore"))]

    # 4. Selección
    comp = comparar_configs(full, s["candidatas"])
    sec += ["## 4. Selección en calibración y resultado en prueba\n",
            "Candidatas elegidas SOLO con calibración (min WAPE del paso final): `mejor_<UCP>` por UCP, "
            "`mejor_conjunta` = min del promedio de WAPE entre UCP (única candidata posible a default global, "
            "porque los DEFAULTS son globales), `apt_*` = la misma config + `alpha_por_tipo` elegido en calibración "
            "(corrida REAL del pipeline con ese parámetro, no reconstruida).\n",
            T(pd.DataFrame([{"candidata": k, "config": v} for k, v in s["candidatas"].items()])),
            T(comp, 3)]
    reglas = [{"candidata": k, **{kk: vv for kk, vv in regla_default(comp, k).items() if kk != "detalle"},
               "detalle (ΔWAPE, ΔMAPE sem) por UCP": str(regla_default(comp, k).get("detalle"))}
              for k in s["candidatas"] if k != "default"]
    sec += ["### Regla para cambiar DEFAULTS (en prueba)\n",
            f"Cumple si mejora el WAPE del final en TODAS las UCP y no empeora el MAPE semanal > "
            f"{TOL_MAPE_SEMANAL_PP} pp en ninguna. Solo `mejor_conjunta`/`apt_conjunta`/`apt_default` pueden "
            "convertirse en default global.\n", T(pd.DataFrame(reglas))]

    # 5. alpha_por_tipo
    sec += ["## 5. `alpha_por_tipo` (elegido en calibración)\n",
            f"Para cada candidata se fija su fuente/n y se elige el mejor α por tipo de día filtrando días del tipo "
            f"(promedio de WAPE entre las UCP del grupo). Festivo: si alguna UCP tiene < {MIN_FESTIVOS_CAL} días "
            "festivos en calibración, NO se fija α propio (usa el global). Claves iguales al α global se omiten.\n"]
    for clave, info in s["apt"].items():
        t = info["tabla"]
        if t.empty:
            continue
        piv = t.pivot_table(index=["ucp", "tipo_dia", "n_dias"], columns="alpha", values="wape_pct").reset_index()
        piv.columns = [str(c) for c in piv.columns]
        sec += [f"### {clave} (sobre `{info['sobre']}`, UCP {', '.join(info['ucps'])}) → "
                f"`{json.dumps(info['alpha_por_tipo'], sort_keys=True) or '{}'}`\n", T(piv)]
    # cada config con alpha_por_tipo junto a la config sobre la que se eligió (y el default)
    cfg_apt = {dflt} | {i["sobre"] for i in s["apt"].values()} | {i["config"] for i in s["apt"].values()}
    apt_rows = full[(full["base"] == "modelo_k1") & (full["paso"] == "forma") & (full["config"].isin(cfg_apt))]
    sec += ["### Efecto por tipo de día en prueba (WAPE del final)\n",
            T(ba.resumen_por(apt_rows[apt_rows["particion"] == "prueba"], ["ucp", "tipo_dia", "config"])
              [["ucp", "tipo_dia", "config", "n_dias", "wape_pct", "mape_energia_dia_pct", "sesgo_energia_dia_pct"]], 3)]

    # 6. Banda
    bt = s["banda_tabla"]
    sec += ["## 6. Banda del control (`banda_referencia_pp`, alerta TENDENCIA_REFERENCIA_FUERA_BANDA)\n",
            "Con la config default de modelo_k1 (el diagnóstico `diferencia_tendencia_pp` no depende de la banda): "
            "alerta = |diferencia_tendencia_pp| > b; evento real = |error de energía semanal del final| > b. "
            "precision = P(evento | alerta), recall = P(alerta | evento), tasa_base = P(evento), lift = precision − "
            "tasa_base. Se elige en calibración (promedio entre UCP: max lift, desempate F1).\n",
            f"**Banda elegida en calibración: {s['banda']} pp** (default actual "
            f"{paso_control.DEFAULTS['banda_referencia_pp']} pp).\n", T(bt)]

    # 7. Sensibilidad k
    cand_k = {s["candidatas"][k] for k in ("default", "mejor_conjunta", "apt_conjunta", "apt_default")
              if k in s["candidatas"]}
    dk = full[(full["base"].isin(["modelo_k1", "modelo_k4"])) & (full["config"].isin(cand_k))]
    filas = []
    for (u, part, cfg, b), g in pd.concat([dk, dk.assign(particion="total")]).groupby(
            ["ucp", "particion", "config", "base"]):
        fila = {"ucp": u, "particion": part, "config": cfg, "base": b}
        for paso in ba.PASOS_EVALUADOS:
            fila[f"wape_{paso}"] = wape(g[g["paso"] == paso])
        fila["mape_sem_forma"] = mape_semanal(g[g["paso"] == "forma"])
        filas.append(fila)
    sec += ["## 7. Sensibilidad: reentreno semanal (k=1) vs cada 4 semanas (k=4)\n", T(pd.DataFrame(filas), 3)]

    # 8. R2
    r2 = r2_segmentos(full[(full["base"] == "modelo_k1") & (full["config"] == dflt)])
    sec += ["## 8. R2: qué pasos pueden ser automáticos por segmento (modelo_k1, config default)\n",
            f"ΔWAPE = WAPE(paso) − WAPE(paso previo) (negativo = el paso ayuda). `automático` si ayuda en calibración "
            f"Y en prueba, empeora en ≤ {int(MAX_SEMANAS_PEORES * 100)} % de las semanas de prueba y hay ≥ 10 días "
            "de prueba; `no aplicar` si empeora en ambas; `sugerido` en otro caso.\n", T(r2, 3)]
    if "mejor_conjunta" in s["candidatas"] and s["candidatas"]["mejor_conjunta"] != dflt:
        r2b = r2_segmentos(full[(full["base"] == "modelo_k1") & (full["config"] == s["candidatas"]["mejor_conjunta"])])
        sec += [f"### Con `{s['candidatas']['mejor_conjunta']}` (mejor conjunta)\n", T(r2b, 3)]
    md.write_text("\n".join(sec), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generar", help="curvas base del modelo (sandbox) → cache CSV")
    g.add_argument("--ucp", nargs="+", default=list(UCPS_DEFAULT))
    g.add_argument("--k", type=int, default=1)
    g.add_argument("--semanas", type=int, default=52)
    g.add_argument("--out-dir", type=Path, default=OUT_DIR)
    e = sub.add_parser("evaluar", help="grid de configs sobre el cache del modelo + ingenuo/archivo")
    e.add_argument("--ucp", nargs="+", default=list(UCPS_DEFAULT))
    e.add_argument("--base", nargs="+", default=["modelo_k1", "modelo_k4", "ingenuo", "archivo"])
    e.add_argument("--semanas", type=int, default=52)
    e.add_argument("--procesos", type=int, default=4)
    e.add_argument("--out-dir", type=Path, default=OUT_DIR)
    r = sub.add_parser("recalibrar", help="holdout 60/40, selección en calibración y reporte")
    r.add_argument("--ucp", nargs="+", default=list(UCPS_DEFAULT))
    r.add_argument("--semanas", type=int, default=52)
    r.add_argument("--procesos", type=int, default=4)
    r.add_argument("--fecha", default=date.today().strftime("%Y%m%d"))
    r.add_argument("--reusar-extra", action="store_true", help="no re-correr las configs alpha_por_tipo")
    r.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.cmd == "generar":
        generar(args.ucp, args.k, args.semanas, args.out_dir)
    elif args.cmd == "evaluar":
        df = evaluar(args.base, args.ucp, grid_configs(), args.semanas, args.out_dir, args.procesos)
        df.to_csv(Path(args.out_dir) / NOMBRE_GRID, index=False)
        logger.info("escrito %s (%d filas)", NOMBRE_GRID, len(df))
    else:
        md = recalibrar(args.ucp, args.semanas, args.out_dir, args.fecha, args.procesos, args.reusar_extra)
        logger.info("escrito %s", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
