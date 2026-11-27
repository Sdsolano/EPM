"""
Backtest reproducible de la Mesa de ajustes (perfil `jamer` y variantes) contra los envíos de Jamer y contra
la demanda real (fase festivos_r1, ver `docs/work/ajustes/festivos_r1/PLAN.md`, sección C).

Parte 1 - vs Jamer. Cada semana objetivo (lunes L) desde `--desde` (default 2026-04-06: las 33 semanas del PLAN,
60/40 = 20 de calibración / 13 de prueba) con un envío de Jamer con fecha <= domingo:
  * VF    = primer envío de la semana; la Mesa corre con corte = fecha_envío - 1 (misma información que tuvo
            Jamer) y se evalúan los días reales de la semana con fecha > corte (los <= corte se descartan y se
            cuentan en el reporte);
  * final = último envío antes del domingo; la Mesa se re-corre con corte = fecha_envío_final - 1 y se evalúan
            solo los días > fecha_envío_final.
  Series: `jamer_envio` (lo que envió Jamer) + una por config. Base `ingenuo` (sin modelo), sin fuga (assert).
  Métricas: MAPE horario, |error de energía diaria|, sesgo de energía, p90 |err día|; por tipo de día
  (+ post_festivo) y por lead; comparación pareada por semana con IC bootstrap (semilla); partición temporal
  60/40 (semanas antiguas = calibración, recientes = prueba).
  Con `--muestra-informativa` se reporta además la muestra desde `--desde-informativa` (2025-12-01, 48 semanas)
  rotulada "INFORMATIVO - no decide": los veredictos salen SOLO de la muestra del PLAN.

Parte 2 - evaluación ampliada contra real (sin Jamer), desde `--desde-ampliada` (2022-01-01) hasta el último dato:
  semanas con (a) algún festivo, (b) Semana Santa o algún día cuya R1 de producción cambia al activar
  `r1_limpia`, y (c) una muestra aleatoria (con semilla) de semanas normales de control.
  Corte = lunes - 6 (martes; emisión miércoles).

Segmento "R1 saltada" de una config con `r1_limpia`: los días cuya R1 de nivel (diagnostics["dias"][f]["r1"])
en la corrida de esa config difiere de la de `jamer` (sin copia propia de la regla).
Segmento "festivos por tipo": festivos de tipo `fijo` o `semana_santa` (`data.tipo_festivo`; los `lunes` no
cambian con `festivo_por_tipo`).

Criterio de aceptación (PLAN + Enmienda 2026-11-27), por regla:
  1. mejora el MAPE h (diferencia < 0) de su segmento objetivo:
     - festivos por tipo: en la muestra ampliada (parte 2) partida por fecha objetivo, calibración 2022-2024
       Y prueba 2025-2026 (las semanas de prueba de la parte 1 solo traen festivos lunes);
     - R1 saltada / R1 cambia: en la parte 2 completa Y en las semanas de prueba de la parte 1 (VF);
     - `jamer+ambas`: las dos mitades anteriores (festivos por tipo; días con R1 distinta que no son de ese
       segmento) deben pasar;
  2. MAPE h global (parte 1, VF, todas las semanas del PLAN) no empeora, lectura estricta: diferencia pareada
     media <= 0 Y el IC95 no es enteramente positivo;
  3. ningún tipo de día (laboral/sabado/domingo/festivo; parte 1, VF) empeora > 0,1 pp.

Las configs se definen como datos (`CONFIGS`). Si un paso rechaza una clave nueva (ValueError "params
desconocidos"), esa config se salta con aviso.

Uso (desde la raíz del repo; los CSV viven FUERA del repo, nada de credenciales):
    python scripts/backtest_mesa_vs_jamer.py --historico <historico.csv> --festivos <festivos.csv> \
        --envios <envios.csv> [--desde 2026-04-06] [--muestra-informativa] [--partes 1,2] [--seed 7]

Entradas: historico (FECHA,P1..P24), festivos (fecha[,nombre]), envios (sesion,envio,version,fecha,P1..P24).
Salida: reports/ajustes/backtest_mesa_vs_jamer_<YYYYMMDD>.md (solo agregados, versionable con `git add -f`) y
el detalle por día en `--salida-detalle` (default reports/ajustes/jamer/, ignorado por git: trae energía real).
NO escribe data/raw ni models/.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import backtest_adjustments as bta  # noqa: E402  (base ingenua, caché, tabla_md)
from backtest_jamer import partir_calibracion  # noqa: E402  (60/40 temporal, >= 1 semana por lado)
from src.adjustments import data, perfiles, pipeline  # noqa: E402
from src.adjustments.data import PERIODOS, es_semana_santa  # noqa: E402
from src.adjustments.referencia_reciente import seleccionar_recientes  # noqa: E402

logger = logging.getLogger("backtest_mesa_vs_jamer")

UCP = "Antioquia"
REPORTS_DIR = ROOT / "reports" / "ajustes"
DETALLE_DIR = REPORTS_DIR / "jamer"
TIPOS_DIA = ("laboral", "sabado", "domingo", "festivo")
SERIE_JAMER = "jamer_envio"
CONFIG_BASE = "jamer"
CONFIG_R1_LIMPIA = "jamer+r1_limpia"  # su corrida define la columna informativa `r1_saltada` (parte 2)
HASTA_PLAN = "2026-11-16"  # último lunes de la muestra del PLAN: fija las 33 semanas aunque lleguen datos nuevos
DESDE_PLAN = "2026-04-06"  # muestra del PLAN: 33 semanas (20 calibración / 13 prueba); decide los veredictos
DESDE_INFORMATIVA = "2025-12-01"  # muestra ampliada con VF (48 semanas): solo informativa
TOLERANCIA_TIPO_DIA_PP = 0.1
N_BOOT = 4000
CASOS_CLAVE = ("2026-04-13", "2026-04-18", "2026-05-01", "2026-08-07")
# Enmienda 2026-11-27: la regla 1 del segmento de festivos por tipo usa un corte temporal de la parte 2 por año de
# la fecha objetivo (las 13 semanas de prueba de la parte 1 solo tienen festivos lunes, que la regla no cambia).
TIPOS_FESTIVO_SEGMENTO = (data.TIPO_FESTIVO_FIJO, data.TIPO_FESTIVO_SEMANA_SANTA)
ANIOS_CAL_FESTIVOS = (2022, 2024)
ANIOS_PRUEBA_FESTIVOS = (2025, 2026)


def _ambos(extra: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Mismos params extra para nivel y forma (patrón de `saltar_puente`)."""
    return {"nivel": dict(extra), "forma": dict(extra)}


@dataclass(frozen=True)
class Config:
    nombre: str
    extra: Dict[str, Dict[str, Any]]
    segmento: Optional[str]  # segmento objetivo del criterio; None = informativa (sin veredicto)
    perfil: str = "jamer"


CONFIGS: Tuple[Config, ...] = (
    Config(CONFIG_BASE, {}, None),
    Config("jamer+festivo_por_tipo", _ambos({"festivo_por_tipo": True}), "festivo_tipo"),
    Config(CONFIG_R1_LIMPIA, _ambos({"r1_limpia": True}), "r1_saltada"),
    Config("jamer+saltar_puente", _ambos({"saltar_puente": True}), "r1_cambia"),
    Config("jamer+ambas", _ambos({"festivo_por_tipo": True, "r1_limpia": True}), "festivo_tipo|r1_saltada"),
    # sensibilidad del umbral de atípico (solo información, sin veredicto)
    Config("jamer+r1_limpia_u04", _ambos({"r1_limpia": True, "r1_umbral_atipico": 0.04}), None),
    Config("jamer+r1_limpia_u08", _ambos({"r1_limpia": True, "r1_umbral_atipico": 0.08}), None),
)

SEGMENTO_DESC = {
    "festivo": "días festivos",
    "festivo_tipo": "festivos de tipo fijo o semana_santa",
    "r1_saltada": "días cuya R1 de nivel (corrida de la config) difiere de la de `jamer` (R1 saltada)",
    "r1_cambia": "días cuya R1 de nivel difiere de la de `jamer`",
    "festivo_tipo|r1_saltada": "festivos fijo/semana_santa + (aparte) días fuera de ese segmento con R1 de nivel distinta",
}


class ConfigNoSoportada(Exception):
    """El código actual rechaza alguna clave de la config (p. ej. el flag aún no existe)."""


# ---------------------------------------------------------------------------
# Carga de insumos (CSV locales, sin red)
# ---------------------------------------------------------------------------

def cargar_historico(path: Path) -> pd.DataFrame:
    return data.normalizar_historico(pd.read_csv(path), ucp=UCP)


def cargar_festivos(path: Path) -> List[str]:
    df = pd.read_csv(path)
    col = "fecha" if "fecha" in df.columns else df.columns[0]
    return sorted({str(x)[:10] for x in df[col].dropna()})


def cargar_envios(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c if not c.lower().startswith("p") or not c[1:].isdigit() else c.upper() for c in df.columns]
    faltan = [c for c in ("sesion", "envio", "fecha", *PERIODOS) if c not in df.columns]
    if faltan:
        raise ValueError(f"envios: faltan columnas {faltan}")
    df["envio"] = pd.to_datetime(df["envio"])
    df["fecha"] = df["fecha"].astype(str).str[:10]
    return df


# ---------------------------------------------------------------------------
# Selección de semanas / envíos (puro)
# ---------------------------------------------------------------------------

def seleccionar_envios(envios: pd.DataFrame, desde: Optional[date] = None) -> pd.DataFrame:
    """
    Una fila por semana objetivo (lunes = menor fecha de la sesión, debe ser lunes) con su envío VF (el primero)
    y final (el último), considerando solo envíos con fecha <= domingo de la semana (los envíos retroactivos
    no cuentan). Columnas: lunes, vf_sesion, vf_envio, final_sesion, final_envio, n_envios.
    """
    ses = envios.groupby("sesion").agg(envio=("envio", "min"), lunes=("fecha", "min")).reset_index()
    ses["lunes"] = ses["lunes"].map(date.fromisoformat)
    ses = ses[ses["lunes"].map(lambda d: d.weekday() == 0)]
    ses = ses[ses.apply(lambda r: r["envio"].date() <= r["lunes"] + timedelta(days=6), axis=1)]
    if desde is not None:
        ses = ses[ses["lunes"] >= desde]
    filas = []
    for lunes, g in ses.groupby("lunes"):
        g = g.sort_values(["envio", "sesion"])
        vf, fin = g.iloc[0], g.iloc[-1]
        filas.append({"lunes": lunes, "vf_sesion": vf["sesion"], "vf_envio": vf["envio"],
                      "final_sesion": fin["sesion"], "final_envio": fin["envio"], "n_envios": len(g)})
    cols = ["lunes", "vf_sesion", "vf_envio", "final_sesion", "final_envio", "n_envios"]
    return pd.DataFrame(filas, columns=cols).sort_values("lunes").reset_index(drop=True)


def curvas_envio(envios: pd.DataFrame, sesion) -> Dict[str, np.ndarray]:
    g = envios[envios["sesion"] == sesion]
    return {str(r["fecha"]): np.asarray([r[p] for p in PERIODOS], dtype=float) for _, r in g.iterrows()}


def corte_de_envio(envio) -> date:
    """Corte = día anterior a la fecha del envío (la Mesa usa la misma información que tuvo Jamer)."""
    return pd.Timestamp(envio).date() - timedelta(days=1)


def corte_martes(lunes: date) -> date:
    """Parte 2: corte = martes anterior (lunes - 6), simula un envío del miércoles."""
    return lunes - timedelta(days=6)


def dias_a_evaluar(lunes: date, reales: Mapping[str, Any], envio_final=None) -> List[str]:
    """Días de la semana con real; con `envio_final`, solo los posteriores a la fecha del envío."""
    dias = data.dias_semana(lunes)
    if envio_final is not None:
        f_env = pd.Timestamp(envio_final).date()
        dias = [d for d in dias if d > f_env]
    return [d.isoformat() for d in dias if d.isoformat() in reales]


def con_tramo(df: pd.DataFrame) -> pd.DataFrame:
    """Columna `tramo` (calibracion / prueba) con la partición temporal 60/40 de las semanas de `df`."""
    if df.empty:
        return df
    df = df.copy()
    _, prueba = partir_calibracion(sorted(df["lunes"].unique()))
    df["tramo"] = np.where(df["lunes"].isin(prueba), "prueba", "calibracion")
    return df


# ---------------------------------------------------------------------------
# R1 de producción con y sin `r1_limpia` (sin copia de la regla)
# ---------------------------------------------------------------------------

def r1_cambia_con_limpia(d: date, reales: Mapping[str, Any], corte: date, festivos: Set[str]) -> bool:
    """True si la R1 de `seleccionar_recientes` (la función de producción) cambia al activar `r1_limpia`."""
    sin = seleccionar_recientes(d, reales, corte, festivos, n=1)
    con = seleccionar_recientes(d, reales, corte, festivos, n=1, r1_limpia=True)
    return sin.fechas[:1] != con.fechas[:1]


def r1_cambia_con_puente(d: date, reales: Mapping[str, Any], corte: date, festivos: Set[str]) -> bool:
    """True si la R1 de producción cambia al activar `saltar_puente` (selección de semanas de la parte 2)."""
    sin = seleccionar_recientes(d, reales, corte, festivos, n=1)
    con = seleccionar_recientes(d, reales, corte, festivos, n=1, saltar_puente=True)
    return sin.fechas[:1] != con.fechas[:1]


# ---------------------------------------------------------------------------
# Corrida de la Mesa (sin fuga)
# ---------------------------------------------------------------------------

def recortar_sin_fuga(hist: pd.DataFrame, corte: date) -> pd.DataFrame:
    hc = hist[hist["fecha"] <= pd.Timestamp(corte)]
    assert hc.empty or hc["fecha"].max().date() <= corte, "fuga: histórico posterior al corte"
    return hc


def correr_config(ctx, prov, cfg: Config) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, Any]]]:
    """
    (curvas finales, {fecha: {"r1": R1 de nivel, "saltados": candidatos saltados por nivel}}).
    ConfigNoSoportada si un paso rechaza una clave.
    """
    try:
        recs = pipeline.run_pipeline(ctx, prov, params=perfiles.expandir(cfg.perfil, cfg.extra or None))
    except ValueError as e:
        if "desconocid" in str(e):
            raise ConfigNoSoportada(str(e)) from None
        raise
    r1: Dict[str, Dict[str, Any]] = {}
    for r in recs:
        if r.paso == "nivel":
            for f, v in ((r.result.diagnostics or {}).get("dias") or {}).items():
                v = v or {}
                r1[f] = {"r1": (v.get("r1") or {}).get("fecha"), "saltados": list(v.get("saltados") or [])}
    return recs[-1].result.curves, r1


@dataclass
class Corredor:
    """Corre todas las configs activas; desactiva (una vez, con aviso) las que el código no soporta."""
    hist: pd.DataFrame
    festivos: List[str]
    configs: Sequence[Config]
    saltadas: Dict[str, str] = field(default_factory=dict)

    def activas(self) -> List[Config]:
        return [c for c in self.configs if c.nombre not in self.saltadas]

    def correr(self, lunes: date, corte: date):
        hc = recortar_sin_fuga(self.hist, corte)
        ctx = pipeline.build_context(UCP, lunes, fecha_emision=corte + timedelta(days=1), historico=hc,
                                     festivos=self.festivos)
        assert ctx.historico["fecha"].max().date() <= corte, "fuga"
        prov = bta.con_cache(bta.proveedor_ingenuo(ctx.historico))
        curvas, r1s = {}, {}
        for cfg in self.activas():
            try:
                curvas[cfg.nombre], r1s[cfg.nombre] = correr_config(ctx, prov, cfg)
            except ConfigNoSoportada as e:
                self.saltadas[cfg.nombre] = str(e)
                logger.warning("config %s saltada (el código actual no la soporta): %s", cfg.nombre, e)
        return curvas, r1s, ctx.fecha_corte


# ---------------------------------------------------------------------------
# Métricas (puras)
# ---------------------------------------------------------------------------

def errores_dia(pred, real) -> Dict[str, float]:
    """MAPE horario (%), error de energía del día (%), energías."""
    f = np.asarray(pred, dtype=float)
    r = np.asarray(real, dtype=float)
    return {"mape_h": float(np.mean(np.abs(f - r) / r) * 100), "err_dia": float((f.sum() / r.sum() - 1) * 100),
            "energia": float(f.sum()), "real": float(r.sum())}


def resumen(df: pd.DataFrame, claves: Sequence[str] = ()) -> pd.DataFrame:
    """Por serie (+claves): días, MAPE h, |err día| medio, sesgo de energía, p90 |err día| (todo en %)."""
    if df.empty:
        return pd.DataFrame()
    filas = []
    for k, g in df.groupby(list(claves) + ["serie"], sort=True):
        k = k if isinstance(k, tuple) else (k,)
        filas.append({**dict(zip(list(claves) + ["serie"], k)), "dias": len(g),
                      "mape_h": g["mape_h"].mean(), "abs_err_dia": g["err_dia"].abs().mean(),
                      "sesgo_energia": 100 * (g["energia"].sum() / g["real"].sum() - 1),
                      "p90_abs_err_dia": g["err_dia"].abs().quantile(0.9)})
    return pd.DataFrame(filas)


def bootstrap_media(dif: np.ndarray, semilla: int, n_boot: int = N_BOOT, nivel: float = 0.95) -> Tuple[float, float]:
    """IC percentil de la media por bootstrap (determinista para una semilla)."""
    dif = np.asarray(dif, dtype=float)
    if len(dif) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(semilla)
    idx = rng.integers(0, len(dif), size=(n_boot, len(dif)))
    medias = dif[idx].mean(axis=1)
    a = (1 - nivel) / 2 * 100
    return float(np.percentile(medias, a)), float(np.percentile(medias, 100 - a))


def pareado(df: pd.DataFrame, a: str, b: str, semilla: int, n_boot: int = N_BOOT) -> Dict[str, Any]:
    """MAPE h semanal a - b (negativo = a mejor), pareado por semana (solo días que tienen ambas series)."""
    sub = df[df["serie"].isin([a, b])]
    dias = sub.groupby(["lunes", "fecha"])["serie"].nunique()
    comunes = dias[dias == 2].index
    sub = sub.set_index(["lunes", "fecha"]).loc[lambda x: x.index.isin(comunes)].reset_index()
    w = sub.groupby(["lunes", "serie"])["mape_h"].mean().unstack()
    if w.empty or a not in w or b not in w:
        return {"semanas": 0, "dif_media_pp": float("nan"), "ic95": (float("nan"), float("nan")), "gana_a": 0}
    dif = (w[a] - w[b]).dropna().to_numpy()
    return {"semanas": int(len(dif)), "dif_media_pp": float(dif.mean()) if len(dif) else float("nan"),
            "ic95": bootstrap_media(dif, semilla, n_boot), "gana_a": int((dif < 0).sum())}


def es_festivo_tipo(df: pd.DataFrame) -> pd.Series:
    """Filas de festivos de tipo `fijo` / `semana_santa` (segmento de `festivo_por_tipo`; los lunes no cambian)."""
    return df["tipo_festivo"].isin(TIPOS_FESTIVO_SEGMENTO)


def mascara_segmento(df: pd.DataFrame, segmento: str, config: str) -> pd.Series:
    """
    Filas (de cualquier serie) cuyo día pertenece al segmento `segmento` de `config`. `r1_saltada_no_festivo_tipo`
    es la mitad R1 de `jamer+ambas` (días con R1 distinta que no son festivos fijo/semana_santa).
    """
    if segmento == "festivo":
        return df["tipo_dia"] == "festivo"
    if segmento == "festivo_tipo":
        return es_festivo_tipo(df)
    if segmento in ("r1_saltada", "r1_cambia"):
        return _r1_distinta(df, config)
    if segmento == "r1_saltada_no_festivo_tipo":
        return _r1_distinta(df, config) & ~es_festivo_tipo(df)
    if segmento == "festivo_tipo|r1_saltada":
        return es_festivo_tipo(df) | _r1_distinta(df, config)
    raise ValueError(f"segmento desconocido {segmento!r}")


def en_anios(df: pd.DataFrame, anios: Tuple[int, int]) -> pd.DataFrame:
    """Filas cuya fecha objetivo cae en los años [anios[0], anios[1]] (inclusive)."""
    if df.empty:
        return df
    anio = pd.to_datetime(df["fecha"]).dt.year
    return df[(anio >= anios[0]) & (anio <= anios[1])]


def _r1_distinta(df: pd.DataFrame, config: str) -> pd.Series:
    """Filas cuyo día tiene, en la corrida de `config`, una R1 de nivel distinta de la de `jamer`."""
    claves = ["parte", "lunes", "momento", "fecha"]
    cfg = df[df["serie"] == config]
    cambia = cfg[cfg["r1"].fillna("") != cfg["r1_jamer"].fillna("")][claves]
    if cambia.empty:
        return pd.Series(False, index=df.index)
    idx = pd.MultiIndex.from_frame(cambia)
    return pd.Series(pd.MultiIndex.from_frame(df[claves]).isin(idx), index=df.index)


def comparar_segmento(df: pd.DataFrame, segmento: str, config: str, base: str = CONFIG_BASE) -> Dict[str, Any]:
    """MAPE h de config y base en los días del segmento que tienen ambas series."""
    sub = df[mascara_segmento(df, segmento, config) & df["serie"].isin([config, base])]
    piv = sub.pivot_table(index=["parte", "lunes", "momento", "fecha"], columns="serie", values="mape_h")
    if piv.empty or config not in piv or base not in piv:
        return {"dias": 0, "mape_config": float("nan"), "mape_base": float("nan"), "dif_pp": float("nan")}
    piv = piv.dropna()
    mc, mb = float(piv[config].mean()), float(piv[base].mean())
    return {"dias": int(len(piv)), "mape_config": mc, "mape_base": mb, "dif_pp": mc - mb}


def dif_por_tipo_dia(df: pd.DataFrame, config: str, base: str = CONFIG_BASE) -> Dict[str, float]:
    """MAPE h config - base por tipo de día (pp), sobre días comunes."""
    out: Dict[str, float] = {}
    for t in TIPOS_DIA:
        sub = df[(df["tipo_dia"] == t) & df["serie"].isin([config, base])]
        piv = sub.pivot_table(index=["parte", "lunes", "momento", "fecha"], columns="serie", values="mape_h")
        if piv.empty or config not in piv or base not in piv:
            continue
        piv = piv.dropna()
        if len(piv):
            out[t] = float(piv[config].mean() - piv[base].mean())
    return out


# ---------------------------------------------------------------------------
# Criterio de aceptación (puro)
# ---------------------------------------------------------------------------

def veredicto_reglas(partes_r1: Sequence[Tuple[str, Dict[str, Any]]], par_global: Dict[str, Any],
                     dif_tipos: Dict[str, float], tolerancia_pp: float = TOLERANCIA_TIPO_DIA_PP) -> Dict[str, Any]:
    """
    Aplica las 3 reglas del PLAN (con la Enmienda 2026-11-27). Sin datos para evaluar una regla: NO PASA (no
    evaluable, conservador).
      1. `partes_r1` = [(etiqueta, {"dias", "dif_pp"}), ...] (p. ej. ampliada + prueba, o calibración 2022-2024 +
         prueba 2025-2026 para festivos): pasa si TODAS tienen días y dif_pp < 0;
      2. lectura estricta (Enmienda 2026-11-27): media pareada global <= 0 Y el IC95 no es enteramente > 0;
      3. max(dif por tipo de día) <= tolerancia.
    """
    def _ok(x) -> bool:
        return x is not None and not (isinstance(x, float) and math.isnan(x))

    r1_ok = bool(partes_r1) and all(_ok(seg.get("dif_pp")) and seg.get("dias", 0) > 0 for _, seg in partes_r1)
    r1 = bool(r1_ok and all(seg["dif_pp"] < 0 for _, seg in partes_r1))
    m = par_global.get("dif_media_pp")
    lo, hi = par_global.get("ic95", (float("nan"), float("nan")))
    r2_ok = _ok(m) and _ok(lo) and par_global.get("semanas", 0) > 0
    # Regla 2, lectura estricta (Enmienda 2026-11-27 del PLAN): las DOS condiciones, media <= 0 Y el IC95 no
    # enteramente sobre 0 (un IC cuyo extremo inferior es <= 0 cuenta como "no enteramente > 0").
    r2 = bool(r2_ok and m <= 0 and not lo > 0)
    peor = max(dif_tipos.items(), key=lambda kv: kv[1]) if dif_tipos else None
    r3 = bool(peor is not None and peor[1] <= tolerancia_pp)
    return {
        "reglas": [
            {"regla": "1 segmento mejora (en cada parte)", "pasa": r1,
             "valor": "; ".join(f"{e} {_fmt(seg.get('dif_pp'))} pp ({seg.get('dias', 0)} d)" for e, seg in partes_r1)
                      or "sin datos",
             "evaluable": r1_ok},
            {"regla": "2 global no empeora (pareado, lectura estricta)", "pasa": r2,
             "valor": f"media {_fmt(m, 3)} pp, IC95 [{_fmt(lo, 3)}, {_fmt(hi, 3)}], {par_global.get('semanas', 0)} sem",
             "evaluable": r2_ok},
            {"regla": f"3 ningún tipo de día empeora > {tolerancia_pp} pp", "pasa": r3,
             "valor": ", ".join(f"{k} {_fmt(v)}" for k, v in dif_tipos.items()) or "sin datos",
             "evaluable": peor is not None},
        ],
        "pasa": r1 and r2 and r3,
    }


def _fmt(v, dec: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:+.{dec}f}"


def pasa_txt(b: bool) -> str:
    return "PASA" if b else "NO PASA"


# ---------------------------------------------------------------------------
# Parte 1 — vs Jamer
# ---------------------------------------------------------------------------

def _r1(r1s: Mapping[str, Mapping[str, Mapping[str, Any]]], config: str, d: str) -> Optional[str]:
    return (r1s.get(config, {}).get(d) or {}).get("r1")


def _filas_dia(base: Dict[str, Any], d: str, series: Dict[str, Any], r1s: Dict[str, Dict[str, Dict[str, Any]]],
               reales: Mapping[str, np.ndarray], fset: Set[str]) -> List[Dict[str, Any]]:
    """
    Filas (una por serie) del día `d`. `r1_saltada` (informativa): la R1 de nivel de `jamer+r1_limpia` difiere de
    la de `jamer` (None si esa config no corrió); `motivo_r1`: el motivo con que nivel la saltó en esa corrida.
    """
    dd = date.fromisoformat(d)
    r1_j = _r1(r1s, CONFIG_BASE, d)
    saltada: Optional[bool] = None
    motivo: Optional[str] = None
    if CONFIG_R1_LIMPIA in r1s:
        saltada = _r1(r1s, CONFIG_R1_LIMPIA, d) != r1_j
        if saltada:
            saltados = (r1s[CONFIG_R1_LIMPIA].get(d) or {}).get("saltados") or []
            motivo = next((x.get("motivo") for x in saltados if x.get("fecha") == r1_j), None)
    comun = {**base, "fecha": d, "tipo_dia": data.tipo_dia(dd, fset),
             "post_festivo": data.clase_dia(dd, fset) == data.CLASE_POST_FESTIVO,
             "tipo_festivo": data.tipo_festivo(dd, fset), "r1_jamer": r1_j,
             "r1_saltada": saltada, "motivo_r1": motivo}
    out = []
    for n, curv in series.items():
        if d not in curv:
            continue
        out.append({**comun, "serie": n, "r1": _r1(r1s, n, d), **errores_dia(curv[d], reales[d])})
    return out


def correr_parte1(corredor: Corredor, envios: pd.DataFrame, reales: Mapping[str, np.ndarray],
                  desde: Optional[date] = None, progreso: bool = False,
                  descartes_vf: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    Filas de la parte 1 (con `tramo` 60/40 sobre sus semanas). VF evalúa solo días > corte; los descartados por
    semana (lunes iso -> n) se anotan en `descartes_vf` si se pasa.
    """
    fset = set(corredor.festivos)
    sem = seleccionar_envios(envios, desde)
    filas: List[Dict[str, Any]] = []
    for _, s in sem.iterrows():
        lunes = s["lunes"]
        for momento, ses, env in (("VF", s["vf_sesion"], s["vf_envio"]), ("final", s["final_sesion"], s["final_envio"])):
            corte = corte_de_envio(env)
            evaluar = dias_a_evaluar(lunes, reales, env if momento == "final" else None)
            if momento == "VF":
                n_antes = len(evaluar)
                evaluar = [d for d in evaluar if date.fromisoformat(d) > corte]
                if descartes_vf is not None and n_antes > len(evaluar):
                    descartes_vf[lunes.isoformat()] = n_antes - len(evaluar)
            if not evaluar:
                continue
            try:
                curvas, r1s, corte_ef = corredor.correr(lunes, corte)
            except Exception as e:  # noqa: BLE001 - una semana rota no tumba el backtest
                logger.error("parte 1 %s %s: %s", lunes, momento, e)
                continue
            if CONFIG_BASE not in curvas:
                continue
            series = {SERIE_JAMER: curvas_envio(envios, ses), **curvas}
            base = {"parte": 1, "lunes": lunes.isoformat(), "momento": momento,
                    "envio": pd.Timestamp(env).date().isoformat(), "corte": corte_ef.isoformat()}
            for d in evaluar:
                for f in _filas_dia(base, d, series, r1s, reales, fset):
                    f["lead"] = (date.fromisoformat(d) - corte_ef).days
                    filas.append(f)
        if progreso:
            print(f"parte 1: {lunes}", file=sys.stderr)
    return con_tramo(pd.DataFrame(filas))


# ---------------------------------------------------------------------------
# Parte 2 - ampliada contra real
# ---------------------------------------------------------------------------

COLUMNAS_SEMANAS_P2 = ["lunes", "festivo", "semana_santa", "r1_saltada", "r1_puente", "control"]


def semanas_ampliadas(reales: Mapping[str, np.ndarray], festivos: Iterable[str], desde: date,
                      n_control: int, semilla: int) -> pd.DataFrame:
    """
    Semanas de la parte 2 con sus banderas: festivo, semana_santa, r1_saltada (algún día no festivo cuya R1 de
    producción cambia al activar `r1_limpia`, `r1_cambia_con_limpia`), r1_puente (ídem con `saltar_puente`), control (muestra con semilla de las semanas
    sin ninguna de las anteriores). Sin semanas: DataFrame vacío con esas columnas.
    """
    fset = festivos if isinstance(festivos, (set, frozenset)) else set(festivos)
    vacio = pd.DataFrame(columns=COLUMNAS_SEMANAS_P2)
    if not reales:
        return vacio
    ultimo = max(date.fromisoformat(f) for f in reales)
    lunes = data.lunes_de(desde)
    filas = []
    while lunes <= ultimo:
        dias = data.dias_semana(lunes)
        con_real = [d for d in dias if d.isoformat() in reales]
        corte = corte_martes(lunes)
        if con_real and corte.isoformat() in reales:
            no_fest = [d for d in con_real if d.isoformat() not in fset]
            r1_salt = any(r1_cambia_con_limpia(d, reales, corte, fset) for d in no_fest)
            r1_pte = any(r1_cambia_con_puente(d, reales, corte, fset) for d in no_fest)
            filas.append({"lunes": lunes, "festivo": any(d.isoformat() in fset for d in dias),
                          "semana_santa": any(es_semana_santa(d) for d in dias), "r1_saltada": r1_salt,
                          "r1_puente": r1_pte})
        lunes += timedelta(days=7)
    if not filas:
        return vacio
    df = pd.DataFrame(filas)
    normales = df[~(df["festivo"] | df["semana_santa"] | df["r1_saltada"] | df["r1_puente"])]["lunes"].tolist()
    rng = np.random.default_rng(semilla)
    k = min(n_control, len(normales))
    control = set(rng.choice(np.array(normales, dtype=object), size=k, replace=False).tolist()) if k else set()
    df["control"] = df["lunes"].isin(control)
    return df[df["festivo"] | df["semana_santa"] | df["r1_saltada"] | df["r1_puente"] | df["control"]].reset_index(drop=True)


def correr_parte2(corredor: Corredor, reales: Mapping[str, np.ndarray], semanas: pd.DataFrame,
                  progreso: bool = False) -> pd.DataFrame:
    fset = set(corredor.festivos)
    filas: List[Dict[str, Any]] = []
    for i, s in semanas.reset_index(drop=True).iterrows():
        lunes = s["lunes"]
        evaluar = dias_a_evaluar(lunes, reales)
        try:
            curvas, r1s, corte_ef = corredor.correr(lunes, corte_martes(lunes))
        except Exception as e:  # noqa: BLE001
            logger.error("parte 2 %s: %s", lunes, e)
            continue
        if CONFIG_BASE not in curvas:
            continue
        base = {"parte": 2, "lunes": lunes.isoformat(), "momento": "martes", "envio": None,
                "corte": corte_ef.isoformat(), "sem_festivo": bool(s["festivo"]),
                "sem_semana_santa": bool(s["semana_santa"]), "sem_r1_saltada": bool(s["r1_saltada"]),
                "sem_control": bool(s["control"])}
        for d in evaluar:
            for f in _filas_dia(base, d, curvas, r1s, reales, fset):
                f["lead"] = (date.fromisoformat(d) - corte_ef).days
                filas.append(f)
        if progreso and i % 20 == 0:
            print(f"parte 2: {i + 1}/{len(semanas)} ({lunes})", file=sys.stderr)
    return pd.DataFrame(filas)


SEGMENTOS_P2: Tuple[Tuple[str, Callable[[pd.DataFrame], pd.Series]], ...] = (
    ("festivo lunes", lambda d: d["tipo_festivo"] == data.TIPO_FESTIVO_LUNES),
    ("festivo fijo", lambda d: d["tipo_festivo"] == data.TIPO_FESTIVO_FIJO),
    ("festivo semana_santa", lambda d: d["tipo_festivo"] == data.TIPO_FESTIVO_SEMANA_SANTA),
    ("festivo fijo+semana_santa {}-{} (calibración)".format(*ANIOS_CAL_FESTIVOS),
     lambda d: es_festivo_tipo(d) & d.index.isin(en_anios(d, ANIOS_CAL_FESTIVOS).index)),
    ("festivo fijo+semana_santa {}-{} (prueba)".format(*ANIOS_PRUEBA_FESTIVOS),
     lambda d: es_festivo_tipo(d) & d.index.isin(en_anios(d, ANIOS_PRUEBA_FESTIVOS).index)),
    ("festivos (todos)", lambda d: d["tipo_dia"] == "festivo"),
    ("semanas de Semana Santa", lambda d: d["sem_semana_santa"]),
    ("días con R1 saltada (R1 de jamer+r1_limpia != jamer)", lambda d: d["r1_saltada"].fillna(False).astype(bool)),
    ("control (semanas normales)", lambda d: d["sem_control"]),
    ("todo (muestra sesgada)", lambda d: pd.Series(True, index=d.index)),
)


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def _tabla(df: pd.DataFrame, dec: int = 2) -> str:
    return bta.tabla_md(df, dec) if df is not None and not df.empty else "_(sin datos)_\n"


def _orden_series(nombres: Iterable[str], configs: Sequence[Config]) -> List[str]:
    orden = [SERIE_JAMER] + [c.nombre for c in configs]
    return [n for n in orden if n in set(nombres)]


def _pivot_mape(df: pd.DataFrame, clave: str, series: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    t = df.groupby([clave, "serie"])["mape_h"].mean().unstack()
    n = df[df["serie"] == CONFIG_BASE].groupby(clave).size().rename("dias")
    t = t[[s for s in series if s in t]]
    return pd.concat([n, t], axis=1).reset_index()


def _seg(df: Optional[pd.DataFrame], segmento: str, config: str) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"dias": 0, "dif_pp": float("nan")}
    return comparar_segmento(df, segmento, config)


def partes_regla1(cfg: Config, p2: pd.DataFrame, vf_prueba: pd.DataFrame) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Partes de la regla 1 de `cfg` (todas deben mejorar):
    - `festivo_tipo` (Enmienda 2026-11-27): parte 2 con fecha objetivo 2022-2024 (calibración) y 2025-2026 (prueba);
    - `r1_saltada` / `r1_cambia`: parte 2 completa y semanas de prueba de la parte 1 (VF);
    - `festivo_tipo|r1_saltada` (`jamer+ambas`): las 2 de festivos + las 2 de R1 sobre los días con R1 distinta que no
      son festivos fijo/semana_santa.
    """
    def festivos() -> List[Tuple[str, Dict[str, Any]]]:
        return [("festivos {}-{} (ampliada, calibración)".format(*ANIOS_CAL_FESTIVOS),
                 _seg(en_anios(p2, ANIOS_CAL_FESTIVOS), "festivo_tipo", cfg.nombre)),
                ("festivos {}-{} (ampliada, prueba)".format(*ANIOS_PRUEBA_FESTIVOS),
                 _seg(en_anios(p2, ANIOS_PRUEBA_FESTIVOS), "festivo_tipo", cfg.nombre))]

    def r1(segmento: str, pref: str = "") -> List[Tuple[str, Dict[str, Any]]]:
        return [(f"{pref}ampliada", _seg(p2, segmento, cfg.nombre)),
                (f"{pref}prueba parte 1", _seg(vf_prueba, segmento, cfg.nombre))]

    if cfg.segmento == "festivo_tipo":
        return festivos()
    if cfg.segmento in ("r1_saltada", "r1_cambia"):
        return r1(cfg.segmento)
    if cfg.segmento == "festivo_tipo|r1_saltada":
        return festivos() + r1("r1_saltada_no_festivo_tipo", "R1 distinta ")
    raise ValueError(f"segmento sin regla 1 definida: {cfg.segmento!r}")


def nota_festivos_prueba(vf_prueba: Optional[pd.DataFrame]) -> Optional[str]:
    """
    Si las semanas de prueba de la parte 1 (VF) solo tienen festivos lunes (o ninguno), la nota que lo dice
    (calculada, no fija); si traen algún festivo fijo/semana_santa, None.
    """
    if vf_prueba is None or vf_prueba.empty:
        return None
    base = vf_prueba[vf_prueba["serie"] == CONFIG_BASE]
    tipos = base.loc[base["tipo_dia"] == "festivo", "tipo_festivo"]
    if tipos.isin(TIPOS_FESTIVO_SEGMENTO).any():
        return None
    n_sem = int(vf_prueba["lunes"].nunique())
    return (f"Semanas de prueba de la parte 1 ({n_sem}): solo contienen festivos lunes ({len(tipos)} día(s) "
            "festivo(s) evaluado(s), todos de tipo `lunes`, que `festivo_por_tipo` no cambia): por eso la regla 1 del "
            "segmento de festivos usa el corte temporal de la parte 2 (Enmienda 2026-11-27).")


def evaluar_criterios(p1: pd.DataFrame, p2: pd.DataFrame, configs: Sequence[Config], semilla: int) -> Dict[str, Any]:
    vf = p1[p1["momento"] == "VF"] if not p1.empty else p1
    vf_prueba = vf[vf["tramo"] == "prueba"] if not vf.empty else vf
    out = {}
    for cfg in configs:
        if cfg.segmento is None:
            continue
        presente = (not vf.empty and cfg.nombre in set(vf["serie"])) or (not p2.empty and cfg.nombre in set(p2["serie"]))
        if not presente:
            continue
        partes = partes_regla1(cfg, p2, vf_prueba)
        par = pareado(vf, cfg.nombre, CONFIG_BASE, semilla) if not vf.empty else {"semanas": 0}
        tipos = dif_por_tipo_dia(vf, cfg.nombre) if not vf.empty else {}
        out[cfg.nombre] = {**veredicto_reglas(partes, par, tipos), "segmento": cfg.segmento, "partes_regla1": partes}
    return out


def md_parte1(p1: pd.DataFrame, configs: Sequence[Config], semilla: int,
              titulo: str = "Parte 1 — Mesa vs envíos de Jamer (muestra del PLAN: decide)",
              descartes_vf: Optional[Mapping[str, int]] = None) -> List[str]:
    out = [f"## {titulo}\n"]
    if p1.empty:
        return out + ["_(sin datos)_\n"]
    sem = sorted(p1["lunes"].unique())
    cal, prueba = partir_calibracion(sem)
    series = _orden_series(p1["serie"].unique(), configs)
    n_desc = sum(v for k, v in (descartes_vf or {}).items() if sem[0] <= k <= sem[-1])
    out.append(f"Semanas: {len(sem)} ({sem[0]} .. {sem[-1]}); calibración {len(cal)} (hasta {cal[-1] if cal else '—'}), "
               f"prueba {len(prueba)} (desde {prueba[0] if prueba else '—'}). Base `ingenuo`; corte = envío - 1. "
               f"VF evalúa solo días > corte: {n_desc} día(s) VF con fecha <= corte descartados.\n")
    vf = p1[p1["momento"] == "VF"]
    for titulo, d in (("VF, todas las semanas", vf), ("VF, calibración", vf[vf["tramo"] == "calibracion"]),
                      ("VF, prueba (fuera de muestra)", vf[vf["tramo"] == "prueba"]),
                      ("final (días posteriores al último envío)", p1[p1["momento"] == "final"])):
        r = resumen(d)
        if not r.empty:
            r = r.set_index("serie").loc[[s for s in series if s in set(r["serie"])]].reset_index()
        out += [f"### {titulo}\n", _tabla(r)]
    out.append("### Pareado por semana (MAPE h semanal, a - b; negativo = a mejor; IC95 bootstrap)\n")
    filas = []
    for momento, d in (("VF todas", vf), ("VF prueba", vf[vf["tramo"] == "prueba"]), ("final", p1[p1["momento"] == "final"])):
        for a in [s for s in series if s != SERIE_JAMER]:
            for b in ([CONFIG_BASE, SERIE_JAMER] if a != CONFIG_BASE else [SERIE_JAMER]):
                r = pareado(d, a, b, semilla)
                filas.append({"muestra": momento, "a": a, "b": b, "semanas": r["semanas"],
                              "dif_media_pp": r["dif_media_pp"], "ic95_inf": r["ic95"][0], "ic95_sup": r["ic95"][1],
                              "semanas_gana_a": r["gana_a"]})
    out.append(_tabla(pd.DataFrame(filas), 3))
    out.append("### VF por tipo de día — MAPE h %\n")
    vf2 = vf.copy()
    vf2["tipo"] = np.where(vf2["post_festivo"].astype(bool), vf2["tipo_dia"] + "+post_festivo", vf2["tipo_dia"])
    out.append(_tabla(_pivot_mape(vf2, "tipo", series)))
    out.append("### VF por tipo de día — sesgo de energía %\n")
    t = vf.groupby(["tipo_dia", "serie"]).apply(lambda x: 100 * (x["energia"].sum() / x["real"].sum() - 1),
                                                 include_groups=False).unstack()
    out.append(_tabla(t[[s for s in series if s in t]].reset_index()))
    out.append("### VF por lead (días desde el corte) — MAPE h %\n")
    out.append(_tabla(_pivot_mape(vf, "lead", series)))
    casos = vf[vf["fecha"].isin(CASOS_CLAVE)]
    if not casos.empty:
        out.append("### Casos revisados día a día (VF) — error de energía del día %\n")
        t = casos.pivot_table(index="fecha", columns="serie", values="err_dia")
        out.append(_tabla(t[[s for s in series if s in t]].reset_index()))
    return out


def md_parte2(p2: pd.DataFrame, semanas: pd.DataFrame, configs: Sequence[Config], desde: date) -> List[str]:
    out = ["## Parte 2 — muestra ampliada contra real (sin Jamer)\n"]
    if p2 is None or p2.empty or semanas is None or semanas.empty:
        return out + ["_(sin datos: ninguna semana califica o no hubo corridas)_\n"]
    series = _orden_series(p2["serie"].unique(), configs)
    out.append(f"Desde {desde.isoformat()}; corte = martes anterior (lunes - 6). Semanas: {len(semanas)} "
               f"(festivo {int(semanas['festivo'].sum())}, Semana Santa {int(semanas['semana_santa'].sum())}, "
               f"con R1 saltada {int(semanas['r1_saltada'].sum())}, control {int(semanas['control'].sum())}; "
               "una semana puede estar en varios grupos). R1 saltada = la R1 de nivel de "
               f"`{CONFIG_R1_LIMPIA}` difiere de la de `{CONFIG_BASE}`.\n")
    filas = []
    for nombre, fn in SEGMENTOS_P2:
        d = p2[fn(p2).astype(bool)]
        if d.empty:
            continue
        fila = {"segmento": nombre, "dias": int((d["serie"] == CONFIG_BASE).sum())}
        for s in series:
            fila[s] = d.loc[d["serie"] == s, "mape_h"].mean()
        filas.append(fila)
    out += ["### MAPE h % por segmento\n", _tabla(pd.DataFrame(filas))]
    filas = []
    for nombre, fn in SEGMENTOS_P2:
        d = p2[fn(p2).astype(bool)]
        if d.empty:
            continue
        fila = {"segmento": nombre}
        for s in series:
            g = d[d["serie"] == s]
            fila[s] = 100 * (g["energia"].sum() / g["real"].sum() - 1) if len(g) else float("nan")
        filas.append(fila)
    out += ["### Sesgo de energía % por segmento\n", _tabla(pd.DataFrame(filas))]
    mot = p2[(p2["serie"] == CONFIG_BASE) & p2["r1_saltada"].fillna(False).astype(bool)]["motivo_r1"].value_counts()
    if len(mot):
        out.append("Días con R1 saltada por motivo: " + ", ".join(f"{k} {v}" for k, v in mot.items()) + ".\n")
    out += ["### Por tipo de día — MAPE h %\n", _tabla(_pivot_mape(p2, "tipo_dia", series))]
    return out


def md_criterios(crit: Dict[str, Any], saltadas: Dict[str, str], nota_prueba: Optional[str] = None) -> List[str]:
    out = ["## Criterio de aceptación (PLAN, fijado antes de correr; Enmienda 2026-11-27)\n",
           "Veredictos SOLO con la muestra del PLAN (parte 1 desde `--desde`); la muestra informativa no decide. "
           "Regla 1: el segmento objetivo mejora (diferencia de MAPE h < 0) en CADA parte listada (con sus días). "
           "Festivos por tipo (segmento = festivos `fijo`/`semana_santa`; los `lunes` no cambian): parte 2 partida "
           "por fecha objetivo, calibración {}-{} y prueba {}-{}. ".format(*ANIOS_CAL_FESTIVOS, *ANIOS_PRUEBA_FESTIVOS)
           + "R1 saltada / R1 cambia: parte 2 completa y semanas de prueba de la parte 1 (VF). `jamer+ambas`: deben "
           "pasar las dos mitades (festivos por tipo; días con R1 distinta fuera de ese segmento). "
           "Regla 2 (lectura estricta, Enmienda 2026-11-27): pareado global parte 1 VF (config - jamer) con media "
           "<= 0 Y IC95 no enteramente > 0. "
           f"Regla 3: ningún tipo de día empeora > {TOLERANCIA_TIPO_DIA_PP} pp (parte 1, VF). "
           "Sin datos para una regla: NO PASA.\n"]
    if nota_prueba:
        out.append(nota_prueba + "\n")
    filas = []
    for n, c in crit.items():
        for r in c["reglas"]:
            filas.append({"config": n, "segmento": SEGMENTO_DESC.get(c["segmento"], c["segmento"]),
                          "regla": r["regla"], "valor": r["valor"], "veredicto": pasa_txt(r["pasa"])})
        filas.append({"config": n, "segmento": "", "regla": "**global**", "valor": "",
                      "veredicto": f"**{pasa_txt(c['pasa'])}**"})
    out.append(_tabla(pd.DataFrame(filas)))
    if saltadas:
        out.append("Configs no evaluadas (el código actual rechaza sus params):\n")
        out += [f"- `{n}`: {m.replace('|', '/')}" for n, m in saltadas.items()]
        out.append("")
    return out


def escribir_reporte(path: Path, meta: Dict[str, Any], p1: Optional[pd.DataFrame], p2: Optional[pd.DataFrame],
                     semanas2: Optional[pd.DataFrame], crit: Dict[str, Any], configs: Sequence[Config],
                     saltadas: Dict[str, str], p1_info: Optional[pd.DataFrame] = None,
                     descartes_vf: Optional[Mapping[str, int]] = None) -> None:
    vf_prueba = None
    if p1 is not None and not p1.empty:
        vf_prueba = p1[(p1["momento"] == "VF") & (p1["tramo"] == "prueba")]
    nota_prueba = nota_festivos_prueba(vf_prueba)
    out = [f"# Backtest Mesa (perfil jamer + variantes) vs Jamer y vs real — {meta['fecha']}\n",
           "Generado por `scripts/backtest_mesa_vs_jamer.py` (fase `docs/work/ajustes/festivos_r1/`). Solo agregados "
           "(errores %, no curvas ni demanda real). El detalle por día está en un CSV ignorado por git.\n",
           f"Entradas: `{meta['historico']}`, `{meta['festivos']}`, `{meta['envios']}` (fuera del repo). "
           f"Semilla {meta['seed']}. Duración {meta['duracion_s']:.0f} s. Configs evaluadas: "
           + ", ".join(f"`{c.nombre}`" for c in configs if c.nombre not in saltadas) + ".\n"]
    out += md_criterios(crit, saltadas, nota_prueba)
    if p1 is not None:
        out += md_parte1(p1, configs, meta["seed"], descartes_vf=descartes_vf)
    if p2 is not None:
        out += md_parte2(p2, semanas2, configs, meta["desde_ampliada"])
    if p1_info is not None:
        out += md_parte1(p1_info, configs, meta["seed"], descartes_vf=descartes_vf,
                         titulo=f"Parte 1 INFORMATIVO — no decide (muestra desde {meta['desde_informativa']})")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _fecha(s: Optional[str]) -> Optional[date]:
    return date.fromisoformat(s) if s else None


def _salida_tolerante() -> None:
    """Consola Windows (cp1252): que un carácter no representable no tumbe el script."""
    for st in (sys.stdout, sys.stderr):
        if hasattr(st, "reconfigure"):
            try:
                st.reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    _salida_tolerante()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--historico", required=True, help="CSV FECHA,P1..P24 (demanda real)")
    ap.add_argument("--festivos", required=True, help="CSV fecha[,nombre] (festivos usados en producción)")
    ap.add_argument("--envios", help="CSV sesion,envio,version,fecha,P1..P24 (envíos de Jamer; requerido para la parte 1)")
    ap.add_argument("--desde", default=DESDE_PLAN,
                    help=f"lunes mínimo de la parte 1 (YYYY-MM-DD; default {DESDE_PLAN}: muestra del PLAN, decide)")
    ap.add_argument("--muestra-informativa", action="store_true",
                    help="reporta además la parte 1 desde --desde-informativa, rotulada INFORMATIVO (no decide)")
    ap.add_argument("--desde-informativa", default=DESDE_INFORMATIVA,
                    help=f"inicio de la muestra informativa (default {DESDE_INFORMATIVA})")
    ap.add_argument("--desde-ampliada", default="2022-01-01", help="inicio de la parte 2 (default 2022-01-01)")
    ap.add_argument("--partes", default="1,2", help="partes a correr: 1, 2 o 1,2")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-control", type=int, default=40, help="semanas normales de control en la parte 2")
    ap.add_argument("--fecha", default=None, help="sufijo YYYYMMDD del reporte (default: hoy)")
    ap.add_argument("--salida", default=str(REPORTS_DIR), help="carpeta del .md (agregados)")
    ap.add_argument("--salida-detalle", default=str(DETALLE_DIR),
                    help="carpeta del CSV de detalle (trae energía real; default ignorado por git)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for nombre in ("src", "src.adjustments"):  # el pipeline es verboso por semana
        logging.getLogger(nombre).setLevel(logging.ERROR)
    partes = {p.strip() for p in args.partes.split(",") if p.strip()}
    if "1" in partes and not args.envios:
        ap.error("--envios es requerido para la parte 1")

    t0 = time.time()
    hist = cargar_historico(Path(args.historico))
    festivos = cargar_festivos(Path(args.festivos))
    reales = data.curvas_reales(hist)
    corredor = Corredor(hist, festivos, CONFIGS)

    desde = _fecha(args.desde)
    desde_info = _fecha(args.desde_informativa) if args.muestra_informativa else None
    p1 = p1_info = p2 = semanas2 = None
    p1_todo = None
    descartes_vf: Dict[str, int] = {}
    if "1" in partes:
        envios = cargar_envios(Path(args.envios))
        # una sola corrida que cubre ambas muestras; los veredictos usan solo la del PLAN (`desde`)
        inicio = min(d for d in (desde, desde_info) if d is not None) if (desde or desde_info) else None
        p1_todo = correr_parte1(corredor, envios, reales, inicio, progreso=True, descartes_vf=descartes_vf)
        p1 = p1_todo if (p1_todo.empty or desde is None) else con_tramo(
            p1_todo[(p1_todo["lunes"] >= desde.isoformat()) & (p1_todo["lunes"] <= HASTA_PLAN)])
        if desde_info is not None:
            p1_info = p1_todo if p1_todo.empty else con_tramo(p1_todo[p1_todo["lunes"] >= desde_info.isoformat()])
    desde2 = date.fromisoformat(args.desde_ampliada)
    if "2" in partes:
        semanas2 = semanas_ampliadas(reales, festivos, desde2, args.n_control, args.seed)
        print(f"parte 2: {len(semanas2)} semanas", file=sys.stderr)
        p2 = correr_parte2(corredor, reales, semanas2, progreso=True)

    vacio = pd.DataFrame(columns=["serie", "momento", "tramo"])
    crit = evaluar_criterios(p1 if p1 is not None else vacio, p2 if p2 is not None else vacio,
                             CONFIGS, args.seed)
    dur = time.time() - t0

    stamp = args.fecha or date.today().strftime("%Y%m%d")
    salida, salida_det = Path(args.salida), Path(args.salida_detalle)
    salida.mkdir(parents=True, exist_ok=True)
    salida_det.mkdir(parents=True, exist_ok=True)
    detalle = salida_det / f"backtest_mesa_vs_jamer_{stamp}_detalle.csv"
    partes_df = []
    if p1_todo is not None and not p1_todo.empty:
        p1_det = p1_todo.drop(columns=["tramo"], errors="ignore").copy()
        p1_det["muestra_plan"] = p1_det["lunes"] >= desde.isoformat() if desde is not None else True
        partes_df.append(p1_det)
    if p2 is not None and not p2.empty:
        partes_df.append(p2)
    if partes_df:
        pd.concat(partes_df, ignore_index=True).to_csv(detalle, index=False)
    else:
        detalle = "(sin filas, no se escribió)"
    md = salida / f"backtest_mesa_vs_jamer_{stamp}.md"
    meta = {"fecha": stamp, "historico": Path(args.historico).name, "festivos": Path(args.festivos).name,
            "envios": Path(args.envios).name if args.envios else "—", "seed": args.seed, "duracion_s": dur,
            "desde_ampliada": desde2, "desde_informativa": args.desde_informativa}
    escribir_reporte(md, meta, p1, p2, semanas2, crit, CONFIGS, corredor.saltadas, p1_info=p1_info,
                     descartes_vf=descartes_vf)

    print("\n=== Criterio de aceptación (muestra del PLAN) ===")
    if not crit:
        print("(ninguna config con regla evaluable: todas las variantes fueron saltadas)")
    for n, c in crit.items():
        print(f"{n}: {pasa_txt(c['pasa'])}")
        for r in c["reglas"]:
            print(f"   [{pasa_txt(r['pasa'])}] {r['regla']}: {r['valor']}")
    for n, m in corredor.saltadas.items():
        print(f"SALTADA {n}: {m}")
    print(f"Días VF descartados (fecha <= corte): {sum(descartes_vf.values())}")
    print(f"Reporte: {md}\nDetalle: {detalle}\nDuración: {dur:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
