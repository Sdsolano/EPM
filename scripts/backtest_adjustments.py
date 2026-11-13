"""
Backtest del pipeline de ajustes por pasos (src/adjustments).

Para cada UCP y cada una de las últimas N semanas completas con datos reales:
- corte = sábado de la semana ANTERIOR a la objetivo (fecha_emision = domingo),
  igual que el analista: corta el sábado y pronostica la semana que empieza el
  lunes siguiente (el domingo intermedio es brecha estimada).
- El contexto se construye SOLO con datos <= corte (se verifica que no haya fuga).
- Se corre `run_pipeline` para cada configuración (alpha del paso nivel x fuente
  del paso forma) y se comparan las curvas de base / nivel / forma con el real.

Bases disponibles (`--base`, se pueden pasar varias):
- `ingenuo`: cada día = mismo día de semana de la última semana real (con la
  misma brecha que usa el paso nivel como ancla). Sin red, sin modelos.
- `archivo`: pronóstico horario guardado en el repo
  (reports/performance/antioquia_ene_jun_2026/resultados_horarios_ene_jun_2026.csv:
  Antioquia, 2026-01-01..2026-06-01, una sola emisión con corte 2025-12-31).
- `modelo`: `run_predict_flow` de src/api/main.py (con `full_update_csv`
  desactivado para no tocar red ni data/raw). En el venv actual falla
  (pandas 3 sin `fillna(method=)` en feature_engineering) — ver reporte.

Uso (desde la raíz del repo):
    .venv/bin/python scripts/backtest_adjustments.py                       # ingenuo, 52 semanas
    .venv/bin/python scripts/backtest_adjustments.py --base ingenuo archivo
    .venv/bin/python scripts/backtest_adjustments.py --ucp Antioquia --semanas 10 --sin-barrido

Salida: reports/ajustes/backtest_<YYYYMMDD>.csv (detalle por semana/día/paso/config)
y reports/ajustes/backtest_<YYYYMMDD>.md (resumen).
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adjustments import data, pipeline  # noqa: E402
from src.adjustments.steps import forma as paso_forma, nivel as paso_nivel  # noqa: E402
from src.adjustments.types import Curves  # noqa: E402

logger = logging.getLogger("backtest_adjustments")

UCPS_DEFAULT = ("Antioquia", "Atlantico")
ALPHAS_DEFAULT = (0.0, 0.25, 0.5, 0.75, 1.0)
FUENTES_DEFAULT = ("reciente", "referencia", "promedio_n")
CONFIG_DEFAULT = (paso_nivel.DEFAULTS["alpha"], paso_forma.DEFAULTS["fuente"])  # defaults del pipeline
PASOS_EVALUADOS = ("base", "nivel", "forma")
ARCHIVO_DEFAULT = (
    ROOT / "reports" / "performance" / "antioquia_ene_jun_2026" / "resultados_horarios_ene_jun_2026.csv"
)
ARCHIVO_UCP = "Antioquia"

ProviderFactory = Callable[[str, pd.DataFrame], Optional[pipeline.BaseProvider]]


# ---------------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------------

def proveedor_ingenuo(historico_cortado: pd.DataFrame) -> pipeline.BaseProvider:
    """Cada día = mismo día de semana de la última semana real <= corte."""
    reales = data.curvas_reales(historico_cortado)

    def provider(ucp: str, end_date: date, n_days: int) -> Curves:
        out: Curves = {}
        for i in range(1, n_days + 1):
            d = end_date + timedelta(days=i)
            a = data.ancla_mismo_dow(d, end_date)  # misma regla que el paso nivel
            # si falta el ancla se retrocede de a 7 días (hasta 4 semanas)
            for _ in range(5):
                if a.isoformat() in reales:
                    out[d.isoformat()] = reales[a.isoformat()].tolist()
                    break
                a -= timedelta(days=7)
        return out

    return provider


def cargar_archivo(path: Path) -> Curves:
    """Lee el pronóstico horario guardado (fecha, periodo, pronostico_mwh) a Curves."""
    df = pd.read_csv(path)
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d")
    df["p"] = df["periodo"].astype(str).str.replace("P", "", regex=False).astype(int)
    tabla = df.pivot_table(index="fecha", columns="p", values="pronostico_mwh", aggfunc="first")
    tabla = tabla.reindex(columns=range(1, 25))
    out: Curves = {}
    for f, fila in tabla.iterrows():
        v = fila.to_numpy(dtype=float)
        if np.isfinite(v).all():
            out[f] = v.tolist()
    return out


def proveedor_desde_curvas(curvas: Curves) -> pipeline.BaseProvider:
    def provider(ucp: str, end_date: date, n_days: int) -> Curves:
        fechas = [(end_date + timedelta(days=i)).isoformat() for i in range(1, n_days + 1)]
        return {f: list(curvas[f]) for f in fechas if f in curvas}

    return provider


def proveedor_modelo() -> pipeline.BaseProvider:  # pragma: no cover - requiere modelos entrenados
    """`run_predict_flow` real, sin `full_update_csv` (red + escribe data/raw)."""
    import asyncio

    import src.api.main as api_main

    api_main.full_update_csv = lambda ucp: None  # no red, no escribir datos.csv

    def provider(ucp: str, end_date: date, n_days: int) -> Curves:
        curves, _meta = asyncio.run(api_main._adjustments_base_provider(ucp, end_date, n_days))
        return curves

    return provider


def con_cache(provider: pipeline.BaseProvider) -> pipeline.BaseProvider:
    cache: Dict[Tuple[str, date, int], Curves] = {}

    def wrapped(ucp: str, end_date: date, n_days: int) -> Curves:
        key = (ucp, end_date, n_days)
        if key not in cache:
            cache[key] = provider(ucp, end_date, n_days)
        return {f: list(v) for f, v in cache[key].items()}

    return wrapped


# ---------------------------------------------------------------------------
# Semanas y métricas
# ---------------------------------------------------------------------------

def semanas_objetivo(historico: pd.DataFrame, n: int) -> List[date]:
    """Últimas `n` semanas (lunes) con los 7 días reales válidos."""
    validas = set(data.curvas_reales(historico))
    if not validas:
        return []
    ultimo = max(date.fromisoformat(f) for f in validas)
    lunes = data.lunes_de(ultimo)
    if (lunes + timedelta(days=6)) > ultimo:
        lunes -= timedelta(days=7)
    out: List[date] = []
    while len(out) < n and lunes.year >= 2000:
        dias = [(lunes + timedelta(days=i)).isoformat() for i in range(7)]
        if all(d in validas for d in dias):
            out.append(lunes)
        lunes -= timedelta(days=7)
        if len(out) == 0 and (ultimo - lunes).days > 400:
            break
    return sorted(out)


def metricas_dia(pred: Sequence[float], real: Sequence[float]) -> Dict[str, float]:
    p = np.asarray(pred, dtype=float)
    r = np.asarray(real, dtype=float)
    return {
        "energia_pred": float(p.sum()),
        "energia_real": float(r.sum()),
        "abs_err_horario": float(np.abs(p - r).sum()),
        "pico_pred": float(p.max()),
        "pico_real": float(r.max()),
        "hora_pico_pred": int(p.argmax()) + 1,
        "hora_pico_real": int(r.argmax()) + 1,
    }


# ---------------------------------------------------------------------------
# Núcleo
# ---------------------------------------------------------------------------

def _fuera_banda(valor: Optional[float], banda: Optional[float]) -> bool:
    return valor is not None and banda is not None and abs(valor) > banda


def construir_contexto(ucp: str, lunes: date, historico: pd.DataFrame, festivos: Optional[List[str]]):
    """Contexto con corte = sábado anterior; histórico recortado y verificado (sin fuga)."""
    corte_objetivo = lunes - timedelta(days=2)
    hist_cortado = historico[historico["fecha"] <= pd.Timestamp(corte_objetivo)]
    ctx = pipeline.build_context(
        ucp, lunes, fecha_emision=corte_objetivo + timedelta(days=1),
        historico=hist_cortado, festivos=festivos,
    )
    # Verificación de no fuga: nada posterior al corte, ni de la semana objetivo.
    if ctx.fecha_corte > corte_objetivo:
        raise AssertionError(f"fecha_corte {ctx.fecha_corte} > sábado {corte_objetivo}")
    max_hist = ctx.historico["fecha"].max().date() if not ctx.historico.empty else None
    if max_hist is None or max_hist > ctx.fecha_corte:
        raise AssertionError(f"Fuga de datos: histórico hasta {max_hist}, corte {ctx.fecha_corte}")
    return ctx


def evaluar_semana(
    ucp: str,
    lunes: date,
    historico: pd.DataFrame,
    reales: Dict[str, np.ndarray],
    provider_factory: ProviderFactory,
    base_nombre: str,
    configs: Sequence[Tuple[float, str]],
    festivos: Optional[List[str]] = None,
) -> List[Dict]:
    """Filas de detalle (día x paso x config) para una semana objetivo."""
    ctx = construir_contexto(ucp, lunes, historico, festivos)
    provider = provider_factory(ucp, ctx.historico)
    if provider is None:
        return []
    provider = con_cache(provider)

    fechas = [d.isoformat() for d in data.dias_semana(lunes)]
    festivos_set = set(ctx.festivos)
    # semana previa 100% real (para la tendencia real)
    previas = [(d - timedelta(days=7)).isoformat() for d in data.dias_semana(lunes)]
    e_prev_real = sum(float(reales[f].sum()) for f in previas) if all(f in reales for f in previas) else None
    e_sem_real = sum(float(reales[f].sum()) for f in fechas)

    filas: List[Dict] = []
    for alpha, fuente in configs:
        records = pipeline.run_pipeline(
            ctx, provider, params={"nivel": {"alpha": alpha}, "forma": {"fuente": fuente}}
        )
        por_paso = {r.paso: r for r in records}
        base_curves = por_paso["base"].result.curves
        if any(f not in base_curves for f in fechas):
            logger.warning("%s %s base incompleta; semana omitida", ucp, lunes)
            return []
        ctrl = por_paso["control"].result
        codigos = sorted({a.codigo for r in records for a in r.result.alerts})
        semana_meta = {
            "ucp": ucp,
            "base": base_nombre,
            "semana_inicio": lunes.isoformat(),
            "fecha_corte": ctx.fecha_corte.isoformat(),
            "semana_ref_inicio": ctx.semana_ref_inicio.isoformat() if ctx.semana_ref_inicio else None,
            "alpha": alpha,
            "fuente": fuente,
            "es_default": (alpha, fuente) == CONFIG_DEFAULT,
            "alerta_tendencia": "TENDENCIA_FUERA_BANDA" in codigos,
            "banda_pp": ctrl.diagnostics.get("banda_pp"),
            "desvio_pp": ctrl.diagnostics.get("desvio_pp"),
            "tendencia_resultante_pct": ctrl.diagnostics.get("tendencia_resultante_pct"),
            "tendencia_esperada_pct": ctrl.diagnostics.get("tendencia_esperada_pct"),
            "diferencia_tendencia_pp": ctrl.diagnostics.get("diferencia_tendencia_pp"),
            # alerta hipotética = la de v2 (diagnóstico de tendencia de referencia con la misma banda)
            "alerta_tendencia_referencia": _fuera_banda(
                ctrl.diagnostics.get("diferencia_tendencia_pp"), ctrl.diagnostics.get("banda_pp")
            ),
            # alerta real del control (info, default banda_referencia_pp = 2.0 pp; fix ronda 2)
            "alerta_tendencia_referencia_info": "TENDENCIA_REFERENCIA_FUERA_BANDA" in codigos,
            "banda_referencia_pp": ctrl.diagnostics.get("banda_referencia_pp"),
            "tendencia_real_pct": (e_sem_real / e_prev_real - 1) * 100 if e_prev_real else None,
            "bloqueos": "|".join(a.codigo for a in pipeline.approval_blockers(records)),
            "alertas": "|".join(codigos),
        }
        for paso in PASOS_EVALUADOS:
            curves = por_paso[paso].result.curves
            for f in fechas:
                d = date.fromisoformat(f)
                filas.append({
                    **semana_meta,
                    "paso": paso,
                    "fecha": f,
                    "dia_semana": d.weekday(),
                    "tipo_dia": data.tipo_dia(d, festivos_set),
                    **metricas_dia(curves[f], reales[f]),
                })
    return filas


def correr_backtest(
    ucp: str,
    historico: pd.DataFrame,
    semanas: Iterable[date],
    provider_factory: ProviderFactory,
    base_nombre: str,
    configs: Sequence[Tuple[float, str]],
    festivos: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Backtest de una UCP con una base. `historico` = histórico completo normalizado."""
    historico = data.normalizar_historico(historico, ucp=ucp)
    reales = data.curvas_reales(historico)
    if festivos is None:
        anios = range(int(historico["fecha"].dt.year.min()) - 1, int(historico["fecha"].dt.year.max()) + 2)
        festivos = data.festivos_colombia(anios)
    filas: List[Dict] = []
    for lunes in semanas:
        try:
            filas.extend(evaluar_semana(ucp, lunes, historico, reales, provider_factory,
                                        base_nombre, configs, festivos))
        except AssertionError:
            raise
        except Exception as exc:  # una semana rota no tumba el backtest
            logger.warning("%s %s %s: error %s", base_nombre, ucp, lunes, exc)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Resúmenes
# ---------------------------------------------------------------------------

def _dif_hora(a: pd.Series, b: pd.Series) -> pd.Series:
    """Diferencia horaria circular (0..12 h): 24 h y 1 h están a 1 h."""
    d = (a - b).abs() % 24
    return np.minimum(d, 24 - d)


def _agg_dias(g: pd.DataFrame) -> pd.Series:
    err_e = g["energia_pred"] / g["energia_real"] - 1
    return pd.Series({
        "n_dias": len(g),
        "wape_pct": 100 * g["abs_err_horario"].sum() / g["energia_real"].sum(),
        "mape_energia_dia_pct": 100 * err_e.abs().mean(),
        "sesgo_energia_dia_pct": 100 * err_e.mean(),
        "err_pico_pct": 100 * (g["pico_pred"] / g["pico_real"] - 1).abs().mean(),
        "dif_hora_pico_h": _dif_hora(g["hora_pico_pred"], g["hora_pico_real"]).mean(),
    })


def resumen_por(df: pd.DataFrame, claves: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.groupby(claves, sort=False).apply(_agg_dias, include_groups=False).reset_index()


def resumen_semanal(df: pd.DataFrame, claves: List[str]) -> pd.DataFrame:
    """Error % de energía semanal (MAPE y sesgo) por claves (además de la semana)."""
    if df.empty:
        return pd.DataFrame()
    sem = (
        df.groupby(claves + ["semana_inicio"], sort=False)[["energia_pred", "energia_real"]].sum().reset_index()
    )
    sem["err"] = 100 * (sem["energia_pred"] / sem["energia_real"] - 1)
    return (
        sem.groupby(claves, sort=False)["err"]
        .agg(n_semanas="count", mape_energia_semana_pct=lambda s: s.abs().mean(), sesgo_energia_semana_pct="mean")
        .reset_index()
    )


def evaluar_alerta(
    df: pd.DataFrame, columna: str = "alerta_tendencia", columna_banda: str = "banda_pp"
) -> pd.DataFrame:
    """
    Utilidad de una alerta semanal (config default, curvas finales = forma):
    evento real = |error energía semanal del final| > banda (pp ~ %), con la
    banda tomada de `columna_banda`.
    `columna`: "alerta_tendencia" (TENDENCIA_FUERA_BANDA, enclavamiento),
    "alerta_tendencia_referencia" (hipotética: diagnóstico de tendencia de
    referencia con banda_pp = la alerta de v2) o
    "alerta_tendencia_referencia_info" (alerta info real del control, con
    `columna_banda="banda_referencia_pp"`).
    """
    d = df[(df["es_default"]) & (df["paso"] == "forma")]
    if d.empty or columna not in d.columns or columna_banda not in d.columns:
        return pd.DataFrame()
    sem = d.groupby(["base", "ucp", "semana_inicio"], sort=False).agg(
        e_pred=("energia_pred", "sum"), e_real=("energia_real", "sum"),
        alerta=(columna, "first"), banda=(columna_banda, "first"),
    ).reset_index()
    sem["alerta"] = sem["alerta"].astype(bool)
    sem["err_sem_pct"] = 100 * (sem["e_pred"] / sem["e_real"] - 1)
    sem["fuera_banda_real"] = sem["err_sem_pct"].abs() > sem["banda"].fillna(1.0)
    filas = []
    for (base, ucp), g in sem.groupby(["base", "ucp"], sort=False):
        tp = int((g["alerta"] & g["fuera_banda_real"]).sum())
        fp = int((g["alerta"] & ~g["fuera_banda_real"]).sum())
        fn = int((~g["alerta"] & g["fuera_banda_real"]).sum())
        tn = int((~g["alerta"] & ~g["fuera_banda_real"]).sum())
        filas.append({
            "base": base, "ucp": ucp, "semanas": len(g), "alertas": tp + fp,
            "fuera_banda_real": tp + fn, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "tasa_base": (tp + fn) / len(g),
            "err_sem_abs_con_alerta": g.loc[g["alerta"], "err_sem_pct"].abs().mean(),
            "err_sem_abs_sin_alerta": g.loc[~g["alerta"], "err_sem_pct"].abs().mean(),
        })
    return pd.DataFrame(filas)


def mejores_configs(df: pd.DataFrame) -> pd.DataFrame:
    """Mejor (alpha, fuente) por base/UCP/tipo de día según WAPE del paso final (forma)."""
    f = df[df["paso"] == "forma"]
    if f.empty:
        return pd.DataFrame()
    r = resumen_por(f, ["base", "ucp", "tipo_dia", "alpha", "fuente"])
    tot = resumen_por(f, ["base", "ucp", "alpha", "fuente"]).assign(tipo_dia="(todos)")
    r = pd.concat([tot, r], ignore_index=True)
    filas = []
    for (base, ucp, tipo), g in r.groupby(["base", "ucp", "tipo_dia"], sort=False):
        best = g.loc[g["wape_pct"].idxmin()]
        dflt = g[(g["alpha"] == CONFIG_DEFAULT[0]) & (g["fuente"] == CONFIG_DEFAULT[1])]
        filas.append({
            "base": base, "ucp": ucp, "tipo_dia": tipo, "n_dias": int(best["n_dias"]),
            "mejor_alpha": best["alpha"], "mejor_fuente": best["fuente"], "wape_mejor": best["wape_pct"],
            "wape_default": float(dflt["wape_pct"].iloc[0]) if not dflt.empty else float("nan"),
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def tabla_md(df: pd.DataFrame, decimales: int = 2) -> str:
    if df is None or df.empty:
        return "_(sin datos)_\n"
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "—" if math.isnan(v) else f"{v:.{decimales}f}"
        if isinstance(v, (bool, np.bool_)):
            return "sí" if v else "no"
        return str(v)

    lineas = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lineas += ["| " + " | ".join(fmt(v) for v in fila) + " |" for fila in df.itertuples(index=False)]
    return "\n".join(lineas) + "\n"


def escribir_reporte(df: pd.DataFrame, path: Path, meta: Dict) -> None:
    dflt = df[df["es_default"]]
    secciones: List[str] = [
        f"# Backtest del pipeline de ajustes — {meta['fecha']}\n",
        "Generado por `scripts/backtest_adjustments.py`. Detalle en "
        f"`{meta['csv_nombre']}` (una fila por UCP × semana × día × paso × config).\n",
        "## Configuración\n",
        f"- Bases: {', '.join(meta['bases'])}. Base usada y por qué: {meta['nota_base']}\n"
        f"- Semanas: últimas {meta['semanas']} completas por UCP; corte = sábado anterior a la semana "
        "objetivo (fecha_emision = domingo; el domingo es brecha estimada por la base).\n"
        f"- Barrido: alpha ∈ {list(meta['alphas'])}, fuente de forma ∈ {list(meta['fuentes'])}; "
        f"default = alpha {CONFIG_DEFAULT[0]}, `{CONFIG_DEFAULT[1]}`.\n"
        "- Sin fuga: el contexto se construye con el histórico recortado al corte y se verifica "
        "`max(historico.fecha) <= fecha_corte <= sábado`.\n",
    ]
    rangos = df.groupby(["base", "ucp"])["semana_inicio"].agg(["min", "max", "nunique"]).reset_index()
    rangos.columns = ["base", "ucp", "primera_semana", "ultima_semana", "n_semanas"]
    secciones += ["### Semanas evaluadas\n", tabla_md(rangos)]

    secciones += ["## 1. Resultado por paso (config default)\n",
                  "WAPE horario = Σ|pred−real| / Σreal. Energía diaria: MAPE y sesgo de E(d). "
                  "Pico: |error %| del máximo horario y diferencia absoluta de hora del pico.\n"]
    r = resumen_por(dflt, ["base", "ucp", "paso"])
    s = resumen_semanal(dflt, ["base", "ucp", "paso"])
    if not r.empty:
        r = r.merge(s, on=["base", "ucp", "paso"], how="left")
    secciones.append(tabla_md(r))

    secciones += ["### Por tipo de día (config default)\n",
                  tabla_md(resumen_por(dflt, ["base", "ucp", "tipo_dia", "paso"]))]

    secciones += ["## 2. Barrido alpha × fuente (WAPE % del paso final = forma)\n"]
    forma = df[df["paso"] == "forma"]
    for (base, ucp), g in forma.groupby(["base", "ucp"], sort=False):
        piv = resumen_por(g, ["alpha", "fuente"]).pivot(index="alpha", columns="fuente", values="wape_pct")
        piv = piv.reset_index()
        secciones += [f"### {base} — {ucp}\n", tabla_md(piv)]

    secciones += ["### Energía diaria del paso nivel por alpha (MAPE / sesgo %, no depende de la fuente)\n"]
    niv = df[(df["paso"] == "nivel") & (df["fuente"] == CONFIG_DEFAULT[1])]
    secciones.append(tabla_md(
        resumen_por(niv, ["base", "ucp", "alpha"])[
            ["base", "ucp", "alpha", "mape_energia_dia_pct", "sesgo_energia_dia_pct", "wape_pct"]
        ] if not niv.empty else pd.DataFrame()
    ))
    sem_alpha = resumen_semanal(niv, ["base", "ucp", "alpha"])
    secciones += ["### Energía semanal del paso nivel por alpha\n", tabla_md(sem_alpha)]

    secciones += ["### Mejor configuración por UCP y tipo de día (WAPE del final)\n", tabla_md(mejores_configs(df)),
                  nota_festivos(df)]

    secciones += [
        "## 3. Alertas de tendencia (config default)\n",
        "El control emite dos alertas de tendencia distintas: el **enclavamiento** "
        "`TENDENCIA_FUERA_BANDA` (warning, control de edición) y la **coherencia con la referencia** "
        "`TENDENCIA_REFERENCIA_FUERA_BANDA` (info, control de proceso).\n",
        "Evento real = |error de energía semanal del final| > banda de la alerta (pp ≈ %). "
        "precision = P(evento | alerta); recall = P(alerta | evento); "
        "tasa_base = P(evento) (precision debe superarla para que la alerta aporte).\n",
        "### Enclavamiento (TENDENCIA_FUERA_BANDA, warning)\n",
        "Compara la energía semanal final con la que fijó el paso nivel (desvio_pp = E_final/E_nivel − 1). "
        "Solo se dispara cuando algo posterior a nivel (ediciones del analista en nivel/forma, o una forma "
        "que no reconcilia) mueve la energía semanal. El backtest corre **sin ediciones**, así que lo "
        "esperado es 0 alertas: su utilidad solo se puede medir con ediciones reales del analista "
        "(datos de la Mesa de ajustes).\n",
        tabla_md(evaluar_alerta(df)),
        "### Coherencia con la referencia (TENDENCIA_REFERENCIA_FUERA_BANDA, info, default ±2 pp)\n",
        "Alerta real del control con los defaults (`banda_referencia_pp` = 2.0; null la desactiva): "
        "|tendencia resultante − tendencia de la semana de referencia| > 2 pp. Evento real con la misma "
        "banda (|error semanal| > 2 %).\n",
        tabla_md(evaluar_alerta(df, "alerta_tendencia_referencia_info", "banda_referencia_pp")),
        "### Hipotética: misma coherencia con banda ±1 pp (= alerta de v2)\n",
        "Se calcula a partir del diagnóstico `diferencia_tendencia_pp` con `banda_pp` (1 pp); no es una "
        "alerta que emita el control.\n",
        tabla_md(evaluar_alerta(df, "alerta_tendencia_referencia")),
    ]
    secciones += ["## 4. Recomendación automática (WAPE del final, todas las semanas)\n",
                  meta.get("recomendacion") or "_(sin datos)_\n"]
    secciones += ["## 5. Limitaciones\n", meta.get("limitaciones", "")]
    if meta.get("secciones_manuales"):
        secciones.append(meta["secciones_manuales"])
    path.write_text("\n".join(secciones), encoding="utf-8")


_RE_SECCION_MANUAL = re.compile(r"^## (\d+)\.", re.MULTILINE)
PRIMERA_SECCION_MANUAL = 6  # 1..5 las genera escribir_reporte; >= 6 son escritas a mano


def secciones_manuales(path: Path) -> str:
    """Texto desde la primera sección "## N." con N >= 6 de un reporte existente ('' si no hay)."""
    if not path.exists():
        return ""
    texto = path.read_text(encoding="utf-8")
    for m in _RE_SECCION_MANUAL.finditer(texto):
        if int(m.group(1)) >= PRIMERA_SECCION_MANUAL:
            return texto[m.start():].rstrip() + "\n"
    return ""


def _wape_festivo(df: pd.DataFrame, base: str, ucp: str, paso: str, **config) -> Optional[float]:
    g = df[(df["base"] == base) & (df["ucp"] == ucp) & (df["tipo_dia"] == "festivo") & (df["paso"] == paso)]
    for k, v in config.items():
        g = g[g[k] == v]
    if g.empty:
        return None
    return float(100 * g["abs_err_horario"].sum() / g["energia_real"].sum())


def nota_festivos(df: pd.DataFrame) -> str:
    """
    Advertencia sobre las filas de festivo de "mejor configuración por tipo de
    día" (números tomados del propio detalle).
    """
    f = df[df["tipo_dia"] == "festivo"]
    if f.empty:
        return ""
    lineas = ["**Cómo leer las filas de festivo** (no se deben usar para fijar α por tipo de día):\n"]
    dflt = {"es_default": True}
    for (base, ucp), g in f.groupby(["base", "ucp"], sort=False):
        n_dias = int(g[(g["es_default"]) & (g["paso"] == "forma")].shape[0])
        w_base = _wape_festivo(df, base, ucp, "base", **dflt)
        w_nivel = _wape_festivo(df, base, ucp, "nivel", **dflt)
        w_forma = _wape_festivo(df, base, ucp, "forma", **dflt)
        mejor = mejores_configs(df[(df["base"] == base) & (df["ucp"] == ucp)])
        mejor = mejor[mejor["tipo_dia"] == "festivo"]
        txt_mejor = (
            f"mejor α={mejor['mejor_alpha'].iloc[0]} `{mejor['mejor_fuente'].iloc[0]}` "
            f"{mejor['wape_mejor'].iloc[0]:.2f} % vs default {mejor['wape_default'].iloc[0]:.2f} %"
            if not mejor.empty else "sin mejor config"
        )
        pasos = " → ".join(f"{w:.2f} %" for w in (w_base, w_nivel, w_forma) if w is not None)
        if base == "ingenuo":
            lineas.append(
                f"- **{base} / {ucp}: NO interpretable.** La base ingenua pronostica cada festivo con un día "
                f"NO festivo (mismo día de la semana anterior), así que arranca con un error enorme "
                f"(WAPE festivo base→nivel→forma {pasos}, {n_dias} días). Cualquier α alto \"gana\" solo "
                f"porque corrige ese error artificial ({txt_mejor}); no dice nada de cómo se comporta α "
                f"sobre el pronóstico del modelo, que sí trata los festivos."
            )
        else:
            lineas.append(
                f"- **{base} / {ucp}: no concluyente.** Solo {n_dias} días festivos; WAPE festivo "
                f"base→nivel→forma {pasos} con el default ({txt_mejor}). La diferencia es de décimas de "
                f"punto sobre una muestra mínima y la forma default empeora levemente al nivel."
            )
    lineas.append(
        "- La recomendación \"α por tipo de día\" (p. ej. α alto en festivos, α=0 en domingos) queda "
        "**pendiente de un backtest con el modelo real**; con estas bases no se puede sostener.\n"
    )
    return "\n".join(lineas) + "\n"


def recomendacion_auto(df: pd.DataFrame) -> str:
    """Mejor config global por base/UCP vs default, y aporte de cada paso."""
    lineas = []
    f = df[df["paso"] == "forma"]
    for (base, ucp), g in f.groupby(["base", "ucp"], sort=False):
        r = resumen_por(g, ["alpha", "fuente"])
        best = r.loc[r["wape_pct"].idxmin()]
        d = r[(r["alpha"] == CONFIG_DEFAULT[0]) & (r["fuente"] == CONFIG_DEFAULT[1])]
        wd = float(d["wape_pct"].iloc[0]) if not d.empty else float("nan")
        pasos = resumen_por(df[(df["base"] == base) & (df["ucp"] == ucp) & df["es_default"]], ["paso"])
        w = dict(zip(pasos["paso"], pasos["wape_pct"]))
        lineas.append(
            f"- **{base} / {ucp}**: mejor alpha={best['alpha']}, fuente=`{best['fuente']}` "
            f"(WAPE {best['wape_pct']:.2f} %) vs default {wd:.2f} % "
            f"(diferencia {wd - best['wape_pct']:+.2f} pp). Default por paso: "
            + ", ".join(f"{p} {w[p]:.2f} %" for p in PASOS_EVALUADOS if p in w)
        )
    return "\n".join(lineas) + "\n"


LIMITACIONES = """\
- **Defaults provisionales, elegidos in-sample**: α = 0.25 y `promedio_n` se eligieron mirando el
  WAPE de estas mismas 52 semanas (no hay muestra de validación separada) → el WAPE del default es
  optimista. Deben revalidarse fuera de muestra y con el modelo real antes de fijarlos.
- **Base no-modelo**: ninguna base evaluada es el pronóstico que recibe el analista (ver abajo); la
  mejora relativa entre pasos sobre el modelo real puede ser distinta (probablemente menor).
- **Corte siempre en sábado**: todas las semanas usan corte = sábado previo (1 día de brecha). Otras
  fechas de emisión (más brecha) no están evaluadas; nivel/forma usan anclas más lejanas en ese caso.
- **Alerta de enclavamiento sin evaluar**: requiere ediciones del analista, que el backtest no tiene.
- **Base**: con `ingenuo` la base es la semana real anterior (no el modelo); mide cuánto aportan
  nivel y forma sobre una base débil. La mejora relativa sobre el modelo real será menor.
- **Modelo real no evaluado** (`--base modelo`): `run_predict_flow` llama a `full_update_csv`
  (red + reescribe `data/raw/{ucp}/datos.csv`), consulta festivos por red, y en el venv actual
  (pandas 3.0.6) falla en `src/pipeline/feature_engineering.py:493` (`fillna(method='ffill')`).
  Además los modelos guardados (models/Antioquia, entrenados 2026-07-08) vieron estas semanas →
  aunque corriera, el error base sería optimista (fuga de entrenamiento).
- **Base `archivo`**: una sola emisión con corte 2025-12-31 (horizonte crece hasta ~5 meses) y con
  ajustes post-hoc ("régimen +5.3 % desde 2026-03-25", calibrado mirando datos reales) → no es el
  pronóstico semanal que recibe el analista; solo Antioquia, ~21 semanas.
- **Clima**: no aplica a la base ingenua; para la base modelo se usaría clima observado (no hay
  pronóstico archivado) → optimista.
- **Festivos**: calendario oficial (`holidays`, sin red). Pocos festivos en 52 semanas (n chico por tipo).
- **Festivos con base `ingenuo` NO interpretables**: la base ingenua usa un día no festivo como
  pronóstico del festivo (WAPE festivo base ≈ 16 % en Antioquia, ≈ 11 % en Atlántico), así que la
  "mejor configuración" de festivos (α = 1) solo refleja que corrige un error artificial.
- **Festivos con base `archivo` no concluyentes**: ver la nota bajo "Mejor configuración por UCP y
  tipo de día" (6 días; diferencias de décimas de punto).
- **α por tipo de día**: pendiente de backtest con el modelo real; no se recomienda con estos datos.
- Semanas con días reales faltantes se excluyen; Atlántico termina 2026-05-07.
"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ucp", nargs="+", default=list(UCPS_DEFAULT))
    ap.add_argument("--semanas", type=int, default=52)
    ap.add_argument("--base", nargs="+", default=["ingenuo"], choices=["ingenuo", "archivo", "modelo"])
    ap.add_argument("--archivo", type=Path, default=ARCHIVO_DEFAULT)
    ap.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS_DEFAULT))
    ap.add_argument("--fuentes", nargs="+", default=list(FUENTES_DEFAULT))
    ap.add_argument("--sin-barrido", action="store_true", help="solo la config default")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "ajustes")
    ap.add_argument("--fecha", default=date.today().strftime("%Y%m%d"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    configs = [CONFIG_DEFAULT] if args.sin_barrido else [(a, f) for a in args.alphas for f in args.fuentes]
    if CONFIG_DEFAULT not in configs:
        configs.append(CONFIG_DEFAULT)

    frames = []
    for base_nombre in args.base:
        if base_nombre == "ingenuo":
            factory: ProviderFactory = lambda ucp, hist: proveedor_ingenuo(hist)
        elif base_nombre == "archivo":
            curvas_archivo = cargar_archivo(args.archivo)
            prov_archivo = proveedor_desde_curvas(curvas_archivo)
            factory = lambda ucp, hist, _p=prov_archivo: _p if ucp == ARCHIVO_UCP else None
        else:
            prov_modelo = proveedor_modelo()
            factory = lambda ucp, hist, _p=prov_modelo: _p

        for ucp in args.ucp:
            if base_nombre == "archivo" and ucp != ARCHIVO_UCP:
                continue
            historico = data.cargar_historico(ucp)
            semanas = semanas_objetivo(historico, args.semanas if base_nombre != "archivo" else 10_000)
            if base_nombre == "archivo":
                # solo semanas cuya brecha + semana estén cubiertas por el archivo
                semanas = [
                    l for l in semanas
                    if all((l - timedelta(days=1) + timedelta(days=i)).isoformat() in curvas_archivo
                           for i in range(8))
                ][-args.semanas:]
            logger.info("%s %s: %d semanas (%s .. %s)", base_nombre, ucp, len(semanas),
                        semanas[0] if semanas else "-", semanas[-1] if semanas else "-")
            frames.append(correr_backtest(ucp, historico, semanas, factory, base_nombre, configs))

    df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        logger.error("Backtest sin resultados")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"backtest_{args.fecha}.csv"
    md_path = args.out_dir / f"backtest_{args.fecha}.md"
    df.to_csv(csv_path, index=False)
    nota = (
        "`ingenuo` (y `archivo` como contraste para Antioquia) — `run_predict_flow` no es viable "
        "offline (ver Limitaciones)." if "modelo" not in args.base else "`modelo` (run_predict_flow)."
    )
    escribir_reporte(df, md_path, {
        "fecha": args.fecha, "csv_nombre": csv_path.name, "bases": args.base, "nota_base": nota,
        "semanas": args.semanas, "alphas": args.alphas, "fuentes": args.fuentes,
        "limitaciones": LIMITACIONES, "recomendacion": recomendacion_auto(df),
        # las secciones escritas a mano (## 6. en adelante) del reporte previo se preservan
        "secciones_manuales": secciones_manuales(md_path),
    })
    logger.info("Escrito %s (%d filas) y %s", csv_path, len(df), md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
