"""
Comparador ASR actual (Sphere) vs mesa de ajustes vs real.

Dos modos (desde la raíz del repo):

1) Una semana (p. ej. el caso real del analista):
    python scripts/comparar_asr.py --ucp Antioquia --semana 2026-11-16 \
        --base-csv base.csv [--analista-csv final.csv] [--fecha-emision 2026-11-14] \
        [--seleccion 2026-11-16=2026-11-09,2025-11-17,2025-11-10 ...] [--out semana.md]
   - `--base-csv`/`--analista-csv`: CSV largo (fecha, periodo P1..P24 o 1..24, valor) o ancho
     (fecha, P1..P24). Sin `--base-csv` se usa `--base ingenuo|archivo|modelo|modelo_k1`.
   - `--seleccion`: por día objetivo, las fechas de los históricos que marcó el analista EN ORDEN DE
     CLIC (el ASR usa h1 = el más reciente y congela desvi2 con el orden de clic). Sin `--seleccion`
     se usan los escenarios (`--escenarios`, default todos).
   Imprime (o escribe en --out) tablas por día y por hora.

2) Histórico (~52 semanas por UCP):
    python scripts/comparar_asr.py --historico --base ingenuo archivo --fecha 20261112
    python scripts/comparar_asr.py --historico --semanas 52 --base modelo_k1 --fecha 20261113_modelo
   Escribe reports/ajustes/asr_comparacion_<fecha>.csv (no se versiona) y .md (secciones 1..5
   automáticas; las secciones "## 6." en adelante del .md previo se conservan).

Sin red: festivos de la librería `holidays` (también para `is_special_day` del validador); el
validador usa los modelos de desagregación de models/{ucp} (solo lectura).
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adjustments import asr, data, pipeline  # noqa: E402
from src.adjustments import comparar_asr as cmp  # noqa: E402
from src.adjustments.types import Curves  # noqa: E402

logger = logging.getLogger("comparar_asr")

MODELO_DIR_DEFAULT = ROOT / "reports" / "ajustes"


def _cargar_backtest():
    """Reutiliza proveedores, semanas, contexto sin fuga y utilidades de reporte del backtest."""
    nombre = "backtest_adjustments"
    if nombre in sys.modules:
        return sys.modules[nombre]
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / "backtest_adjustments.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


bt = _cargar_backtest()


def series_principales(df: pd.DataFrame) -> List[str]:
    """base, mesa y el ASR con validador de cada escenario PRESENTE en `df` (respeta --escenarios)."""
    presentes = set(df["serie"])
    return ["base", "mesa"] + [cmp.nombre_asr(e) for e in asr.ESCENARIOS if cmp.nombre_asr(e) in presentes]


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------

def leer_curvas_csv(path: Path) -> Curves:
    """CSV largo (fecha, periodo, valor) o ancho (fecha, P1..P24) → Curves."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    fecha_col = cols.get("fecha") or df.columns[0]
    df[fecha_col] = pd.to_datetime(df[fecha_col]).dt.strftime("%Y-%m-%d")
    if all(f"P{i}" in df.columns for i in range(1, 25)):
        return {r[fecha_col]: [float(r[f"P{i}"]) for i in range(1, 25)] for _, r in df.iterrows()}
    per_col = cols.get("periodo") or cols.get("hora")
    val_col = next((cols[c] for c in ("valor", "pronostico_mwh", "mw", "mwh") if c in cols), None)
    if per_col is None or val_col is None:
        raise ValueError(f"{path}: se espera fecha + P1..P24, o fecha + periodo + valor")
    df["_p"] = df[per_col].astype(str).str.replace("P", "", regex=False).astype(int)
    tabla = df.pivot_table(index=fecha_col, columns="_p", values=val_col, aggfunc="first")
    tabla = tabla.reindex(columns=range(1, 25))
    return {f: fila.to_numpy(dtype=float).tolist() for f, fila in tabla.iterrows()}


def parsear_seleccion(items: Sequence[str]) -> Dict[str, List[str]]:
    """["2026-11-16=2026-11-09,2025-11-17,2025-11-10", ...] → {objetivo: [fechas en orden de clic]}."""
    out: Dict[str, List[str]] = {}
    for it in items or []:
        objetivo, _, resto = it.partition("=")
        fechas = [date.fromisoformat(x.strip()).isoformat() for x in resto.split(",") if x.strip()]
        if len(fechas) < 3:
            raise ValueError(f"--seleccion {it!r}: se necesitan al menos 3 históricos")
        out[date.fromisoformat(objetivo.strip()).isoformat()] = fechas
    return out


def _festivos(historico: pd.DataFrame) -> List[str]:
    anios = range(int(historico["fecha"].dt.year.min()) - 1, int(historico["fecha"].dt.year.max()) + 2)
    return data.festivos_colombia(anios)


def _modelo_k1(ucp: str, modelo_dir: Path):
    try:
        return bt.cargar_curvas_modelo_k1(ucp, modelo_dir)
    except FileNotFoundError:
        logger.warning("modelo_k1: sin curvas para %s en %s", ucp, modelo_dir)
        return None


def _factory(base: str, archivo: Path, modelo_dir: Path = MODELO_DIR_DEFAULT, cache: Optional[Dict] = None):
    if base == "ingenuo":
        return lambda ucp, hist: bt.proveedor_ingenuo(hist)
    if base == "archivo":
        prov = bt.proveedor_desde_curvas(bt.cargar_archivo(archivo))
        return lambda ucp, hist, _p=prov: _p if ucp == bt.ARCHIVO_UCP else None
    if base == "modelo":  # enchufable: depende de la Fase 2a (run_predict_flow offline)
        prov = bt.proveedor_modelo()
        return lambda ucp, hist, _p=prov: _p
    if base == "modelo_k1":
        cache = {} if cache is None else cache

        def factory(ucp, hist):
            if ucp not in cache:
                cache[ucp] = _modelo_k1(ucp, modelo_dir)
            cargadas = cache[ucp]
            return None if cargadas is None else bt.proveedor_desde_curvas(cargadas[0])

        return factory
    raise ValueError(f"base desconocida {base!r}")


# ---------------------------------------------------------------------------
# Modo semana
# ---------------------------------------------------------------------------

def _tabla_dias(res: Dict) -> pd.DataFrame:
    m = pd.DataFrame(res["metricas"])
    cols = [c for c in ["fecha", "tipo_dia", "serie", "energia_pred", "pico_pred", "hora_pico_pred", "wape_pct",
                        "err_energia_pct", "err_pico_pct", "dif_hora_pico_h", "err_forma_pct"] if c in m.columns]
    return m[cols].sort_values(["fecha", "serie"]) if not m.empty else m


def _tabla_horas(res: Dict, fecha: str) -> pd.DataFrame:
    filas = {"hora": list(range(1, 25))}
    for nombre, curvas in res["series"].items():
        if fecha in curvas:
            filas[nombre] = [round(float(v), 1) for v in curvas[fecha]]
    return pd.DataFrame(filas)


def _validar_csv(curvas: Curves, campo: str) -> None:
    try:
        cmp.validar_curvas(curvas, campo)
    except ValueError as e:
        raise SystemExit(str(e))


def modo_semana(args) -> int:
    historico = data.cargar_historico(args.ucp)
    festivos = _festivos(historico)
    reales_todos = data.curvas_reales(historico)
    lunes = date.fromisoformat(args.semana)
    if args.fecha_emision:
        emision = date.fromisoformat(args.fecha_emision)
        ctx = pipeline.build_context(args.ucp, lunes, fecha_emision=emision,
                                     historico=historico,
                                     festivos=festivos)
    else:
        ctx = bt.construir_contexto(args.ucp, lunes, historico, festivos)
    if args.base_csv:
        base = leer_curvas_csv(args.base_csv)
        necesarias = [(ctx.fecha_corte + timedelta(days=i)).isoformat()
                      for i in range(1, (lunes - ctx.fecha_corte).days + 7)]
        faltan = [f for f in necesarias if f not in base]
        if faltan:
            raise SystemExit(f"--base-csv debe cubrir {necesarias[0]}..{necesarias[-1]} (corte {ctx.fecha_corte}); "
                             f"faltan {faltan}")
        _validar_csv({f: base[f] for f in necesarias}, "--base-csv")
        provider = bt.proveedor_desde_curvas(base)
    else:
        provider = _factory(args.base[0], args.archivo, args.modelo_dir)(args.ucp, ctx.historico)
        if provider is None:
            raise SystemExit(f"La base {args.base[0]} no aplica a {args.ucp}")
    seleccion = None
    if args.seleccion:
        reales_corte = data.curvas_reales(ctx.historico)
        seleccion = {}
        for objetivo, fechas in parsear_seleccion(args.seleccion).items():
            faltan = [f for f in fechas if f not in reales_corte]
            if faltan:
                raise SystemExit(f"{objetivo}: históricos sin dato real ≤ corte {ctx.fecha_corte}: {faltan}")
            seleccion[objetivo] = [asr.Historico(f, list(reales_corte[f])) for f in fechas]
    analista = leer_curvas_csv(args.analista_csv) if args.analista_csv else None
    if analista:
        _validar_csv(analista, "--analista-csv")
    try:
        res = _comparar(ctx, provider, reales_todos, seleccion, analista, args)
    except ValueError as e:
        raise SystemExit(str(e))
    return _escribir_semana(res, args)


def _comparar(ctx, provider, reales_todos, seleccion, analista, args) -> Dict:
    return cmp.comparar_semana(
        ctx, provider, reales=reales_todos,
        escenarios=args.escenarios, seleccion=seleccion, festivos_especial=ctx.festivos,
        analista=analista, descomponer=args.descomponer,
    )


def _escribir_semana(res: Dict, args) -> int:
    lineas = [
        f"# Comparación ASR vs mesa — {args.ucp}, semana {res['semana_inicio']}\n",
        f"Corte {res['fecha_corte']}; semana de referencia de la mesa {res['semana_ref_inicio']} "
        f"(desplazamiento {res['desplazamiento_ref_dias']} días).\n",
        *[f"**Advertencia:** {a}\n" for a in res["advertencias"]],
        "## Por día\n", bt.tabla_md(_tabla_dias(res)),
        "## Semana\n", bt.tabla_md(pd.DataFrame(res["semana"])),
    ]
    if res["omitidos"]:
        lineas += ["## Omitidos\n"] + [f"- `{s}`: {om}" for s, om in res["omitidos"].items()] + [""]
    det = [{"serie": s, "fecha": f, **d} for s, dd in res["asr_detalle"].items()
           for f, d in dd.items()]
    lineas += ["## Detalle del ASR (h1, h2, h3 por fecha; desvi2 = a − b)\n", bt.tabla_md(pd.DataFrame(det))]
    if args.horas:
        for f in res["fechas"]:
            lineas += [f"## Horas {f}\n", bt.tabla_md(_tabla_horas(res, f), decimales=1)]
    texto = "\n".join(lineas)
    if args.out:
        args.out.write_text(texto, encoding="utf-8")
        logger.info("Escrito %s", args.out)
    else:
        sys.stdout.reconfigure(encoding="utf-8")  # la consola de Windows (cp1252) no imprime "−"
        print(texto)
    return 0


# ---------------------------------------------------------------------------
# Modo histórico
# ---------------------------------------------------------------------------

def correr_historico(ucp: str, historico: pd.DataFrame, semanas: Sequence[date], factory, base_nombre: str,
                     motor=None, senda_fn=None, festivos: Optional[List[str]] = None,
                     escenarios: Sequence[str] = asr.ESCENARIOS) -> pd.DataFrame:
    historico = data.normalizar_historico(historico, ucp=ucp)
    reales = data.curvas_reales(historico)
    festivos = festivos if festivos is not None else _festivos(historico)
    festivos_set = set(festivos)
    filas: List[Dict] = []
    for lunes in semanas:
        try:
            ctx = bt.construir_contexto(ucp, lunes, historico, festivos)
            provider = factory(ucp, ctx.historico)
            if provider is None:
                continue
            res = cmp.comparar_semana(
                ctx, bt.con_cache(provider), reales=reales, escenarios=escenarios, motor=motor,
                senda_fn=senda_fn, festivos_especial=festivos, descomponer=True,
            )
        except AssertionError:
            raise
        except Exception as exc:  # una semana rota no tumba la corrida
            logger.warning("%s %s %s: %s", base_nombre, ucp, lunes, exc)
            continue
        meta = {"ucp": ucp, "base": base_nombre, "semana_inicio": res["semana_inicio"],
                "fecha_corte": res["fecha_corte"], "semana_ref_inicio": res["semana_ref_inicio"],
                "desplazamiento_ref_dias": res["desplazamiento_ref_dias"]}
        for m in res["metricas"]:
            det = res["asr_detalle"].get(m["serie"], {}).get(m["fecha"], {})
            filas.append({
                **meta, **m,
                "periodos_modificados": det.get("periodos_modificados"),
                "cluster_id": det.get("cluster_id"), "metodo": det.get("metodo"),
                "energia_objetivo": det.get("energia_objetivo"),
                "h_fechas": "|".join(det.get("h_fechas", [])),
                # tipo de día de h1, h2, h3: un histórico de otro tipo (festivo vs laboral) contamina el delta
                "h_tipos": "|".join(data.tipo_dia(date.fromisoformat(h), festivos_set)
                                    for h in det.get("h_fechas", [])),
                "desvi2_fechas": "|".join(det.get("desvi2_fechas", [])),
            })
    return pd.DataFrame(filas)


def _agg(g: pd.DataFrame) -> pd.Series:
    err_e = g["energia_pred"] / g["energia_real"] - 1
    return pd.Series({
        "n_dias": len(g),
        "wape_pct": 100 * g["abs_err_horario"].sum() / g["energia_real"].sum(),
        "mape_energia_dia_pct": 100 * err_e.abs().mean(),
        "sesgo_energia_dia_pct": 100 * err_e.mean(),
        "err_forma_pct": g["err_forma_pct"].mean(),
        "err_pico_pct": g["err_pico_pct"].abs().mean(),
        "dif_hora_pico_h": g["dif_hora_pico_h"].mean(),
    })


def resumen(df: pd.DataFrame, claves: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.groupby(claves, sort=False).apply(_agg, include_groups=False).reset_index()


def resumen_semanal(df: pd.DataFrame, claves: List[str]) -> pd.DataFrame:
    """MAPE/sesgo de energía semanal (solo semanas con los 7 días en la serie)."""
    if df.empty:
        return pd.DataFrame()
    sem = df.groupby(claves + ["semana_inicio"], sort=False).agg(
        n=("fecha", "count"), e_pred=("energia_pred", "sum"), e_real=("energia_real", "sum")).reset_index()
    sem = sem[sem["n"] == 7]
    sem["err"] = 100 * (sem["e_pred"] / sem["e_real"] - 1)
    return sem.groupby(claves, sort=False)["err"].agg(
        n_semanas="count", mape_energia_semana_pct=lambda s: s.abs().mean(), sesgo_energia_semana_pct="mean"
    ).reset_index()


def comparacion_pareada(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """% de días en que la serie `a` tiene menor WAPE que `b` (mismos días), por base/UCP."""
    # índice "_base": la serie "base" chocaría con la columna "base" al hacer reset_index
    x = df[df["serie"].isin([a, b])].rename(columns={"base": "_base"}).pivot_table(
        index=["_base", "ucp", "semana_inicio", "fecha"], columns="serie", values="wape_pct")
    x = x.dropna()
    if x.empty or a not in x or b not in x:
        return pd.DataFrame()
    x = x.reset_index()
    filas = []
    for (base, ucp), g in x.groupby(["_base", "ucp"], sort=False):
        filas.append({"base": base, "ucp": ucp, "comparacion": f"{a} vs {b}", "n_dias": len(g),
                      f"gana_{a}_pct": 100 * float((g[a] < g[b]).mean()),
                      "dif_wape_media_pp": float((g[a] - g[b]).mean())})
    return pd.DataFrame(filas)


def dias_comunes(df: pd.DataFrame, series: Sequence[str]) -> pd.DataFrame:
    """Filas de `series` solo en los días (base, ucp, semana, fecha) donde TODAS tienen dato."""
    d = df[df["serie"].isin(series)]
    claves = ["base", "ucp", "semana_inicio", "fecha"]
    n = d.groupby(claves)["serie"].transform("nunique")
    return d[n == len(set(series))]


def correccion_vs_necesaria(df: pd.DataFrame, series: Sequence[str]) -> pd.DataFrame:
    """
    Corrección de energía diaria que aplica cada serie sobre la base (E_serie/E_base − 1) frente a la
    que hacía falta (E_real/E_base − 1): correlación, pendiente y desviación estándar. Una corrección
    útil correlaciona con la necesaria y es más chica que ella.
    """
    filas = []
    for (base, ucp), g in df.groupby(["base", "ucp"], sort=False):
        e = g.pivot_table(index=["semana_inicio", "fecha"], columns="serie", values="energia_pred")
        real = g[g["serie"] == "base"].set_index(["semana_inicio", "fecha"])["energia_real"]
        if "base" not in e:
            continue
        for s in series:
            if s not in e:
                continue
            x = pd.concat([e[s] / e["base"] - 1, real / e["base"] - 1], axis=1,
                          keys=["aplicada", "necesaria"]).dropna()
            if len(x) < 3:
                continue
            filas.append({
                "base": base, "ucp": ucp, "serie": s, "n_dias": len(x),
                "corr": float(np.corrcoef(x["aplicada"], x["necesaria"])[0, 1]),
                "pendiente": float(np.polyfit(x["aplicada"], x["necesaria"], 1)[0]),
                "std_aplicada_pct": 100 * float(x["aplicada"].std()),
                "std_necesaria_pct": 100 * float(x["necesaria"].std()),
            })
    return pd.DataFrame(filas)


def contaminacion(df: pd.DataFrame, serie: str = "asr_anual") -> pd.DataFrame:
    """WAPE de base, mesa y `serie` según si algún histórico (h1, h2, h3) es de otro tipo de día."""
    if "h_tipos" not in df:
        return pd.DataFrame()
    g = df[df["serie"] == serie]
    if g.empty:
        return pd.DataFrame()
    claves = ["base", "ucp", "semana_inicio", "fecha"]
    cont = pd.Series(
        [any(t != td for t in str(ht).split("|")) for td, ht in zip(g["tipo_dia"], g["h_tipos"])],
        index=pd.MultiIndex.from_frame(g[claves]), name="contaminado")
    filas = []
    abs_err = df[df["serie"].isin(["base", "mesa", serie])].set_index(claves + ["serie"])["abs_err_horario"]
    real = df[df["serie"] == "base"].set_index(claves)["energia_real"]
    for (base, ucp), c in cont.groupby(level=["base", "ucp"], sort=False):
        for valor in (False, True):
            idx = c[c == valor].index
            if len(idx) == 0:
                continue
            er = real.reindex(idx).sum()
            fila = {"base": base, "ucp": ucp, "historico_de_otro_tipo": valor, "n_dias": len(idx)}
            for s in ("base", "mesa", serie):
                err = abs_err.xs(s, level="serie").reindex(idx).sum()
                fila[f"wape_{s}"] = 100 * err / er
            filas.append(fila)
    return pd.DataFrame(filas)


def escribir_reporte(df: pd.DataFrame, path: Path, meta: Dict) -> None:
    principales = dias_comunes(df, series_principales(df))
    sec: List[str] = [
        f"# Comparación ASR actual vs mesa de ajustes vs real — {meta['fecha']}\n",
        "Generado por `scripts/comparar_asr.py --historico`. Detalle en "
        f"`{meta['csv_nombre']}` (no versionado; una fila por UCP × base × semana × día × serie).\n",
        "## 1. Configuración\n",
        f"- Bases: {', '.join(meta['bases'])}."
        + (" `modelo_k1` = pronóstico del modelo real de la Fase 2a, sin fuga (corte previo verificado)."
           if "modelo_k1" in meta["bases"] else "")
        + f" Semanas: últimas {meta['semanas']} completas por UCP; corte = "
        "sábado anterior (el domingo es brecha). Contexto sin fuga (histórico ≤ corte; los históricos del "
        "ASR salen de ese mismo histórico).\n"
        "- `mesa` = `run_pipeline` con los defaults vigentes sobre la misma base.\n"
        "- `asr_<escenario>` = ASR de Sphere reproducido (`src/adjustments/asr.py`) + validador del endpoint "
        "(models/{ucp}, festivos de librería para `is_special_day`). `_sin_validador` = objetivo h1 + desvi2 "
        "sin el validador. `_solo_nivel` = forma de la base con la energía del ASR; `_solo_forma` = forma del "
        "ASR con la energía de la base.\n"
        "- Escenarios (orden de clic): `anual` h1 = último mismo día de semana real, h2 = d−364, h3 = h1−364; "
        "`anual_alineado` = con el desplazamiento de la semana de referencia de la mesa (festivo homólogo); "
        "`anual_desfase` = h3 = d−371 aunque h1 sea d−14; `anual_clic_asc` = misma selección marcada de "
        "abajo arriba (desvi2 congelado = h2 − h1); `recientes` = los 3 últimos mismos días de semana.\n",
    ]
    rangos = df.groupby(["base", "ucp"])["semana_inicio"].agg(["min", "max", "nunique"]).reset_index()
    rangos.columns = ["base", "ucp", "primera_semana", "ultima_semana", "n_semanas"]
    sec += ["### Semanas evaluadas\n", bt.tabla_md(rangos)]

    r = resumen(principales, ["base", "ucp", "serie"])
    s = resumen_semanal(principales, ["base", "ucp", "serie"])
    if not r.empty:
        r = r.merge(s, on=["base", "ucp", "serie"], how="left")
    n_total = df[df["serie"] == "base"].groupby(["base", "ucp"]).size().rename("dias_total")
    n_com = principales[principales["serie"] == "base"].groupby(["base", "ucp"]).size().rename("dias_comunes")
    cobertura = pd.concat([n_total, n_com], axis=1).fillna(0).astype(int).reset_index()
    sec += ["### Cobertura\n", "Las secciones 2, 3 y 5 usan solo los días en que todas las series comparadas "
            "tienen dato (el ASR necesita h2/h3 de hace un año; Atlántico tiene huecos en 2024).\n",
            bt.tabla_md(cobertura)]
    sec += ["## 2. Error por serie (días comunes)\n",
            "WAPE = Σ|pred−real|/Σreal. Energía día/semana: MAPE y sesgo. err_forma = 100·Σ|p/Σp − r/Σr| "
            "(media diaria). Pico: |error %| del máximo y diferencia de hora.\n", bt.tabla_md(r)]

    tipo = principales[principales["serie"].isin(["base", "mesa", "asr_anual", "asr_anual_alineado",
                                                    "asr_recientes"])]
    piv = resumen(tipo, ["base", "ucp", "tipo_dia", "serie"])
    if not piv.empty:
        piv["serie"] = "wape_" + piv["serie"]  # la serie "base" chocaría con la columna "base"
        piv = piv.pivot_table(index=["base", "ucp", "tipo_dia"], columns="serie", values="wape_pct").reset_index()
        n = tipo[tipo["serie"] == "base"].groupby(["base", "ucp", "tipo_dia"]).size().rename("n_dias").reset_index()
        piv = piv.merge(n, on=["base", "ucp", "tipo_dia"], how="left")
    sec += ["## 3. WAPE % por tipo de día\n", bt.tabla_md(piv)]

    pares = pd.concat([comparacion_pareada(df, "asr_anual", "base"), comparacion_pareada(df, "asr_anual", "mesa"),
                       comparacion_pareada(df, "mesa", "base")], ignore_index=True)
    sec += ["### Comparación día a día (WAPE)\n", bt.tabla_md(pares)]

    val = []
    for e in asr.ESCENARIOS:
        con, sin = cmp.nombre_asr(e), cmp.nombre_asr(e, False)
        g = df[df["serie"].isin([con, sin])]
        if g.empty:
            continue
        w = resumen(g, ["base", "ucp", "serie"]).pivot_table(index=["base", "ucp"], columns="serie",
                                                             values="wape_pct").reset_index()
        mod = df[df["serie"] == con].groupby(["base", "ucp"]).agg(
            dias_con_clip_pct=("periodos_modificados", lambda x: 100 * float((x > 0).mean())),
            periodos_modificados_media=("periodos_modificados", "mean"),
            dias_especiales=("metodo", lambda x: int((x == "special").sum()))).reset_index()
        w = w.merge(mod, on=["base", "ucp"], how="left").rename(columns={con: "wape_con_validador",
                                                                          sin: "wape_sin_validador"})
        w.insert(2, "escenario", e)
        val.append(w)
    sec += ["## 4. Efecto del validador (cluster ±15 %)\n",
            bt.tabla_md(pd.concat(val, ignore_index=True) if val else pd.DataFrame())]

    dec = dias_comunes(df, ["base", "asr_anual", "asr_anual_solo_nivel", "asr_anual_solo_forma", "mesa"])
    sec += ["## 5. Descomposición nivel / forma del ASR `anual`\n",
            "`solo_nivel`: la base reescalada a la energía diaria del ASR (aporta solo el nivel). `solo_forma`: "
            "la forma del ASR con la energía de la base (aporta solo la forma).\n",
            bt.tabla_md(resumen(dec, ["base", "ucp", "serie"]))]
    sec += ["### Corrección de energía aplicada vs necesaria\n",
            "aplicada = E_serie/E_base − 1; necesaria = E_real/E_base − 1 (por día). Una corrección útil "
            "correlaciona con la necesaria y es más chica que ella.\n",
            bt.tabla_md(correccion_vs_necesaria(df, ["mesa", "asr_anual_sin_validador",
                                                     "asr_anual_alineado_sin_validador",
                                                     "asr_recientes_sin_validador"]))]
    sec += ["### Históricos de otro tipo de día en h1, h2 o h3 (`asr_anual`)\n",
            "Día contaminado = alguno de h1, h2, h3 es de distinto tipo (laboral/sábado/domingo/festivo) que el "
            "día objetivo (p. ej. h1 festivo para un laboral, o h3 festivo del año anterior).\n",
            bt.tabla_md(contaminacion(df))]
    if meta.get("secciones_manuales"):
        sec.append(meta["secciones_manuales"])
    path.write_text("\n".join(sec), encoding="utf-8")


def modo_historico(args) -> int:
    frames = []
    for base_nombre in args.base:
        cache: Dict = {}
        factory = _factory(base_nombre, args.archivo, args.modelo_dir, cache)
        for ucp in args.ucp:
            if base_nombre == "archivo" and ucp != bt.ARCHIVO_UCP:
                continue
            historico = data.cargar_historico(ucp)
            if base_nombre == "modelo_k1":
                cargadas = _modelo_k1(ucp, args.modelo_dir)
                cache[ucp] = cargadas
                if cargadas is None:
                    continue
                semanas = bt.semanas_modelo_k1(historico, args.semanas, cargadas[1], cargadas[0])
            else:
                semanas = bt.semanas_objetivo(historico, args.semanas if base_nombre != "archivo" else 10_000)
                if base_nombre == "archivo":
                    cubiertas = bt.cargar_archivo(args.archivo)
                    semanas = [l for l in semanas
                               if all((l - timedelta(days=1) + timedelta(days=i)).isoformat() in cubiertas
                                      for i in range(8))][-args.semanas:]
            logger.info("%s %s: %d semanas", base_nombre, ucp, len(semanas))
            motor = asr.cargar_motor(ucp)
            frames.append(correr_historico(ucp, historico, semanas, factory, base_nombre, motor=motor,
                                           escenarios=args.escenarios))
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        logger.error("Sin resultados")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"asr_comparacion_{args.fecha}.csv"
    md_path = args.out_dir / f"asr_comparacion_{args.fecha}.md"
    df.to_csv(csv_path, index=False)
    escribir_reporte(df, md_path, {"fecha": args.fecha, "csv_nombre": csv_path.name, "bases": args.base,
                                   "semanas": args.semanas, "secciones_manuales": bt.secciones_manuales(md_path)})
    logger.info("Escrito %s (%d filas) y %s", csv_path, len(df), md_path)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--historico", action="store_true", help="corrida sobre N semanas por UCP")
    ap.add_argument("--ucp", nargs="+", default=list(bt.UCPS_DEFAULT))
    ap.add_argument("--semana", help="lunes de la semana objetivo (modo semana)")
    ap.add_argument("--fecha-emision", help="modo semana: corte = último dato real anterior a esta fecha")
    ap.add_argument("--semanas", type=int, default=52)
    ap.add_argument("--base", nargs="+", default=["ingenuo"],
                    choices=["ingenuo", "archivo", "modelo", "modelo_k1"])
    ap.add_argument("--modelo-dir", type=Path, default=MODELO_DIR_DEFAULT,
                    help="directorio de curvas_modelo_k1_<ucp>.csv")
    ap.add_argument("--archivo", type=Path, default=bt.ARCHIVO_DEFAULT)
    ap.add_argument("--base-csv", type=Path, help="modo semana: pronóstico base del analista")
    ap.add_argument("--analista-csv", type=Path, help="modo semana: resultado final ajustado a mano")
    ap.add_argument("--seleccion", nargs="+", help="OBJETIVO=F1,F2,F3 (históricos en orden de clic)")
    ap.add_argument("--escenarios", nargs="+", default=list(asr.ESCENARIOS), choices=list(asr.ESCENARIOS))
    ap.add_argument("--descomponer", action="store_true", help="modo semana: series solo_nivel/solo_forma")
    ap.add_argument("--horas", action="store_true", help="modo semana: tabla hora a hora por día")
    ap.add_argument("--out", type=Path, help="modo semana: archivo .md de salida (default: stdout)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "ajustes")
    ap.add_argument("--fecha", default=date.today().strftime("%Y%m%d"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("src").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
    if args.historico:
        return modo_historico(args)
    if not args.semana or len(args.ucp) != 1:
        ap.error("modo semana: indique --semana y una sola --ucp (o use --historico)")
    args.ucp = args.ucp[0]
    return modo_semana(args)


if __name__ == "__main__":
    raise SystemExit(main())
