"""
Backtest del buscador de días análogos (R3, Fase 2b) dentro del pipeline de ajustes.

Compara, con holdout temporal, las fuentes de forma y el ancla de nivel por
análogos contra los defaults actuales del pipeline:

- Forma (con el nivel default): `promedio_n` (default, n=4), `referencia` y
  `analogos` automático con variantes de k (`n`) y de ventana de temporada
  (`filtros_analogos.temporada_dias`).
- Nivel (con la forma default): ancla `actual` vs `analogos`, α 0.25 (default) y 0.5.
- Combinación: la mejor variante de forma + la mejor de nivel elegidas en
  CALIBRACIÓN (base ingenuo, ambas UCP), si las dos difieren del default.

Para cada UCP y base, últimas N semanas completas (default 104); corte =
sábado anterior (igual que `construir_contexto` de `backtest_adjustments.py`,
que verifica que no haya fuga). Calibración = la mitad más antigua (52) y
holdout = la mitad más reciente (52). La base `archivo` (Antioquia, ene–may 2026)
se reporta aparte (partición "archivo"): no se elige nada ahí.

Cada día se etiqueta con `analogos.descriptor_dia` (festivos del contexto):
tipo de día, relación con festivo, distancia, Semana Santa, diciembre y "normal".

Regla para proponer una variante como default de un segmento (PLAN §5): gana en
calibración Y holdout, en ambas UCP con base ingenuo, por >= `--umbral` % (2 %)
relativo del WAPE del paso afectado (forma para fuentes de forma; nivel y forma
para el ancla/α de nivel) y sin empeorar el error de pico. El script SOLO
recomienda: no cambia defaults.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe scripts/backtest_analogos.py                     # ingenuo + archivo, 104 semanas
    .venv/Scripts/python.exe scripts/backtest_analogos.py --semanas 52 --base modelo_k1 --base-regla modelo_k1
    .venv/Scripts/python.exe scripts/backtest_analogos.py --ucp Antioquia --semanas 4 --ns 5 --temporadas 0
    .venv/Scripts/python.exe scripts/backtest_analogos.py --desde-csv reports/ajustes/backtest_analogos_20261113.csv --fecha 20261113  # solo regenera el .md

Salida: reports/ajustes/backtest_analogos_<AAAAMMDD>.csv (detalle; no se
commitea) y reports/ajustes/backtest_analogos_<AAAAMMDD>.md (resumen).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.adjustments import analogos, data, pipeline  # noqa: E402
from backtest_adjustments import (  # noqa: E402
    ARCHIVO_DEFAULT,
    ARCHIVO_UCP,
    UCPS_DEFAULT,
    _agg_dias,
    cargar_archivo,
    cargar_curvas_modelo_k1,
    con_cache,
    construir_contexto,
    metricas_dia,
    proveedor_desde_curvas,
    proveedor_ingenuo,
    proveedor_modelo,
    resumen_semanal,
    semanas_modelo_k1,
    semanas_objetivo,
    tabla_md,
)

logger = logging.getLogger("backtest_analogos")

ProviderFactory = Any  # Callable[[str, pd.DataFrame], Optional[BaseProvider]]

NS_DEFAULT = (3, 5, 8)
TEMPORADAS_DEFAULT = (0, 45)  # 0 = sin ventana de temporada
ALPHAS_DEFAULT = (0.25, 0.5)
UMBRAL_DEFAULT = 2.0
MODELO_DIR_DEFAULT = ROOT / "reports" / "ajustes"
BASES_REGLA = ("ingenuo", "modelo_k1", "archivo")
PARTICIONES = ("calibracion", "holdout")
SEGMENTOS_DIFICILES = (
    "rel:festivo", "rel:despues", "despues_dist1", "rel:antes", "rel:puente", "semana_santa", "diciembre",
)


# ---------------------------------------------------------------------------
# Configuraciones
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    nombre: str
    grupo: str  # "default" | "forma" | "nivel" | "combo"
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict, hash=False, compare=False)

    def pasos_afectados(self) -> Tuple[str, ...]:
        return ("forma",) if self.grupo == "forma" else ("nivel", "forma")


DEFAULT = Config("default", "default", {})


def _filtros_forma(temporada: int) -> Optional[Dict[str, Any]]:
    if not temporada:
        return None  # = FILTROS_AUTO del paso
    return {"dia_semana": "auto", "relacion_festivo": "auto", "distancia_festivo": "auto",
            "temporada_dias": int(temporada)}


def configs_barrido(ns: Sequence[int], temporadas: Sequence[int], alphas: Sequence[float]) -> List[Config]:
    """default + variantes de forma (nivel default) + variantes de nivel (forma default)."""
    out = [DEFAULT, Config("forma_referencia", "forma", {"forma": {"fuente": "referencia"}})]
    for n in ns:
        for t in temporadas:
            forma = {"fuente": "analogos", "n": int(n)}
            filtros = _filtros_forma(t)
            if filtros is not None:
                forma["filtros_analogos"] = filtros
            out.append(Config(f"forma_analogos_n{n}" + (f"_t{t}" if t else ""), "forma", {"forma": forma}))
    for ancla in ("actual", "analogos"):
        for a in alphas:
            if ancla == "actual" and a == 0.25:
                continue  # = default
            nivel: Dict[str, Any] = {"alpha": float(a)}
            if ancla == "analogos":
                nivel["ancla"] = "analogos"
            out.append(Config(f"nivel_{ancla}_a{int(round(a * 100)):03d}", "nivel", {"nivel": nivel}))
    return out


def config_combo(forma: Config, nivel: Config) -> Config:
    return Config(f"combo[{forma.nombre}+{nivel.nombre}]", "combo", {**forma.params, **nivel.params})


# ---------------------------------------------------------------------------
# Particiones y segmentos
# ---------------------------------------------------------------------------

def particionar(semanas: Sequence[date]) -> Dict[date, str]:
    """Mitad más antigua = calibración, mitad más reciente = holdout (impar: la extra va a holdout)."""
    orden = sorted(semanas)
    n_cal = len(orden) // 2
    return {l: ("calibracion" if i < n_cal else "holdout") for i, l in enumerate(orden)}


def etiquetas_dia(d: date, festivos) -> Dict[str, Any]:
    """Columnas de segmentación de un día (a partir de `analogos.descriptor_dia`)."""
    desc = analogos.descriptor_dia(d, festivos)
    rel = desc["relacion_festivo"]
    ss = bool(desc["semana_santa"])
    dic = d.month == 12
    return {
        "tipo_dia": data.tipo_dia(d, set(festivos)),
        "relacion_festivo": rel,
        "distancia_festivo": desc.get("distancia_festivo"),
        "semana_santa": ss,
        "diciembre": dic,
        "normal": rel == "ninguna" and not ss and not dic,
    }


def mascaras_segmentos(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """{segmento: máscara booleana} sobre filas día-nivel."""
    m: Dict[str, pd.Series] = {"todos": pd.Series(True, index=df.index)}
    for t in ("laboral", "sabado", "domingo", "festivo"):
        m[f"tipo:{t}"] = df["tipo_dia"] == t
    for r in ("festivo", "puente", "antes", "despues", "ninguna"):
        m[f"rel:{r}"] = df["relacion_festivo"] == r
    dist = pd.to_numeric(df["distancia_festivo"], errors="coerce")
    m["despues_dist1"] = (df["relacion_festivo"] == "despues") & (dist == 1)
    m["semana_santa"] = df["semana_santa"].astype(bool)
    m["diciembre"] = df["diciembre"].astype(bool)
    m["normal"] = df["normal"].astype(bool)
    return m


def resumen_segmentos(df: pd.DataFrame, claves: List[str], segmentos: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Métricas (`_agg_dias`) por segmento × claves."""
    if df.empty:
        return pd.DataFrame()
    frames = []
    for seg, mask in mascaras_segmentos(df).items():
        if segmentos is not None and seg not in segmentos:
            continue
        sub = df[mask]
        if sub.empty:
            continue
        r = sub.groupby(claves, sort=False).apply(_agg_dias, include_groups=False).reset_index()
        r.insert(0, "segmento", seg)
        frames.append(r)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Núcleo
# ---------------------------------------------------------------------------

def _codigos(records) -> List[str]:
    return sorted({a.codigo for r in records for a in r.result.alerts})


def evaluar_semana(
    ucp: str,
    lunes: date,
    historico: pd.DataFrame,
    reales: Dict[str, np.ndarray],
    provider_factory: ProviderFactory,
    base_nombre: str,
    configs: Sequence[Config],
    festivos: List[str],
    particion: str,
) -> List[Dict]:
    """Filas (día × paso × config) de una semana. El contexto y la base se calculan una vez."""
    ctx = construir_contexto(ucp, lunes, historico, festivos)  # verifica que no haya fuga
    provider = provider_factory(ucp, ctx.historico)
    if provider is None:
        return []
    provider = con_cache(provider)
    corte = ctx.fecha_corte
    fechas = [d.isoformat() for d in data.dias_semana(lunes)]
    etiquetas = {f: etiquetas_dia(date.fromisoformat(f), ctx.festivos) for f in fechas}

    filas: List[Dict] = []
    for cfg in configs:
        records = pipeline.run_pipeline(ctx, provider, params=cfg.params)
        por_paso = {r.paso: r for r in records}
        if any(f not in por_paso["base"].result.curves for f in fechas):
            logger.warning("%s %s %s base incompleta; semana omitida", base_nombre, ucp, lunes)
            return []
        codigos = "|".join(_codigos(records))
        diag_forma = por_paso["forma"].result.diagnostics.get("dias", {})
        diag_nivel = por_paso["nivel"].result.diagnostics.get("dias", {})
        # Sin fuga: los análogos usados deben ser <= corte.
        for f in fechas:
            usados = list(diag_forma.get(f, {}).get("fechas_fuente") or [])
            usados += [a.get("fecha") for a in (diag_nivel.get(f, {}).get("analogos") or []) if a.get("fecha")]
            if any(u > corte.isoformat() for u in usados):
                raise AssertionError(f"Fuga: análogos > corte {corte} en {f} ({cfg.nombre})")
        pasos = ("base", "nivel", "forma") if cfg.grupo == "default" else ("nivel", "forma")
        for paso in pasos:
            curves = por_paso[paso].result.curves
            for f in fechas:
                filas.append({
                    "ucp": ucp, "base": base_nombre, "particion": particion,
                    "semana_inicio": lunes.isoformat(), "fecha_corte": corte.isoformat(),
                    "config": cfg.nombre, "grupo": cfg.grupo, "paso": paso, "fecha": f,
                    "dia_semana": date.fromisoformat(f).weekday(),
                    **etiquetas[f],
                    "forma_fuente_usada": diag_forma.get(f, {}).get("fuente"),
                    "nivel_regla": diag_nivel.get(f, {}).get("regla"),
                    "alertas": codigos,
                    **metricas_dia(curves[f], reales[f]),
                })
    return filas


def correr(
    ucp: str,
    historico: pd.DataFrame,
    semanas: Sequence[date],
    particiones: Dict[date, str],
    provider_factory: ProviderFactory,
    base_nombre: str,
    configs: Sequence[Config],
) -> pd.DataFrame:
    historico = data.normalizar_historico(historico, ucp=ucp)
    reales = data.curvas_reales(historico)
    anios = range(int(historico["fecha"].dt.year.min()) - 1, int(historico["fecha"].dt.year.max()) + 2)
    festivos = data.festivos_colombia(anios)
    filas: List[Dict] = []
    t0 = time.perf_counter()
    for i, lunes in enumerate(semanas, 1):
        try:
            filas.extend(evaluar_semana(ucp, lunes, historico, reales, provider_factory, base_nombre,
                                        configs, festivos, particiones[lunes]))
        except AssertionError:
            raise
        except Exception as exc:  # una semana rota no tumba el backtest
            logger.warning("%s %s %s: error %s", base_nombre, ucp, lunes, exc)
        if i % 10 == 0 or i == len(semanas):
            logger.info("%s %s: %d/%d semanas (%.0f s)", base_nombre, ucp, i, len(semanas),
                        time.perf_counter() - t0)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------

def _wape(g: pd.DataFrame) -> float:
    return float(100 * g["abs_err_horario"].sum() / g["energia_real"].sum()) if len(g) else float("nan")


def tabla_comparacion(df: pd.DataFrame, segmentos: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Por base/ucp/partición/segmento/config/paso: WAPE y pico de la variante y del
    default (mismo paso, mismos días) y mejora relativa de WAPE en % (positivo = mejor).
    """
    r = resumen_segmentos(df[df["paso"].isin(["nivel", "forma"])],
                          ["base", "ucp", "particion", "config", "grupo", "paso"], segmentos)
    if r.empty:
        return r
    d = r[r["config"] == DEFAULT.nombre][["segmento", "base", "ucp", "particion", "paso", "wape_pct", "err_pico_pct"]]
    d = d.rename(columns={"wape_pct": "wape_default", "err_pico_pct": "pico_default"})
    out = r.merge(d, on=["segmento", "base", "ucp", "particion", "paso"], how="left")
    out["mejora_rel_pct"] = 100 * (out["wape_default"] - out["wape_pct"]) / out["wape_default"]
    out["delta_pico_pp"] = out["err_pico_pct"] - out["pico_default"]
    return out


def evaluar_regla(
    comp: pd.DataFrame, ucps: Sequence[str], umbral: float, configs: Sequence[Config],
    base_regla: str = "ingenuo",
) -> pd.DataFrame:
    """
    Regla del PLAN §5 por segmento × variante: gana en calibración y holdout, en todas
    las `ucps`, por >= umbral % relativo de WAPE en cada paso afectado y sin empeorar el
    error de pico (delta_pico_pp <= 0) en esos pasos. `base_regla` elige con qué base se
    evalúa la regla (default `ingenuo`, sin cambiar los defaults del producto).
    """
    c = comp[comp["base"] == base_regla]
    grupos = {cfg.nombre: cfg for cfg in configs}
    filas = []
    for (seg, nombre), g in c.groupby(["segmento", "config"], sort=False):
        cfg = grupos.get(nombre)
        if cfg is None or cfg.grupo == "default":
            continue
        pasos = cfg.pasos_afectados()
        celdas, mejoras, picos, n_dias = 0, [], [], []
        ok = True
        for ucp in ucps:
            for part in PARTICIONES:
                for paso in pasos:
                    x = g[(g["ucp"] == ucp) & (g["particion"] == part) & (g["paso"] == paso)]
                    if x.empty:
                        ok = False
                        continue
                    celdas += 1
                    m, dp = float(x["mejora_rel_pct"].iloc[0]), float(x["delta_pico_pp"].iloc[0])
                    mejoras.append(m)
                    picos.append(dp)
                    n_dias.append(int(x["n_dias"].iloc[0]))
                    if not (m >= umbral and dp <= 0):
                        ok = False
        esperadas = len(ucps) * len(PARTICIONES) * len(pasos)
        filas.append({
            "segmento": seg, "config": nombre, "pasos": "+".join(pasos),
            "n_dias_min": min(n_dias) if n_dias else 0,
            "celdas": f"{celdas}/{esperadas}",
            "celdas_que_ganan": sum(1 for m, p in zip(mejoras, picos) if m >= umbral and p <= 0),
            "mejora_min_pct": min(mejoras) if mejoras else float("nan"),
            "mejora_media_pct": float(np.mean(mejoras)) if mejoras else float("nan"),
            "delta_pico_max_pp": max(picos) if picos else float("nan"),
            "cumple_regla": bool(ok and celdas == esperadas),
        })
    return pd.DataFrame(filas)


def elegir_en_calibracion(
    comp: pd.DataFrame, grupo: str, configs: Sequence[Config], base: str = "ingenuo"
) -> Optional[Config]:
    """
    Mejor variante del `grupo` en calibración (segmento "todos", base `base`):
    mayor mejora relativa media del WAPE del paso final (forma) sobre las UCP.
    None si ninguna mejora en promedio.
    """
    nombres = {c.nombre: c for c in configs if c.grupo == grupo}
    c = comp[(comp["base"] == base) & (comp["particion"] == "calibracion") & (comp["segmento"] == "todos")
             & (comp["paso"] == "forma") & (comp["config"].isin(list(nombres)))]
    if c.empty:
        return None
    media = c.groupby("config")["mejora_rel_pct"].mean().sort_values(ascending=False)
    if media.iloc[0] <= 0:
        return None
    return nombres[media.index[0]]


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def _pivot_wape(comp: pd.DataFrame, paso: str, segmento: str, configs: Sequence[str]) -> pd.DataFrame:
    c = comp[(comp["paso"] == paso) & (comp["segmento"] == segmento) & (comp["config"].isin(list(configs)))]
    if c.empty:
        return pd.DataFrame()
    c = c.assign(col=c["base"] + "·" + c["ucp"] + "·" + c["particion"])
    piv = c.pivot_table(index="config", columns="col", values="wape_pct", aggfunc="first")
    piv = piv.reindex([x for x in configs if x in piv.index])
    cols = sorted(piv.columns, key=lambda s: (not s.startswith("ingenuo"), s))
    return piv[cols].reset_index()


def _pivot_n(comp: pd.DataFrame, segmentos: Sequence[str]) -> pd.DataFrame:
    c = comp[(comp["config"] == DEFAULT.nombre) & (comp["paso"] == "forma") & (comp["segmento"].isin(list(segmentos)))]
    if c.empty:
        return pd.DataFrame()
    c = c.assign(col=c["base"] + "·" + c["ucp"] + "·" + c["particion"])
    piv = c.pivot_table(index="segmento", columns="col", values="n_dias", aggfunc="first")
    piv = piv.reindex([s for s in segmentos if s in piv.index])
    cols = sorted(piv.columns, key=lambda s: (not s.startswith("ingenuo"), s))
    return piv[cols].reset_index()


def _pivot_mejora(comp: pd.DataFrame, segmento: str, configs: Sequence[Config]) -> pd.DataFrame:
    """Mejora relativa % del WAPE (paso forma; para nivel/combo también nivel) vs default."""
    filas = []
    for cfg in configs:
        if cfg.grupo == "default":
            continue
        for paso in cfg.pasos_afectados():
            c = comp[(comp["config"] == cfg.nombre) & (comp["paso"] == paso) & (comp["segmento"] == segmento)]
            if c.empty:
                continue
            fila = {"config": cfg.nombre, "paso": paso}
            for _, x in c.iterrows():
                fila[f"{x['base']}·{x['ucp']}·{x['particion']}"] = x["mejora_rel_pct"]
            filas.append(fila)
    if not filas:
        return pd.DataFrame()
    df = pd.DataFrame(filas)
    cols = ["config", "paso"] + sorted([c for c in df.columns if c not in ("config", "paso")],
                                       key=lambda s: (not s.startswith("ingenuo"), s))
    return df[cols]


def _detalle_global(df: pd.DataFrame, configs: Sequence[Config]) -> pd.DataFrame:
    """Todas las métricas del paso afectado (forma; nivel también para nivel/combo) por config."""
    frames = []
    for cfg in configs:
        pasos = ("base", "nivel", "forma") if cfg.grupo == "default" else cfg.pasos_afectados()
        sub = df[(df["config"] == cfg.nombre) & (df["paso"].isin(pasos))]
        if sub.empty:
            continue
        r = sub.groupby(["base", "ucp", "particion", "config", "paso"], sort=False).apply(
            _agg_dias, include_groups=False).reset_index()
        s = resumen_semanal(sub, ["base", "ucp", "particion", "config", "paso"])
        frames.append(r.merge(s, on=["base", "ucp", "particion", "config", "paso"], how="left"))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["base", "ucp", "particion"], kind="stable", ascending=[False, True, True])


def _uso_fuentes(df: pd.DataFrame) -> pd.DataFrame:
    """% de días en que la fuente/regla pedida se usó de verdad (fallbacks)."""
    filas = []
    f = df[(df["paso"] == "forma") & (df["grupo"].isin(["forma", "combo"])) & df["config"].str.contains("analogos")]
    for (base, ucp, cfg), g in f.groupby(["base", "ucp", "config"], sort=False):
        filas.append({"base": base, "ucp": ucp, "config": cfg, "dias": len(g),
                      "pct_forma_analogos": 100 * (g["forma_fuente_usada"] == "analogos").mean(),
                      "pct_forma_otra": 100 * (g["forma_fuente_usada"] != "analogos").mean()})
    n = df[(df["paso"] == "nivel") & df["config"].str.contains("nivel_analogos")]
    for (base, ucp, cfg), g in n.groupby(["base", "ucp", "config"], sort=False):
        filas.append({"base": base, "ucp": ucp, "config": cfg, "dias": len(g),
                      "pct_nivel_regla_analogos": 100 * (g["nivel_regla"] == "analogos").mean()})
    return pd.DataFrame(filas)


def escribir_reporte(df: pd.DataFrame, comp: pd.DataFrame, regla: pd.DataFrame, configs: Sequence[Config],
                     path: Path, meta: Dict[str, Any]) -> None:
    nombres = [c.nombre for c in configs]
    base_regla = meta.get("base_regla", "ingenuo")
    s: List[str] = [
        f"# Backtest de días análogos (R3) — {meta['fecha']}\n",
        "Generado por `scripts/backtest_analogos.py` (importa `scripts/backtest_adjustments.py`). Detalle en "
        f"`{meta['csv_nombre']}` (fuera de git; una fila por UCP × semana × día × paso × config).\n",
        "## Configuración\n",
        f"- Bases: {', '.join(meta['bases'])}. `modelo_k1` = pronóstico del modelo real de la Fase 2a, sin fuga "
        "(curvas generadas con corte previo y verificadas; ver `docs/work/ajustes/fase2a/`).\n"
        f"- Semanas: últimas {meta['semanas']} completas por UCP; corte = sábado anterior (fecha_emision = "
        "domingo). Calibración = mitad más antigua, holdout = mitad más reciente. Base `archivo`: partición "
        "`archivo` aparte (no se elige configuración ahí).\n"
        "- Sin fuga: `construir_contexto` verifica `max(historico.fecha) <= fecha_corte <= sábado`; además se "
        "verifica que toda fecha análoga usada por forma/nivel sea <= corte (aserción por día y config).\n"
        f"- Configs ({len(configs)}): {', '.join('`' + n + '`' for n in nombres)}.\n"
        "  - `default` = forma `promedio_n` n=4 + nivel ancla actual α=0.25.\n"
        "  - `forma_*` cambian solo la forma (nivel default); `_nK` = k análogos; `_t45` = filtro "
        "`temporada_dias` 45 además de los automáticos (mismo día de semana y misma relación/distancia con festivo).\n"
        "  - `nivel_*` cambian solo el nivel (forma default): ancla `actual`/`analogos`, α en centésimas.\n"
        "  - `combo[...]` = mejor forma + mejor nivel elegidas en calibración (si ambas mejoran).\n"
        f"- Regla para proponer default por segmento: gana en calibración y holdout, en ambas UCP con base "
        f"`{base_regla}`, por >= {meta['umbral']} % relativo del WAPE de cada paso afectado (forma; nivel y "
        "forma para configs de nivel/combo) y sin empeorar el error de pico (|error %| del máximo) en esos pasos. "
        f"La base de la regla es `{base_regla}` (no cambia defaults del producto).\n"
        f"- Tiempo de corrida: {meta['tiempo_s']:.0f} s.\n",
    ]
    rangos = (df[df["config"] == DEFAULT.nombre].groupby(["base", "ucp", "particion"])["semana_inicio"]
              .agg(["min", "max", "nunique"]).reset_index())
    rangos.columns = ["base", "ucp", "particion", "primera_semana", "ultima_semana", "n_semanas"]
    s += ["### Semanas evaluadas\n", tabla_md(rangos)]

    s += ["## 1. Resultado global (todos los días)\n",
          "WAPE horario % del paso final (forma) por config; columnas = base·UCP·partición.\n",
          tabla_md(_pivot_wape(comp, "forma", "todos", nombres)),
          "WAPE % del paso nivel (solo cambia en configs de nivel/combo):\n",
          tabla_md(_pivot_wape(comp, "nivel", "todos",
                               [c.nombre for c in configs if c.grupo in ("default", "nivel", "combo")])),
          "Mejora relativa % del WAPE vs default (positivo = mejor):\n",
          tabla_md(_pivot_mejora(comp, "todos", configs)),
          "### Todas las métricas del paso afectado\n",
          "WAPE horario; energía diaria (MAPE y sesgo %); pico (|error %| y diferencia de hora); energía "
          "semanal (MAPE y sesgo %).\n",
          tabla_md(_detalle_global(df, configs))]

    s += ["### Uso real de la fuente pedida (fallbacks)\n", tabla_md(_uso_fuentes(df), 1)]

    s += ["## 2. Elección en calibración y resultado en holdout\n", meta["eleccion"]]

    s += ["## 3. Por segmento\n",
          "Días por segmento (config default; los segmentos se solapan):\n",
          tabla_md(_pivot_n(comp, meta["segmentos"]), 0),
          "WAPE % del paso forma con el default por segmento:\n"]
    c_def = comp[(comp["config"] == DEFAULT.nombre) & (comp["paso"] == "forma")]
    c_def = c_def.assign(col=c_def["base"] + "·" + c_def["ucp"] + "·" + c_def["particion"])
    piv = c_def.pivot_table(index="segmento", columns="col", values="wape_pct", aggfunc="first")
    piv = piv.reindex([x for x in meta["segmentos"] if x in piv.index])
    piv = piv[sorted(piv.columns, key=lambda x: (not x.startswith("ingenuo"), x))].reset_index()
    s.append(tabla_md(piv))
    for seg in meta["segmentos"]:
        if seg == "todos":
            continue
        s += [f"### Segmento `{seg}` — mejora relativa % del WAPE vs default\n",
              tabla_md(_pivot_mejora(comp, seg, configs), 1)]

    s += ["## 4. Días difíciles\n", meta["dificiles"]]
    s += [f"## 5. Regla de cambio de default (base `{base_regla}`, ambas UCP, calibración y holdout)\n",
          "Una fila por segmento × variante. `celdas_que_ganan` = UCP × partición × paso afectado que mejoran "
          f">= {meta['umbral']} % sin empeorar el pico. Solo se listan las variantes con al menos la mitad de "
          "las celdas ganadas; `cumple_regla` = todas.\n"]
    if not regla.empty:
        tot = regla["celdas"].str.split("/").str[1].astype(int)
        r = regla[regla["celdas_que_ganan"] >= tot / 2].sort_values(
            ["cumple_regla", "celdas_que_ganan", "mejora_min_pct"], ascending=False)
        s.append(tabla_md(r))
    else:
        s.append("_(sin datos)_\n")
    s += ["## 6. Conclusión\n", meta["conclusion"]]
    s += ["## 7. Cómo correr con la base modelo_k1 (Fase 2a)\n",
          "```\n.venv/Scripts/python.exe scripts/backtest_analogos.py --semanas 52 --base modelo_k1 "
          "--base-regla modelo_k1 --fecha <AAAAMMDD>\n```\n"
          "`modelo_k1` usa las curvas `reports/ajustes/curvas_modelo_k1_<ucp>.csv` de la Fase 2a (sin fuga, "
          "corte previo verificado). `--base-regla` elige la base con la que se evalúa la regla de la §5; "
          "`--modelo-dir` cambia el directorio de las curvas.\n"]
    s += ["## 8. Limitaciones\n", LIMITACIONES]
    path.write_text("\n".join(s), encoding="utf-8")


def texto_eleccion(
    comp: pd.DataFrame, elegidas: Dict[str, Optional[Config]], ucps: Sequence[str], base: str = "ingenuo"
) -> str:
    lineas = []
    c = comp[(comp["segmento"] == "todos") & (comp["paso"] == "forma")]
    for ucp in ucps:
        cal = c[(c["base"] == base) & (c["ucp"] == ucp) & (c["particion"] == "calibracion")]
        if cal.empty:
            continue
        best = cal.loc[cal["wape_pct"].idxmin()]
        hold = c[(c["base"] == base) & (c["ucp"] == ucp) & (c["particion"] == "holdout")]
        hb = hold[hold["config"] == best["config"]]
        hd = hold[hold["config"] == DEFAULT.nombre]
        txt_h = (f"holdout {hb['wape_pct'].iloc[0]:.3f} % vs default {hd['wape_pct'].iloc[0]:.3f} % "
                 f"(mejora {hb['mejora_rel_pct'].iloc[0]:+.2f} %)" if not hb.empty and not hd.empty else "sin holdout")
        lineas.append(f"- **{base} / {ucp}**: mejor config en calibración = `{best['config']}` "
                      f"(WAPE forma {best['wape_pct']:.3f} %, mejora {best['mejora_rel_pct']:+.2f} % vs default) → "
                      f"{txt_h}.")
    for grupo, cfg in elegidas.items():
        if grupo == "combo":
            continue
        lineas.append(f"- Mejor variante de **{grupo}** en calibración (media de mejora relativa sobre las UCP, "
                      f"{base}): {'`' + cfg.nombre + '`' if cfg else 'ninguna mejora al default'}.")
    combo = elegidas.get("combo")
    if combo is not None:
        cc = c[(c["config"] == combo.nombre)]
        partes = [f"{x['base']}·{x['ucp']}·{x['particion']} {x['wape_pct']:.3f} % ({x['mejora_rel_pct']:+.2f} %)"
                  for _, x in cc.iterrows()]
        lineas.append(f"- Combinación `{combo.nombre}` (WAPE forma, mejora vs default): " + "; ".join(partes) + ".")
    else:
        lineas.append("- No se corrió combinación (alguna de las dos mejores variantes no mejora al default).")
    return "\n".join(lineas) + "\n"


def texto_dificiles(comp: pd.DataFrame, configs: Sequence[Config], ucps: Sequence[str]) -> str:
    """Para cada segmento difícil: n de días, WAPE default y la mejor variante por celda (ingenuo y archivo)."""
    lineas = ["Muestras chicas: con 52 semanas por partición hay pocos festivos, puentes y días de Semana Santa "
              "por UCP (ver n). Diferencias en estos segmentos son ruidosas; un solo día atípico mueve el WAPE. "
              "Con base `ingenuo` los festivos arrancan de un día no festivo (error artificial grande).\n"]
    c = comp[comp["paso"] == "forma"]
    for seg in SEGMENTOS_DIFICILES:
        g = c[c["segmento"] == seg]
        if g.empty:
            lineas.append(f"- **{seg}**: sin días.")
            continue
        partes = []
        for (base, ucp, part), x in g.groupby(["base", "ucp", "particion"], sort=False):
            d = x[x["config"] == DEFAULT.nombre]
            if d.empty:
                continue
            b = x.loc[x["wape_pct"].idxmin()]
            partes.append(f"{base}·{ucp}·{part}: n={int(d['n_dias'].iloc[0])}, default {d['wape_pct'].iloc[0]:.2f} %, "
                          f"mejor `{b['config']}` {b['wape_pct']:.2f} % ({b['mejora_rel_pct']:+.1f} %)")
        lineas.append(f"- **{seg}**: " + "; ".join(partes) + ".")
    return "\n".join(lineas) + "\n"


def texto_conclusion(regla: pd.DataFrame, umbral: float, base: str = "ingenuo") -> str:
    if regla.empty:
        return "Sin datos para evaluar la regla.\n"
    ok = regla[regla["cumple_regla"]]
    lineas = []
    if ok.empty:
        lineas.append(
            f"**No hay evidencia para cambiar ningún default.** Ninguna variante gana por >= {umbral} % relativo "
            f"en calibración y holdout, en ambas UCP con base `{base}`, sin empeorar el pico, en ningún segmento. "
            "Los defaults actuales (forma `promedio_n` n=4, nivel ancla actual α=0.25) se mantienen; las fuentes "
            "de análogos quedan como herramienta opcional del analista (selección explícita o automática)."
        )
    else:
        lineas.append("**Variantes que cumplen la regla (candidatas a default por segmento; decide el coordinador):**\n")
        for _, x in ok.sort_values(["segmento", "mejora_min_pct"], ascending=[True, False]).iterrows():
            lineas.append(f"- Segmento `{x['segmento']}`: `{x['config']}` (pasos {x['pasos']}, n mínimo "
                          f"{x['n_dias_min']} días, mejora mínima {x['mejora_min_pct']:.1f} %, media "
                          f"{x['mejora_media_pct']:.1f} %, Δ pico máx {x['delta_pico_max_pp']:+.2f} pp).")
        lineas.append(
            "\nSi se aplica, debe ser por segmento y retrocompatible (p. ej. un default condicionado al tipo de "
            "día), y conviene revalidar con la base `modelo_k1` antes de fijarlo. Segmentos con n chico (< 15 días "
            "por celda) deben tomarse como indicio, no como evidencia."
        )
        lineas += _lectura_cumplidas(ok)
    return "\n".join(lineas) + "\n"


SEGMENTOS_FESTIVOS = ("tipo:festivo", "rel:festivo", "semana_santa")


def _lectura_cumplidas(ok: pd.DataFrame) -> List[str]:
    """Advertencias de lectura sobre las variantes que cumplen la regla (α vs ancla; festivos con ingenuo)."""
    out: List[str] = []
    pares = []
    for _, x in ok.iterrows():
        nombre = str(x["config"])
        if nombre.startswith("nivel_analogos_a"):
            gemela = nombre.replace("nivel_analogos_", "nivel_actual_")
            g = ok[(ok["segmento"] == x["segmento"]) & (ok["config"] == gemela)]
            if not g.empty:
                pares.append((x["segmento"], nombre, gemela, float(x["mejora_media_pct"]),
                              float(g["mejora_media_pct"].iloc[0])))
    if pares:
        out.append("\n**Lectura: la ganancia viene sobre todo de α, no del ancla por análogos.** En estos segmentos "
                   "también cumple la variante con ancla actual y el mismo α:")
        for seg, an, ac, m_an, m_ac in pares:
            out.append(f"- `{seg}`: `{an}` mejora media {m_an:.1f} % vs `{ac}` {m_ac:.1f} % → aporte propio del "
                       f"ancla por análogos ≈ {m_an - m_ac:+.1f} pp.")
    if set(ok["segmento"]) & set(SEGMENTOS_FESTIVOS):
        out.append(
            "\n**Festivos y Semana Santa con base `ingenuo` no son interpretables por sí solos**: la base ingenua "
            "pronostica cada festivo con un día NO festivo, así que cualquier α más alto \"gana\" porque corrige ese "
            "error artificial (mismo hallazgo de `backtest_20261112_v3.md`). La base `archivo` (Antioquia, n chico) "
            "apunta en la misma dirección, pero es una sola emisión con ajustes post-hoc. Recomendación: NO cambiar "
            "defaults todavía; dejar α = 0.5 con ancla por análogos en festivos / Semana Santa como candidato a "
            "revalidar con `--base modelo_k1` (y coordinarlo con el α por tipo de día de la Fase 2a)."
        )
    return out


LIMITACIONES = """\
- **Base ingenua ≠ modelo real**: la base `ingenuo` (mismo día de la semana anterior) es débil y en festivos
  parte de un día no festivo; las mejoras medidas sobre ella pueden no trasladarse al pronóstico del modelo.
  La base `modelo_k1` (Fase 2a) sí es el pronóstico del modelo real sin fuga, pero solo cubre las 52 semanas
  generadas y su corrida es más cara; la regla se evalúa con `--base-regla` (default `ingenuo`).
- **Base `modelo_k1`**: curvas pronosticadas con corte previo (sábado) por la Fase 2a; no se regeneran aquí.
  Si faltan los CSV, la base se omite para esa UCP.
- **Base `archivo`**: una sola emisión con corte 2025-12-31 y ajustes post-hoc; solo Antioquia, ~21 semanas
  (ene–may 2026). Se reporta como contraste, sin elegir nada ahí.
- **Clima no usado** en la búsqueda automática (`seleccionar` no puntúa clima; además el clima real solo
  cubre 2025-12..2026-07). El filtro de clima del endpoint queda sin evaluar.
- **Muestras chicas** en festivos, puentes, Semana Santa y diciembre (ver n por segmento): diferencias de
  pocos puntos en esos segmentos no son concluyentes (n < 15 días por celda).
- **Corte siempre en sábado** (1 día de brecha); otras emisiones no se evalúan.
- **Calibración y holdout contiguos** (la mitad de las semanas cada una): un cambio de régimen en el holdout
  afecta a todas las configs por igual, pero la comparación es de una sola partición temporal (sin validación
  cruzada).
- La elección de la mejor variante y la combinación se hacen solo en calibración con la base de la regla
  (`--base-regla`, default `ingenuo`); el holdout no se usa para elegir.
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _factory(base_nombre: str, archivo: Path, modelo_dir: Path = MODELO_DIR_DEFAULT):
    if base_nombre == "ingenuo":
        return (lambda ucp, hist: proveedor_ingenuo(hist)), None
    if base_nombre == "archivo":
        curvas = cargar_archivo(archivo)
        prov = proveedor_desde_curvas(curvas)
        return (lambda ucp, hist, _p=prov: _p if ucp == ARCHIVO_UCP else None), curvas
    if base_nombre == "modelo_k1":
        cargadas: Dict[str, Tuple[Any, Dict[str, date]]] = {}
        for ucp in UCPS_DEFAULT:
            try:
                cargadas[ucp] = cargar_curvas_modelo_k1(ucp, modelo_dir)
            except FileNotFoundError:
                logger.warning("modelo_k1: sin curvas para %s en %s", ucp, modelo_dir)
        return (lambda ucp, hist, _c=cargadas:
                proveedor_desde_curvas(_c[ucp][0]) if ucp in _c else None), cargadas
    prov_modelo = proveedor_modelo()
    return (lambda ucp, hist, _p=prov_modelo: _p), None


def semanas_de(
    base_nombre: str, historico: pd.DataFrame, n: int, extra, ucp: str
) -> Tuple[List[date], Dict[date, str]]:
    if base_nombre == "archivo":
        semanas = semanas_objetivo(historico, 10_000)
        semanas = [l for l in semanas
                   if all((l - timedelta(days=1) + timedelta(days=i)).isoformat() in extra
                          for i in range(8))][-n:]
        return semanas, {l: "archivo" for l in semanas}
    if base_nombre == "modelo_k1":
        if ucp not in extra:
            return [], {}
        curvas, semanas_corte = extra[ucp]
        semanas = semanas_modelo_k1(historico, n, semanas_corte, curvas)
        return semanas, particionar(semanas)
    semanas = semanas_objetivo(historico, n)
    return semanas, particionar(semanas)


def _reportar(df: pd.DataFrame, configs: Sequence[Config], elegidas: Dict[str, Optional[Config]],
              args: argparse.Namespace, md_path: Path, csv_nombre: str, tiempo_s: float) -> None:
    segmentos = list(mascaras_segmentos(df.head(1)).keys())
    comp = tabla_comparacion(df)
    ucps = [u for u in args.ucp if u in set(df[df["base"] == args.base_regla]["ucp"])]
    regla = evaluar_regla(comp, ucps, args.umbral, configs, args.base_regla)
    escribir_reporte(df, comp, regla, configs, md_path, {
        "fecha": args.fecha, "csv_nombre": csv_nombre, "bases": sorted(df["base"].unique(), reverse=True),
        "semanas": args.semanas, "umbral": args.umbral, "tiempo_s": tiempo_s, "segmentos": segmentos,
        "base_regla": args.base_regla,
        "eleccion": texto_eleccion(comp, elegidas, ucps, args.base_regla),
        "dificiles": texto_dificiles(comp, configs, ucps),
        "conclusion": texto_conclusion(regla, args.umbral, args.base_regla),
    })


def regenerar_reporte(args: argparse.Namespace) -> int:
    """Reescribe el .md desde el CSV de detalle (sin correr el pipeline)."""
    df = pd.read_csv(args.desde_csv, dtype={"semana_inicio": str, "fecha_corte": str, "fecha": str})
    pares = df[["config", "grupo"]].drop_duplicates()
    configs = [Config(c, g) for c, g in pares.itertuples(index=False)]
    configs.sort(key=lambda c: c.grupo != "default")
    comp = tabla_comparacion(df, ["todos"])
    combo = next((c for c in configs if c.grupo == "combo"), None)
    elegidas = {"forma": elegir_en_calibracion(comp, "forma", configs, args.base_regla),
                "nivel": elegir_en_calibracion(comp, "nivel", configs, args.base_regla), "combo": combo}
    md_path = args.out_dir / f"backtest_analogos_{args.fecha}.md"
    tiempo = args.tiempo_s if args.tiempo_s is not None else float("nan")
    _reportar(df, configs, elegidas, args, md_path, args.desde_csv.name, tiempo)
    logger.info("Reporte regenerado desde %s: %s", args.desde_csv, md_path)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ucp", nargs="+", default=list(UCPS_DEFAULT))
    ap.add_argument("--semanas", type=int, default=104, help="semanas completas por UCP (mitad calibración)")
    ap.add_argument("--base", nargs="+", default=["ingenuo", "archivo"],
                    choices=["ingenuo", "archivo", "modelo", "modelo_k1"])
    ap.add_argument("--modelo-dir", type=Path, default=MODELO_DIR_DEFAULT,
                    help="directorio de curvas_modelo_k1_<ucp>.csv")
    ap.add_argument("--base-regla", default="ingenuo", choices=list(BASES_REGLA),
                    help="base con la que se evalúa la regla de cambio de default")
    ap.add_argument("--archivo", type=Path, default=ARCHIVO_DEFAULT)
    ap.add_argument("--ns", type=int, nargs="+", default=list(NS_DEFAULT), help="k análogos (forma.n)")
    ap.add_argument("--temporadas", type=int, nargs="+", default=list(TEMPORADAS_DEFAULT),
                    help="temporada_dias (0 = sin ventana)")
    ap.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS_DEFAULT))
    ap.add_argument("--umbral", type=float, default=UMBRAL_DEFAULT, help="mejora relativa mínima de WAPE (%%)")
    ap.add_argument("--sin-combo", action="store_true", help="no correr la combinación ganadora")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "ajustes")
    ap.add_argument("--fecha", default=date.today().strftime("%Y%m%d"))
    ap.add_argument("--desde-csv", type=Path, default=None,
                    help="no corre el backtest: regenera el .md desde un CSV de detalle ya escrito")
    ap.add_argument("--tiempo-s", type=float, default=None, help="con --desde-csv: tiempo de la corrida original")
    args = ap.parse_args(argv)
    if args.desde_csv is not None:
        return regenerar_reporte(args)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = time.perf_counter()
    configs = configs_barrido(args.ns, args.temporadas, args.alphas)

    entradas = []  # (base, ucp, historico, semanas, particiones, factory)
    for base_nombre in args.base:
        factory, extra = _factory(base_nombre, args.archivo, args.modelo_dir)
        for ucp in args.ucp:
            if base_nombre == "archivo" and ucp != ARCHIVO_UCP:
                continue
            historico = data.cargar_historico(ucp)
            semanas, parts = semanas_de(base_nombre, historico, args.semanas, extra, ucp)
            logger.info("%s %s: %d semanas (%s .. %s)", base_nombre, ucp, len(semanas),
                        semanas[0] if semanas else "-", semanas[-1] if semanas else "-")
            entradas.append((base_nombre, ucp, historico, semanas, parts, factory))

    def correr_todo(cfgs: Sequence[Config]) -> pd.DataFrame:
        frames = [correr(ucp, hist, sem, parts, fac, base, cfgs) for base, ucp, hist, sem, parts, fac in entradas]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    df = correr_todo(configs)
    if df.empty:
        logger.error("Backtest sin resultados")
        return 1
    comp = tabla_comparacion(df, ["todos"])
    elegidas: Dict[str, Optional[Config]] = {
        "forma": elegir_en_calibracion(comp, "forma", configs, args.base_regla),
        "nivel": elegir_en_calibracion(comp, "nivel", configs, args.base_regla),
    }
    elegidas["combo"] = None
    if not args.sin_combo and elegidas["forma"] and elegidas["nivel"]:
        combo = config_combo(elegidas["forma"], elegidas["nivel"])
        logger.info("Combinación elegida en calibración: %s", combo.nombre)
        extra = correr_todo([combo])
        if not extra.empty:
            df = pd.concat([df, extra], ignore_index=True)
            configs = [*configs, combo]
            elegidas["combo"] = combo

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"backtest_analogos_{args.fecha}.csv"
    md_path = args.out_dir / f"backtest_analogos_{args.fecha}.md"
    df.to_csv(csv_path, index=False)
    _reportar(df, configs, elegidas, args, md_path, csv_path.name, time.perf_counter() - t0)
    logger.info("Escrito %s (%d filas) y %s en %.0f s", csv_path, len(df), md_path, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
