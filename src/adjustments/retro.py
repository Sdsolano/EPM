"""P1 "Lo que aprendió": diagnóstico informativo del error reportado vs real.

Funciones puras (sin red, sin store, sin efecto sobre curvas ni corridas).
Signo del error: (reportado - real) / real * 100 (positivo = el pronóstico quedó
por encima del real). Los promedios y la EMA usan errores winsorizados a ±5 %.
Solo se usan datos con fecha <= `hasta` (sin fuga de futuro).
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from src.adjustments import data

WINSOR_PCT = 5.0
MIN_CASOS = 8
SEMANAS_DEFAULT = 12
SEMANAS_MIN, SEMANAS_MAX = 4, 26
VENTANAS = (4, 8, 12)
EMA_H = (56, 112)
LAGS = {"dia1": 1, "dia5": 5}
REZAGO_AVISO_DIAS = 3
DECIMALES = 2
TZ_BOGOTA = ZoneInfo("America/Bogota")
TIPOS = ("laboral", "sabado", "domingo", "festivo", "post_festivo")
BLOQUES = (("madrugada", 1, 6), ("manana", 7, 12), ("tarde", 13, 18), ("noche", 19, 24))


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------

def _fecha(valor: Any, nombre: str) -> date:
    """Fecha estricta: `date` o cadena exactamente `YYYY-MM-DD` (rechaza basura tras los 10 caracteres)."""
    try:
        if isinstance(valor, datetime):
            return valor.date()
        if isinstance(valor, date):
            return valor
        if isinstance(valor, str) and len(valor) == 10:
            return date.fromisoformat(valor)
    except ValueError:
        pass
    raise ValueError(f"{nombre}: fecha inválida {valor!r} (se espera ISO YYYY-MM-DD)")


def _curva(valores: Any, nombre: str) -> np.ndarray:
    if not isinstance(valores, (list, tuple, np.ndarray)) or len(valores) != 24:
        raise ValueError(f"{nombre}: la curva debe tener exactamente 24 valores")
    try:
        arr = np.asarray(valores, dtype=float)
    except (ValueError, TypeError):
        raise ValueError(f"{nombre}: la curva contiene valores no numéricos")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{nombre}: la curva contiene NaN o infinitos")
    if np.any(arr < 0):
        raise ValueError(f"{nombre}: la curva contiene valores negativos")
    return arr


def validar_entrada(
    ucp: Any,
    emisiones: Any,
    reales: Any,
    semanas: Any = SEMANAS_DEFAULT,
    hasta: Any = None,
    min_casos: Any = MIN_CASOS,
    festivos: Any = None,
) -> Dict[str, Any]:
    """Valida y normaliza la entrada. Lanza ValueError con mensajes claros (422).

    Devuelve `{ucp, emisiones: {fecha: (version, ndarray24)}, reales: {fecha: ndarray24},
    semanas, hasta, min_casos, festivos}`.
    """
    if not isinstance(ucp, str) or not ucp.strip():
        raise ValueError("ucp: no puede estar vacío")
    if isinstance(semanas, bool) or not isinstance(semanas, int) or not SEMANAS_MIN <= semanas <= SEMANAS_MAX:
        raise ValueError(f"semanas: debe ser un entero entre {SEMANAS_MIN} y {SEMANAS_MAX}")
    if isinstance(min_casos, bool) or not isinstance(min_casos, int) or min_casos < 1:
        raise ValueError("min_casos: debe ser un entero >= 1")
    hasta_d = _fecha(hasta, "hasta") if hasta is not None else None
    if festivos is None:
        festivos_l: Optional[List[str]] = None
    elif isinstance(festivos, (list, tuple, set)):
        festivos_l = sorted({_fecha(f, "festivos").isoformat() for f in festivos})
    else:
        raise ValueError("festivos: debe ser una lista de fechas ISO")

    if not isinstance(emisiones, (list, tuple)):
        raise ValueError("emisiones: debe ser una lista")
    if not isinstance(reales, Mapping):
        raise ValueError("reales: debe ser un objeto {fecha: [24 valores]}")

    emis: Dict[str, Any] = {}
    for i, e in enumerate(emisiones):
        if not isinstance(e, Mapping):
            raise ValueError(f"emisiones[{i}]: debe ser un objeto")
        inicio = _fecha(e.get("semana_inicio"), f"emisiones[{i}].semana_inicio")
        if inicio.weekday() != 0:
            raise ValueError(f"emisiones[{i}].semana_inicio: {inicio.isoformat()} no es lunes")
        version = e.get("version", 1)
        if version is None:
            version = 1
        if isinstance(version, bool) or not isinstance(version, (int, float)):
            raise ValueError(f"emisiones[{i}].version: debe ser numérica")
        curvas = e.get("curvas")
        if not isinstance(curvas, Mapping):
            raise ValueError(f"emisiones[{i}].curvas: debe ser un objeto {{fecha: [24 valores]}}")
        for k, v in curvas.items():
            f = _fecha(k, f"emisiones[{i}].curvas").isoformat()
            arr = _curva(v, f"emisiones[{i}].curvas[{f}]")
            previa = emis.get(f)
            if previa is None or version < previa[0]:  # ex-ante = menor versión
                emis[f] = (float(version), arr)
    reales_n: Dict[str, np.ndarray] = {}
    for k, v in reales.items():
        f = _fecha(k, "reales").isoformat()
        reales_n[f] = _curva(v, f"reales[{f}]")
    return {
        "ucp": ucp.strip(), "emisiones": emis, "reales": reales_n, "semanas": semanas,
        "hasta": hasta_d, "min_casos": min_casos, "festivos": festivos_l,
    }


# ---------------------------------------------------------------------------
# Estadística básica
# ---------------------------------------------------------------------------

def winsorizar(valor: float, tope: float = WINSOR_PCT) -> float:
    """Recorta un error % a ±tope."""
    return float(min(max(valor, -tope), tope))


def _r(x: Optional[float], n: int = DECIMALES) -> Optional[float]:
    return None if x is None or not math.isfinite(x) else round(float(x), n)


def _tipo(fecha: date, festivos: set) -> str:
    clase = data.clase_dia(fecha, festivos)
    if clase == data.CLASE_FESTIVO:
        return "festivo"
    if clase == data.CLASE_POST_FESTIVO:
        return "post_festivo"
    return data.tipo_dia(fecha, festivos)


def _err(rep: float, real: float) -> float:
    return (rep - real) / real * 100.0


# ---------------------------------------------------------------------------
# Historial
# ---------------------------------------------------------------------------

def historial_error(
    emisiones: Mapping[str, Any],
    reales: Mapping[str, Any],
    festivos: Iterable[str],
) -> List[Dict[str, Any]]:
    """Error con signo por día y por periodo, ordenado por fecha.

    `emisiones` y `reales` son los mapas normalizados de `validar_entrada`
    (`{fecha: (version, arr24)}` / `{fecha: arr24}`); también se aceptan
    `{fecha: lista24}` en ambos. Solo los días con emisión y real y energía real > 0.
    Cada fila: fecha, tipo_dia, reportado_mwh, real_mwh, error_pct,
    abs_err_mwh, errores_periodo (24, % o None si real=0), rep/real (arrays), d (date, parseada una vez).
    """
    fest = set(festivos)
    filas: List[Dict[str, Any]] = []
    for f in sorted(set(emisiones) & set(reales)):
        e = emisiones[f]
        rep = np.asarray(e[1] if isinstance(e, tuple) else e, dtype=float)
        real = np.asarray(reales[f], dtype=float)
        if rep.shape != (24,) or real.shape != (24,) or real.sum() <= 0:
            continue
        d = _fecha(f, "fecha")
        with np.errstate(divide="ignore", invalid="ignore"):
            per = np.where(real > 0, (rep - real) / real * 100.0, np.nan)
        filas.append({
            "fecha": f,
            "d": d,
            "tipo_dia": _tipo(d, fest),
            "reportado_mwh": float(rep.sum()),
            "real_mwh": float(real.sum()),
            "error_pct": _err(float(rep.sum()), float(real.sum())),
            "abs_err_mwh": float(np.abs(rep - real).sum()),
            "errores_periodo": [None if np.isnan(x) else float(x) for x in per],
            "rep": rep,
            "real": real,
        })
    return filas


def sesgo_ema(
    errores: Mapping[str, float],
    h_dias: float,
    hasta: Any = None,
    tope: float = WINSOR_PCT,
) -> Optional[float]:
    """Sesgo EMA (media ponderada 0.5^(edad/h)) de errores diarios % winsorizados.

    `errores`: `{fecha_iso: error_pct}`. Solo se usan fechas <= `hasta`
    (default: la última). edad = días entre la fecha y `hasta`. None si no hay datos.
    Usa todo el historial que reciba: la vida media h112 no puede superar lo enviado
    (el proxy recorta a `semanas`, así que con 12 semanas h112 apenas se distingue de h56).
    """
    if h_dias <= 0:
        raise ValueError("h_dias debe ser > 0")
    items = [(_fecha(k, "errores"), v) for k, v in errores.items() if v is not None and math.isfinite(v)]
    if not items:
        return None
    corte = _fecha(hasta, "hasta") if hasta is not None else max(d for d, _ in items)
    items = [(d, v) for d, v in items if d <= corte]  # sin fuga
    if not items:
        return None
    w = np.array([0.5 ** ((corte - d).days / h_dias) for d, _ in items])
    x = np.array([winsorizar(v, tope) for _, v in items])
    return float(np.sum(w * x) / np.sum(w))


def _stats_error(filas: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(filas)
    if n == 0:
        return {"sesgo_pct": None, "wape_pct": None, "n": 0}
    sesgo = float(np.mean([winsorizar(f["error_pct"]) for f in filas]))
    tot = sum(f["real_mwh"] for f in filas)
    wape = sum(f["abs_err_mwh"] for f in filas) / tot * 100.0
    return {"sesgo_pct": _r(sesgo), "wape_pct": _r(wape), "n": n}


def _autocorr(serie: Mapping[date, float], lag: int) -> Optional[float]:
    pares = [(v, serie[d + timedelta(days=lag)]) for d, v in serie.items() if d + timedelta(days=lag) in serie]
    if len(pares) < 8:
        return None
    a, b = np.array([p[0] for p in pares]), np.array([p[1] for p in pares])
    if a.std() == 0 or b.std() == 0:
        return None
    return _r(float(np.corrcoef(a, b)[0, 1]))


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------

def resumen_aprendizaje(
    ucp: Any,
    emisiones: Any,
    reales: Any,
    festivos: Any = None,
    semanas: int = SEMANAS_DEFAULT,
    hasta: Any = None,
    min_casos: int = MIN_CASOS,
    hoy: Any = None,
) -> Dict[str, Any]:
    """Resumen informativo (forma de CONTRATO.md). No modifica nada.

    `emisiones`: lista `[{semana_inicio, version?, curvas}]` y `reales`: `{fecha: [24]}` (crudos;
    se validan aquí). `festivos`: lista ISO o None (se calculan los de Colombia).
    `hasta`: solo se usan fechas <= hasta (default: último día con emisión y real).
    `hoy`: referencia para el aviso de rezago (default: hoy en America/Bogota).
    """
    v = validar_entrada(ucp, emisiones, reales, semanas, hasta, min_casos, festivos)
    ucp_n, semanas, min_casos = v["ucp"], v["semanas"], v["min_casos"]

    def sin_datos(motivo: str) -> Dict[str, Any]:
        return {"estado": "sin_datos", "ucp": ucp_n, "motivo": motivo,
                "avisos": [{"codigo": "RETRO_SIN_HISTORIAL", "nivel": "info", "mensaje": motivo}]}

    if not v["emisiones"] or not v["reales"]:
        return sin_datos("No hay emisiones o datos reales para calcular el error.")

    tope_f = v["hasta"]
    fechas_comunes = [date.fromisoformat(f) for f in set(v["emisiones"]) & set(v["reales"])]
    if tope_f is not None:
        fechas_comunes = [d for d in fechas_comunes if d <= tope_f]
    if not fechas_comunes:
        return sin_datos("Ninguna fecha tiene emisión y dato real (hasta la fecha pedida).")

    if v["festivos"] is None:
        anios = {d.year for d in fechas_comunes} | {(min(fechas_comunes) - timedelta(days=1)).year}
        festivos_l = data.festivos_colombia(anios)
    else:
        festivos_l = v["festivos"]
    fest = set(festivos_l)

    todas = historial_error(v["emisiones"], v["reales"], fest)
    if tope_f is not None:
        todas = [f for f in todas if f["d"] <= tope_f]
    if not todas:
        return sin_datos("No hay días válidos (con energía real > 0) para calcular el error.")
    fin = tope_f if tope_f is not None else todas[-1]["d"]

    def desde_dias(dias: int) -> List[Dict[str, Any]]:
        corte = fin - timedelta(days=dias)
        return [f for f in todas if f["d"] > corte]

    base = desde_dias(7 * semanas)
    if not base:
        return sin_datos(f"La fecha pedida ({fin.isoformat()}) queda más de {semanas} semanas después del "
                         f"último dato con emisión y real ({todas[-1]['fecha']}).")
    # Último día con dato real (no con emisión), sin pasar de `hasta`.
    ultimo_real = max(date.fromisoformat(f) for f, arr in v["reales"].items()
                      if (tope_f is None or date.fromisoformat(f) <= tope_f) and float(arr.sum()) > 0)
    # Semanas: tramos de 7 días anclados en `fin` (exactamente `semanas` posibles, sin desfase a lunes).
    tramos = {(fin - f["d"]).days // 7 for f in base}

    por_tipo = {}
    insuf = []
    for t in TIPOS:
        s = _stats_error([f for f in base if f["tipo_dia"] == t])
        suf = s["n"] >= min_casos
        if not suf:
            insuf.append(t)
            s = {"sesgo_pct": None, "wape_pct": None, "n": s["n"]}
        por_tipo[t] = {**s, "suficiente": suf}

    por_bloque = []
    for nombre, a, b in BLOQUES:
        errs = []
        for f in base:
            real_b = f["real"][a - 1:b].sum()
            if real_b > 0:
                errs.append(winsorizar(_err(float(f["rep"][a - 1:b].sum()), float(real_b))))
        por_bloque.append({"bloque": nombre, "periodos": [a, b],
                           "sesgo_pct": _r(float(np.mean(errs))) if errs else None, "n": len(errs)})

    serie_ema = {f["fecha"]: f["error_pct"] for f in todas}
    serie_lag = {f["d"]: winsorizar(f["error_pct"]) for f in base}
    ult = [f for f in todas if f["d"] > fin - timedelta(days=7)]

    g = _stats_error(base)
    g = {"pct": g["sesgo_pct"], "wape_pct": g["wape_pct"], "n": g["n"]}
    avisos: List[Dict[str, str]] = []
    if len(tramos) < semanas:
        avisos.append({"codigo": "RETRO_MUESTRA_CORTA", "nivel": "info",
                       "mensaje": f"Muestra corta: la base cubre {len(tramos)} de {semanas} semanas pedidas."})
    if insuf:
        avisos.append({"codigo": "RETRO_TIPO_INSUFICIENTE", "nivel": "info",
                       "mensaje": f"Menos de {min_casos} casos en: {', '.join(insuf)}; no se muestra su sesgo."})
    hoy_d = _fecha(hoy, "hoy") if hoy is not None else datetime.now(TZ_BOGOTA).date()
    if (hoy_d - ultimo_real).days > REZAGO_AVISO_DIAS:
        avisos.append({"codigo": "RETRO_REZAGO_REAL", "nivel": "info",
                       "mensaje": f"El último dato real es del {ultimo_real.isoformat()} "
                                  f"({(hoy_d - ultimo_real).days} días de rezago)."})

    def ventana(w: int) -> Dict[str, Any]:
        # Una ventana mayor que `semanas` repetiría los mismos datos de la base: no disponible.
        if w > semanas:
            return {"semanas": w, "sesgo_pct": None, "wape_pct": None, "n": 0}
        return {"semanas": w, **_stats_error(desde_dias(7 * w))}

    return {
        "estado": "ok",
        "ucp": ucp_n,
        "modo": "error",
        "base": {"fuente": "emisiones", "desde": base[0]["fecha"], "hasta": base[-1]["fecha"],
                 "n_dias": len(base), "semanas": len(tramos), "ultimo_real": ultimo_real.isoformat()},
        "sesgo_global": g,
        "ventanas": [ventana(w) for w in VENTANAS],
        "ema": {f"h{h}": _r(sesgo_ema(serie_ema, h, fin)) for h in EMA_H},
        "por_tipo": por_tipo,
        "por_bloque": por_bloque,
        "ultima_semana": [{"fecha": f["fecha"], "tipo_dia": f["tipo_dia"],
                           "reportado_mwh": round(f["reportado_mwh"], 1), "real_mwh": round(f["real_mwh"], 1),
                           "error_pct": _r(f["error_pct"], 1)} for f in ult],
        "autocorr_lag": {k: _autocorr(serie_lag, lag) for k, lag in LAGS.items()},
        "min_casos": min_casos,
        "avisos": avisos,
    }
