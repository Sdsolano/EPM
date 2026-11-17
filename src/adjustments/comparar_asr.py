"""
Comparador ASR actual vs mesa de ajustes vs real (y ajuste del analista, si se tiene) para una semana.

Para una UCP, una semana objetivo y un pronóstico base produce, por día y hora:
- `base`: el pronóstico base (el mismo para todos);
- `mesa`: curvas finales del pipeline de ajustes (`run_pipeline`, defaults vigentes) sobre esa base;
- `asr_<escenario>` / `asr_<escenario>_sin_validador`: el ASR de Sphere reproducido (`asr.py`) con la
  selección de históricos del escenario (o `asr` con una selección explícita);
- `real` (si la semana ya está observada) y `analista` (si se pasan sus curvas finales).
Más métricas por día (energía, pico MW y hora, WAPE, error de energía, error de forma) y por semana.

Error de forma = 100·Σ|p/Σp − r/Σr|: el WAPE que quedaría si la energía del día fuera exacta.
Sin IO: el histórico viene en `ctx` (≤ corte, así los históricos del ASR no tienen fuga) y el real se
pasa aparte.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.adjustments import asr, data, pipeline
from src.adjustments.types import Curves, RunContext

SERIE_BASE = "base"
SERIE_MESA = "mesa"
SERIE_REAL = "real"
SERIE_ANALISTA = "analista"
SERIE_ASR_EXPLICITA = "asr"
SUFIJO_SIN_VALIDADOR = "_sin_validador"
SUFIJO_SOLO_NIVEL = "_solo_nivel"  # forma de la base con la energía del ASR
SUFIJO_SOLO_FORMA = "_solo_forma"  # forma del ASR con la energía de la base

MAX_BRECHA_DIAS = 6  # más días entre el corte y el lunes objetivo = histórico desactualizado

SendaFn = Callable[[str], Tuple[np.ndarray, int, str]]
Seleccion = Mapping[str, Sequence[asr.Historico]]  # {fecha objetivo: históricos en orden de clic}


def nombre_asr(escenario: Optional[str], validador: bool = True) -> str:
    base = SERIE_ASR_EXPLICITA if escenario is None else f"asr_{escenario}"
    return base if validador else base + SUFIJO_SIN_VALIDADOR


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def _forma(v: np.ndarray) -> Optional[np.ndarray]:
    s = float(v.sum())
    return v / s if np.isfinite(s) and s > 0 else None


def metricas_dia(pred: Sequence[float], real: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """
    Energía y pico de `pred`; con `real`, errores (abs_err_horario es sumable). Los errores relativos
    son None si el real no tiene energía (día real todo en cero).
    """
    p = np.asarray(pred, dtype=float)
    out: Dict[str, Any] = {
        "energia_pred": float(p.sum()),
        "pico_pred": float(p.max()),
        "hora_pico_pred": int(p.argmax()) + 1,
    }
    if real is None:
        return out
    r = np.asarray(real, dtype=float)
    fp, fr = _forma(p), _forma(r)
    dif = abs(int(p.argmax()) - int(r.argmax()))
    e_real, pico_real = float(r.sum()), float(r.max())
    out.update({
        "energia_real": e_real,
        "pico_real": pico_real,
        "hora_pico_real": int(r.argmax()) + 1,
        "abs_err_horario": float(np.abs(p - r).sum()),
        "wape_pct": 100 * float(np.abs(p - r).sum()) / e_real if e_real > 0 else None,
        "err_energia_pct": 100 * (float(p.sum()) / e_real - 1) if e_real > 0 else None,
        "err_pico_pct": 100 * (float(p.max()) / pico_real - 1) if pico_real > 0 else None,
        "dif_hora_pico_h": int(min(dif, 24 - dif)),
        "err_forma_pct": 100 * float(np.abs(fp - fr).sum()) if fp is not None and fr is not None else None,
    })
    return out


# ---------------------------------------------------------------------------
# Selección de históricos
# ---------------------------------------------------------------------------

def seleccion_escenario(
    escenario: str, ctx: RunContext, reales_corte: Mapping[str, Any], desplazamiento_ref: int
) -> Tuple[Dict[str, List[asr.Historico]], Dict[str, str]]:
    """Selección (orden de clic) por día de la semana objetivo; días sin los 3 históricos → omitidos."""
    seleccion: Dict[str, List[asr.Historico]] = {}
    omitidos: Dict[str, str] = {}
    for d in data.dias_semana(ctx.semana_inicio):
        fechas = asr.fechas_escenario(escenario, d, ctx.fecha_corte, desplazamiento_ref)
        faltan = [f.isoformat() for f in fechas if f.isoformat() not in reales_corte]
        if faltan:
            omitidos[d.isoformat()] = f"históricos sin dato real: {faltan}"
            continue
        seleccion[d.isoformat()] = [asr.Historico(f.isoformat(), list(reales_corte[f.isoformat()])) for f in fechas]
    return seleccion, omitidos


def proveedor_ingenuo(historico_cortado) -> pipeline.BaseProvider:
    """
    Base ingenua (misma regla que `scripts/backtest_adjustments.proveedor_ingenuo`, que vive en un
    script y no se puede importar desde src): cada día = mismo día de semana de la última semana real
    ≤ corte (ancla con el desfase del corte; si falta, se retrocede de a 7 días hasta 4 semanas).
    """
    reales = data.curvas_reales(historico_cortado)

    def provider(ucp: str, end_date: date, n_days: int) -> Curves:
        out: Curves = {}
        for i in range(1, n_days + 1):
            d = end_date + timedelta(days=i)
            a = data.ancla_mismo_dow(d, end_date)
            for _ in range(5):
                if a.isoformat() in reales:
                    out[d.isoformat()] = reales[a.isoformat()].tolist()
                    break
                a -= timedelta(days=7)
        return out

    return provider


def validar_curvas(curvas: Mapping[str, Sequence[float]], campo: str) -> None:
    """ValueError si alguna fecha no trae 24 periodos finitos (NaN/inf romperían métricas y JSON)."""
    mal = [f for f, v in curvas.items()
           if len(v) != asr.N_PERIODOS or not np.isfinite(np.asarray(v, dtype=float)).all()]
    if mal:
        raise ValueError(f"{campo}: se esperan {asr.N_PERIODOS} periodos finitos por fecha; fallan {mal}")


def _escalar(curva: Sequence[float], energia: float) -> List[float]:
    v = np.asarray(curva, dtype=float)
    s = float(v.sum())
    return (v * (energia / s)).tolist() if s > 0 else v.tolist()


# ---------------------------------------------------------------------------
# Núcleo
# ---------------------------------------------------------------------------

def comparar_semana(
    ctx: RunContext,
    base_provider: pipeline.BaseProvider,
    *,
    reales: Optional[Mapping[str, Sequence[float]]] = None,
    escenarios: Sequence[str] = ("anual",),
    seleccion: Optional[Seleccion] = None,
    validador: Sequence[bool] = (True, False),
    senda_fn: Optional[SendaFn] = None,
    motor: Any = None,
    festivos_especial: Optional[Iterable[str]] = None,
    params_mesa: Optional[Dict[str, Dict[str, Any]]] = None,
    analista: Optional[Mapping[str, Sequence[float]]] = None,
    descomponer: bool = False,
) -> Dict[str, Any]:
    """
    Compara base, ASR (por escenario o con `seleccion` explícita), mesa, real y analista en la semana
    de `ctx`. Con `validador` True se usa `senda_fn(fecha)` o, si no se da, el motor de desagregación
    (`motor` o `asr.cargar_motor(ucp)`) con `festivos_especial` para `is_special_day` (None = API).
    Si la senda no se puede obtener (sin modelos), la serie con validador queda en `omitidos`.
    `descomponer`: agrega las series solo_nivel / solo_forma de cada ASR con validador.
    """
    records = pipeline.run_pipeline(ctx, base_provider, params_mesa)
    por_paso = {r.paso: r for r in records}
    fechas = [d.isoformat() for d in data.dias_semana(ctx.semana_inicio)]
    base = por_paso["base"].result.curves
    faltan = [f for f in fechas if f not in base]
    if faltan:
        raise ValueError(f"la base no cubre la semana objetivo: faltan {faltan}")
    mesa = pipeline.final_curves(records)
    desplazamiento = (ctx.semana_ref_inicio - ctx.semana_inicio).days if ctx.semana_ref_inicio else -364

    series: Dict[str, Curves] = {
        SERIE_BASE: {f: list(base[f]) for f in fechas},
        SERIE_MESA: {f: list(mesa[f]) for f in fechas},
    }
    detalle: Dict[str, Dict[str, Dict[str, Any]]] = {}
    omitidos: Dict[str, Dict[str, str]] = {}

    reales_corte = data.curvas_reales(ctx.historico)
    selecciones: List[Tuple[Optional[str], Dict[str, List[asr.Historico]], Dict[str, str]]] = []
    if seleccion is not None:
        fuera = sorted(set(seleccion) - set(fechas))
        if fuera:  # no descartar en silencio: el analista creería que su selección se usó
            raise ValueError(f"seleccion: fechas objetivo fuera de la semana {fechas[0]}..{fechas[-1]}: {fuera}")
        selecciones.append((None, {f: list(v) for f, v in seleccion.items()}, {}))
    else:
        for esc in escenarios:
            sel, om = seleccion_escenario(esc, ctx, reales_corte, desplazamiento)
            selecciones.append((esc, sel, om))

    senda_cache: Dict[str, Tuple[np.ndarray, int, str]] = {}
    motor_estado: Dict[str, Any] = {}  # motor cargado una vez, o el error de carga (no se reintenta)

    def _motor() -> Any:
        if "motor" not in motor_estado and "error" not in motor_estado:
            try:
                m = motor if motor is not None else asr.cargar_motor(ctx.ucp)
                asr.verificar_motor(m, ctx.ucp)
                motor_estado["motor"] = m
            except Exception as e:  # noqa: BLE001 — pickle ausente/corrupto/sin entrenar → serie omitida
                motor_estado["error"] = f"motor de desagregación no disponible: {type(e).__name__}: {e}"
        if "error" in motor_estado:
            raise ValueError(motor_estado["error"])
        return motor_estado["motor"]

    def senda(fecha: str) -> Tuple[np.ndarray, int, str]:
        if fecha not in senda_cache:
            if senda_fn is not None:
                senda_cache[fecha] = senda_fn(fecha)
            else:
                senda_cache[fecha] = asr.senda_cluster(_motor(), fecha, festivos_especial)
        return senda_cache[fecha]

    for esc, sel, om in selecciones:
        for con_validador in validador:
            nombre = nombre_asr(esc, con_validador)
            curvas: Curves = {}
            det: Dict[str, Dict[str, Any]] = {}
            om_serie = dict(om)
            for f in fechas:
                if f not in sel:
                    om_serie.setdefault(f, "sin selección de históricos")
                    continue
                try:
                    r = asr.asr_dia(ctx.ucp, f, base[f], sel[f], validador=con_validador, senda_fn=senda)
                except (ValueError, ArithmeticError, KeyError, AttributeError) as e:
                    om_serie[f] = f"{type(e).__name__}: {e}"
                    continue
                curvas[f] = r.final
                det[f] = {
                    "h_fechas": r.h_fechas, "desvi2_fechas": r.desvi2_fechas, "cluster_id": r.cluster_id,
                    "metodo": r.metodo, "periodos_modificados": r.periodos_modificados,
                    "energia_objetivo": float(np.sum(r.objetivo)),
                }
            series[nombre] = curvas
            detalle[nombre] = det
            if om_serie:
                omitidos[nombre] = om_serie
            if descomponer and con_validador:
                series[nombre + SUFIJO_SOLO_NIVEL] = {
                    f: _escalar(base[f], float(np.sum(c))) for f, c in curvas.items()
                }
                series[nombre + SUFIJO_SOLO_FORMA] = {
                    f: _escalar(c, float(np.sum(base[f]))) for f, c in curvas.items()
                }

    if analista:
        series[SERIE_ANALISTA] = {f: list(analista[f]) for f in fechas if f in analista}
    reales_semana = {f: list(reales[f]) for f in fechas if reales and f in reales}
    if reales_semana:
        series[SERIE_REAL] = reales_semana

    festivos = set(ctx.festivos)
    metricas: List[Dict[str, Any]] = []
    for nombre, curvas in series.items():
        if nombre == SERIE_REAL:
            continue
        for f in fechas:
            if f not in curvas:
                continue
            fila = {"serie": nombre, "fecha": f, "tipo_dia": data.tipo_dia(date.fromisoformat(f), festivos)}
            fila.update(metricas_dia(curvas[f], reales_semana.get(f)))
            metricas.append(fila)

    brecha = (ctx.semana_inicio - ctx.fecha_corte).days - 1
    advertencias = []
    if brecha > MAX_BRECHA_DIAS:
        advertencias.append(
            f"El último dato real ({ctx.fecha_corte.isoformat()}) está {brecha} días antes de la semana: el "
            "histórico no está al día; los históricos del ASR y la mesa no son los que vería el analista. "
            "Actualice data/raw/{ucp}/datos.csv (o use la API del servidor) antes de comparar.")
    return {
        "ucp": ctx.ucp,
        "advertencias": advertencias,
        "semana_inicio": ctx.semana_inicio.isoformat(),
        "fecha_corte": ctx.fecha_corte.isoformat(),
        "semana_ref_inicio": ctx.semana_ref_inicio.isoformat() if ctx.semana_ref_inicio else None,
        "desplazamiento_ref_dias": desplazamiento,
        "fechas": fechas,
        "series": series,
        "asr_detalle": detalle,
        "omitidos": omitidos,
        "metricas": metricas,
        "semana": resumen_semana(series, fechas),
    }


def resumen_semana(series: Mapping[str, Curves], fechas: Sequence[str]) -> List[Dict[str, Any]]:
    """Energía semanal por serie (solo series con los 7 días) y, si hay real, error % y WAPE semanal."""
    real = series.get(SERIE_REAL) or {}
    completo_real = all(f in real for f in fechas)
    e_real = sum(float(np.sum(real[f])) for f in fechas) if completo_real else None
    out = []
    for nombre, curvas in series.items():
        if nombre == SERIE_REAL or not all(f in curvas for f in fechas):
            continue
        e = sum(float(np.sum(curvas[f])) for f in fechas)
        fila: Dict[str, Any] = {"serie": nombre, "energia_semana": e}
        if e_real:
            abs_err = sum(float(np.abs(np.asarray(curvas[f]) - np.asarray(real[f])).sum()) for f in fechas)
            fila.update({"energia_real_semana": e_real, "err_energia_semana_pct": 100 * (e / e_real - 1),
                         "wape_semana_pct": 100 * abs_err / e_real})
        out.append(fila)
    return out
