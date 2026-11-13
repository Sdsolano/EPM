"""
Paso 3 — forma: reemplaza la forma horaria conservando la energía diaria del
paso anterior (suma diaria idéntica, tolerancia 1e-6).

curva(d) = [(1 − β)·forma_previa(d) + β·forma_fuente(d)] · E(d)
con formas normalizadas (suman 1) y E(d) = suma de la curva previa.

Fuente de forma por día:
- festivo → `average_holiday_hourly_profile` (festivos homólogos), usado SOLO
  como forma normalizada. Sin homólogos → domingo real más reciente.
- resto, según `params["fuente"]`:
  * "reciente": último día real <= corte con el MISMO día de semana
    y no festivo (más estricto que "mismo tipo": un lunes tiene una
    madrugada distinta a un miércoles).
  * "referencia": día equivalente de la semana de referencia (d − desfase);
    si no tiene datos o es festivo → cae a "reciente" con alerta info.
  * "promedio_n" (default, calibrado en el backtest 2026-11-12): promedio de
    las formas de los últimos n días reales del mismo día de semana, no
    festivos (n default 4).
β default 1.0 (forma de la fuente pura).
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.adjustments.data import PERIODOS, ancla_mismo_dow, curvas_reales, dias_semana, iso
from src.adjustments.steps import _params
from src.adjustments.types import Alert, Curves, RunContext, StepResult, validar_curva
from src.prediction.holiday_homology import average_holiday_hourly_profile

FUENTES = ("reciente", "referencia", "promedio_n")
DEFAULTS: Dict[str, Any] = {"fuente": "promedio_n", "n": 4, "beta": 1.0}
MAX_SEMANAS_BUSQUEDA = 12
MAX_N = 60  # más de ~1 año de días del mismo día de semana no aporta y alarga la búsqueda


def validar_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """fuente en FUENTES; n entero en [1, MAX_N]; beta número finito en [0, 1]."""
    p = _params.completar("forma", DEFAULTS, params)
    p["fuente"] = _params.opcion("forma", "fuente", p["fuente"], FUENTES)
    p["n"] = _params.entero("forma", "n", p["n"], minimo=1, maximo=MAX_N)
    p["beta"] = _params.numero("forma", "beta", p["beta"], minimo=0.0, maximo=1.0)
    return p


def _normalizar(v: np.ndarray) -> Optional[np.ndarray]:
    total = float(np.sum(v))
    if not math.isfinite(total) or total <= 0:
        return None
    return np.asarray(v, dtype=float) / total


def _pico(v) -> Dict[str, float]:
    arr = np.asarray(v, dtype=float)
    i = int(np.argmax(arr))
    return {"mw": float(arr[i]), "hora": i + 1}


def _dias_mismo_dow(
    d: date, ctx: RunContext, curvas: Dict[str, np.ndarray], festivos: set, n: int
) -> Dict[str, np.ndarray]:
    """
    Hasta n días reales <= corte, mismo día de semana que d, no festivos
    (recientes primero) → {fecha: forma normalizada}.
    """
    primera = ancla_mismo_dow(d, ctx.fecha_corte)
    encontrados: Dict[str, np.ndarray] = {}
    for k in range(MAX_SEMANAS_BUSQUEDA + n):
        f = iso(primera - timedelta(days=7 * k))
        forma = _normalizar(curvas[f]) if f in curvas and f not in festivos else None
        if forma is not None:
            encontrados[f] = forma
            if len(encontrados) >= n:
                break
    return encontrados


def _forma_fuente(
    d: date,
    fuente: str,
    n: int,
    ctx: RunContext,
    curvas: Dict[str, np.ndarray],
    festivos: set,
) -> Tuple[Optional[np.ndarray], str, List[str], Optional[str]]:
    """(forma normalizada, fuente usada, fechas fuente, motivo de fallback)."""
    if iso(d) in festivos:
        perfil, meta = average_holiday_hourly_profile(
            ctx.historico, d, ctx.festivos, period_cols=PERIODOS,
            date_column="fecha", scale_with_recent_sundays=False,
        )
        forma = _normalizar(perfil) if meta.get("n_rows", 0) > 0 else None
        if forma is not None:
            return forma, f"festivo_{meta.get('match_mode')}", list(meta.get("match_dates", [])), None
        domingo = d + timedelta(days=6 - d.weekday())  # mismo día de semana que un domingo
        formas = _dias_mismo_dow(domingo, ctx, curvas, festivos, 1)
        if formas:
            return next(iter(formas.values())), "festivo_domingo_reciente", list(formas), "festivo_sin_homologo"
        return None, "sin_forma", [], "festivo_sin_homologo"

    motivo = None
    if fuente == "referencia":
        if ctx.semana_ref_inicio is not None:
            dia_ref = d - (ctx.semana_inicio - ctx.semana_ref_inicio)
            f = iso(dia_ref)
            forma_ref = _normalizar(curvas[f]) if f in curvas and f not in festivos else None
            if forma_ref is not None:
                return forma_ref, "referencia", [f], None
        motivo = "referencia_no_disponible"
        fuente = "reciente"

    cantidad = n if fuente == "promedio_n" else 1
    formas = _dias_mismo_dow(d, ctx, curvas, festivos, cantidad)
    if not formas:
        return None, "sin_forma", [], motivo or "sin_dias_comparables"
    forma = np.mean(list(formas.values()), axis=0)
    return forma / forma.sum(), fuente, list(formas), motivo


def step(ctx: RunContext, prev: Curves, params: Dict[str, Any]) -> StepResult:
    p = validar_params(params)
    fuente, n, beta = p["fuente"], p["n"], p["beta"]

    curves: Curves = {f: list(v) for f, v in prev.items()}
    curvas = curvas_reales(ctx.historico)
    festivos = set(ctx.festivos)
    por_dia: Dict[str, Any] = {}
    fallbacks: List[str] = []
    sin_forma: List[str] = []

    for d in dias_semana(ctx.semana_inicio):
        f = iso(d)
        if f not in prev or validar_curva(prev[f]) is not None:
            continue  # faltante o inválida: se deja igual y el control la bloquea
        curva_prev = np.asarray(prev[f], dtype=float)
        energia = float(curva_prev.sum())
        forma_prev = _normalizar(curva_prev)
        forma_src, usada, fechas, motivo = _forma_fuente(d, fuente, n, ctx, curvas, festivos)

        if forma_src is None or forma_prev is None:
            sin_forma.append(f)
            nueva = curva_prev
        else:
            forma = (1 - beta) * forma_prev + beta * forma_src
            nueva = forma / forma.sum() * energia
            curves[f] = nueva.tolist()
        if motivo:
            fallbacks.append(f)

        por_dia[f] = {
            "fuente": usada,
            "fechas_fuente": fechas,
            "fallback": motivo,
            "energia": energia,
            "pico_antes": _pico(curva_prev),
            "pico_despues": _pico(nueva),
            "diferencia_energia": float(np.sum(nueva) - energia),
        }

    alerts: List[Alert] = []
    if fallbacks:
        alerts.append(
            Alert(
                nivel="info",
                codigo="FORMA_FUENTE_ALTERNA",
                mensaje="Algunos días usaron una fuente de forma alterna a la pedida",
                fechas=fallbacks,
                detalle={f: por_dia[f]["fallback"] for f in fallbacks},
            )
        )
    if sin_forma:
        alerts.append(
            Alert(
                nivel="warning",
                codigo="SIN_FORMA",
                mensaje="No se encontró forma de referencia; se conserva la curva previa",
                fechas=sin_forma,
            )
        )

    diagnostics = {"fuente": fuente, "n": n, "beta": beta, "dias": por_dia}
    return StepResult(curves=curves, diagnostics=diagnostics, alerts=alerts)
