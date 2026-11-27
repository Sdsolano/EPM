"""Acumulado anual de días con error fuera de banda (módulo puro, apagado por defecto).

Cuenta, sobre el año corrido, los días en que ``|aprobada - real| / real`` supera una banda configurable y
devuelve un estado y una alerta informativa. No accede a BD ni a red.

Apagado por defecto: ``banda_pct=None`` y ``cupo_dias=None``. El "cupo de 25 días" NO es un valor confirmado
(solo un ejemplo hablado, sin confirmar): no hay default y nada se enciende solo.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, List, Mapping, Optional

CODIGO_ALERTA = "ACUMULADO_ANUAL_FUERA_RANGO"
FRACCION_AVISO_DEFAULT = 0.8  # provisional: avisa al llegar al 80 % del cupo

# estados del resultado: apagado | sin_cupo | dentro | cerca | excedido


def error_pct_dia(aprobada: float, real: float) -> Optional[float]:
    """``|aprobada - real| / real * 100``; None si el real no es evaluable (<= 0 o no finito)."""
    if real is None or aprobada is None:
        return None
    if not math.isfinite(real) or not math.isfinite(aprobada) or real <= 0:
        return None
    return abs(aprobada - real) / real * 100.0


def calcular_acumulado_anual(
    aprobada: Mapping[str, float],
    real: Mapping[str, float],
    anio: int,
    banda_pct: Optional[float] = None,
    cupo_dias: Optional[int] = None,
    hasta: Optional[str] = None,
    fraccion_aviso: float = FRACCION_AVISO_DEFAULT,
) -> Dict[str, Any]:
    """Cuenta días del año ``anio`` (1-ene..``hasta``) con error > ``banda_pct``.

    Parámetros: ``aprobada``/``real`` = energía diaria por fecha ISO. ``banda_pct`` (None = apagado).
    ``cupo_dias`` (None = sin cupo: cuenta pero no alerta). ``hasta`` = fecha ISO tope inclusive (default:
    sin tope dentro del año). ``fraccion_aviso`` en (0, 1]: umbral de "cerca" respecto al cupo.

    Devuelve ``{estado, anio, banda_pct, cupo_dias, dias_fuera, dias_evaluados, dias_no_evaluables,
    fechas_fuera: [{fecha, error_pct}], restantes, alerta}``; ``alerta`` es None o un dict con las claves de
    ``types.Alert`` (nivel="info", codigo, mensaje, fechas, detalle).
    """
    if banda_pct is not None and not (math.isfinite(banda_pct) and banda_pct >= 0):
        raise ValueError("banda_pct debe ser None o un número >= 0")
    if cupo_dias is not None and (isinstance(cupo_dias, bool) or int(cupo_dias) != cupo_dias or cupo_dias < 0):
        raise ValueError("cupo_dias debe ser None o un entero >= 0")
    if not (0 < fraccion_aviso <= 1):
        raise ValueError("fraccion_aviso debe estar en (0, 1]")
    if hasta is not None:
        date.fromisoformat(str(hasta)[:10])  # valida formato
    base: Dict[str, Any] = {
        "anio": int(anio), "banda_pct": banda_pct, "cupo_dias": cupo_dias,
        "dias_fuera": 0, "dias_evaluados": 0, "dias_no_evaluables": 0,
        "fechas_fuera": [], "restantes": None, "alerta": None,
    }
    if banda_pct is None:
        return {**base, "estado": "apagado"}

    prefijo = f"{int(anio):04d}-"
    tope = str(hasta)[:10] if hasta is not None else None
    fuera: List[Dict[str, Any]] = []
    evaluados = no_eval = 0
    for fecha in sorted(set(map(str, aprobada)) & set(map(str, real))):
        f = fecha[:10]
        if not f.startswith(prefijo) or (tope is not None and f > tope):
            continue
        err = error_pct_dia(aprobada[fecha], real[fecha])
        if err is None:
            no_eval += 1
            continue
        evaluados += 1
        if err > banda_pct:
            fuera.append({"fecha": f, "error_pct": round(err, 4)})

    n = len(fuera)
    base.update(dias_fuera=n, dias_evaluados=evaluados, dias_no_evaluables=no_eval, fechas_fuera=fuera)
    if cupo_dias is None:
        return {**base, "estado": "sin_cupo"}

    base["restantes"] = max(int(cupo_dias) - n, 0)
    if n > cupo_dias:
        estado = "excedido"
    elif n >= math.ceil(fraccion_aviso * cupo_dias) and n > 0:
        estado = "cerca"
    else:
        estado = "dentro"
    alerta = None
    if estado != "dentro":
        verbo = "superó" if estado == "excedido" else "se acerca a"
        alerta = {
            "nivel": "info",
            "codigo": CODIGO_ALERTA,
            "mensaje": (f"En {int(anio)} hay {n} día(s) con error > {banda_pct:g} % y el cupo configurado es "
                        f"{int(cupo_dias)}: el acumulado {verbo} el cupo (informativo)."),
            "fechas": [d["fecha"] for d in fuera],
            "detalle": {"dias_fuera": n, "cupo_dias": int(cupo_dias), "banda_pct": banda_pct,
                        "estado": estado, "restantes": base["restantes"]},
        }
    return {**base, "estado": estado, "alerta": alerta}
