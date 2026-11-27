"""
Interpretación de `p_i` (clima horario) para la mesa de ajustes.

`p_i` NO son mm: es el código de condición del tiempo (OpenWeather `weather_id`
desde 2025-07; legado numeración de iconos AccuWeather 2020-2024; 0 = sin dato).
Ver `docs/work/ajustes/replica_jamer/fase_clima/C0/REPORTE.md`.

Regla por periodo:
- `p_i` NaN / no numérico / 0  -> desconocido (dato faltante, no "sin lluvia").
- 200 <= p_i < 600             -> lluvia (tormenta 2xx, llovizna 3xx, lluvia 5xx).
- p_i >= 600                   -> sin lluvia (niebla 7xx, despejado/nubes 8xx).
- 1 <= p_i < 200 (legado)      -> lluvia si p_i en CODIGOS_LLUVIA_LEGADO; si no, sin lluvia.

Por día: `con_lluvia` si hay >= UMBRAL_PERIODOS periodos con lluvia y al menos
MIN_CONOCIDOS periodos conocidos; con menos conocidos el día es desconocido
(`con_lluvia` None, calidad "insuficiente"/"sin_dato").
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable

UMBRAL_PERIODOS = 3
MIN_CONOCIDOS = 12
CODIGOS_LLUVIA_LEGADO = frozenset({12, 13, 14, 15, 16, 17, 18, 39, 40, 41, 42})
LLUVIA_OWM_MIN = 200
LLUVIA_OWM_MAX = 600  # exclusivo


def _como_codigo(p_i: Any) -> Any:
    """Entero del código, o None si es desconocido (NaN, no numérico, 0 o negativo)."""
    try:
        v = float(p_i)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    return int(v)


def periodo_conocido(p_i: Any) -> bool:
    return _como_codigo(p_i) is not None


def periodo_con_lluvia(p_i: Any) -> bool:
    """True si el código de `p_i` indica lluvia; False si no hay lluvia o el dato es desconocido."""
    c = _como_codigo(p_i)
    if c is None:
        return False
    if c >= LLUVIA_OWM_MAX:
        return False
    if c >= LLUVIA_OWM_MIN:
        return True
    return c in CODIGOS_LLUVIA_LEGADO


def resumen_lluvia_dia(valores_p_i: Iterable[Any]) -> Dict[str, Any]:
    """Resumen de lluvia de un día a partir de sus `p_i` por periodo."""
    conocidos = 0
    lluvia = 0
    for v in valores_p_i:
        if not periodo_conocido(v):
            continue
        conocidos += 1
        if periodo_con_lluvia(v):
            lluvia += 1
    if conocidos == 0:
        calidad = "sin_dato"
    elif conocidos < MIN_CONOCIDOS:
        calidad = "insuficiente"
    else:
        calidad = "suficiente"
    suficiente = calidad == "suficiente"
    return {
        "periodos_lluvia": lluvia,
        "periodos_conocidos": conocidos,
        "frac_lluvia": (lluvia / conocidos) if conocidos else None,
        "con_lluvia": (lluvia >= UMBRAL_PERIODOS) if suficiente else None,
        "calidad": calidad,
    }
