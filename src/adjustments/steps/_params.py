"""
Helpers de validación de params de los pasos.

Cada paso expone `validar_params(params) -> dict` construida con estos
helpers: completa defaults, castea tipos y lanza `ValueError` (nunca
TypeError ni OverflowError) ante claves desconocidas, None, valores no finitos,
tipos o rangos inválidos. Los enteros se validan SIN pasar por float (un
`10**400` no desborda: se rechaza por rango) y todos llevan un máximo, para
que ningún valor que pase la validación pueda desbordar la aritmética de
fechas o de bucles de los pasos. El pipeline la llama antes de ejecutar cada paso; el router
convierte el ValueError en 422.
"""

from __future__ import annotations

import math
import numbers
from datetime import date
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

# Rango de fechas admitido en params (semana de referencia, años excluidos).
# Acota la aritmética de fechas de los pasos lejos de date.min/date.max.
FECHA_MIN = date(1990, 1, 1)
FECHA_MAX = date(2100, 12, 31)


def completar(
    paso: str,
    defaults: Dict[str, Any],
    params: Optional[Dict[str, Any]],
    obsoletas: Iterable[str] = (),
) -> Dict[str, Any]:
    """
    {**defaults, **params}; rechaza params que no son dict y claves
    desconocidas (un typo como "alfa" no debe ignorarse en silencio). Las
    claves `obsoletas` (de records guardados con versiones previas) se descartan.
    """
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError(f"params de {paso!r} debe ser un objeto, llegó {type(params).__name__}")
    obsoletas = set(obsoletas)
    params = {k: v for k, v in params.items() if k not in obsoletas}
    desconocidas = sorted(str(k) for k in params if k not in defaults)
    if desconocidas:
        raise ValueError(
            f"params desconocidos para {paso!r}: {desconocidas}; válidos: {sorted(defaults)}"
        )
    return {**defaults, **params}


def _es_numero(v: Any) -> bool:
    """int/float (incl. numpy) pero no bool."""
    return isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))


def numero(
    paso: str,
    clave: str,
    v: Any,
    *,
    minimo: Optional[float] = None,
    maximo: Optional[float] = None,
    minimo_exclusivo: bool = False,
) -> float:
    """float finito dentro de [minimo, maximo] (o (minimo, …] si minimo_exclusivo)."""
    if v is None or not _es_numero(v):
        raise ValueError(f"{paso}.{clave} debe ser numérico, llegó {_corto(v)}")
    try:
        f = float(v)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{paso}.{clave} fuera de rango, llegó {_corto(v)}") from None
    if not math.isfinite(f):
        raise ValueError(f"{paso}.{clave} debe ser finito, llegó {v!r}")
    if minimo is not None and (f <= minimo if minimo_exclusivo else f < minimo):
        signo = ">" if minimo_exclusivo else ">="
        raise ValueError(f"{paso}.{clave} debe ser {signo} {minimo}, llegó {f}")
    if maximo is not None and f > maximo:
        raise ValueError(f"{paso}.{clave} debe ser <= {maximo}, llegó {f}")
    return f


def _corto(v: Any, n: int = 60) -> str:
    """repr acotado (un int de 400 dígitos no debe inflar el mensaje del 422)."""
    try:
        r = repr(v)
    except Exception:  # noqa: BLE001 - repr de objetos arbitrarios
        r = f"<{type(v).__name__}>"
    return r if len(r) <= n else r[: n - 3] + "..."


def entero(paso: str, clave: str, v: Any, *, minimo: int, maximo: int) -> int:
    """
    int en [minimo, maximo] (acepta float con valor entero, p. ej. 4.0). Los
    int de Python se comparan sin convertir a float (sin OverflowError).
    """
    if v is None or not _es_numero(v):
        raise ValueError(f"{paso}.{clave} debe ser entero, llegó {_corto(v)}")
    if isinstance(v, numbers.Integral):
        n = int(v)
    else:
        f = numero(paso, clave, v)
        if not f.is_integer():
            raise ValueError(f"{paso}.{clave} debe ser entero, llegó {_corto(v)}")
        n = int(f)
    if not minimo <= n <= maximo:
        raise ValueError(f"{paso}.{clave} debe estar entre {minimo} y {maximo}, llegó {_corto(n)}")
    return n


def lista_enteros(
    paso: str, clave: str, v: Any, *, minimo: int, maximo: int, max_items: int = 200
) -> List[int]:
    """Lista (posiblemente vacía) de enteros en [minimo, maximo]; rechaza escalares, None y strings."""
    if v is None or isinstance(v, (str, bytes, dict)) or not isinstance(v, (list, tuple, set, frozenset)):
        raise ValueError(f"{paso}.{clave} debe ser una lista de enteros, llegó {_corto(v)}")
    if len(v) > max_items:
        raise ValueError(f"{paso}.{clave} admite a lo sumo {max_items} elementos, llegaron {len(v)}")
    return sorted({entero(paso, f"{clave}[]", x, minimo=minimo, maximo=maximo) for x in v})


def opcion(paso: str, clave: str, v: Any, opciones: Iterable[str]) -> str:
    opciones = tuple(opciones)
    if not isinstance(v, str) or v not in opciones:
        raise ValueError(f"{paso}.{clave} debe ser una de {opciones}, llegó {_corto(v)}")
    return v
