"""
Paso 0 — base: pronóstico raw del modelo para fecha_corte+1 … domingo objetivo.

El proveedor del pronóstico (`base_provider`) se inyecta como argumento
keyword (no va en `params`, para que los params sigan siendo JSON
persistible). El pipeline se lo pasa al ejecutar este paso.

La versión del modelo NO es un param: el llamador (router) la deja en
`ctx.extras["modelo_version"]` y este paso la copia a diagnostics.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Callable, Dict, Optional

import numpy as np

from src.adjustments.data import energia_curvas, iso
from src.adjustments.steps import _params
from src.adjustments.types import Alert, Curves, RunContext, StepResult

DEFAULTS: Dict[str, Any] = {}
_OBSOLETAS = ("modelo_version",)  # era param hasta la ronda 1; ahora va en ctx.extras


def validar_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """El paso base no tiene params (descarta `modelo_version` de records viejos)."""
    return _params.completar("base", DEFAULTS, params, obsoletas=_OBSOLETAS)


def rango_base(ctx: RunContext):
    """(primer día pronosticado, domingo objetivo, n_dias)."""
    inicio = ctx.fecha_corte + timedelta(days=1)
    fin = ctx.semana_inicio + timedelta(days=6)
    return inicio, fin, (fin - ctx.fecha_corte).days


def step(
    ctx: RunContext,
    prev: Curves,
    params: Dict[str, Any],
    *,
    base_provider: Optional[Callable[[str, Any, int], Curves]] = None,
) -> StepResult:
    validar_params(params)
    if base_provider is None:
        raise ValueError("El paso base requiere un base_provider")

    inicio, fin, n_dias = rango_base(ctx)
    raw = base_provider(ctx.ucp, ctx.fecha_corte, n_dias) or {}
    raw = {str(k)[:10]: v for k, v in raw.items()}

    curves: Curves = {}
    faltantes = []
    for i in range(n_dias):
        f = iso(inicio + timedelta(days=i))
        if f in raw:
            # Valores no convertibles quedan como NaN: el control/approval los bloquea.
            try:
                curves[f] = np.asarray(raw[f], dtype=float).ravel().tolist()
            except (TypeError, ValueError):
                curves[f] = [float("nan")] * 24
        else:
            faltantes.append(f)

    alerts = []
    if faltantes:
        alerts.append(
            Alert(
                nivel="warning",
                codigo="BASE_INCOMPLETA",
                mensaje=f"El pronóstico base no trae {len(faltantes)} día(s) del rango",
                fechas=faltantes,
            )
        )

    diagnostics = {
        "fecha_corte": iso(ctx.fecha_corte),
        "fecha_emision": iso(ctx.fecha_emision),
        "semana_inicio": iso(ctx.semana_inicio),
        "dias_brecha": (ctx.semana_inicio - ctx.fecha_corte).days - 1,
        "rango": [iso(inicio), iso(fin)],
        "n_dias": n_dias,
        "modelo_version": ctx.extras.get("modelo_version"),
        "energia": energia_curvas(curves),
    }
    return StepResult(curves=curves, diagnostics=diagnostics, alerts=alerts)
