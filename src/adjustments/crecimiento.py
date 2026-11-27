"""
Crecimiento interanual reciente `Y` y `g` dinámico de la réplica de Jamer.

`Y` = ΣE(x) / ΣE(x−364) − 1 sobre los últimos `ventana_dias` días terminados en
el corte, usando solo pares con 24 periodos válidos y clase `normal` en ambos
lados (sin festivos ni post-festivos). `g` = crecimiento que la tendencia anual
implica entre R1 y el día objetivo: `(1+Y)^(lag/364) − 1`, multiplicado por
`g_k` y acotado a [g_min, g_max]. Funciones puras (sin disco ni red).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from src.adjustments.data import CLASE_NORMAL, LAG_ANIO, clase_dia, energia_curva
from src.adjustments.steps import _params
from src.adjustments.types import Alert, parse_date

ALERTA_G_ACOTADO = "G_ACOTADO"
ALERTA_G_SIN_EVIDENCIA = "G_SIN_EVIDENCIA"
FUENTES_G = ("interanual", "interanual_corto", "interanual_estacional", "fijo")

# Defaults de los params de nivel relacionados con g (PLAN_IMPL §2.2).
DEFAULTS_G: Dict[str, Any] = {
    "g_fuente": "interanual",
    "g_fijo": 0.005,
    "g_ventana_dias": 28,
    "g_k": 1.0,
    "g_min": -0.01,
    "g_max": 0.02,
    "g_min_pares": 14,
}


@dataclass
class Crecimiento:
    Y: Optional[float]
    ventana: Tuple[str, str]
    n_pares: int
    fallback: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Y": self.Y,
            "ventana": list(self.ventana),
            "n_pares": self.n_pares,
            "fallback": self.fallback,
        }


@dataclass
class ResultadoG:
    g_raw: float
    g_usado: float
    acotado: bool
    alertas: List[Alert] = field(default_factory=list)


def crecimiento_interanual(
    curvas_reales: Mapping[str, Any],
    corte,
    festivos: Iterable[str],
    ventana_dias: int = 28,
    min_pares: int = 14,
) -> Crecimiento:
    """`Y` reciente; con menos de `min_pares` pares válidos: `Y=None`, `fallback=True`."""
    if ventana_dias < 1:
        raise ValueError("ventana_dias debe ser >= 1")
    corte = parse_date(corte)
    festivos_set = festivos if isinstance(festivos, (set, frozenset)) else set(festivos)
    desde = corte - timedelta(days=ventana_dias - 1)

    num = den = 0.0
    n_pares = 0
    for i in range(ventana_dias):
        x = desde + timedelta(days=i)
        ref = x - timedelta(days=LAG_ANIO)
        if (
            clase_dia(x, festivos_set) != CLASE_NORMAL
            or clase_dia(ref, festivos_set) != CLASE_NORMAL
        ):
            continue
        ex, er = energia_curva(curvas_reales.get(x.isoformat())), energia_curva(curvas_reales.get(ref.isoformat()))
        if ex is None or er is None or er <= 0:
            continue
        num += ex
        den += er
        n_pares += 1

    ventana = (desde.isoformat(), corte.isoformat())
    if n_pares < max(1, min_pares) or den <= 0:
        return Crecimiento(Y=None, ventana=ventana, n_pares=n_pares, fallback=True)
    return Crecimiento(Y=num / den - 1.0, ventana=ventana, n_pares=n_pares, fallback=False)


def g_crudo(Y: float, lag_dias: float, s_sem: Optional[float] = None) -> float:
    """`(1+Y)^(lag/364) − 1`; con `s_sem` (factor estacional R2/R3): `s_sem·(1+Y)^(lag/364) − 1`."""
    if Y <= -1.0:
        raise ValueError("Y debe ser > -1")
    crec = (1.0 + Y) ** (lag_dias / LAG_ANIO)
    if s_sem is not None:
        crec *= float(s_sem)
    return crec - 1.0


def calcular_g(
    Y: Optional[float],
    lag_dias: float,
    *,
    fuente: str = DEFAULTS_G["g_fuente"],
    g_fijo: float = DEFAULTS_G["g_fijo"],
    g_k: float = DEFAULTS_G["g_k"],
    g_min: float = DEFAULTS_G["g_min"],
    g_max: float = DEFAULTS_G["g_max"],
    s_sem: Optional[float] = None,
    fecha: Optional[str] = None,
) -> ResultadoG:
    """
    g usado = clip(g_k · g_raw, g_min, g_max). `fuente="fijo"` o sin evidencia
    (`Y is None`): g_raw = g_fijo (con `G_SIN_EVIDENCIA` solo si la
    fuente era dinámica). Si el clip actúa: alerta `G_ACOTADO` (info).
    `interanual_corto` solo cambia la ventana con que el llamador calculó `Y`;
    `interanual_estacional` exige `s_sem`.
    """
    if fuente not in FUENTES_G:
        raise ValueError(f"g_fuente inválida: {fuente!r}; opciones {FUENTES_G}")
    if g_min > g_max:
        raise ValueError("g_min no puede ser mayor que g_max")
    fechas = [fecha] if fecha else []
    alertas: List[Alert] = []

    sin_evidencia = Y is None
    if fuente == "fijo":
        g_raw = float(g_fijo)
    elif sin_evidencia:
        g_raw = float(g_fijo)
        alertas.append(
            Alert(
                nivel="info",
                codigo=ALERTA_G_SIN_EVIDENCIA,
                mensaje="Sin pares interanuales suficientes: se usa g fijo.",
                fechas=fechas,
                detalle={"g_fijo": float(g_fijo), "fuente": fuente},
            )
        )
    else:
        if fuente == "interanual_estacional" and s_sem is None:
            raise ValueError("interanual_estacional requiere s_sem")
        g_raw = g_crudo(float(Y), lag_dias, s_sem if fuente == "interanual_estacional" else None)

    g_k_g = float(g_k) * g_raw
    g_usado = float(min(max(g_k_g, g_min), g_max))
    acotado = g_usado != g_k_g
    if acotado:
        alertas.append(
            Alert(
                nivel="info",
                codigo=ALERTA_G_ACOTADO,
                mensaje=f"g ({g_k_g:.4f}) acotado a [{g_min:.4f}, {g_max:.4f}].",
                fechas=fechas,
                detalle={"g_raw": g_raw, "g_k": float(g_k), "g_min": g_min, "g_max": g_max, "g_usado": g_usado},
            )
        )
    return ResultadoG(g_raw=g_raw, g_usado=g_usado, acotado=acotado, alertas=alertas)


# Cotas máximas de los params g_* (ningún valor validado puede desbordar bucles ni la potencia de `g_crudo`).
G_FIJO_MAX = 0.5  # |g| de 50 % entre R1 y el objetivo ya es absurdo; las cotas son de cordura, no de negocio
G_K_MAX = 1000.0
G_VENTANA_MAX = LAG_ANIO  # la ventana no puede abarcar más de un año (los pares son x vs x−364)
G_MIN_PARES_MAX = LAG_ANIO


def validar_params_g(params: Optional[Mapping[str, Any]], paso: str = "nivel") -> Dict[str, Any]:
    """
    Valida/normaliza las claves `g_*` con los helpers de `steps._params` (ValueError acotado, con cotas
    máximas); las demás claves se ignoran. `paso` solo rotula el mensaje.
    """
    p = {**DEFAULTS_G, **{k: v for k, v in (params or {}).items() if k in DEFAULTS_G}}
    p["g_fuente"] = _params.opcion(paso, "g_fuente", p["g_fuente"], FUENTES_G)
    p["g_fijo"] = _params.numero(paso, "g_fijo", p["g_fijo"], minimo=-G_FIJO_MAX, maximo=G_FIJO_MAX)
    p["g_k"] = _params.numero(paso, "g_k", p["g_k"], minimo=0.0, maximo=G_K_MAX)
    p["g_min"] = _params.numero(paso, "g_min", p["g_min"], minimo=-G_FIJO_MAX, maximo=G_FIJO_MAX)
    p["g_max"] = _params.numero(paso, "g_max", p["g_max"], minimo=-G_FIJO_MAX, maximo=G_FIJO_MAX)
    if p["g_min"] > p["g_max"]:
        raise ValueError(f"{paso}.g_min no puede ser mayor que g_max")
    p["g_ventana_dias"] = _params.entero(paso, "g_ventana_dias", p["g_ventana_dias"], minimo=7, maximo=G_VENTANA_MAX)
    p["g_min_pares"] = _params.entero(paso, "g_min_pares", p["g_min_pares"], minimo=1, maximo=G_MIN_PARES_MAX)
    if p["g_ventana_dias"] % 7 != 0:
        raise ValueError(f"{paso}.g_ventana_dias debe ser múltiplo de 7")
    return p
