"""
Paso 2 — nivel: ajusta la energía diaria de los 7 días objetivo (no la forma).

Para cada día objetivo d (desfase = semana objetivo − semana referencia):
- Ancla actual a: último día real del mismo día de semana <= fecha_corte
  (a = d − 7·ceil((d − corte)/7)). lag = d − a.
- Día ref d' = d − desfase y ancla ref a' = d' − lag (mismo desfase de corte
  en el año de referencia).
- ratio = E(d') / E(a');  E*(d) = E(a) · ratio.
- E(d) = (1 − α)·E_base(d) + α·E*(d), α default 0.25 (calibrado en el
  backtest 2026-11-12). La curva previa se escala a E(d) (forma intacta).
- E*(d) no depende de α ni de la base: `evidencias(ctx)` la expone para que el
  paso control derive de ahí la tendencia semanal esperada.

Casos borde (regla elegida):
- Anclas (a y a') nunca son festivos: si alguna lo es (o falta el dato) se
  retrocede una semana más ambas a la vez (lag + 7), hasta 6 semanas extra (junio tiene lunes festivos seguidos).
- d laboral/fin de semana cuyo d' es festivo: se prueba d' desplazado
  −7, +7, −14, +14 días (manteniendo el lag) hasta hallar un d' no festivo.
- d festivo: d' es el festivo de la semana ref en la misma posición si lo es;
  si no, el festivo homólogo de d en el año de referencia
  (`find_homologous_holiday_date`). El ratio queda "festivo vs día normal".
  Sin homólogo con datos → sin evidencia.
- Años excluidos (`ctx.extras["anios_excluidos"]`, que fija el paso
  referencia; default [2020]): d', a' y los desplazamientos nunca caen en
  esos años (si caen, ese candidato/ancla se descarta).
- Sin evidencia para un día (o sin semana de referencia): α = 0 ese día y
  alerta "SIN_EVIDENCIA_NIVEL" (warning).
- Los días de brecha (antes de la semana objetivo) se conservan tal cual.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Set

import numpy as np

from src.adjustments.data import (
    ANIOS_EXCLUIDOS_DEFAULT,
    ancla_mismo_dow,
    dias_semana,
    energia_diaria,
    iso,
)
from src.adjustments.steps import _params
from src.adjustments.types import Alert, Curves, RunContext, StepResult, validar_curva
from src.prediction.holiday_homology import find_homologous_holiday_date

DEFAULTS: Dict[str, Any] = {"alpha": 0.25}
MAX_SEMANAS_EXTRA_ANCLA = 6
DESPLAZAMIENTOS_REF = (0, -7, 7, -14, 14)


def validar_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """alpha: número finito en [0, 1]."""
    p = _params.completar("nivel", DEFAULTS, params)
    p["alpha"] = _params.numero("nivel", "alpha", p["alpha"], minimo=0.0, maximo=1.0)
    return p


def anios_excluidos(ctx: RunContext) -> Set[int]:
    """Años que no se usan como referencia (los fija el paso referencia en ctx.extras)."""
    valor = ctx.extras.get("anios_excluidos")
    return {int(a) for a in (ANIOS_EXCLUIDOS_DEFAULT if valor is None else valor)}


def _evidencia(
    d: date,
    ctx: RunContext,
    energia: Dict[str, float],
    festivos: set,
    desfase: int,
    excluidos: Set[int] = frozenset(),
) -> Optional[Dict[str, Any]]:
    def ok(x: date, permitir_festivo: bool = False, lado_ref: bool = False) -> bool:
        if lado_ref and x.year in excluidos:
            return False
        return iso(x) in energia and energia[iso(x)] > 0 and (
            permitir_festivo or iso(x) not in festivos
        )

    if iso(d) in festivos:
        directo = d - timedelta(days=desfase)
        if iso(directo) in festivos and ok(directo, permitir_festivo=True, lado_ref=True):
            candidatos, regla = [directo], "festivo_ref_misma_posicion"
        else:
            hom = find_homologous_holiday_date(d, directo.year, ctx.festivos)
            if hom is None or not ok(date.fromisoformat(hom), permitir_festivo=True, lado_ref=True):
                return None
            candidatos, regla = [date.fromisoformat(hom)], "festivo_homologo"
    else:
        candidatos = [
            d - timedelta(days=desfase - s)
            for s in DESPLAZAMIENTOS_REF
            if ok(d - timedelta(days=desfase - s), lado_ref=True)
        ]
        regla = "mismo_dia_semana"

    primera_ancla = ancla_mismo_dow(d, ctx.fecha_corte)
    for extra in range(MAX_SEMANAS_EXTRA_ANCLA + 1):
        ancla = primera_ancla - timedelta(days=7 * extra)
        if not ok(ancla):
            continue
        lag = (d - ancla).days
        for dia_ref in candidatos:
            ancla_ref = dia_ref - timedelta(days=lag)
            if not ok(ancla_ref, lado_ref=True):
                continue
            ratio = energia[iso(dia_ref)] / energia[iso(ancla_ref)]
            return {
                "ancla": iso(ancla),
                "dia_ref": iso(dia_ref),
                "ancla_ref": iso(ancla_ref),
                "lag_dias": lag,
                "ratio": ratio,
                "E_ancla": energia[iso(ancla)],
                "E_objetivo": energia[iso(ancla)] * ratio,
                "regla": (
                    "mismo_dia_semana_desplazado"
                    if regla == "mismo_dia_semana" and (d - dia_ref).days != desfase
                    else regla
                ),
            }
    return None


def evidencias(ctx: RunContext) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Evidencia (ancla, día ref, ratio, E_objetivo = E*(d)) por día de la semana
    objetivo; None si no hay evidencia o no hay semana de referencia.
    """
    fechas = [iso(d) for d in dias_semana(ctx.semana_inicio)]
    if not ctx.semana_ref_inicio:
        return {f: None for f in fechas}
    desfase = (ctx.semana_inicio - ctx.semana_ref_inicio).days
    if not desfase:
        return {f: None for f in fechas}
    energia = energia_diaria(ctx.historico)
    festivos = set(ctx.festivos)
    excluidos = anios_excluidos(ctx)
    return {
        iso(d): _evidencia(d, ctx, energia, festivos, desfase, excluidos)
        for d in dias_semana(ctx.semana_inicio)
    }


def step(ctx: RunContext, prev: Curves, params: Dict[str, Any]) -> StepResult:
    alpha = validar_params(params)["alpha"]

    curves: Curves = {f: list(v) for f, v in prev.items()}
    desfase = (
        (ctx.semana_inicio - ctx.semana_ref_inicio).days if ctx.semana_ref_inicio else None
    )
    evidencia_dias = evidencias(ctx)

    por_dia: Dict[str, Any] = {}
    sin_evidencia: List[str] = []
    base_nula: List[str] = []

    for d in dias_semana(ctx.semana_inicio):
        f = iso(d)
        if f not in prev or validar_curva(prev[f]) is not None:
            continue  # faltante o inválida: se deja igual y el control la bloquea
        curva = np.asarray(prev[f], dtype=float)
        e_base = float(curva.sum())
        ev = evidencia_dias.get(f)
        alpha_d = alpha if ev is not None else 0.0
        if ev is None:
            sin_evidencia.append(f)
        e_obj = ev["E_objetivo"] if ev else None
        e_final = (1 - alpha_d) * e_base + alpha_d * e_obj if ev else e_base

        if e_base > 0:
            curves[f] = (curva * (e_final / e_base)).tolist()
        elif alpha_d > 0:
            base_nula.append(f)
            e_final = e_base

        por_dia[f] = {
            "E_base": e_base,
            "E_objetivo": e_obj,
            "E_final": e_final,
            "ratio": ev["ratio"] if ev else None,
            "alpha": alpha_d,
            "ancla": ev["ancla"] if ev else None,
            "ancla_ref": ev["ancla_ref"] if ev else None,
            "dia_ref": ev["dia_ref"] if ev else None,
            "lag_dias": ev["lag_dias"] if ev else None,
            "regla": ev["regla"] if ev else "sin_evidencia",
        }

    alerts: List[Alert] = []
    if sin_evidencia:
        alerts.append(
            Alert(
                nivel="warning",
                codigo="SIN_EVIDENCIA_NIVEL",
                mensaje=(
                    "Sin evidencia en la semana de referencia para ajustar el nivel; "
                    "se conserva la energía del paso anterior (α = 0)"
                ),
                fechas=sin_evidencia,
                detalle={"semana_ref_inicio": iso(ctx.semana_ref_inicio) if ctx.semana_ref_inicio else None},
            )
        )
    if base_nula:
        alerts.append(
            Alert(
                nivel="warning",
                codigo="CURVA_BASE_NULA",
                mensaje="La curva previa suma 0; no se puede escalar al nuevo nivel",
                fechas=base_nula,
            )
        )

    diagnostics = {
        "alpha": alpha,
        "semana_ref_inicio": iso(ctx.semana_ref_inicio) if ctx.semana_ref_inicio else None,
        "desfase_dias": desfase,
        "anios_excluidos": sorted(anios_excluidos(ctx)),
        "dias": por_dia,
    }
    return StepResult(curves=curves, diagnostics=diagnostics, alerts=alerts)
