"""
Paso 1 — referencia: elige la semana de referencia (no modifica curvas).

Reglas (Fase 1, sin ranking):
1. `params["semana_ref_inicio"]` la fuerza (si no es lunes se lleva al lunes
   de esa semana, con alerta info). Debe estar en 1990-01-01..2100-12-31
   (`validar_params`) y no ser posterior a la semana objetivo (`step`);
   fuera de eso → ValueError (422 en el router).
2. Si la semana objetivo no tiene festivos: lunes objetivo − 364 días
   (misma semana del año anterior alineada por día de semana). Si esa semana
   toca un año excluido se retrocede de a 364 días (−728, …) hasta
   `max_anios_atras`.
3. Si tiene festivos: para k = 1..`max_anios_atras` (default 4) años atrás se
   busca el festivo homólogo (`find_homologous_holiday_date`) de cada festivo
   de la semana. Se elige el año más reciente que maximiza la cantidad de
   homólogos que caen en el mismo día de semana Y cuya semana tiene los 7
   días en el histórico y ningún día en un año excluido. La semana de
   referencia es la que contiene el homólogo del primer festivo que coincide.
   Si ningún año sirve, cae al default de la regla 2 con alerta info.

Años excluidos (`anios_excluidos`, default [2020] — pandemia): ninguna semana
elegida automáticamente puede tener días en esos años. Si el analista fuerza
una semana en un año excluido se respeta, con alerta warning (el paso nivel
tampoco usa días de años excluidos, así que quedará sin evidencia salvo que
se pase `anios_excluidos: []`).

La semana elegida queda en `diagnostics["semana_ref_inicio"]` y los años
excluidos en `diagnostics["anios_excluidos"]`; el pipeline los copia a
`ctx.semana_ref_inicio` y `ctx.extras["anios_excluidos"]` (el paso no muta el
contexto) para que nivel/control los respeten.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from src.adjustments.data import ANIOS_EXCLUIDOS_DEFAULT, dias_semana, energia_diaria, iso, lunes_de
from src.adjustments.steps import _params
from src.adjustments.types import Alert, Curves, RunContext, StepResult, parse_date
from src.prediction.holiday_homology import find_homologous_holiday_date

DEFAULTS: Dict[str, Any] = {
    "semana_ref_inicio": None,
    "max_anios_atras": 4,
    "anios_excluidos": list(ANIOS_EXCLUIDOS_DEFAULT),
}
_OBSOLETAS = ("max_anios",)  # reemplazado por max_anios_atras (fix H2)
MAX_ANIOS_ATRAS = 20


def validar_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    semana_ref_inicio: None o fecha ISO en [FECHA_MIN, FECHA_MAX] (1990-01-01 ..
    2100-12-31; se guarda como "YYYY-MM-DD"). Que no sea posterior a la semana
    objetivo se verifica en `step` (necesita el contexto) y también es ValueError.
    max_anios_atras: entero en [1, MAX_ANIOS_ATRAS]; anios_excluidos: lista de
    años en [FECHA_MIN.year, FECHA_MAX.year].
    """
    p = _params.completar("referencia", DEFAULTS, params, obsoletas=_OBSOLETAS)
    forzada = p["semana_ref_inicio"]
    if forzada is not None:
        try:
            if not isinstance(forzada, (str, date)):
                raise TypeError
            fecha = parse_date(forzada)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f"referencia.semana_ref_inicio debe ser una fecha YYYY-MM-DD, llegó {_params._corto(forzada)}"
            ) from None
        if not _params.FECHA_MIN <= fecha <= _params.FECHA_MAX:
            raise ValueError(
                f"referencia.semana_ref_inicio debe estar entre {_params.FECHA_MIN} y "
                f"{_params.FECHA_MAX}, llegó {fecha.isoformat()}"
            )
        p["semana_ref_inicio"] = fecha.isoformat()
    p["max_anios_atras"] = _params.entero(
        "referencia", "max_anios_atras", p["max_anios_atras"], minimo=1, maximo=MAX_ANIOS_ATRAS
    )
    p["anios_excluidos"] = _params.lista_enteros(
        "referencia", "anios_excluidos", p["anios_excluidos"],
        minimo=_params.FECHA_MIN.year, maximo=_params.FECHA_MAX.year,
    )
    return p


def _faltantes(lunes: date, energia: Dict[str, float]) -> List[str]:
    return [iso(d) for d in dias_semana(lunes) if iso(d) not in energia]


def semana_excluida(lunes: date, excluidos: Set[int]) -> bool:
    """True si algún día de la semana que empieza en `lunes` cae en un año excluido."""
    return bool(excluidos) and any(d.year in excluidos for d in dias_semana(lunes))


def _default_364(semana_inicio: date, max_anios: int, excluidos: Set[int]) -> Tuple[date, bool]:
    """Primera semana −364·k (k = 1..max) fuera de años excluidos; (lunes, encontrada)."""
    for k in range(1, max(1, max_anios) + 1):
        lunes = semana_inicio - timedelta(days=364 * k)
        if not semana_excluida(lunes, excluidos):
            return lunes, True
    return semana_inicio - timedelta(days=364), False


def _buscar_por_festivo(
    ctx: RunContext,
    festivos_semana: List[date],
    energia: Dict[str, float],
    max_anios: int,
    excluidos: Set[int],
) -> Optional[Dict[str, Any]]:
    mejor: Optional[Dict[str, Any]] = None
    for k in range(1, max_anios + 1):
        homologos: Dict[str, str] = {}
        coinciden: List[date] = []
        for h in festivos_semana:
            hom = find_homologous_holiday_date(h, h.year - k, ctx.festivos)
            if hom is None:
                continue
            homologos[iso(h)] = hom
            if date.fromisoformat(hom).weekday() == h.weekday():
                coinciden.append(h)
        if not coinciden:
            continue
        lunes_ref = lunes_de(date.fromisoformat(homologos[iso(coinciden[0])]))
        if semana_excluida(lunes_ref, excluidos) or _faltantes(lunes_ref, energia):
            continue
        if mejor is None or len(coinciden) > mejor["coincidencias"]:
            mejor = {
                "lunes": lunes_ref,
                "anios_atras": k,
                "coincidencias": len(coinciden),
                "homologos": homologos,
            }
        if len(coinciden) == len(festivos_semana):
            break  # coincidencia total en el año más reciente posible
    return mejor


def _alerta_anio_excluido(lunes_ref: date, mensaje: str) -> Alert:
    return Alert(nivel="warning", codigo="REFERENCIA_ANIO_EXCLUIDO", mensaje=mensaje, fechas=[iso(lunes_ref)])


def _default_con_alerta(
    semana_inicio: date, max_anios: int, excluidos: Set[int], alerts: List[Alert]
) -> date:
    """Regla −364·k; si todas caen en años excluidos agrega la alerta warning."""
    lunes_ref, ok = _default_364(semana_inicio, max_anios, excluidos)
    if not ok:
        alerts.append(_alerta_anio_excluido(
            lunes_ref,
            f"Todas las semanas alineadas dentro de max_anios_atras caen en años excluidos "
            f"{sorted(excluidos)}; se usa {iso(lunes_ref)} y el paso nivel no usará esos días",
        ))
    return lunes_ref


def step(ctx: RunContext, prev: Curves, params: Dict[str, Any]) -> StepResult:
    p = validar_params(params)
    max_anios = p["max_anios_atras"]
    excluidos = set(p["anios_excluidos"])
    festivos_set = set(ctx.festivos)
    energia = energia_diaria(ctx.historico)
    festivos_semana = [d for d in dias_semana(ctx.semana_inicio) if iso(d) in festivos_set]
    alerts: List[Alert] = []
    diag: Dict[str, Any] = {
        "festivos_objetivo": [iso(d) for d in festivos_semana],
        "anios_excluidos": sorted(excluidos),
        "max_anios_atras": max_anios,
    }

    forzada = parse_date(p.get("semana_ref_inicio"))
    if forzada is not None:
        if lunes_de(forzada) > ctx.semana_inicio:
            raise ValueError(
                f"referencia.semana_ref_inicio ({iso(forzada)}) no puede ser posterior a la "
                f"semana objetivo ({iso(ctx.semana_inicio)})"
            )
        lunes_ref = lunes_de(forzada)
        metodo = "parametro"
        if lunes_ref != forzada:
            alerts.append(
                Alert(
                    nivel="info",
                    codigo="REFERENCIA_AJUSTADA_A_LUNES",
                    mensaje=f"La semana de referencia {iso(forzada)} se llevó al lunes {iso(lunes_ref)}",
                    fechas=[iso(lunes_ref)],
                )
            )
        if semana_excluida(lunes_ref, excluidos):
            alerts.append(_alerta_anio_excluido(
                lunes_ref,
                f"La semana forzada {iso(lunes_ref)} cae en un año excluido "
                f"{sorted(excluidos)}; el paso nivel no usará esos días",
            ))
    elif not festivos_semana:
        lunes_ref = _default_con_alerta(ctx.semana_inicio, max_anios, excluidos, alerts)
        metodo = "anio_anterior_364"
    else:
        hallado = _buscar_por_festivo(ctx, festivos_semana, energia, max_anios, excluidos)
        if hallado is not None:
            lunes_ref = hallado["lunes"]
            metodo = "festivo_homologo"
            diag.update(
                anios_atras=hallado["anios_atras"],
                homologos=hallado["homologos"],
                coincidencias=hallado["coincidencias"],
            )
        else:
            lunes_ref = _default_con_alerta(ctx.semana_inicio, max_anios, excluidos, alerts)
            metodo = "anio_anterior_364"
            alerts.append(
                Alert(
                    nivel="info",
                    codigo="REFERENCIA_SIN_HOMOLOGO",
                    mensaje=(
                        "Ningún año previo tiene el festivo homólogo en el mismo día de "
                        f"semana con datos completos (hasta {max_anios} años atrás, sin "
                        f"{sorted(excluidos)}); se usa la semana {iso(lunes_ref)}"
                    ),
                    fechas=[iso(d) for d in festivos_semana],
                )
            )

    faltantes = _faltantes(lunes_ref, energia)
    if faltantes:
        alerts.append(
            Alert(
                nivel="warning",
                codigo="REFERENCIA_SIN_DATOS",
                mensaje=f"La semana de referencia no tiene datos reales para {len(faltantes)} día(s)",
                fechas=faltantes,
            )
        )

    diag.update(
        semana_ref_inicio=iso(lunes_ref),
        semana_ref_fin=iso(lunes_ref + timedelta(days=6)),
        metodo=metodo,
        desfase_dias=(ctx.semana_inicio - lunes_ref).days,
        festivos_ref=[iso(d) for d in dias_semana(lunes_ref) if iso(d) in festivos_set],
        dias_con_datos=7 - len(faltantes),
    )
    curves = {f: list(v) for f, v in prev.items()}
    return StepResult(curves=curves, diagnostics=diag, alerts=alerts)
