"""
Referencia reciente por clase de día (R1 de la réplica de Jamer).

R1 de un día objetivo `d` = el día real MÁS RECIENTE (<= corte) con el mismo día
de semana y la MISMA clase de día que `d` (festivo / post_festivo / normal; con
`saltar_puente`, también `puente`). Un objetivo normal salta festivos y
post-festivos; un objetivo festivo o post-festivo usa el último de su clase.

Funciones puras: no leen disco ni red. `curvas_reales` = {fecha_iso: array(24)}
(ver `data.curvas_reales`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.adjustments.data import (
    CLASE_FESTIVO,
    CLASE_NORMAL,
    LAG_ANIO,
    ancla_mismo_dow,
    clase_dia,
    energia_curva,
)
from src.adjustments.types import Alert, parse_date
from src.prediction.holiday_homology import find_homologous_holiday_date

ALERTA_SIN_MISMA_CLASE = "R1_SIN_MISMA_CLASE"
MAX_SEMANAS_DEFAULT = 12


@dataclass
class RecientesDia:
    fechas: List[date]
    clase: str
    saltados: List[Dict[str, str]] = field(default_factory=list)
    alertas: List[Alert] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fechas": [f.isoformat() for f in self.fechas],
            "clase": self.clase,
            "saltados": [dict(s) for s in self.saltados],
            "alertas": [a.to_dict() for a in self.alertas],
        }


def _curva_valida(curvas_reales: Mapping[str, Any], d: date) -> bool:
    return energia_curva(curvas_reales.get(d.isoformat())) is not None


def desfase_r2(semana_inicio, semana_ref_inicio, *, anual_si_falta: bool = False) -> Optional[int]:
    """
    Desfase en días con que se define R2 (`d − desfase`): semana_inicio − semana_ref_inicio. Sin semana de
    referencia: `None`, o 364 con `anual_si_falta=True`; con `anual_si_falta=True` un desfase 0 también es 364.
    Definición única (la usan `fecha_r2` y el diagnóstico `desfase_dias` de nivel).
    """
    if semana_ref_inicio is None:
        return LAG_ANIO if anual_si_falta else None
    desfase = (parse_date(semana_inicio) - parse_date(semana_ref_inicio)).days
    return LAG_ANIO if (desfase == 0 and anual_si_falta) else desfase


def fecha_r2(
    d,
    semana_inicio,
    semana_ref_inicio,
    festivos: Optional[Iterable[str]] = None,
    *,
    anual_si_falta: bool = False,
) -> Optional[date]:
    """
    R2 de `d`: el mismo día de la semana de referencia (`d − desfase`, con
    desfase = semana_inicio − semana_ref_inicio). Definición ÚNICA de R2 (nivel,
    forma y control la comparten).
    - Sin semana de referencia: `None`; con `anual_si_falta=True` (perfil Jamer), `d − 364`
      (también si el desfase fuera 0).
    - `festivos` (opcional): un día festivo cuyo R2 no es festivo usa el festivo homólogo del
      año de R2 (`find_homologous_holiday_date`). Sin `festivos` no se hace ese reemplazo
      (comportamiento del control estándar).
    Solo calcula la fecha: quien la usa verifica que haya curva.
    """
    d = parse_date(d)
    desfase = desfase_r2(semana_inicio, semana_ref_inicio, anual_si_falta=anual_si_falta)
    if desfase is None:
        return None
    r2 = d - timedelta(days=desfase)
    if festivos is not None:
        festivos_set = festivos if isinstance(festivos, (set, frozenset)) else set(festivos)
        if d.isoformat() in festivos_set and r2.isoformat() not in festivos_set:
            hom = find_homologous_holiday_date(d, r2.year, festivos_set)
            if hom:
                r2 = date.fromisoformat(hom)
    return r2


def fecha_r3(r1) -> date:
    """R3 = R1 − 364 días (la referencia reciente llevada al año anterior)."""
    return parse_date(r1) - timedelta(days=LAG_ANIO)


def unir_alertas(alertas: Iterable[Alert]) -> List[Alert]:
    """
    Una alerta por código (fechas unidas) para no repetir 7 veces lo mismo. `detalle["por_fecha"][f]`
    trae el detalle y el mensaje de cada día; si los mensajes de los días difieren, el mensaje unido es
    genérico (no el del primer día).
    """
    por_codigo: Dict[str, Alert] = {}
    mensajes: Dict[str, set] = {}
    for a in alertas:
        if a.codigo not in por_codigo:
            por_codigo[a.codigo] = Alert(
                nivel=a.nivel, codigo=a.codigo, mensaje=a.mensaje, fechas=[], detalle={"por_fecha": {}}
            )
            mensajes[a.codigo] = set()
        u = por_codigo[a.codigo]
        mensajes[a.codigo].add(a.mensaje)
        for f in a.fechas:
            if f not in u.fechas:
                u.fechas.append(f)
            u.detalle["por_fecha"][f] = {**(a.detalle or {}), "mensaje": a.mensaje}
    for codigo, u in por_codigo.items():
        if len(mensajes[codigo]) > 1:
            u.mensaje = f"{len(u.fechas)} día(s) con esta condición; el mensaje de cada día está en detalle.por_fecha"
    return list(por_codigo.values())


def seleccionar_recientes(
    d,
    curvas_reales: Mapping[str, Any],
    corte,
    festivos: Iterable[str],
    n: int = 1,
    *,
    saltar_puente: bool = False,
    max_semanas: int = MAX_SEMANAS_DEFAULT,
) -> RecientesDia:
    """
    Los `n` días reales más recientes (de más reciente a más antiguo) con mismo
    día de semana y misma clase que `d`, retrocediendo de 7 en 7 desde
    `ancla_mismo_dow(d, corte)` hasta `max_semanas` pasos. `saltados` lista cada
    candidato descartado con su motivo ("festivo", "post_festivo", "puente",
    "sin datos"). Si no hay ninguno de la misma clase: festivo -> homólogo del
    año anterior con datos; si tampoco, el último `normal` con alerta
    `R1_SIN_MISMA_CLASE`. Si ni así hay datos, `fechas` queda vacío.
    """
    if n < 1:
        raise ValueError("n debe ser >= 1")
    d = parse_date(d)
    corte = parse_date(corte)
    festivos_set = festivos if isinstance(festivos, (set, frozenset)) else set(festivos)

    clase = clase_dia(d, festivos_set, puente=saltar_puente)
    ancla = ancla_mismo_dow(d, corte)

    fechas: List[date] = []
    saltados: List[Dict[str, str]] = []
    ultimo_normal: Optional[date] = None
    for i in range(max_semanas):
        cand = ancla - timedelta(days=7 * i)
        c_clase = clase_dia(cand, festivos_set, puente=saltar_puente)
        if not _curva_valida(curvas_reales, cand):
            saltados.append({"fecha": cand.isoformat(), "motivo": "sin datos"})
            continue
        if c_clase == clase:
            fechas.append(cand)
            if len(fechas) >= n:
                break
        else:
            if c_clase == CLASE_NORMAL and ultimo_normal is None:
                ultimo_normal = cand
            saltados.append({"fecha": cand.isoformat(), "motivo": c_clase})

    alertas: List[Alert] = []
    if not fechas:
        homologo = None
        if clase == CLASE_FESTIVO:
            iso = find_homologous_holiday_date(d, d.year - 1, festivos_set)
            if iso and _curva_valida(curvas_reales, date.fromisoformat(iso)):
                homologo = date.fromisoformat(iso)
        if homologo is not None:
            fechas = [homologo]
            saltados.append({"fecha": homologo.isoformat(), "motivo": "homologo"})
        elif ultimo_normal is not None:
            fechas = [ultimo_normal]
            alertas.append(
                Alert(
                    nivel="info",
                    codigo=ALERTA_SIN_MISMA_CLASE,
                    mensaje=(
                        f"No hay un día real de clase '{clase}' en las últimas "
                        f"{max_semanas} semanas; la referencia reciente es el último día normal."
                    ),
                    fechas=[d.isoformat()],
                    detalle={"clase": clase, "referencia": ultimo_normal.isoformat()},
                )
            )
    return RecientesDia(fechas=fechas, clase=clase, saltados=saltados, alertas=alertas)
