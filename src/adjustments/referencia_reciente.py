"""
Referencia reciente por clase de día (R1 de la réplica de Jamer).

R1 de un día objetivo `d` = el día real MÁS RECIENTE (<= corte) con el mismo día
de semana y la MISMA clase de día que `d` (festivo / post_festivo / normal; con
`saltar_puente`, también `puente`). Un objetivo normal salta festivos y
post-festivos; un objetivo festivo o post-festivo usa el último de su clase.

Opt-in (default off, ver `seleccionar_recientes`):
- `festivo_por_tipo`: un festivo fijo o de Semana Santa usa R1 = día normal reciente y el factor
  F = E(H)/E(N_H) del homólogo H (hasta 3 años atrás, mismo tipo de día; si el objetivo no es lunes, H no puede
  ser un festivo lunes de puente) frente a un día normal N_H junto a él.
- `r1_limpia`: un objetivo no festivo salta candidatos atípicos, y los de Semana Santa / 24/dic–6/ene cuando el
  objetivo está fuera de ese periodo.

Funciones puras: no leen disco ni red. `curvas_reales` = {fecha_iso: array(24)}
(ver `data.curvas_reales`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from src.adjustments.data import (
    CLASE_FESTIVO,
    CLASE_NORMAL,
    LAG_ANIO,
    TIPO_FESTIVO_FIJO,
    TIPO_FESTIVO_SEMANA_SANTA,
    ancla_mismo_dow,
    clase_dia,
    energia_curva,
    es_semana_santa,
    tipo_dia,
    tipo_festivo,
)
from src.adjustments.steps import _params
from src.adjustments.types import Alert, parse_date
from src.prediction.holiday_homology import find_homologous_holiday_date

ALERTA_SIN_MISMA_CLASE = "R1_SIN_MISMA_CLASE"
ALERTA_FESTIVO_SIN_HOMOLOGO = "FESTIVO_SIN_HOMOLOGO"
ALERTA_R1_SIN_LIMPIA = "R1_SIN_LIMPIA"
MAX_SEMANAS_DEFAULT = 12

# R1 limpia: un candidato es atípico si |E(c)/mediana(E de los N previos normales del mismo día de semana) − 1|
# supera el umbral (6 %: magnitud física, fijado antes del backtest). Con menos de MIN previos no se juzga.
UMBRAL_ATIPICO = 0.06
N_PREVIOS_ATIPICO = 4
MIN_PREVIOS_ATIPICO = 3
MAX_SEMANAS_PREVIOS_ATIPICO = 8
MOTIVOS_NO_LIMPIO = ("semana_santa", "temporada", "atipico")
# Festivo por tipo: el homólogo H se busca en los ANIOS_HOMOLOGO años anteriores (el más reciente que caiga en el
# mismo tipo de día que el objetivo: laboral / sábado / domingo; si el objetivo no es lunes, H tampoco puede ser un
# festivo lunes de puente) y el día normal N_H junto a él a −7, −14, +7, +14 días (antes primero).
ANIOS_HOMOLOGO = 3
DESPLAZAMIENTOS_NORMAL_HOMOLOGO = (-7, -14, 7, 14)
# Códigos de `detalle.motivo` de FESTIVO_SIN_HOMOLOGO (snake_case; el texto legible solo va en el mensaje).
MOTIVO_SIN_HOMOLOGO = "sin_homologo"
MOTIVO_SIN_HOMOLOGO_MISMO_TIPO = "sin_homologo_mismo_tipo_dia"
MOTIVO_HOMOLOGO_LUNES_PUENTE = "homologo_lunes_puente"
MOTIVO_SIN_NORMAL_HOMOLOGO = "sin_normal_homologo"
MOTIVO_SIN_NORMAL_RECIENTE = "sin_normal_reciente"
TEXTO_MOTIVO_SIN_HOMOLOGO: Dict[str, str] = {
    MOTIVO_SIN_HOMOLOGO: f"homólogo con datos en los últimos {ANIOS_HOMOLOGO} años",
    MOTIVO_SIN_HOMOLOGO_MISMO_TIPO: (
        f"homólogo del mismo tipo de día (laboral/sábado/domingo) en los últimos {ANIOS_HOMOLOGO} años"
    ),
    MOTIVO_HOMOLOGO_LUNES_PUENTE: (
        f"homólogo que no sea festivo lunes de puente en los últimos {ANIOS_HOMOLOGO} años"
    ),
    MOTIVO_SIN_NORMAL_HOMOLOGO: "día normal junto al homólogo",
    MOTIVO_SIN_NORMAL_RECIENTE: "día normal reciente del mismo día de semana",
}
# Descarte de un año en `anios_probados` (None = homólogo usado).
DESCARTE_SIN_FESTIVO = "sin_festivo"
DESCARTE_SIN_DATOS = "sin_datos"
DESCARTE_OTRO_TIPO_DIA = "otro_tipo_dia"
DESCARTE_LUNES_PUENTE = "lunes_puente"

# Params de la selección de R1 que comparten nivel (modo reciente) y forma (fuente reciente_clase); default =
# comportamiento de siempre.
DEFAULTS_R1: Dict[str, Any] = {"festivo_por_tipo": False, "r1_limpia": False, "r1_umbral_atipico": UMBRAL_ATIPICO}


def validar_params_r1(p: Dict[str, Any], paso: str) -> Dict[str, Any]:
    """festivo_por_tipo y r1_limpia: bool; r1_umbral_atipico: número en (0, 0.5]. `p` trae las claves de DEFAULTS_R1."""
    return {
        "festivo_por_tipo": _params.booleano(paso, "festivo_por_tipo", p["festivo_por_tipo"]),
        "r1_limpia": _params.booleano(paso, "r1_limpia", p["r1_limpia"]),
        "r1_umbral_atipico": _params.numero(paso, "r1_umbral_atipico", p["r1_umbral_atipico"],
                                            minimo=0.0, maximo=0.5, minimo_exclusivo=True),
    }


def kwargs_r1(p: Dict[str, Any]) -> Dict[str, Any]:
    """Kwargs de `seleccionar_recientes` desde params validados (las claves que falten → default)."""
    q = {**DEFAULTS_R1, **{k: p[k] for k in DEFAULTS_R1 if k in p}}
    return {"festivo_por_tipo": q["festivo_por_tipo"], "r1_limpia": q["r1_limpia"],
            "umbral_atipico": q["r1_umbral_atipico"]}


@dataclass
class RecientesDia:
    fechas: List[date]
    clase: str
    saltados: List[Dict[str, str]] = field(default_factory=list)
    alertas: List[Alert] = field(default_factory=list)
    # Solo con `festivo_por_tipo` y objetivo festivo (`info_festivo`): tipo_festivo, homologo, normal_homologo,
    # E_homologo, E_normal_homologo, factor (None si el factor no se aplica y `fechas` es la selección de siempre) y
    # anios_probados ([{anio, homologo, descarte}] de cada año en que se buscó el homólogo; descarte None = usado).
    festivo: Optional[Dict[str, Any]] = None

    @property
    def factor_festivo(self) -> Optional[float]:
        """F = E(H)/E(N_H) si se aplica el festivo por tipo (entonces `fechas` son días normales); si no, None."""
        return None if self.festivo is None else self.festivo.get("factor")

    def diag_festivo(self) -> Dict[str, Any]:
        """Claves del festivo por tipo en el diagnóstico por día (definición única para nivel y forma); {} sin él."""
        if self.festivo is None:
            return {}
        return {"tipo_festivo": self.festivo["tipo_festivo"], "homologo": self.festivo["homologo"],
                "normal_homologo": self.festivo["normal_homologo"], "factor_festivo": self.festivo["factor"],
                "homologo_anios_probados": [dict(x) for x in self.festivo["anios_probados"]]}

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "fechas": [f.isoformat() for f in self.fechas],
            "clase": self.clase,
            "saltados": [dict(s) for s in self.saltados],
            "alertas": [a.to_dict() for a in self.alertas],
        }
        if self.festivo is not None:
            out["festivo"] = {**self.festivo, "anios_probados": [dict(x) for x in self.festivo["anios_probados"]]}
        return out


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


def en_temporada(d: date) -> bool:
    """Temporada de fin de año: 24/dic .. 6/ene (inclusive)."""
    return (d.month == 12 and d.day >= 24) or (d.month == 1 and d.day <= 6)


def _energia_normal_referencia(
    x: date, curvas_reales: Mapping[str, Any], festivos: set, puente: bool
) -> Optional[float]:
    """
    E(x) si `x` sirve como día normal de referencia (clase normal, fuera de Semana Santa y de 24/dic–6/ene, curva
    válida con energía > 0); si no, None. La usan la mediana de `_es_atipico` y N_H del festivo por tipo.
    """
    if clase_dia(x, festivos, puente=puente) != CLASE_NORMAL or es_semana_santa(x) or en_temporada(x):
        return None
    e = energia_curva(curvas_reales.get(x.isoformat()))
    return e if e is not None and e > 0 else None


def _es_atipico(c: date, curvas_reales: Mapping[str, Any], festivos: set, puente: bool, umbral: float) -> bool:
    """
    |E(c)/mediana(E de hasta N_PREVIOS_ATIPICO días normales de referencia con el mismo día de semana, antes de c)
    − 1| > umbral (`_energia_normal_referencia`: sin Semana Santa ni 24/dic–6/ene). Los previos se buscan hasta
    MAX_SEMANAS_PREVIOS_ATIPICO semanas atrás; con menos de MIN_PREVIOS_ATIPICO no se juzga (False).
    """
    e_c = energia_curva(curvas_reales.get(c.isoformat()))
    if e_c is None:
        return False
    previos: List[float] = []
    for k in range(1, MAX_SEMANAS_PREVIOS_ATIPICO + 1):
        e = _energia_normal_referencia(c - timedelta(days=7 * k), curvas_reales, festivos, puente)
        if e is not None:
            previos.append(e)
            if len(previos) >= N_PREVIOS_ATIPICO:
                break
    if len(previos) < MIN_PREVIOS_ATIPICO:
        return False
    mediana = float(np.median(previos))
    return mediana > 0 and abs(e_c / mediana - 1.0) > umbral


def _motivo_no_limpio(
    c: date, d: date, curvas_reales: Mapping[str, Any], festivos: set, puente: bool, umbral: float
) -> Optional[str]:
    """
    Motivo por el que `c` no sirve como R1 limpia del objetivo `d` ("semana_santa" / "temporada" / "atipico") o
    None. Semana Santa y 24/dic–6/ene solo descartan si el objetivo `d` está FUERA de ese periodo (un objetivo dentro
    del periodo se parece a sus días); el atípico se juzga siempre.
    """
    if es_semana_santa(c) and not es_semana_santa(d):
        return "semana_santa"
    if en_temporada(c) and not en_temporada(d):
        return "temporada"
    # el atípico se mide contra días normales: solo se juzga a candidatos de clase normal
    if clase_dia(c, festivos, puente=puente) == CLASE_NORMAL and _es_atipico(
        c, curvas_reales, festivos, puente, umbral
    ):
        return "atipico"
    return None


def _buscar(
    d: date,
    ancla: date,
    clase: str,
    curvas_reales: Mapping[str, Any],
    festivos: set,
    n: int,
    puente: bool,
    max_semanas: int,
    limpia: bool,
    umbral: float,
) -> Tuple[List[date], List[Dict[str, str]], Optional[date]]:
    """
    Retrocede de 7 en 7 desde `ancla` (`max_semanas` pasos) → (fechas de clase `clase`, saltados, último normal
    visto entre los de otra clase). Con `limpia`, los de la clase pedida que no son limpios para el objetivo `d` se
    saltan con su motivo (`_motivo_no_limpio`).
    """
    fechas: List[date] = []
    saltados: List[Dict[str, str]] = []
    ultimo_normal: Optional[date] = None
    for i in range(max_semanas):
        cand = ancla - timedelta(days=7 * i)
        c_clase = clase_dia(cand, festivos, puente=puente)
        if not _curva_valida(curvas_reales, cand):
            saltados.append({"fecha": cand.isoformat(), "motivo": "sin datos"})
            continue
        if c_clase == clase:
            motivo = _motivo_no_limpio(cand, d, curvas_reales, festivos, puente, umbral) if limpia else None
            if motivo is not None:
                saltados.append({"fecha": cand.isoformat(), "motivo": motivo})
                continue
            fechas.append(cand)
            if len(fechas) >= n:
                break
        else:
            if c_clase == CLASE_NORMAL and ultimo_normal is None:
                ultimo_normal = cand
            saltados.append({"fecha": cand.isoformat(), "motivo": c_clase})
    return fechas, saltados, ultimo_normal


def _buscar_con_limpieza(
    d: date,
    ancla: date,
    clase: str,
    curvas_reales: Mapping[str, Any],
    festivos: set,
    n: int,
    puente: bool,
    max_semanas: int,
    limpia: bool,
    umbral: float,
) -> Tuple[List[date], List[Dict[str, str]], Optional[date], List[Alert]]:
    """
    `_buscar` con limpieza opcional → (fechas, saltados, último normal, alertas). Si la limpieza deja sin
    candidatos pero sin ella sí los hay, se usa la búsqueda sin limpieza (la de siempre) con alerta info
    `R1_SIN_LIMPIA` (detalle: los candidatos descartados y su motivo).
    """
    fechas, saltados, ultimo_normal = _buscar(
        d, ancla, clase, curvas_reales, festivos, n, puente, max_semanas, limpia, umbral
    )
    if not limpia or fechas:
        return fechas, saltados, ultimo_normal, []
    descartados = [s for s in saltados if s["motivo"] in MOTIVOS_NO_LIMPIO]
    fechas, saltados, ultimo_normal = _buscar(
        d, ancla, clase, curvas_reales, festivos, n, puente, max_semanas, False, umbral
    )
    if not fechas:
        return fechas, saltados, ultimo_normal, []
    alerta = Alert(
        nivel="info",
        codigo=ALERTA_R1_SIN_LIMPIA,
        mensaje=(
            f"Ningún día de clase '{clase}' de las últimas {max_semanas} semanas es una referencia limpia "
            "(Semana Santa, 24/dic–6/ene o atípico); se usa la referencia de siempre."
        ),
        fechas=[d.isoformat()],
        detalle={"clase": clase, "referencia": fechas[0].isoformat(), "descartados": descartados},
    )
    return fechas, saltados, ultimo_normal, [alerta]


def info_festivo(tipo: Optional[str]) -> Dict[str, Any]:
    """Dict `RecientesDia.festivo` vacío (definición ÚNICA de sus claves) para un festivo de tipo `tipo`."""
    return {"tipo_festivo": tipo, "homologo": None, "normal_homologo": None, "E_homologo": None,
            "E_normal_homologo": None, "factor": None, "anios_probados": []}


def _homologo_con_normal(
    d: date, curvas_reales: Mapping[str, Any], corte: date, festivos: set, puente: bool, info: Dict[str, Any]
) -> Optional[str]:
    """
    Completa en `info` (`info_festivo`) el homólogo H y el día normal N_H del factor F = E(H)/E(N_H) → None si hay
    factor, o el código de motivo (MOTIVO_*) por el que no lo hay.
    - H: en los años d.year−1 .. d.year−ANIOS_HOMOLOGO (`find_homologous_holiday_date`), el MÁS RECIENTE <= corte,
      con curva válida de energía > 0 (así su forma siempre es usable), que cae en el mismo tipo de día que `d`
      (laboral lun–vie / sábado / domingo) y, si `d` no es lunes, que no sea lunes (festivo de puente). Cada año va a
      `anios_probados` como {anio, homologo, descarte} (DESCARTE_*; None = el usado). Sin H: `sin_homologo` si ningún
      año tuvo homólogo con datos; `homologo_lunes_puente` si todos los que tenían datos eran lunes de puente; si no,
      `sin_homologo_mismo_tipo_dia`.
    - N_H: mismo día de semana que H a −7, −14, +7, +14 días (el primero <= corte que sea día normal de referencia,
      `_energia_normal_referencia`); si no hay → `sin_normal_homologo`.
    """
    tipo_obj = tipo_dia(d, ())
    h: Optional[date] = None
    for anio in range(d.year - 1, d.year - 1 - ANIOS_HOMOLOGO, -1):
        iso_h = find_homologous_holiday_date(d, anio, festivos)
        probado: Dict[str, Any] = {"anio": anio, "homologo": iso_h or None, "descarte": None}
        info["anios_probados"].append(probado)
        if not iso_h:
            probado["descarte"] = DESCARTE_SIN_FESTIVO
            continue
        cand = date.fromisoformat(iso_h)
        e_h = energia_curva(curvas_reales.get(iso_h)) if cand <= corte else None
        if e_h is None or not e_h > 0:
            probado["descarte"] = DESCARTE_SIN_DATOS
        elif tipo_dia(cand, ()) != tipo_obj:
            probado["descarte"] = DESCARTE_OTRO_TIPO_DIA
        elif cand.weekday() == 0 and d.weekday() != 0:
            probado["descarte"] = DESCARTE_LUNES_PUENTE
        else:
            h = cand
            info.update(homologo=iso_h, E_homologo=e_h)
            break
    if h is None:
        con_datos = [x["descarte"] for x in info["anios_probados"]
                     if x["descarte"] in (DESCARTE_OTRO_TIPO_DIA, DESCARTE_LUNES_PUENTE)]
        if not con_datos:
            return MOTIVO_SIN_HOMOLOGO
        if all(x == DESCARTE_LUNES_PUENTE for x in con_datos):
            return MOTIVO_HOMOLOGO_LUNES_PUENTE
        return MOTIVO_SIN_HOMOLOGO_MISMO_TIPO
    for desp in DESPLAZAMIENTOS_NORMAL_HOMOLOGO:
        x = h + timedelta(days=desp)
        if x > corte:
            continue
        e_x = _energia_normal_referencia(x, curvas_reales, festivos, puente)
        if e_x is not None:
            info.update(normal_homologo=x.isoformat(), E_normal_homologo=e_x, factor=info["E_homologo"] / e_x)
            return None
    return MOTIVO_SIN_NORMAL_HOMOLOGO


def _alerta_sin_homologo(d: date, info: Dict[str, Any], motivo: str) -> Alert:
    """Alerta info FESTIVO_SIN_HOMOLOGO: `detalle.motivo` = código; el texto legible solo en el mensaje."""
    return Alert(
        nivel="info",
        codigo=ALERTA_FESTIVO_SIN_HOMOLOGO,
        mensaje=(
            f"Festivo '{info['tipo_festivo']}' sin {TEXTO_MOTIVO_SIN_HOMOLOGO[motivo]}: se usa la "
            "referencia de siempre (festivo reciente del mismo día de semana u homólogo)."
        ),
        fechas=[d.isoformat()],
        detalle={"tipo_festivo": info["tipo_festivo"], "motivo": motivo, "homologo": info["homologo"],
                 "normal_homologo": info["normal_homologo"],
                 "anios_probados": [dict(x) for x in info["anios_probados"]]},
    )


def seleccionar_recientes(
    d,
    curvas_reales: Mapping[str, Any],
    corte,
    festivos: Iterable[str],
    n: int = 1,
    *,
    saltar_puente: bool = False,
    max_semanas: int = MAX_SEMANAS_DEFAULT,
    festivo_por_tipo: bool = False,
    r1_limpia: bool = False,
    umbral_atipico: float = UMBRAL_ATIPICO,
) -> RecientesDia:
    """
    Los `n` días reales más recientes (de más reciente a más antiguo) con mismo
    día de semana y misma clase que `d`, retrocediendo de 7 en 7 desde
    `ancla_mismo_dow(d, corte)` hasta `max_semanas` pasos. `saltados` lista cada
    candidato descartado con su motivo ("festivo", "post_festivo", "puente",
    "sin datos"). Si no hay ninguno de la misma clase: festivo -> homólogo del
    año anterior con datos; si tampoco, el último `normal` con alerta
    `R1_SIN_MISMA_CLASE`. Si ni así hay datos, `fechas` queda vacío.

    Opt-in (con ambos en False el resultado es el de siempre):
    - `festivo_por_tipo`: objetivo festivo de tipo `fijo` o `semana_santa` (`data.tipo_festivo`) →
      `fechas` = días NORMALES recientes del mismo día de semana (la búsqueda de un objetivo normal, con la
      limpieza si `r1_limpia`) y `festivo["factor"]` = E(H)/E(N_H) (`_homologo_con_normal`: H del mismo tipo de
      día laboral/sábado/domingo hasta 3 años atrás; si el objetivo no es lunes, H no es lunes de puente). Quien
      usa la selección aplica E = E(R1)·(1+g)·F (nivel) y la forma de H (forma). Sin H, N_H o R1 normal →
      selección de siempre + alerta info `FESTIVO_SIN_HOMOLOGO` (detalle.motivo = código MOTIVO_*). Festivo `lunes`
      → selección de siempre. Con el flag, `festivo` trae el tipo para todo objetivo festivo.
    - `r1_limpia`: objetivo no festivo → se saltan candidatos atípicos ("atipico", `_es_atipico` con
      `umbral_atipico`) y, si el objetivo está fuera de ese periodo, los de Semana Santa ("semana_santa") o
      24/dic–6/ene ("temporada"). Si no queda ninguno limpio → selección de siempre + alerta info
      `R1_SIN_LIMPIA`. El día objetivo en sí no cambia de clase.
    """
    if n < 1:
        raise ValueError("n debe ser >= 1")
    d = parse_date(d)
    corte = parse_date(corte)
    festivos_set = festivos if isinstance(festivos, (set, frozenset)) else set(festivos)

    clase = clase_dia(d, festivos_set, puente=saltar_puente)
    ancla = ancla_mismo_dow(d, corte)
    alertas: List[Alert] = []

    festivo: Optional[Dict[str, Any]] = None
    if festivo_por_tipo and clase == CLASE_FESTIVO:
        festivo = info_festivo(tipo_festivo(d, festivos_set))
        if festivo["tipo_festivo"] in (TIPO_FESTIVO_FIJO, TIPO_FESTIVO_SEMANA_SANTA):
            motivo = _homologo_con_normal(d, curvas_reales, corte, festivos_set, saltar_puente, festivo)
            if motivo is not None:
                alertas.append(_alerta_sin_homologo(d, festivo, motivo))
            else:
                fechas, saltados, _, alertas_limpia = _buscar_con_limpieza(
                    d, ancla, CLASE_NORMAL, curvas_reales, festivos_set, n, saltar_puente, max_semanas,
                    r1_limpia, umbral_atipico,
                )
                if fechas:
                    return RecientesDia(fechas=fechas, clase=clase, saltados=saltados,
                                        alertas=alertas_limpia, festivo=festivo)
                festivo["factor"] = None
                alertas.append(_alerta_sin_homologo(d, festivo, MOTIVO_SIN_NORMAL_RECIENTE))

    fechas, saltados, ultimo_normal, alertas_limpia = _buscar_con_limpieza(
        d, ancla, clase, curvas_reales, festivos_set, n, saltar_puente, max_semanas,
        r1_limpia and clase != CLASE_FESTIVO, umbral_atipico,
    )
    alertas.extend(alertas_limpia)

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
    return RecientesDia(fechas=fechas, clase=clase, saltados=saltados, alertas=alertas, festivo=festivo)
