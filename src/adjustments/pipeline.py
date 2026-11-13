"""
Pipeline de ajustes encadenados: base → referencia → nivel → forma → control.

Cada paso puede estar:
- `aplicado`: se ejecuta la función del paso con sus params.
- `omitido`: pasa las curvas del paso anterior sin cambios (no aplica a `base`).
- `editado`: el paso se calcula como `aplicado` y luego las curvas manuales
  del analista (validadas con `validar_curvas_editadas`: 24 periodos,
  finitos, no negativos, fechas permitidas) reemplazan esos días; los demás
  días conservan el ajuste calculado. La curva calculada de cada día editado
  queda en `result.curvas_calculadas`. Solo para base, nivel y forma
  (referencia se fuerza con el param `semana_ref_inicio`; control no produce
  curvas).

Los params de cada paso se validan SIEMPRE (`<paso>.validar_params`, ValueError
→ 422 en el router) antes de ejecutarlo, también en `rerun_from`.

Los pasos se identifican por NOMBRE (`paso`), no por posición: los records
cargados deben traer exactamente los nombres de `STEP_ORDER` con su orden
coherente (`orden_de`), si no se lanza ValueError.

Al cambiar un paso (`rerun_from`) se recalculan todos los posteriores con
su estado y params guardados.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import pandas as pd

from src.adjustments import data
from src.adjustments.steps import _params, base, control, forma, nivel, referencia
from src.adjustments.types import (
    Alert,
    Curves,
    RunContext,
    StepRecord,
    StepResult,
    StepState,
    parse_date,
    validar_curvas_editadas,
)

STEP_ORDER = ["base", "referencia", "nivel", "forma", "control"]  # orden 0..4
BaseProvider = Callable[[str, date, int], Curves]  # (ucp, end_date=fecha_corte, n_days) -> curvas raw

_MODULOS = {
    "base": base,
    "referencia": referencia,
    "nivel": nivel,
    "forma": forma,
    "control": control,
}
_PASOS_QUE_MODIFICAN = ("base", "nivel", "forma")  # producen curvas; son también los editables


def orden_de(paso: str) -> int:
    """Posición del paso en STEP_ORDER; ValueError si no existe."""
    try:
        return STEP_ORDER.index(paso)
    except ValueError:
        raise ValueError(f"Paso desconocido {paso!r}; válidos: {STEP_ORDER}") from None


def _nombre_paso(clave: Union[int, str]) -> str:
    """Acepta el nombre del paso o su orden (int) y devuelve el nombre."""
    if isinstance(clave, str) and not clave.isdigit():
        orden_de(clave)
        return clave
    i = int(clave)
    if not 0 <= i < len(STEP_ORDER):
        raise ValueError(f"orden fuera de rango: {i}; pasos: {STEP_ORDER}")
    return STEP_ORDER[i]


def _validar_record(record: StepRecord, paso: str) -> None:
    if record.paso != paso:
        raise ValueError(f"Se esperaba el paso {paso!r}, llegó {record.paso!r}")
    if int(record.orden) != orden_de(paso):
        raise ValueError(
            f"El paso {paso!r} tiene orden {record.orden}, se esperaba {orden_de(paso)} "
            "(los records no coinciden con STEP_ORDER)"
        )


def _records_por_paso(records: List[StepRecord]) -> Dict[str, StepRecord]:
    """Indexa por nombre; ValueError si faltan, sobran o se repiten pasos, o el orden no cuadra."""
    nombres = [r.paso for r in records]
    if sorted(nombres) != sorted(STEP_ORDER):
        raise ValueError(
            f"Los records no coinciden con STEP_ORDER por nombre: llegaron {nombres}, se esperan {STEP_ORDER}"
        )
    for r in records:
        _validar_record(r, r.paso)
    return {r.paso: r for r in records}


# ---------------------------------------------------------------------------
# Contexto
# ---------------------------------------------------------------------------

def build_context(
    ucp: str,
    semana_inicio: date,
    fecha_emision: Optional[date] = None,
    historico: Optional[pd.DataFrame] = None,
    festivos: Optional[List[str]] = None,
) -> RunContext:
    """
    Construye el contexto de una corrida.

    - `semana_inicio` debe ser lunes, entre 1990-01-01 y 2100-12-31 (acota la
      aritmética de fechas de los pasos; fuera → ValueError).
    - `fecha_emision` default = min(hoy, semana_inicio).
    - fecha_corte = último día con 24 periodos válidos <= min(fecha_emision, semana_inicio) − 1
      (nunca dentro de la semana objetivo).
    - Si `historico`/`festivos` son None se cargan de data/raw/{ucp}/datos.csv y
      del calendario oficial de Colombia.
    """
    semana_inicio = parse_date(semana_inicio)
    if not _params.FECHA_MIN <= semana_inicio <= _params.FECHA_MAX:
        raise ValueError(
            f"semana_inicio debe estar entre {_params.FECHA_MIN} y {_params.FECHA_MAX}, "
            f"llegó {semana_inicio.isoformat()}"
        )
    if semana_inicio.weekday() != 0:
        raise ValueError(f"semana_inicio debe ser lunes, llegó {semana_inicio.isoformat()}")
    fecha_emision = parse_date(fecha_emision) or min(date.today(), semana_inicio)

    hist = (
        data.cargar_historico(ucp)
        if historico is None
        else data.normalizar_historico(historico, ucp=ucp)
    )
    limite = min(fecha_emision, semana_inicio) - timedelta(days=1)
    fecha_corte = data.calcular_fecha_corte(hist, limite)
    hist = data.recortar(hist, fecha_corte)

    if festivos is None:
        primer_anio = int(hist["fecha"].dt.year.min()) if not hist.empty else semana_inicio.year
        anios = range(min(primer_anio, semana_inicio.year - 7), semana_inicio.year + 2)
        festivos = data.festivos_colombia(anios)

    return RunContext(
        ucp=ucp,
        semana_inicio=semana_inicio,
        fecha_emision=fecha_emision,
        fecha_corte=fecha_corte,
        historico=hist,
        festivos=sorted({str(f)[:10] for f in festivos}),
    )


# ---------------------------------------------------------------------------
# Ejecución de un paso
# ---------------------------------------------------------------------------

def _copiar(curves: Curves) -> Curves:
    return {f: list(v) for f, v in curves.items()}


def _curvas_editadas(record: StepRecord) -> Curves:
    """Días editados por el analista (en records ya ejecutados, solo `fechas_editadas`)."""
    curves = record.result.curves or {}
    fechas = record.result.diagnostics.get("fechas_editadas")
    if fechas is None:
        return curves
    return {f: curves[f] for f in fechas if f in curves}


def _aplicar_edicion(
    ctx: RunContext, paso: str, calculado: StepResult, editadas: Curves
) -> StepResult:
    """
    Valida las curvas del analista y las superpone a las calculadas por el paso.
    La curva calculada de cada día editado se conserva en `curvas_calculadas`.
    """
    if paso == "base":
        inicio, _, n_dias = base.rango_base(ctx)
        permitidas = [data.iso(inicio + timedelta(days=i)) for i in range(n_dias)]
    else:
        permitidas = [data.iso(d) for d in data.dias_semana(ctx.semana_inicio)]
    if not editadas:
        raise ValueError(f"El paso {paso!r} está editado pero no trae curvas")
    validar_curvas_editadas(editadas, permitidas)

    curves = _copiar(calculado.curves)
    calculadas: Curves = {}
    for fecha, valores in editadas.items():
        f = str(fecha)[:10]
        if f in calculado.curves:
            calculadas[f] = list(calculado.curves[f])
        curves[f] = [float(v) for v in valores]
    fechas = sorted(str(f)[:10] for f in editadas)

    diag: Dict[str, Any] = {**calculado.diagnostics, "editado": True, "fechas_editadas": fechas}
    if paso == "base":
        diag["energia"] = data.energia_curvas(curves)  # refleja las curvas finales (editadas)
    return StepResult(curves=curves, diagnostics=diag, alerts=list(calculado.alerts),
                      curvas_calculadas=calculadas)


def _calcular(
    ctx: RunContext,
    paso: str,
    p: Dict[str, Any],
    prev: Curves,
    base_provider: Optional[BaseProvider],
    previos: List[StepRecord],
) -> StepResult:
    if paso == "base":
        return base.step(ctx, prev, p, base_provider=base_provider)
    if paso == "control":
        origen, energia_nivel = _energia_nivel(previos)
        result = control.step(
            ctx, prev, p,
            energia_esperada=_energia_esperada(previos),
            energia_nivel=energia_nivel,
            nivel_dias=_nivel_dias(previos),
        )
        result.diagnostics["semana_inicio"] = data.iso(ctx.semana_inicio)
        result.diagnostics["origen_energia_nivel"] = origen
        return result
    return _MODULOS[paso].step(ctx, prev, p)


def _ejecutar(
    ctx: RunContext,
    paso: str,
    estado: StepState,
    params: Optional[Dict[str, Any]],
    prev: Curves,
    base_provider: Optional[BaseProvider],
    previos: List[StepRecord],
    editadas: Optional[Curves] = None,
) -> StepRecord:
    orden = orden_de(paso)
    try:
        estado = StepState(estado)
    except ValueError:
        raise ValueError(f"Estado inválido {estado!r} para {paso!r}; válidos: "
                         f"{[e.value for e in StepState]}") from None
    p = _MODULOS[paso].validar_params(params)

    if estado == StepState.OMITIDO:
        if paso == "base":
            raise ValueError("El paso base no se puede omitir")
        result = StepResult(curves=_copiar(prev), diagnostics={"omitido": True})
    elif estado == StepState.EDITADO:
        if paso not in _PASOS_QUE_MODIFICAN:
            raise ValueError(f"El paso {paso!r} no admite curvas editadas")
        # El paso se calcula igual que si estuviera aplicado y luego los días
        # editados reemplazan al cálculo (los demás días conservan el ajuste).
        calculado = _calcular(ctx, paso, p, prev, base_provider, previos)
        result = _aplicar_edicion(ctx, paso, calculado, editadas or {})
    else:
        result = _calcular(ctx, paso, p, prev, base_provider, previos)

    if paso == "referencia":
        _aplicar_referencia(ctx, result.diagnostics)

    return StepRecord(orden=orden, paso=paso, estado=estado, params=p, result=result)


def _aplicar_referencia(ctx: RunContext, diagnostics: Optional[Dict[str, Any]]) -> None:
    """La semana de referencia y los años excluidos viven en diagnostics del paso referencia."""
    diagnostics = diagnostics or {}
    ctx.semana_ref_inicio = parse_date(diagnostics.get("semana_ref_inicio"))
    if "anios_excluidos" in diagnostics:
        ctx.extras["anios_excluidos"] = list(diagnostics["anios_excluidos"])
    else:
        ctx.extras.pop("anios_excluidos", None)


def _nivel_dias(previos: List[StepRecord]) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    E*(d) y E_base(d) por día que calculó el paso nivel (para la tendencia
    esperada del control). None si nivel se omitió: el control recalcula E*.
    """
    por_paso = {r.paso: r for r in previos}
    record = por_paso.get("nivel")
    if record is None or record.estado == StepState.OMITIDO:
        return None
    return record.result.diagnostics.get("dias")


def _energia_calculada(record: StepRecord) -> Dict[str, float]:
    """Energía diaria del cálculo del paso (para días editados, la curva pre-edición)."""
    curves = {**record.result.curves, **(record.result.curvas_calculadas or {})}
    return data.energia_curvas(curves)


def _energia_nivel(previos: List[StepRecord]) -> tuple:
    """
    (origen, energía diaria) de referencia para el enclavamiento del control:
    la que calculó el paso nivel (pre-edición); si nivel se omitió, la del paso
    base (sus curvas finales: nivel omitido = identidad sobre la base).
    """
    por_paso = {r.paso: r for r in previos}
    nivel_rec = por_paso.get("nivel")
    if nivel_rec is not None and nivel_rec.estado != StepState.OMITIDO:
        return "nivel", _energia_calculada(nivel_rec)
    if "base" in por_paso:
        return "base", data.energia_curvas(por_paso["base"].result.curves)
    return None, None


def _energia_esperada(previos: List[StepRecord]) -> Optional[Dict[str, float]]:
    """
    Energía diaria final del paso nivel para reconciliar la forma. Los días
    editados a mano en forma no se reconcilian (los vigila el enclavamiento).
    """
    por_paso = {r.paso: r for r in previos}
    if "nivel" not in por_paso or "forma" not in por_paso:
        return None
    energia = data.energia_curvas(por_paso["nivel"].result.curves)
    forma_rec = por_paso["forma"]
    if forma_rec.estado == StepState.EDITADO:
        for f in forma_rec.result.diagnostics.get("fechas_editadas") or []:
            energia.pop(f, None)
    return energia


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def run_pipeline(
    ctx: RunContext,
    base_provider: BaseProvider,
    params: Optional[Dict[str, Dict[str, Any]]] = None,  # params por nombre de paso
    overrides: Optional[Mapping[Union[int, str], StepRecord]] = None,  # por orden o nombre
) -> List[StepRecord]:
    """
    Corre los 5 pasos. `overrides` (omitidos/editados/params) se indexan por
    orden o por nombre de paso; el `paso` de cada override debe coincidir con
    la clave (no se reinterpreta por posición).
    """
    params = params or {}
    if not isinstance(params, dict):
        raise ValueError("params debe ser un objeto {paso: {...}}")
    desconocidos = sorted(set(params) - set(STEP_ORDER))
    if desconocidos:
        raise ValueError(f"pasos desconocidos en params: {desconocidos}; válidos: {STEP_ORDER}")
    por_paso: Dict[str, StepRecord] = {}
    for clave, ov in (overrides or {}).items():
        paso = _nombre_paso(clave)
        if ov.paso != paso:
            raise ValueError(f"El override de {paso!r} trae el paso {ov.paso!r}")
        por_paso[paso] = ov

    _aplicar_referencia(ctx, None)
    records: List[StepRecord] = []
    prev: Curves = {}
    for paso in STEP_ORDER:
        ov = por_paso.get(paso)
        if ov is not None:
            estado = StepState(ov.estado)
            p = {**(params.get(paso) or {}), **(ov.params or {})}
            editadas = _curvas_editadas(ov) if estado == StepState.EDITADO else None
        else:
            estado, p, editadas = StepState.APLICADO, params.get(paso), None
        record = _ejecutar(ctx, paso, estado, p, prev, base_provider, records, editadas)
        records.append(record)
        prev = record.result.curves
    return records


def rerun_from(
    ctx: RunContext,
    records: List[StepRecord],
    orden: Union[int, str],
    base_provider: BaseProvider,
    nuevo: StepRecord,
) -> List[StepRecord]:
    """
    Reemplaza el paso `orden` (int o nombre del paso) por `nuevo` y recalcula
    los posteriores. `nuevo.paso` debe ser ese paso y `records` deben traer
    exactamente los pasos de STEP_ORDER (por nombre); si no, ValueError.
    """
    paso_nuevo = _nombre_paso(orden)
    if nuevo.paso != paso_nuevo:
        raise ValueError(f"El paso {orden_de(paso_nuevo)} es {paso_nuevo!r}, llegó {nuevo.paso!r}")
    por_paso = _records_por_paso(records)
    i0 = orden_de(paso_nuevo)

    nuevos: List[StepRecord] = [por_paso[p] for p in STEP_ORDER[:i0]]
    _aplicar_referencia(ctx, None)
    if "referencia" in STEP_ORDER[:i0]:
        _aplicar_referencia(ctx, por_paso["referencia"].result.diagnostics)
    prev: Curves = nuevos[-1].result.curves if nuevos else {}

    for paso in STEP_ORDER[i0:]:
        fuente = nuevo if paso == paso_nuevo else por_paso[paso]
        estado = StepState(fuente.estado)
        editadas = _curvas_editadas(fuente) if estado == StepState.EDITADO else None
        record = _ejecutar(ctx, paso, estado, fuente.params, prev, base_provider, nuevos, editadas)
        nuevos.append(record)
        prev = record.result.curves
    return nuevos


def final_curves(records: List[StepRecord]) -> Curves:
    """Curvas del último paso que modifica (base/nivel/forma), incluida la brecha."""
    por_paso = {r.paso: r for r in records}
    for paso in reversed(_PASOS_QUE_MODIFICAN):
        if paso in por_paso:
            return _copiar(por_paso[paso].result.curves)
    return {}


def approval_blockers(records: List[StepRecord]) -> List[Alert]:
    """
    Alertas de nivel "bloqueo": las emitidas por los pasos más una validación
    estructural de las curvas finales de la semana objetivo (por si el control
    se omitió).

    Alcance: SOLO la semana objetivo (lunes–domingo). Los días de brecha
    (fecha_corte+1 .. lunes−1) son estimación del modelo, no se aprueban ni se
    reportan, así que nunca bloquean; si traen valores inválidos el control
    emite el warning `BRECHA_INVALIDA` (no bloqueo).
    """
    if not records:
        return [Alert("bloqueo", "SIN_PASOS", "La corrida no tiene pasos")]
    bloqueos = [a for r in records for a in r.result.alerts if a.nivel == "bloqueo"]

    semana = next(
        (r.result.diagnostics.get("semana_inicio") for r in records
         if r.result.diagnostics.get("semana_inicio")),
        None,
    )
    if semana:
        fechas = [data.iso(d) for d in data.dias_semana(parse_date(semana))]
        vistos = {(a.codigo, tuple(a.fechas)) for a in bloqueos}
        for alerta in control.validar_curvas(final_curves(records), fechas):
            if (alerta.codigo, tuple(alerta.fechas)) not in vistos:
                bloqueos.append(alerta)
    return bloqueos
