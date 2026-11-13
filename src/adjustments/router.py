"""Endpoints del servicio de ajustes por pasos (`/adjustments`).

- `POST  /adjustments/preview`                corre el pipeline sin persistir (funciona sin base).
- `POST  /adjustments/run`                    corre y persiste una corrida (borrador, version 1).
- `GET   /adjustments/aprobada?ucp=&semana_inicio=`  corrida aprobada vigente (404 si no hay).
- `GET   /adjustments/automatico?ucp=`        configuración R2 por UCP × tipo de día (solo lectura, sin base).
- `GET   /adjustments/aprendizaje?ucp=&desde=&hasta=&limite=&min_casos=`
                                              diagnóstico de ediciones recurrentes (solo lectura, requiere base).
- `GET   /adjustments/{id}`                   corrida con curvas, diagnósticos y alertas por paso + eventos.
- `GET   /adjustments/{id}/versiones`         historial de versiones (sin curvas).
- `GET   /adjustments/{id}/versiones/{v}`     snapshot completo de una versión.
- `PATCH /adjustments/{id}/steps/{paso}`      omite/edita/re-parametriza un paso (por nombre) y recalcula
                                              los siguientes; exige `version_esperada` (409 si no coincide).
- `POST  /adjustments/{id}/approve`           aprueba (exige `version_esperada`; 422 si hay bloqueos). La
                                              aprobada anterior de la misma (ucp, semana) pasa a `reemplazada`.
                                              `excepciones_justificadas` debe referirse a warnings vigentes (422).
- `POST  /adjustments/analogos`               días análogos a una fecha (R3; solo lectura, sin base ni modelo).

`meta.params_iniciales` = `params` del request original (/run); los params vigentes de cada paso están en
`steps[i].params`. `meta.extras.base_ref` = `base_ref` del request (enlace al pronóstico emitido en Sphere).
`meta.extras.aplicar_automatico` viaja al pipeline (diagnósticos R2 por día) y `meta.extras.eventos` se reusa
al recalcular (PATCH) sin volver a consultar OpenAI; las corridas viejas sin ese campo simplemente no emiten
alertas de eventos. `meta.extras.modos_paso` (`{paso: "automatico"|"sugerido"}`) viaja SIEMPRE: R2 es una
clasificación de revisión para el front, no una instrucción de ejecución (el backend aplica siempre los
pasos). Ni `festivos` ni `eventos` se exponen en las respuestas.

Origen del pronóstico base:
- Sin `base_curvas`: se refrescan los datos (hook `set_data_refresher`), se calcula el último dato real
  y se llama al modelo con `end_date = fecha_corte = último dato real`. Si la semana objetivo ya tiene
  datos reales (corte forzado anterior al último real) → 422. `fecha_emision` no se admite (422).
- Con `base_curvas` ({fecha: [24]}, p. ej. el pronóstico ya emitido que manda el backend intermedio):
  no se llama al modelo ni al refresco; deben cubrir fecha_corte+1..domingo objetivo.

Validación de UCP (antes de cualquier refresco, red, lock o lectura): formato `^[A-Za-z0-9_-]{1,64}$`
(422) y existencia de `data/raw/{ucp}/datos.csv` dentro de `data.RAW_DATA_DIR` (404).

Autenticación servicio-a-servicio: si `AJUSTES_SERVICE_TOKEN` está definido, todos los endpoints exigen la
cabecera `X-Ajustes-Token` con ese valor (401 si falta o no coincide). Si no está definido se permite y se
loguea un warning una sola vez.

Inyección (main.py / tests):
- `set_model_checker(fn)`: fn(ucp) -> bool; si devuelve False → 409 (no se entrena dentro del request).
  main.py: `check_model_exists` + `check_hourly_disaggregation_trained`.
- `set_base_provider(fn)`: fn(ucp, end_date, n_days) -> Curves | (Curves, info) | awaitable de eso.
  `info` puede traer "modelo_version". main.py lo cablea a `run_predict_flow` fuera del event loop.
- `set_data_refresher(fn)`: fn(ucp) (sync, corre en threadpool, o async). main.py: `full_update_csv`.
- `set_history_loader(fn)`: fn(ucp) -> DataFrame del histórico (default `data.cargar_historico`).
- `set_festivos_loader(fn)`: fn(ucp, inicio, fin) -> lista ISO (default API de festivos por UCP; si
  falla, librería `holidays` + alerta info `FESTIVOS_FUENTE_ALTERNA`).
- `set_clima_loader(fn)`: fn(ucp) -> {fecha: clima diario} (default `analogos.cargar_clima`; solo /analogos).
- `get_store_dep`: dependencia FastAPI; los tests la sobreescriben con un InMemoryStore.
"""

from __future__ import annotations

import asyncio
import hmac
import inspect
import logging
import os
import re
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import pandas as pd
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from src.adjustments import analogos, aprendizaje, automatico, data, eventos
from src.adjustments import pipeline as pl
from src.adjustments.steps import base as base_step
from src.adjustments.steps._params import FECHA_MAX, FECHA_MIN
from src.adjustments.store import (
    ApprovalBlockedError,
    RunApprovedError,
    RunNotFoundError,
    Store,
    VersionConflictError,
    get_store,
)
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

logger = logging.getLogger(__name__)

ENV_SERVICE_TOKEN = "AJUSTES_SERVICE_TOKEN"
HEADER_SERVICE_TOKEN = "X-Ajustes-Token"
_aviso_sin_token = False

# Literales (status.HTTP_422_UNPROCESSABLE_CONTENT no existe en Starlette < 0.48)
HTTP_201 = 201
HTTP_401 = 401
HTTP_404 = 404
HTTP_409 = 409
HTTP_422 = 422
HTTP_501 = 501
HTTP_502 = 502
HTTP_503 = 503

N_PERIODOS = 24
UCP_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
_UCP_RE = re.compile(UCP_PATTERN)
ERRORES_ENTRADA = (ValueError, OverflowError)  # errores del pipeline/fechas que se responden como 422
MAX_N_DAYS = 90  # límite de PredictRequest.n_days
EVENTOS_TIMEOUT = 5  # segundos; si OpenAI tarda más, la corrida sigue sin alertas de eventos
FUENTE_BASE_MODELO = "modelo"
FUENTE_BASE_EXTERNA = "externa"
FUENTE_FESTIVOS_API = "api"
FUENTE_FESTIVOS_LIBRERIA = "libreria"

BaseProviderResult = Union[Curves, Tuple[Curves, Dict[str, Any]]]
AsyncBaseProvider = Callable[[str, date, int], Union[BaseProviderResult, Awaitable[BaseProviderResult]]]
DataRefresher = Callable[[str], Any]
ModelChecker = Callable[[str], bool]
HistoryLoader = Callable[[str], pd.DataFrame]
FestivosLoader = Callable[[str, date, date], Iterable[str]]


def _festivos_api(ucp: str, inicio: date, fin: date) -> List[str]:
    """Festivos por UCP desde la misma API que usa el modelo (`FestivosAPIClient`)."""
    from src.prediction.festivos_api import FestivosAPIClient

    return FestivosAPIClient().get_festivos(inicio.isoformat(), fin.isoformat(), ucp)


_base_provider: Optional[AsyncBaseProvider] = None
_data_refresher: Optional[DataRefresher] = None
_model_checker: Optional[ModelChecker] = None
_history_loader: HistoryLoader = data.cargar_historico
_festivos_loader: FestivosLoader = _festivos_api
_ucp_locks: Dict[str, asyncio.Lock] = {}


def set_base_provider(fn: Optional[AsyncBaseProvider]) -> Optional[AsyncBaseProvider]:
    """Fija el proveedor del pronóstico base; devuelve el anterior (para restaurarlo)."""
    global _base_provider
    anterior, _base_provider = _base_provider, fn
    return anterior


def set_data_refresher(fn: Optional[DataRefresher]) -> Optional[DataRefresher]:
    global _data_refresher
    anterior, _data_refresher = _data_refresher, fn
    return anterior


def set_model_checker(fn: Optional[ModelChecker]) -> Optional[ModelChecker]:
    """fn(ucp) -> bool: True si la UCP tiene modelo entrenado. None = no se verifica."""
    global _model_checker
    anterior, _model_checker = _model_checker, fn
    return anterior


def set_history_loader(fn: Optional[HistoryLoader]) -> HistoryLoader:
    """None restaura el default (`data.cargar_historico`)."""
    global _history_loader
    anterior, _history_loader = _history_loader, (fn or data.cargar_historico)
    return anterior


def set_festivos_loader(fn: Optional[FestivosLoader]) -> FestivosLoader:
    """None restaura el default (API de festivos por UCP)."""
    global _festivos_loader
    anterior, _festivos_loader = _festivos_loader, (fn or _festivos_api)
    return anterior


def get_store_dep() -> Optional[Store]:
    return get_store()


def verificar_token(token: Optional[str] = Header(None, alias=HEADER_SERVICE_TOKEN)) -> None:
    """Auth servicio-a-servicio opcional (el backend intermedio manda `X-Ajustes-Token`)."""
    global _aviso_sin_token
    esperado = os.getenv(ENV_SERVICE_TOKEN)
    if not esperado:
        if not _aviso_sin_token:
            _aviso_sin_token = True
            logger.warning("[ajustes] %s no definido: /adjustments acepta peticiones sin autenticación",
                           ENV_SERVICE_TOKEN)
        return
    if token is None or not hmac.compare_digest(token.encode("utf-8"), esperado.encode("utf-8")):
        raise HTTPException(HTTP_401, detail=f"Falta o no coincide la cabecera {HEADER_SERVICE_TOKEN}")


router = APIRouter(prefix="/adjustments", tags=["ajustes"], dependencies=[Depends(verificar_token)])


def _validar_ucp(ucp: str) -> str:
    """Formato (422) y existencia de data/raw/{ucp}/datos.csv dentro de RAW_DATA_DIR (404). Sin efectos."""
    if not isinstance(ucp, str) or not _UCP_RE.fullmatch(ucp):
        raise HTTPException(HTTP_422, detail=f"ucp inválida: debe cumplir {UCP_PATTERN}")
    raiz = Path(data.RAW_DATA_DIR).resolve()
    ruta = (raiz / ucp / "datos.csv").resolve()
    if not ruta.is_relative_to(raiz) or not ruta.is_file():
        raise HTTPException(HTTP_404, detail=f"UCP desconocida: '{ucp}'")
    return ucp


async def _verificar_modelo(ucp: str) -> None:
    """409 si la UCP no tiene modelo entrenado (el request no entrena; eso es /retrain)."""
    if _model_checker is None:
        return
    try:
        ok = bool(await run_in_threadpool(_model_checker, ucp))
    except Exception as e:  # noqa: BLE001
        logger.exception("[ajustes] 502: falló la verificación de modelo de %s", ucp)
        raise HTTPException(HTTP_502, detail={"mensaje": f"No se pudo verificar el modelo de {ucp}",
                                              "causa": _corto(e)})
    if not ok:
        raise HTTPException(HTTP_409, detail="La UCP no tiene modelo entrenado; use /retrain")


def _ucp_lock(ucp: str) -> asyncio.Lock:
    """Serializa refresco + modelo por UCP (ambos reescriben archivos de data/). Solo UCP validadas."""
    return _ucp_locks.setdefault(_validar_ucp(ucp), asyncio.Lock())


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

CurvasIn = Dict[str, List[float]]
CurvasOut = Dict[str, List[Optional[float]]]  # None = valor no finito (bloquea la aprobación)


class RunRequest(BaseModel):
    ucp: str = Field(..., pattern=UCP_PATTERN, description="UCP (ej. Antioquia, Atlantico); debe existir "
                                                            "data/raw/{ucp}/datos.csv")
    semana_inicio: date = Field(..., description="Lunes de la semana objetivo (YYYY-MM-DD)")
    fecha_emision: Optional[date] = Field(
        None, description="Solo con base_curvas: fecha de emisión del pronóstico externo (corte = último "
                          "dato real anterior). Con el modelo, el corte es siempre el último dato real."
    )
    base_curvas: Optional[CurvasIn] = Field(
        None, description="Pronóstico base ya emitido {fecha: [P1..P24]} que cubra corte+1..domingo; "
                          "si viene, no se refrescan datos ni se llama al modelo"
    )
    modelo_version: Optional[str] = Field(None, description="Versión del modelo de base_curvas")
    params: Optional[Dict[str, Dict[str, Any]]] = Field(
        None, description='Parámetros por nombre de paso, ej. {"nivel": {"alpha": 0.5}}'
    )
    usuario: Optional[str] = Field(None, description="Usuario que crea la corrida (solo /run)")
    base_ref: Optional[str] = Field(
        None, max_length=200, description="Referencia al pronóstico emitido en Sphere (se guarda en meta.extras)"
    )
    aplicar_automatico: bool = Field(
        False, description="Añade a nivel/forma/especiales el estado R2 por día en diagnostics.automatico; "
                           "no cambia las curvas"
    )

    @field_validator("semana_inicio")
    @classmethod
    def _lunes(cls, v: date) -> date:
        if v.weekday() != 0:
            raise ValueError("semana_inicio debe ser lunes")
        return v

    @field_validator("semana_inicio", "fecha_emision")
    @classmethod
    def _fecha_en_rango(cls, v: Optional[date]) -> Optional[date]:
        # antes de cualquier aritmética de fechas (rango de festivos, contexto): evita OverflowError
        if v is not None and not FECHA_MIN <= v <= FECHA_MAX:
            raise ValueError(f"fecha fuera de rango ({FECHA_MIN.isoformat()}..{FECHA_MAX.isoformat()})")
        return v

    @field_validator("params")
    @classmethod
    def _pasos_conocidos(cls, v: Optional[Dict[str, Dict[str, Any]]]) -> Optional[Dict[str, Dict[str, Any]]]:
        if v:
            desconocidos = sorted(set(v) - set(pl.STEP_ORDER))
            if desconocidos:
                raise ValueError(f"pasos desconocidos en params: {desconocidos}; válidos: {pl.STEP_ORDER}")
        return v


class StepPatchRequest(BaseModel):
    estado: Literal["aplicado", "omitido", "editado"]
    version_esperada: int = Field(..., ge=1, description="Versión de la corrida sobre la que se edita")
    params: Optional[Dict[str, Any]] = Field(None, description="Reemplaza los params del paso")
    curvas: Optional[CurvasIn] = Field(
        None, description="Solo con estado=editado: {fecha: [P1..P24]}; días ya editados se conservan"
    )
    motivo: Optional[str] = None
    usuario: Optional[str] = None


class ApproveRequest(BaseModel):
    usuario: str = Field(..., min_length=1)
    version_esperada: int = Field(..., ge=1, description="Versión de la corrida que se aprueba")
    motivo: Optional[str] = None
    excepciones_justificadas: Optional[List[str]] = Field(
        None, description="Códigos de alertas warning vigentes en la versión que el analista justifica (422 si "
                          "alguno no está presente). No es obligatorio justificar todas; queda en el evento aprobar "
                          "junto con warnings_vigentes y warnings_sin_justificar"
    )


class AlertOut(BaseModel):
    nivel: str
    codigo: str
    mensaje: str
    fechas: List[str] = Field(default_factory=list)
    detalle: Dict[str, Any] = Field(default_factory=dict)


class StepOut(BaseModel):
    orden: int
    paso: str
    estado: str
    params: Dict[str, Any] = Field(default_factory=dict)
    curves: CurvasOut = Field(default_factory=dict)
    curvas_calculadas: CurvasOut = Field(
        default_factory=dict, description="Paso editado: curva calculada (pre-edición) de los días editados"
    )
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    alerts: List[AlertOut] = Field(default_factory=list)


class RunOut(BaseModel):
    id: Optional[str] = Field(None, description="None en preview")
    persistido: bool
    version: Optional[int] = Field(None, description="Versión actual (None en preview)")
    meta: Dict[str, Any]
    steps: List[StepOut]
    final_curves: CurvasOut
    bloqueos: List[AlertOut] = Field(default_factory=list, description="Alertas que impiden aprobar")
    eventos: List[Dict[str, Any]] = Field(default_factory=list)


class VersionInfo(BaseModel):
    version: int
    creado_por: Optional[str] = None
    motivo: Optional[str] = None
    created_at: Optional[str] = None


class VersionOut(VersionInfo):
    steps: List[StepOut]
    final_curves: CurvasOut


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_store(store: Optional[Store]) -> Store:
    if store is None:
        logger.warning("[ajustes] 503: persistencia no configurada (falta AJUSTES_DATABASE_URL)")
        raise HTTPException(HTTP_503, detail="Persistencia de ajustes no configurada (falta AJUSTES_DATABASE_URL)")
    return store


def _alert_dict(a: Any) -> Dict[str, Any]:
    return a.to_dict() if hasattr(a, "to_dict") else dict(a)


def _corto(e: BaseException) -> str:
    msg = e.detail if isinstance(e, HTTPException) else str(e)
    msg = str(msg).strip().splitlines()[0] if str(msg).strip() else type(e).__name__
    return msg[:300]


async def _call(fn: Callable[..., Any], *args: Any) -> Any:
    """Ejecuta fn: si es async la espera; si es sync la corre en threadpool."""
    if inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(getattr(fn, "__call__", None)):
        return await fn(*args)
    result = await run_in_threadpool(fn, *args)
    if inspect.isawaitable(result):
        result = await result
    return result


async def _refresh_data(ucp: str) -> None:
    if _data_refresher is None:
        return
    try:
        await _call(_data_refresher, ucp)
    except Exception as e:  # noqa: BLE001 — red / fuente externa
        logger.exception("[ajustes] 502: falló el refresco de datos de %s", ucp)
        raise HTTPException(HTTP_502, detail={"mensaje": f"No se pudieron refrescar los datos de {ucp}",
                                              "causa": _corto(e)})


async def _load_history(ucp: str) -> pd.DataFrame:
    try:
        hist = await run_in_threadpool(_history_loader, ucp)
        return data.normalizar_historico(hist, ucp=ucp)
    except FileNotFoundError as e:
        raise HTTPException(HTTP_404, detail=f"Sin histórico para UCP '{ucp}': {e}")
    except ERRORES_ENTRADA as e:
        raise HTTPException(HTTP_422, detail=str(e))


def _ultimo_dato_real(hist: pd.DataFrame, ucp: str) -> date:
    validas = hist[data.filas_validas(hist)]
    if validas.empty:
        raise HTTPException(HTTP_404, detail=f"Sin días con 24 periodos válidos para UCP '{ucp}'")
    return validas["fecha"].max().date()


async def _load_festivos(ucp: str, hist: pd.DataFrame, semana_inicio: date) -> Tuple[List[str], str]:
    """Festivos de la UCP (API, como el modelo); si falla, librería oficial + fuente registrada."""
    primer_anio = int(hist["fecha"].dt.year.min()) if not hist.empty else semana_inicio.year
    inicio = date(min(primer_anio, semana_inicio.year - 7), 1, 1)
    fin = date(semana_inicio.year + 1, 12, 31)
    try:
        festivos = sorted({str(f)[:10] for f in await run_in_threadpool(_festivos_loader, ucp, inicio, fin)})
        if not festivos:
            raise ValueError("la API de festivos no devolvió fechas")
        return festivos, FUENTE_FESTIVOS_API
    except Exception as e:  # noqa: BLE001 — red / API externa
        logger.warning("[ajustes] festivos de %s por API fallaron (%s); se usa la librería", ucp, _corto(e))
        anios = range(inicio.year, fin.year + 1)
        return data.festivos_colombia(anios), FUENTE_FESTIVOS_LIBRERIA


def _build_context(ucp: str, semana_inicio: date, fecha_emision: Optional[date],
                   hist: pd.DataFrame, festivos: List[str]) -> RunContext:
    try:
        return pl.build_context(ucp, semana_inicio, fecha_emision=fecha_emision,
                                historico=hist, festivos=festivos)
    except ERRORES_ENTRADA as e:
        raise HTTPException(HTTP_422, detail=str(e))


def _n_days(ctx: RunContext) -> int:
    n = base_step.rango_base(ctx)[2]
    if n < 1:
        raise HTTPException(HTTP_422, detail=f"La semana objetivo ya está observada "
                                             f"(fecha_corte={ctx.fecha_corte.isoformat()})")
    if n > MAX_N_DAYS:
        raise HTTPException(HTTP_422, detail=f"Brecha demasiado grande: {n} días desde fecha_corte "
                                             f"(máx {MAX_N_DAYS})")
    return n


async def _fetch_base(ctx: RunContext) -> Tuple[Curves, Dict[str, Any]]:
    """Llama al base_provider (sync o async) con end_date = fecha_corte y normaliza a (curves, info)."""
    if _base_provider is None:
        logger.warning("[ajustes] 503: base_provider no configurado")
        raise HTTPException(HTTP_503, detail="base_provider no configurado")
    n = _n_days(ctx)
    try:
        result = _base_provider(ctx.ucp, ctx.fecha_corte, n)
        if inspect.isawaitable(result):
            result = await result
    except Exception as e:  # noqa: BLE001 — HTTPException de run_predict_flow o error del modelo
        logger.exception("[ajustes] 502: falló el modelo base (%s, corte %s, %d días)", ctx.ucp, ctx.fecha_corte, n)
        raise HTTPException(HTTP_502, detail={"mensaje": "No se pudo obtener el pronóstico base del modelo",
                                              "causa": _corto(e)})
    info: Dict[str, Any] = {}
    if isinstance(result, tuple):
        result, info = result[0], dict(result[1] or {})
    return {str(k)[:10]: list(v) for k, v in (result or {}).items()}, info


def _normalizar_fechas(curvas: CurvasIn, campo: str) -> Curves:
    out: Curves = {}
    invalidas = []
    for fecha, valores in curvas.items():
        try:
            out[date.fromisoformat(str(fecha)).isoformat()] = [float(x) for x in valores]
        except ERRORES_ENTRADA:
            invalidas.append(fecha)
    if invalidas:
        raise HTTPException(HTTP_422, detail={"mensaje": f"{campo}: fechas inválidas (YYYY-MM-DD)",
                                              "fechas": invalidas})
    return out


def _fechas_rango_base(ctx: RunContext) -> List[str]:
    inicio, _, n = base_step.rango_base(ctx)
    return [(inicio + timedelta(days=k)).isoformat() for k in range(n)]


def _validar_base_curvas(curvas: CurvasIn, ctx: RunContext) -> Curves:
    """base_curvas debe cubrir fecha_corte+1..domingo con 24 periodos (los no finitos los bloquea control)."""
    norm = _normalizar_fechas(curvas, "base_curvas")
    requeridas = _fechas_rango_base(ctx)
    if not requeridas:
        raise HTTPException(HTTP_422, detail="La semana objetivo ya está observada")
    faltan = [f for f in requeridas if f not in norm]
    if faltan:
        raise HTTPException(HTTP_422, detail={
            "mensaje": f"base_curvas debe cubrir {requeridas[0]}..{requeridas[-1]} "
                       f"(fecha_corte={ctx.fecha_corte.isoformat()})",
            "faltan": faltan,
        })
    mal = [f for f in requeridas if len(norm[f]) != N_PERIODOS]
    if mal:
        raise HTTPException(HTTP_422, detail={"mensaje": f"base_curvas: se esperan {N_PERIODOS} periodos",
                                              "fechas": mal})
    return {f: norm[f] for f in requeridas}


def _const_provider(curves: Curves) -> pl.BaseProvider:
    def provider(*_args: Any, **_kwargs: Any) -> Curves:
        return {k: list(v) for k, v in curves.items()}
    return provider


def _anotar_festivos(records: List[StepRecord], fuente: Optional[str]) -> None:
    """Alerta info en el paso base cuando los festivos no vinieron de la API por UCP."""
    if fuente in (None, FUENTE_FESTIVOS_API):
        return
    for rec in records:
        if rec.paso == "base":
            if not any(a.codigo == "FESTIVOS_FUENTE_ALTERNA" for a in rec.result.alerts):
                rec.result.alerts.append(Alert(
                    nivel="info", codigo="FESTIVOS_FUENTE_ALTERNA",
                    mensaje="La API de festivos por UCP no respondió; se usó el calendario oficial de "
                            "Colombia (librería), que puede diferir de los festivos del modelo",
                    detalle={"fuente_festivos": fuente},
                ))
            return


def _steps_out(records: List[StepRecord]) -> List[StepOut]:
    steps = []
    for rec in records:
        d = rec.to_dict()
        r = d.get("result") or {}
        steps.append(StepOut(
            orden=d["orden"], paso=d["paso"], estado=d["estado"], params=d.get("params") or {},
            curves=r.get("curves") or {}, curvas_calculadas=r.get("curvas_calculadas") or {},
            diagnostics=r.get("diagnostics") or {},
            alerts=[AlertOut(**a) for a in (r.get("alerts") or [])],
        ))
    return steps


def _final_out(records: List[StepRecord]) -> CurvasOut:
    return StepResult(curves=pl.final_curves(records)).to_dict()["curves"]


def _run_out(records: List[StepRecord], meta: Dict[str, Any], run_id: Optional[str],
             eventos: Optional[List[Dict[str, Any]]] = None) -> RunOut:
    meta_out = {k: v for k, v in meta.items() if k != "extras"}
    extras = meta.get("extras") or {}
    extras_out = {k: v for k, v in extras.items()
                  if k not in ("festivos", "eventos")}  # listas/dicts largos: no se exponen
    # Contrato con el front: `modos_paso` siempre presente (también en corridas viejas).
    if not isinstance(extras_out.get("modos_paso"), dict):
        extras_out["modos_paso"] = automatico.modos_paso(meta.get("ucp"))
    meta_out["extras"] = extras_out
    return RunOut(
        id=run_id,
        persistido=run_id is not None,
        version=meta.get("version"),
        meta=meta_out,
        steps=_steps_out(records),
        final_curves=_final_out(records),
        bloqueos=[AlertOut(**_alert_dict(a)) for a in pl.approval_blockers(records)],
        eventos=eventos or [],
    )


def _run_pipeline(ctx: RunContext, base_curves: Curves,
                  params: Optional[Dict[str, Dict[str, Any]]]) -> List[StepRecord]:
    return pl.run_pipeline(ctx, _const_provider(base_curves), params or None)


async def _compute(req: RunRequest) -> Tuple[List[StepRecord], Dict[str, Any]]:
    _validar_ucp(req.ucp)  # antes de cualquier lectura, red o lock
    externa = req.base_curvas is not None
    if not externa and req.fecha_emision is not None:
        raise HTTPException(HTTP_422, detail="fecha_emision solo se admite junto con base_curvas: con el "
                                             "modelo el corte es siempre el último dato real disponible")
    if not externa and req.modelo_version is not None:
        raise HTTPException(HTTP_422, detail="modelo_version solo se admite junto con base_curvas")

    if externa:
        hist = await _load_history(req.ucp)
        festivos, fuente_festivos = await _load_festivos(req.ucp, hist, req.semana_inicio)
        ctx = _build_context(req.ucp, req.semana_inicio, req.fecha_emision, hist, festivos)
        base_curves = _validar_base_curvas(req.base_curvas, ctx)
        modelo_version = req.modelo_version
        ultimo = _ultimo_dato_real(hist, req.ucp)
    else:
        await _verificar_modelo(req.ucp)  # 409 sin modelo: no entrenar dentro del request
        async with _ucp_lock(req.ucp):
            await _refresh_data(req.ucp)  # primero refrescar: el corte sale de los datos nuevos
            hist = await _load_history(req.ucp)
            ultimo = _ultimo_dato_real(hist, req.ucp)
            if req.semana_inicio <= ultimo:
                raise HTTPException(HTTP_422, detail=(
                    f"La semana {req.semana_inicio.isoformat()} ya tiene datos reales (último dato real "
                    f"{ultimo.isoformat()}); el modelo solo emite con corte en el último dato real. "
                    f"Para semanas pasadas envíe base_curvas (y fecha_emision)."))
            festivos, fuente_festivos = await _load_festivos(req.ucp, hist, req.semana_inicio)
            # emisión = día siguiente al último dato real → fecha_corte = último dato real
            ctx = _build_context(req.ucp, req.semana_inicio, ultimo + timedelta(days=1), hist, festivos)
            if ctx.fecha_corte != ultimo:  # defensa: nunca llamar al modelo con un corte pasado
                raise HTTPException(HTTP_422, detail=f"fecha_corte {ctx.fecha_corte} distinta del último dato "
                                                     f"real {ultimo}")
            base_curves, info = await _fetch_base(ctx)
        modelo_version = info.get("modelo_version")

    ctx.extras["modelo_version"] = modelo_version
    ctx.extras["fuente_festivos"] = fuente_festivos
    ctx.extras["aplicar_automatico"] = req.aplicar_automatico
    ctx.extras["modos_paso"] = automatico.modos_paso(ctx.ucp)
    try:
        ctx.extras["eventos"] = await asyncio.wait_for(
            eventos.eventos_de_semana(ctx.ucp, ctx.semana_inicio), timeout=EVENTOS_TIMEOUT
        ) or {}
    except Exception as e:  # noqa: BLE001 — red/OpenAI: degradación silenciosa, sin alertas de eventos
        logger.warning("[ajustes] eventos de %s no disponibles (%s); la corrida sigue sin ellos",
                       req.ucp, _corto(e))
        ctx.extras["eventos"] = {}
    try:
        records = await run_in_threadpool(_run_pipeline, ctx, base_curves, req.params)
    except ERRORES_ENTRADA as e:
        raise HTTPException(HTTP_422, detail=str(e))
    _anotar_festivos(records, fuente_festivos)
    meta = {
        "ucp": ctx.ucp,
        "semana_inicio": ctx.semana_inicio.isoformat(),
        "fecha_corte": ctx.fecha_corte.isoformat(),
        "fecha_emision": ctx.fecha_emision.isoformat(),
        "modelo_version": modelo_version,
        "params_iniciales": req.params or {},  # request original; los params vigentes están en cada paso
        "extras": {
            "fuente_base": FUENTE_BASE_EXTERNA if externa else FUENTE_BASE_MODELO,
            "fuente_festivos": fuente_festivos,
            "ultimo_dato_real": ultimo.isoformat(),
            "festivos": festivos,  # se reusan al recalcular (PATCH) para que la corrida sea reproducible
            "eventos": ctx.extras["eventos"],  # dict fecha→evento; se reusa en PATCH y no se expone en meta
            "aplicar_automatico": req.aplicar_automatico,
            # R2 para el front: qué revisiones puede automatizar (no altera curvas)
            "modos_paso": automatico.modos_paso(ctx.ucp),
        },
        "creado_por": req.usuario,
    }
    if req.base_ref is not None:
        meta["extras"]["base_ref"] = req.base_ref
    return records, meta


def _get_run_or_404(store: Store, run_id: str) -> Dict[str, Any]:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(HTTP_404, detail=f"Corrida {run_id} no existe")
    return run


def _check_editable(meta: Dict[str, Any], version_esperada: int) -> None:
    if meta.get("estado") != "borrador":
        raise HTTPException(HTTP_409, detail=f"La corrida está {meta.get('estado')}; no se puede modificar")
    if meta.get("version") != version_esperada:
        raise HTTPException(HTTP_409, detail={"mensaje": "La corrida cambió desde que se leyó",
                                              "version_esperada": version_esperada,
                                              "version_actual": meta.get("version")})


async def _context_from_meta(meta: Dict[str, Any], hist: Optional[pd.DataFrame] = None) -> RunContext:
    """Reconstruye el contexto de una corrida guardada, fijando la fecha_corte y los festivos originales."""
    ucp = meta["ucp"]
    semana = parse_date(meta["semana_inicio"])
    extras = meta.get("extras") or {}
    if hist is None:
        hist = await _load_history(ucp)
    festivos = extras.get("festivos")
    fuente = extras.get("fuente_festivos")
    if not festivos:  # corridas sin festivos guardados
        festivos, fuente = await _load_festivos(ucp, hist, semana)
    ctx = _build_context(ucp, semana, parse_date(meta.get("fecha_emision")), hist, festivos)
    corte = parse_date(meta["fecha_corte"])
    if ctx.fecha_corte != corte:
        h = ctx.historico
        if isinstance(h, pd.DataFrame) and "fecha" in h.columns:
            h = h[pd.to_datetime(h["fecha"]).dt.date <= corte].copy()
        ctx = replace(ctx, fecha_corte=corte, historico=h)
    ctx.extras["modelo_version"] = meta.get("modelo_version")
    ctx.extras["fuente_festivos"] = fuente
    ctx.extras["aplicar_automatico"] = bool(extras.get("aplicar_automatico"))
    # Corridas viejas sin `modos_paso`: se recalcula (barato, sin store ni datos).
    modos = extras.get("modos_paso")
    ctx.extras["modos_paso"] = dict(modos) if isinstance(modos, dict) else automatico.modos_paso(ucp)
    eventos_extras = extras.get("eventos")
    # Solo las corridas nuevas traen meta.extras.eventos. En las viejas no se
    # vuelve a consultar OpenAI (un PATCH no debe cambiar las alertas de eventos).
    ctx.extras["eventos"] = eventos_extras if isinstance(eventos_extras, dict) else {}
    return ctx


class JustificacionInvalidaError(Exception):
    """excepciones_justificadas con códigos que no son warnings vigentes de la versión aprobada."""

    def __init__(self, invalidos: List[str], vigentes: List[str]) -> None:
        super().__init__(f"códigos no vigentes: {invalidos}")
        self.invalidos = invalidos
        self.vigentes = vigentes


def _warnings_vigentes(records: List[StepRecord]) -> List[Dict[str, Any]]:
    """Alertas de nivel warning presentes en los records (paso, código, fechas)."""
    out = []
    for rec in records:
        for a in rec.result.alerts:
            d = _alert_dict(a)
            if d.get("nivel") == "warning":
                out.append({"paso": rec.paso, "codigo": d.get("codigo"), "fechas": list(d.get("fechas") or [])})
    return out


def _approval_check(justificadas: List[str], detalle: Dict[str, Any]) -> Callable[[List[StepRecord]], List[Any]]:
    """Check que el store corre bajo lock: bloqueos; luego valida las justificaciones contra los warnings de
    esa versión y completa `detalle` (el store inserta el evento `aprobar` después del check)."""
    def check(records: List[StepRecord]) -> List[Any]:
        bloqueos = pl.approval_blockers(records)
        if bloqueos:
            return bloqueos
        vigentes = _warnings_vigentes(records)
        codigos = sorted({w["codigo"] for w in vigentes})
        invalidos = [c for c in justificadas if c not in codigos]
        if invalidos:
            raise JustificacionInvalidaError(invalidos, codigos)
        detalle["warnings_vigentes"] = vigentes
        detalle["warnings_sin_justificar"] = [c for c in codigos if c not in justificadas]
        return []
    return check


def _conflict(e: VersionConflictError) -> HTTPException:
    return HTTPException(HTTP_409, detail={"mensaje": "La corrida cambió desde que se leyó (otra edición o "
                                                      "aprobación concurrente)",
                                           "version_esperada": e.esperada,
                                           "version_actual": e.actual if e.actual >= 0 else None})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/preview", response_model=RunOut)
async def preview(req: RunRequest) -> RunOut:
    """Corre el pipeline completo sin persistir (no requiere base de datos)."""
    records, meta = await _compute(req)
    return _run_out(records, {**meta, "estado": "preview"}, None)


@router.post("/run", response_model=RunOut, status_code=HTTP_201)
async def run(req: RunRequest, store: Optional[Store] = Depends(get_store_dep)) -> RunOut:
    """Corre el pipeline y persiste la corrida como borrador (version 1)."""
    store = _require_store(store)
    records, meta = await _compute(req)
    evento = {
        "usuario": req.usuario, "accion": "crear",
        "detalle": {"params": req.params or {}, "fuente_base": meta["extras"]["fuente_base"],
                    "modelo_version": meta["modelo_version"], "base_ref": req.base_ref},
    }
    run_id = await run_in_threadpool(store.create_run, meta, records, evento)  # misma transacción
    logger.info("[ajustes] corrida %s creada: %s semana %s corte %s (base %s, festivos %s) por %s",
                run_id, meta["ucp"], meta["semana_inicio"], meta["fecha_corte"],
                meta["extras"]["fuente_base"], meta["extras"]["fuente_festivos"], req.usuario)
    saved = await run_in_threadpool(_get_run_or_404, store, run_id)
    return _run_out(saved["records"], saved["meta"], run_id, saved.get("eventos"))


@router.get("/aprobada", response_model=RunOut)
async def get_aprobada(ucp: str = Query(..., pattern=UCP_PATTERN), semana_inicio: date = Query(...),
                       store: Optional[Store] = Depends(get_store_dep)) -> RunOut:
    """Corrida aprobada vigente para (ucp, semana_inicio); 404 si no hay."""
    store = _require_store(store)
    saved = await run_in_threadpool(store.get_approved, ucp, semana_inicio)
    if saved is None:
        raise HTTPException(HTTP_404, detail=f"No hay corrida aprobada para {ucp} semana {semana_inicio.isoformat()}")
    return _run_out(saved["records"], saved["meta"], saved["meta"]["id"], saved.get("eventos"))


@router.get("/automatico")
async def get_automatico(ucp: str = Query(..., pattern=UCP_PATTERN)) -> Dict[str, Any]:
    """Configuración R2 (pasos automáticos por tipo de día) de una UCP; solo lectura, sin base ni modelo."""
    _validar_ucp(ucp)
    return automatico.configuracion(ucp)


@router.get("/aprendizaje")
async def get_aprendizaje(
    ucp: Optional[str] = Query(None, pattern=UCP_PATTERN),
    desde: Optional[date] = Query(None, description="Lunes mínimo de semana_inicio (YYYY-MM-DD)"),
    hasta: Optional[date] = Query(None, description="Lunes máximo de semana_inicio (YYYY-MM-DD)"),
    limite: int = Query(200, ge=1, le=500, description="Máximo de corridas a analizar"),
    min_casos: int = Query(3, ge=1, le=50, description="Días editados distintos para marcar `recurrente`"),
    store: Optional[Store] = Depends(get_store_dep),
) -> Dict[str, Any]:
    """Diagnóstico de solo lectura de las ediciones recurrentes (calculada vs editada)."""
    store = _require_store(store)
    if not hasattr(store, "list_runs"):
        raise HTTPException(HTTP_501, detail="El store no soporta list_runs; no se puede calcular el aprendizaje")
    if desde is not None and hasta is not None and desde > hasta:
        raise HTTPException(HTTP_422, detail="desde no puede ser posterior a hasta")
    if ucp is not None:
        _validar_ucp(ucp)
    runs = await run_in_threadpool(store.list_runs, ucp, desde, hasta, limite)
    resumen = await run_in_threadpool(aprendizaje.resumir, runs, min_casos=min_casos)
    return await run_in_threadpool(aprendizaje.a_dict, resumen)


@router.get("/{run_id}", response_model=RunOut)
async def get_run(run_id: str, store: Optional[Store] = Depends(get_store_dep)) -> RunOut:
    store = _require_store(store)
    saved = await run_in_threadpool(_get_run_or_404, store, run_id)
    return _run_out(saved["records"], saved["meta"], run_id, saved.get("eventos"))


@router.get("/{run_id}/versiones", response_model=List[VersionInfo])
async def list_versiones(run_id: str, store: Optional[Store] = Depends(get_store_dep)) -> List[VersionInfo]:
    store = _require_store(store)
    versiones = await run_in_threadpool(store.list_versions, run_id)
    if versiones is None:
        raise HTTPException(HTTP_404, detail=f"Corrida {run_id} no existe")
    return [VersionInfo(**v) for v in versiones]


@router.get("/{run_id}/versiones/{version}", response_model=VersionOut)
async def get_version(run_id: str, version: int, store: Optional[Store] = Depends(get_store_dep)) -> VersionOut:
    store = _require_store(store)
    ver = await run_in_threadpool(store.get_version, run_id, version)
    if ver is None:
        raise HTTPException(HTTP_404, detail=f"Corrida {run_id} sin versión {version}")
    records = ver["records"]
    return VersionOut(version=ver["version"], creado_por=ver.get("creado_por"), motivo=ver.get("motivo"),
                      created_at=ver.get("created_at"), steps=_steps_out(records),
                      final_curves=_final_out(records))


@router.patch("/{run_id}/steps/{paso}", response_model=RunOut)
async def patch_step(run_id: str, paso: str, body: StepPatchRequest,
                     store: Optional[Store] = Depends(get_store_dep)) -> RunOut:
    """Cambia estado/params/curvas del paso `paso` (nombre) y recalcula los posteriores."""
    store = _require_store(store)
    if paso not in pl.STEP_ORDER:
        raise HTTPException(HTTP_422, detail=f"paso {paso!r} inválido; pasos: {pl.STEP_ORDER}")
    orden = pl.orden_de(paso)
    saved = await run_in_threadpool(_get_run_or_404, store, run_id)
    meta = saved["meta"]
    _check_editable(meta, body.version_esperada)
    records: List[StepRecord] = saved["records"]
    by_paso = {r.paso: r for r in records}
    if paso not in by_paso:
        raise HTTPException(HTTP_422, detail=f"La corrida no tiene el paso {paso!r}")
    actual = by_paso[paso]
    if paso == "base" and body.estado == StepState.OMITIDO.value:
        raise HTTPException(HTTP_422, detail="El paso base no se puede omitir")
    if body.estado == StepState.EDITADO.value and not body.curvas:
        raise HTTPException(HTTP_422, detail="estado=editado requiere curvas")
    if body.estado != StepState.EDITADO.value and body.curvas:
        raise HTTPException(HTTP_422, detail="curvas solo se aceptan con estado=editado")

    extras = meta.get("extras") or {}
    reaplicar_base = paso == "base" and body.estado == StepState.APLICADO.value
    meta_updates: Dict[str, Any] = {}
    if reaplicar_base:
        if extras.get("fuente_base") == FUENTE_BASE_EXTERNA:
            raise HTTPException(HTTP_422, detail="La base de esta corrida vino en base_curvas; para cambiarla "
                                                 "cree una corrida nueva")
        corte = parse_date(meta["fecha_corte"])
        _validar_ucp(meta["ucp"])
        await _verificar_modelo(meta["ucp"])
        async with _ucp_lock(meta["ucp"]):
            await _refresh_data(meta["ucp"])
            hist = await _load_history(meta["ucp"])
            ultimo = _ultimo_dato_real(hist, meta["ucp"])
            if ultimo != corte:
                raise HTTPException(HTTP_422, detail=(
                    f"Hay datos reales posteriores al corte de la corrida (último {ultimo.isoformat()}, "
                    f"corte {corte.isoformat()}); el modelo no se ejecuta con cortes pasados. Cree una corrida nueva."))
            ctx = await _context_from_meta(meta, hist)
            base_curves, info = await _fetch_base(ctx)
        nueva_version_modelo = info.get("modelo_version")
        ctx.extras["modelo_version"] = nueva_version_modelo
        meta_updates["modelo_version"] = nueva_version_modelo
    else:
        ctx = await _context_from_meta(meta)
        # La salida original del modelo: días ya editados del paso base conservan su curva calculada,
        # así una nueva edición de base deja en curvas_calculadas lo que dio el modelo (no la edición previa).
        base_rec = by_paso.get("base")
        base_curves = ({**base_rec.result.curves, **base_rec.result.curvas_calculadas} if base_rec else {})

    if body.estado == StepState.EDITADO.value:
        editadas = _normalizar_fechas(body.curvas, "curvas")
        permitidas = _fechas_rango_base(ctx) if paso == "base" else [
            d.isoformat() for d in data.dias_semana(ctx.semana_inicio)]
        try:
            validar_curvas_editadas(editadas, permitidas)
        except ERRORES_ENTRADA as e:
            raise HTTPException(HTTP_422, detail=str(e))
        # Se acumulan con las ediciones anteriores del mismo paso (diagnostics.fechas_editadas).
        previas: Curves = {}
        if actual.estado == StepState.EDITADO:
            fechas_previas = actual.result.diagnostics.get("fechas_editadas") or []
            previas = {f: list(actual.result.curves[f]) for f in fechas_previas if f in actual.result.curves}
        curvas = {**previas, **editadas}
        diag_nuevo: Dict[str, Any] = {"fechas_editadas": sorted(curvas)}
    else:
        curvas = {k: list(v) for k, v in actual.result.curves.items()}
        diag_nuevo = {}

    nuevo = StepRecord(
        orden=orden,
        paso=paso,
        estado=StepState(body.estado),
        params=dict(body.params) if body.params is not None else dict(actual.params),
        result=StepResult(curves=curvas, diagnostics=diag_nuevo, alerts=[]),
    )
    try:
        new_records = await run_in_threadpool(pl.rerun_from, ctx, records, orden, _const_provider(base_curves), nuevo)
    except ERRORES_ENTRADA as e:
        raise HTTPException(HTTP_422, detail=str(e))
    _anotar_festivos(new_records, ctx.extras.get("fuente_festivos"))

    detalle: Dict[str, Any] = {
        "estado_anterior": StepState(actual.estado).value,
        "estado": body.estado,
        "params": nuevo.params,
        "fechas_editadas": sorted(body.curvas) if body.curvas else [],
    }
    if reaplicar_base:
        detalle["modelo_version_anterior"] = meta.get("modelo_version")
        detalle["modelo_version"] = meta_updates["modelo_version"]
    evento = {
        "usuario": body.usuario,
        "accion": {"aplicado": "aplicar_paso", "omitido": "omitir_paso", "editado": "editar_paso"}[body.estado],
        "paso": paso,
        "motivo": body.motivo,
        "detalle": detalle,
    }
    try:
        version = await run_in_threadpool(store.replace_records, run_id, new_records, evento,
                                          body.version_esperada, meta_updates or None)
    except VersionConflictError as e:
        raise _conflict(e)
    except RunApprovedError:
        raise HTTPException(HTTP_409, detail="La corrida ya está aprobada; no se puede modificar")
    except RunNotFoundError:
        raise HTTPException(HTTP_404, detail=f"Corrida {run_id} no existe")
    logger.info("[ajustes] corrida %s: %s paso %s → v%d por %s", run_id, evento["accion"], paso, version,
                body.usuario)
    saved = await run_in_threadpool(_get_run_or_404, store, run_id)
    return _run_out(saved["records"], saved["meta"], run_id, saved.get("eventos"))


@router.post("/{run_id}/approve", response_model=RunOut)
async def approve(run_id: str, body: ApproveRequest, store: Optional[Store] = Depends(get_store_dep)) -> RunOut:
    """Aprueba la corrida (versión esperada). 422 con bloqueos; 409 si cambió o ya no es borrador."""
    store = _require_store(store)
    justificadas = list(dict.fromkeys(body.excepciones_justificadas or []))  # sin duplicados, en orden
    detalle: Dict[str, Any] = {"excepciones_justificadas": justificadas}
    try:
        version = await run_in_threadpool(
            store.approve, run_id, body.usuario, body.motivo, body.version_esperada,
            _approval_check(justificadas, detalle), detalle,
        )
    except JustificacionInvalidaError as e:
        raise HTTPException(HTTP_422, detail={
            "mensaje": "excepciones_justificadas incluye códigos que no son warnings vigentes de esta versión",
            "codigos_invalidos": e.invalidos, "warnings_vigentes": e.vigentes,
        })
    except ApprovalBlockedError as e:
        bloqueos = [_alert_dict(a) for a in e.bloqueos]
        await run_in_threadpool(store.add_event, run_id, {
            "usuario": body.usuario, "accion": "aprobacion_rechazada", "motivo": body.motivo,
            "detalle": {"bloqueos": bloqueos},
        })
        logger.info("[ajustes] corrida %s: aprobación rechazada (%d bloqueos)", run_id, len(bloqueos))
        raise HTTPException(HTTP_422, detail={"mensaje": "La corrida tiene bloqueos de aprobación",
                                              "bloqueos": bloqueos})
    except VersionConflictError as e:
        raise _conflict(e)
    except RunApprovedError:
        raise HTTPException(HTTP_409, detail="La corrida ya no es borrador (aprobada o reemplazada)")
    except RunNotFoundError:
        raise HTTPException(HTTP_404, detail=f"Corrida {run_id} no existe")
    logger.info("[ajustes] corrida %s aprobada (v%d) por %s", run_id, version, body.usuario)
    saved = await run_in_threadpool(_get_run_or_404, store, run_id)
    return _run_out(saved["records"], saved["meta"], run_id, saved.get("eventos"))


# ---------------------------------------------------------------------------
# Días análogos (R3): POST /adjustments/analogos
# ---------------------------------------------------------------------------

ClimaLoader = Callable[[str], Dict[str, Dict[str, float]]]
_clima_loader: ClimaLoader = analogos.cargar_clima


def set_clima_loader(fn: Optional[ClimaLoader]) -> ClimaLoader:
    """None restaura el default (`analogos.cargar_clima`: data/raw/{ucp}/clima_new.csv)."""
    global _clima_loader
    anterior, _clima_loader = _clima_loader, (fn or analogos.cargar_clima)
    return anterior


class AnalogosRequest(BaseModel):
    ucp: str = Field(..., pattern=UCP_PATTERN, description="UCP; debe existir data/raw/{ucp}/datos.csv")
    fecha: date = Field(..., description="Día objetivo (YYYY-MM-DD)")
    fecha_corte: Optional[date] = Field(
        None, description="Solo días <= corte; ausente o posterior al último dato real → último dato real"
    )
    filtros: Optional[Dict[str, Any]] = Field(
        None, description="Filtros de búsqueda (ver src/adjustments/analogos.py y fase2b/PLAN.md §2.1)"
    )

    @field_validator("fecha", "fecha_corte")
    @classmethod
    def _fecha_en_rango(cls, v: Optional[date]) -> Optional[date]:
        if v is not None and not FECHA_MIN <= v <= FECHA_MAX:
            raise ValueError(f"fecha fuera de rango ({FECHA_MIN.isoformat()}..{FECHA_MAX.isoformat()})")
        return v


@router.post("/analogos")
async def post_analogos(req: AnalogosRequest) -> Dict[str, Any]:
    """Días análogos a `fecha` (solo lectura: no refresca datos, no llama al modelo, no usa base)."""
    ucp = _validar_ucp(req.ucp)
    try:
        analogos.validar_filtros(req.filtros)
    except ERRORES_ENTRADA as e:
        raise HTTPException(HTTP_422, detail=str(e))
    hist = await _load_history(ucp)
    festivos, _fuente = await _load_festivos(ucp, hist, req.fecha)
    notas = []
    try:
        clima = await run_in_threadpool(_clima_loader, ucp) or {}
    except Exception as e:  # noqa: BLE001 — archivo de clima ausente/corrupto: se busca sin clima
        logger.warning("[ajustes] clima de %s no disponible (%s); análogos sin clima", ucp, _corto(e))
        clima, notas = {}, [f"clima no disponible para {ucp}: búsqueda sin clima"]
    try:
        resultado = await run_in_threadpool(
            analogos.buscar_analogos, hist, festivos, req.fecha, req.fecha_corte, req.filtros, clima
        )
    except ERRORES_ENTRADA as e:
        raise HTTPException(HTTP_422, detail=str(e))
    resultado["filtros_aplicados"]["notas"] = [*notas, *resultado["filtros_aplicados"].get("notas", [])]
    return {"ucp": ucp, **resultado}
