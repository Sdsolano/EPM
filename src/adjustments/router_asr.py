"""
`POST /adjustments/comparar-asr`: compara, para una semana, el pronóstico base, el ASR actual de Sphere
(reproducido en `asr.py`), la mesa de ajustes (`/adjustments/preview` con la misma base y defaults) y,
si existen, el real y el resultado final del analista. Pensado para la revisión con el analista: se le
pasa su pronóstico base (`base_curvas`) y, opcionalmente, los históricos que marcó en Sphere y su
resultado ajustado a mano. No persiste nada; misma autenticación (`X-Ajustes-Token`) que `/adjustments`.

Request:
- `ucp`, `semana_inicio` (lunes), `fecha_emision` (opcional: corte = último dato real anterior; default
  = último dato real, a lo sumo el sábado previo a la semana, como el backtest).
- `base_curvas` {fecha: [24]} que cubra corte+1..domingo, o `base: "ingenuo"` (misma regla del backtest).
  `base: "modelo"` no está disponible aquí (refresca datos y llama al modelo; usar `base_curvas`).
- `seleccion` {fecha objetivo: [≥ 3 fechas de históricos en ORDEN DE CLIC]} (deben ser días reales
  ≤ corte). Sin `seleccion` se usan `escenarios` (default `["anual"]`, ver `asr.fechas_escenario`).
- `analista_curvas` {fecha: [24]} (opcional), `params` de la mesa (opcional, mismos que /preview).

Respuesta: `series` {nombre: {fecha: [24]}} (base, mesa, asr…, real, analista), `metricas` por día y
serie (energía, pico, y con real: WAPE, error de energía, pico y forma), `semana`, `asr_detalle`
(h1/h2/h3, par de desvi2, cluster, periodos recortados por el validador) y `omitidos`.
El validador usa models/{ucp}; si no hay modelos, las series con validador quedan en `omitidos`.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from src.adjustments import asr, data
from src.adjustments import comparar_asr as cmp
from src.adjustments import router as ajustes_router
from src.adjustments.types import RunContext

HTTP_422 = ajustes_router.HTTP_422

router = APIRouter(prefix="/adjustments", tags=["ajustes"],
                   dependencies=[Depends(ajustes_router.verificar_token)])


class CompararASRRequest(BaseModel):
    ucp: str = Field(..., pattern=ajustes_router.UCP_PATTERN)
    semana_inicio: date = Field(..., description="Lunes de la semana objetivo")
    fecha_emision: Optional[date] = Field(None, description="Corte = último dato real anterior a esta fecha")
    base_curvas: Optional[Dict[str, List[float]]] = Field(None, description="Pronóstico base {fecha: [24]}")
    base: Literal["ingenuo", "modelo"] = Field("ingenuo", description="Solo si no viene base_curvas")
    seleccion: Optional[Dict[str, List[str]]] = Field(
        None, description="{fecha objetivo: [fechas de históricos en orden de clic]} (≥ 3 por día)")
    escenarios: List[str] = Field(default_factory=lambda: ["anual"])
    analista_curvas: Optional[Dict[str, List[float]]] = None
    params: Optional[Dict[str, Dict[str, Any]]] = None
    descomponer: bool = False

    @field_validator("semana_inicio")
    @classmethod
    def _lunes(cls, v: date) -> date:
        if v.weekday() != 0:
            raise ValueError("semana_inicio debe ser lunes")
        return v

    @field_validator("escenarios")
    @classmethod
    def _escenarios(cls, v: List[str]) -> List[str]:
        malos = [e for e in v if e not in asr.ESCENARIOS]
        if malos or not v:
            raise ValueError(f"escenarios inválidos {malos}; válidos: {list(asr.ESCENARIOS)}")
        return v


def _curvas_finitas(curvas: Dict[str, List[float]], campo: str) -> Dict[str, List[float]]:
    """24 periodos finitos por fecha (422 si no): NaN/inf romperían las métricas y el JSON de respuesta."""
    try:
        cmp.validar_curvas(curvas, campo)
    except ValueError as e:
        raise HTTPException(HTTP_422, detail=str(e))
    return curvas


def _seleccion(req: CompararASRRequest, ctx: RunContext,
               reales: Dict[str, Any]) -> Optional[Dict[str, List[asr.Historico]]]:
    """Selección explícita → históricos; solo días reales ≤ corte (`reales` ya viene recortado)."""
    if req.seleccion is None:
        return None
    fechas_semana = {d.isoformat() for d in data.dias_semana(ctx.semana_inicio)}
    out: Dict[str, List[asr.Historico]] = {}
    for objetivo, fechas in req.seleccion.items():
        try:
            obj = date.fromisoformat(str(objetivo)).isoformat()
            norm = [date.fromisoformat(str(f)).isoformat() for f in fechas]
        except ValueError:
            raise HTTPException(HTTP_422, detail=f"seleccion: fechas inválidas en {objetivo!r}")
        if obj not in fechas_semana:
            raise HTTPException(HTTP_422, detail=f"seleccion: {obj} no está en la semana objetivo")
        if len(norm) < 3:
            raise HTTPException(HTTP_422, detail=f"seleccion {obj}: el ASR necesita al menos 3 históricos")
        faltan = [f for f in norm if f not in reales]
        if faltan:
            raise HTTPException(HTTP_422, detail={
                "mensaje": f"seleccion {obj}: históricos sin dato real ≤ corte {ctx.fecha_corte.isoformat()}",
                "fechas": faltan})
        out[obj] = [asr.Historico(f, reales[f].tolist()) for f in norm]
    return out


@router.post("/comparar-asr")
async def comparar_asr(req: CompararASRRequest) -> Dict[str, Any]:
    ucp = ajustes_router._validar_ucp(req.ucp)
    if req.base_curvas is None and req.base == "modelo":
        raise HTTPException(HTTP_422, detail="base 'modelo' no disponible en comparar-asr: envíe base_curvas "
                                             "(el pronóstico emitido) o use base 'ingenuo'")
    hist = await ajustes_router._load_history(ucp)
    festivos, _fuente = await ajustes_router._load_festivos(ucp, hist, req.semana_inicio)
    emision = req.fecha_emision
    if emision is None:  # corte = último dato real, a lo sumo el sábado previo (como el backtest)
        emision = min(ajustes_router._ultimo_dato_real(hist, ucp) + timedelta(days=1),
                      req.semana_inicio - timedelta(days=1))
    ctx = ajustes_router._build_context(ucp, req.semana_inicio, emision, hist, festivos)
    if req.base_curvas is not None:
        base = _curvas_finitas(ajustes_router._validar_base_curvas(req.base_curvas, ctx), "base_curvas")
        provider = ajustes_router._const_provider(base)
    else:
        provider = cmp.proveedor_ingenuo(ctx.historico)
    analista = (_curvas_finitas(ajustes_router._normalizar_fechas(req.analista_curvas, "analista_curvas"),
                                "analista_curvas")
                if req.analista_curvas else None)
    reales = data.curvas_reales(hist)  # una sola vez: la selección usa ≤ corte; las métricas, la semana
    corte = ctx.fecha_corte.isoformat()
    seleccion = _seleccion(req, ctx, {f: v for f, v in reales.items() if f <= corte})
    try:
        return await run_in_threadpool(
            cmp.comparar_semana, ctx, provider, reales=reales, escenarios=req.escenarios, seleccion=seleccion,
            festivos_especial=ctx.festivos, params_mesa=req.params, analista=analista,
            descomponer=req.descomponer,
        )
    except ajustes_router.ERRORES_ENTRADA as e:
        raise HTTPException(HTTP_422, detail=str(e))
