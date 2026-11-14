"""
`/adjustments/insights/*`: resumen ejecutivo, precisión reciente, recomendaciones, banda de incertidumbre y
series de la semana. Solo lectura. Contrato: docs/work/ajustes/noche2/CONTRATO_B.md; lógica en `insights.py`.

Misma autenticación (`X-Ajustes-Token`) y mismas validaciones de `ucp`/`run_id` que `/adjustments`. Sin
historial real o sin historial de pronósticos emitidos se degrada a null/[]/"sin_datos" (nunca 500).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from src.adjustments import data, insights
from src.adjustments import pipeline as pl
from src.adjustments import router as ajustes_router
from src.adjustments.store import ESTADO_APROBADA, Store

logger = logging.getLogger(__name__)

HTTP_422 = ajustes_router.HTTP_422
router = APIRouter(prefix="/adjustments/insights", tags=["ajustes-insights"],
                   dependencies=[Depends(ajustes_router.verificar_token)])


async def _reales(ucp: str) -> Dict[str, Any]:
    """Curvas reales; sin histórico (o ilegible) → {} para degradar en vez de fallar."""
    try:
        hist = await ajustes_router._load_history(ucp)
    except HTTPException as e:
        logger.warning("[insights] sin histórico real de %s: %s", ucp, e.detail)
        return {}
    return await run_in_threadpool(data.curvas_reales, hist)


def _aprobadas(store: Optional[Store], ucp: str) -> Dict[str, Dict[str, Any]]:
    """{semana_inicio: curvas finales} de las corridas aprobadas (tolerante: sin store o con error → {})."""
    if store is None or not hasattr(store, "list_runs"):
        return {}
    try:
        out: Dict[str, Dict[str, Any]] = {}
        for run in store.list_runs(ucp, None, None, 500):
            meta = run["meta"]
            if meta.get("estado") != ESTADO_APROBADA:
                continue
            out[str(meta["semana_inicio"])[:10]] = pl.final_curves(run["records"])
        return out
    except Exception as e:  # noqa: BLE001 — opcional: la precisión sigue con el CSV
        logger.warning("[insights] no se pudieron leer corridas aprobadas de %s: %s", ucp, e)
        return {}


def _datos(ucp: str, reales: Dict[str, Any], store: Optional[Store], con_aprobadas: bool = True) -> insights.Datos:
    emitidos, semana_de = insights.cargar_emitidos(ucp)
    fechas = list(reales)[-1:] + list(emitidos)[:1] + list(emitidos)[-1:]
    return insights.Datos(
        emitidos=emitidos, semana_de=semana_de, reales=reales,
        festivos=insights.festivos_por_defecto(fechas),
        aprobadas=_aprobadas(store, ucp) if con_aprobadas else {},
    )


async def _cargar_corrida(run_id: str, store: Optional[Store]) -> Tuple[insights.Corrida, insights.Datos]:
    store = ajustes_router._require_store(store)
    saved = await run_in_threadpool(ajustes_router._get_run_or_404, store, run_id)
    records = pl.migrar_records(saved["records"])
    corrida = insights.corrida_desde(saved["meta"], records, pl.final_curves(records))
    reales = await _reales(corrida.ucp)
    datos = await run_in_threadpool(_datos, corrida.ucp, reales, store, False)
    datos.festivos = set(datos.festivos) | set(corrida.festivos)  # los festivos de la corrida también cuentan
    return corrida, datos


@router.get("/resumen")
async def get_resumen(run_id: str = Query(...), store: Optional[Store] = Depends(ajustes_router.get_store_dep)
                      ) -> Dict[str, Any]:
    corrida, datos = await _cargar_corrida(run_id, store)

    def _calc() -> Dict[str, Any]:
        datos.aprobadas = _aprobadas(store, corrida.ucp)
        return {"run_id": run_id, **insights.resumen(corrida, datos)}
    return await run_in_threadpool(_calc)


@router.get("/precision")
async def get_precision(
    ucp: str = Query(..., pattern=ajustes_router.UCP_PATTERN),
    semanas: int = Query(12, ge=1, le=52),
    hasta: Optional[date] = Query(None, description="Lunes máximo (default: última semana con datos completos)"),
    store: Optional[Store] = Depends(ajustes_router.get_store_dep),
) -> Dict[str, Any]:
    ucp = ajustes_router._validar_ucp(ucp)
    if hasta is not None and hasta.weekday() != 0:
        raise HTTPException(HTTP_422, detail="hasta debe ser lunes")
    reales = await _reales(ucp)

    def _calc() -> Dict[str, Any]:
        out = insights.precision(_datos(ucp, reales, store), semanas=semanas, hasta=hasta)
        out["ucp"] = ucp
        return out
    return await run_in_threadpool(_calc)


@router.get("/recomendaciones")
async def get_recomendaciones(run_id: str = Query(...), store: Optional[Store] = Depends(ajustes_router.get_store_dep)
                              ) -> Dict[str, Any]:
    corrida, datos = await _cargar_corrida(run_id, store)

    def _calc() -> Dict[str, Any]:
        recs = insights.recomendaciones(corrida, datos)
        return {"run_id": run_id, "version": corrida.version, "estado": corrida.estado, **recs}
    return await run_in_threadpool(_calc)


@router.get("/banda")
async def get_banda(
    ucp: Optional[str] = Query(None, pattern=ajustes_router.UCP_PATTERN),
    run_id: Optional[str] = Query(None),
    fecha_inicio: Optional[date] = Query(None, description="Acota la muestra de errores (inclusive)"),
    fecha_fin: Optional[date] = Query(None, description="Acota la muestra de errores (inclusive)"),
    tipo_dia: Optional[str] = Query(None, pattern="^(laboral|sabado|domingo|festivo)$"),
    store: Optional[Store] = Depends(ajustes_router.get_store_dep),
) -> Dict[str, Any]:
    if ucp is None and run_id is None:
        raise HTTPException(HTTP_422, detail="Indique ucp o run_id")
    if fecha_inicio and fecha_fin and fecha_inicio > fecha_fin:
        raise HTTPException(HTTP_422, detail="fecha_inicio no puede ser posterior a fecha_fin")
    corrida: Optional[insights.Corrida] = None
    if run_id is not None:
        corrida, datos = await _cargar_corrida(run_id, store)
        if ucp is not None and ucp != corrida.ucp:
            raise HTTPException(HTTP_422, detail=f"La corrida {run_id} es de {corrida.ucp}, no de {ucp}")
        ucp = corrida.ucp
    else:
        ucp = ajustes_router._validar_ucp(ucp)
        datos = await run_in_threadpool(_datos, ucp, await _reales(ucp), store, False)

    def _calc() -> Dict[str, Any]:
        tablas = insights.tablas_banda(datos, fecha_inicio.isoformat() if fecha_inicio else None,
                                       fecha_fin.isoformat() if fecha_fin else None)
        out = {"ucp": ucp, **insights.banda_json(tablas, datos, tipo_dia)}
        if corrida is not None:
            out["dias"] = insights.banda_corrida(corrida, tablas)
        return out
    return await run_in_threadpool(_calc)


@router.get("/series")
async def get_series(run_id: str = Query(...), store: Optional[Store] = Depends(ajustes_router.get_store_dep)
                     ) -> Dict[str, Any]:
    corrida, datos = await _cargar_corrida(run_id, store)

    def _calc() -> Dict[str, Any]:
        return {"run_id": run_id, **insights.series(corrida, datos)}
    return await run_in_threadpool(_calc)
