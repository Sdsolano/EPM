"""
`/adjustments/insights/*`: resumen ejecutivo, precisión reciente, recomendaciones, banda de incertidumbre y
series de la semana. Solo lectura. Contrato: docs/work/ajustes/noche2/CONTRATO_B.md; lógica en `insights.py`.

Misma autenticación (`X-Ajustes-Token`) y mismas validaciones de `ucp`/`run_id` que `/adjustments`. Sin
historial real o sin historial de pronósticos emitidos se degrada a null/[]/"sin_datos" (nunca 500).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
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


def _curvas_reales(ucp: str) -> Dict[str, Any]:
    """Carga + normaliza + curvas reales (síncrono: correr en threadpool)."""
    hist = data.normalizar_historico(ajustes_router._history_loader(ucp), ucp=ucp)
    return data.curvas_reales(hist)


@lru_cache(maxsize=8)
def _curvas_reales_cache(ruta: str, mtime: float, ucp: str) -> Dict[str, Any]:
    return _curvas_reales(ucp)


async def _reales(ucp: str) -> Dict[str, Any]:
    """
    Curvas reales; sin histórico (o ilegible) → {} para degradar en vez de fallar. Con el cargador por defecto se cachea
    por (ruta, mtime) del datos.csv (tamaño acotado); la carga y la normalización corren en threadpool, nunca en el event loop.
    Un cargador inyectado (tests/demo) no se cachea.
    """
    mtime: Optional[float] = None
    ruta = Path(data.RAW_DATA_DIR) / ucp / "datos.csv"
    if ajustes_router._history_loader is data.cargar_historico:
        try:
            mtime = ruta.stat().st_mtime
        except OSError:
            mtime = None
    try:
        if mtime is not None:
            return await run_in_threadpool(_curvas_reales_cache, str(ruta), mtime, ucp)
        return await run_in_threadpool(_curvas_reales, ucp)
    except FileNotFoundError as e:
        logger.warning("[insights] sin histórico real de %s: %s", ucp, e)
    except ajustes_router.ERRORES_ENTRADA as e:
        logger.warning("[insights] histórico real de %s ilegible: %s", ucp, e)
    return {}


def _aprobadas(store: Optional[Store], ucp: str, hasta: date, semanas: int) -> Dict[str, Dict[str, Any]]:
    """
    {semana_inicio: curvas finales} de las corridas aprobadas de las `semanas` semanas que terminan en el lunes `hasta`
    (tolerante: sin store o con error → {}). El store devuelve las más recientes primero (y acota `limite` a 500): ante
    varias aprobadas de la misma semana prevalece la más reciente.
    """
    if store is None or not hasattr(store, "list_runs"):
        return {}
    try:
        out: Dict[str, Dict[str, Any]] = {}
        for run in store.list_runs(ucp, hasta - timedelta(weeks=semanas), hasta, min(500, (semanas + 2) * 6)):
            meta = run["meta"]
            if meta.get("estado") == ESTADO_APROBADA:
                out.setdefault(str(meta["semana_inicio"])[:10], pl.final_curves(run["records"]))
        return out
    except Exception as e:  # noqa: BLE001 — opcional: la precisión sigue con el CSV
        logger.warning("[insights] no se pudieron leer corridas aprobadas de %s: %s", ucp, e)
        return {}


def _datos(ucp: str, reales: Dict[str, Any], store: Optional[Store] = None, semanas: int = 12,
           hasta: Optional[date] = None) -> insights.Datos:
    """Datos de cálculo; con `store` incluye las corridas aprobadas de las `semanas` semanas hasta `hasta`."""
    emitidos, semana_de = insights.cargar_emitidos(ucp)
    fechas = list(reales)[-1:] + list(emitidos)[:1] + list(emitidos)[-1:]
    apr: Dict[str, Dict[str, Any]] = {}
    if store is not None:
        ref = hasta or (date.fromisoformat(max(reales)) if reales else date.today())
        apr = _aprobadas(store, ucp, data.lunes_de(ref), semanas)
    return insights.Datos(
        emitidos=emitidos, semana_de=semana_de, reales=reales,
        festivos=insights.festivos_por_defecto(fechas), aprobadas=apr,
    )


async def _cargar_corrida(run_id: str, store: Optional[Store]) -> Tuple[insights.Corrida, insights.Datos]:
    store = ajustes_router._require_store(store)
    saved = await run_in_threadpool(ajustes_router._get_run_or_404, store, run_id)
    records = pl.migrar_records(saved["records"])
    corrida = insights.corrida_desde(saved["meta"], records, pl.final_curves(records))
    reales = await _reales(corrida.ucp)
    datos = await run_in_threadpool(_datos, corrida.ucp, reales)
    datos.festivos = set(datos.festivos) | set(corrida.festivos)  # los festivos de la corrida también cuentan
    return corrida, datos


@router.get("/resumen")
async def get_resumen(run_id: str = Query(...), store: Optional[Store] = Depends(ajustes_router.get_store_dep)
                      ) -> Dict[str, Any]:
    corrida, datos = await _cargar_corrida(run_id, store)

    def _calc() -> Dict[str, Any]:
        # la confianza usa las SEMANAS_CONFIANZA semanas anteriores a la corrida
        datos.aprobadas = _aprobadas(store, corrida.ucp, corrida.semana_inicio - timedelta(days=7),
                                     insights.SEMANAS_CONFIANZA + 1)
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
        out = insights.precision(_datos(ucp, reales, store, semanas, hasta), semanas=semanas, hasta=hasta)
        out["ucp"] = ucp
        return out
    return await run_in_threadpool(_calc)


@router.get("/recomendaciones")
async def get_recomendaciones(run_id: str = Query(...), store: Optional[Store] = Depends(ajustes_router.get_store_dep)
                              ) -> Dict[str, Any]:
    """
    Recomendaciones por día. Las alertas del sistema (`respaldo: "alerta_del_sistema"`) son las accionables. Las de sesgo son
    TENDENCIAS RECIENTES de referencia (`respaldo: "tendencia_reciente"`, prioridad baja, `validado_fuera_de_muestra: false`): B2 y
    B5 midieron que corregir el sesgo no mejora fuera de muestra, así que se emiten solo con evidencia estricta y sin promesa.
    """
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
        datos = await run_in_threadpool(_datos, ucp, await _reales(ucp))

    # con run_id la muestra termina por defecto en min(semana − 1 día, fecha_corte): sin fuga hacia la semana evaluada
    # y un `fecha_fin` explícito no puede pasar de ese corte (min): con run_id nunca hay fuga hacia la semana evaluada
    fin = fecha_fin.isoformat() if fecha_fin else None
    if corrida is not None:
        fin = min(fin, corrida.fecha_fin_sin_fuga) if fin else corrida.fecha_fin_sin_fuga
    ini = fecha_inicio.isoformat() if fecha_inicio else None

    def _calc() -> Dict[str, Any]:
        tablas = insights.tablas_banda(datos, ini, fin)
        out = {"ucp": ucp, **insights.banda_json(tablas, datos, tipo_dia, ini, fin)}
        out["muestra"] = {"fecha_inicio": ini, "fecha_fin": fin}
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
