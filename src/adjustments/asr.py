"""
Reproducción en Python del botón ASR ("ajuste inteligente por escenario de referencia") de Sphere.

Fuente: `Sphere/src/views/pages/pronosticos/Pronosticos.tsx` (rama `test`, commit 57eb4fb), `handleASR`
y el efecto que congela `valoresDesviDeltaOriginales`, más el endpoint `/validate-hourly-adjustments`
de `src/api/main.py` (`HourlyAdjustmentValidator`). Sirve para diagnosticar el ASR, no para producción.

Qué hace el front, hora a hora (h1, h2, h3 = históricos seleccionados):
- `historicosCombinados` = selección ordenada por fecha DESCENDENTE (`selectedHistoricosOrdenados`):
  h1 es el día más reciente seleccionado.
- `desvi1 = pred − h1` (pred sin redondear).
- `desvi2`: si ya está congelado (lo normal: se congela al haber ≥ 3 seleccionados en modo
  `diario_ajustado`) = `toFixed(1)(s[1] − s[2])` con `s` = [históricos reales en ORDEN DE CLIC,
  luego históricos-pronóstico en orden de clic]; si no, `h2 − h3` del orden por fecha.
- `delta = Math.round(desvi2 − desvi1)`; payload `{ucp, fecha, tipo_dia: "laboral",
  predicciones_actuales: toFixed(1)(pred), ajustes_solicitados: delta}`.
- Objetivo antes del validador = pred + delta ≈ h1 + desvi2: no depende del pronóstico base.

El endpoint ignora `tipo_dia`: elige la "senda" (perfil normalizado del cluster) con
`special_disaggregator.is_special_day(fecha)` y aplica el validador (±15 %, 3 iteraciones).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

N_PERIODOS = 24
TOLERANCIA_ENDPOINT = 0.15  # /validate-hourly-adjustments
MAX_ITER_ENDPOINT = 3
TIPO_DIA_FRONT = "laboral"  # el front lo manda fijo; el endpoint no lo usa
DIAS_MUY_ESPECIALES = ("12-24", "01-02")  # SpecialDaysDisaggregator.is_special_day

MODELS_ROOT = Path(__file__).resolve().parents[2] / "models"


# ---------------------------------------------------------------------------
# Redondeos de JavaScript
# ---------------------------------------------------------------------------

def math_round(x: float) -> int:
    """`Math.round` de JS: mitades hacia +∞ (Math.round(-2.5) = -2, Math.round(2.5) = 3)."""
    return int(math.floor(float(x) + 0.5))


def to_fixed_1(x: float) -> float:
    """`parseFloat(x.toFixed(1))`: valor decimal exacto del double, mitades lejos de cero."""
    return float(Decimal(float(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _vector(v: Optional[Sequence[Optional[float]]]) -> List[float]:
    """`ps[idx] ?? 0` del front: None (o faltante) → 0."""
    v = list(v or [])
    out = [float(v[i]) if i < len(v) and v[i] is not None else 0.0 for i in range(N_PERIODOS)]
    if not all(math.isfinite(x) for x in out):
        raise ValueError("curva con valores no finitos (NaN/inf)")
    return out


# ---------------------------------------------------------------------------
# Selección de históricos (orden del front)
# ---------------------------------------------------------------------------

@dataclass
class Historico:
    fecha: str                       # ISO; define el orden por fecha descendente
    curva: Sequence[Optional[float]]
    tipo: str = "normal"             # "normal" (histórico real) | "pronostico" (histórico-pronóstico)


def ordenar_por_fecha(seleccion: Sequence[Historico]) -> List[Historico]:
    """`selectedHistoricosOrdenados`: fecha descendente (sort estable, como Array.prototype.sort)."""
    return sorted(seleccion, key=lambda h: h.fecha, reverse=True)


def orden_congelado(seleccion_clic: Sequence[Historico]) -> List[Historico]:
    """Orden con el que se congela desvi2: reales en orden de clic y luego históricos-pronóstico."""
    return [h for h in seleccion_clic if h.tipo == "normal"] + [h for h in seleccion_clic if h.tipo != "normal"]


def desvi2_congelado(seleccion_clic: Sequence[Historico]) -> Optional[List[float]]:
    """`valoresDesviDeltaOriginales.desvi2` (toFixed(1)); None si hay menos de 3 seleccionados."""
    s = orden_congelado(seleccion_clic)
    if len(s) < 3:
        return None
    a, b = _vector(s[1].curva), _vector(s[2].curva)
    return [to_fixed_1(a[i] - b[i]) for i in range(N_PERIODOS)]


def ajustes_asr(
    pred: Sequence[Optional[float]],
    seleccion_clic: Sequence[Historico],
    congelar_desvi2: bool = True,
) -> List[int]:
    """
    `ajustes_solicitados` de handleASR. `seleccion_clic` = históricos en el orden en que el analista
    los marcó (el front exige ≥ 3). `congelar_desvi2=False` reproduce la rama sin desvi2 congelado
    (h2 − h3 del orden por fecha).
    """
    if len(seleccion_clic) < 3:
        raise ValueError("ASR requiere al menos 3 históricos")
    p = _vector(pred)
    hs = ordenar_por_fecha(seleccion_clic)
    h1 = _vector(hs[0].curva)
    congelado = desvi2_congelado(seleccion_clic) if congelar_desvi2 else None
    if congelado is None:
        h2, h3 = _vector(hs[1].curva), _vector(hs[2].curva)
        desvi2 = [h2[i] - h3[i] for i in range(N_PERIODOS)]
    else:
        desvi2 = congelado
    return [math_round(desvi2[i] - (p[i] - h1[i])) for i in range(N_PERIODOS)]


def payload_asr(
    ucp: str,
    fecha: str,
    pred: Sequence[Optional[float]],
    seleccion_clic: Sequence[Historico],
    congelar_desvi2: bool = True,
    tipo_dia: str = TIPO_DIA_FRONT,
) -> Dict[str, Any]:
    """Payload exacto que manda Sphere a `/validate-hourly-adjustments`."""
    return {
        "ucp": ucp,
        "fecha": fecha,
        "tipo_dia": tipo_dia,
        "predicciones_actuales": [to_fixed_1(v) for v in _vector(pred)],
        "ajustes_solicitados": ajustes_asr(pred, seleccion_clic, congelar_desvi2),
    }


# ---------------------------------------------------------------------------
# Senda del cluster y validador (misma lógica que /validate-hourly-adjustments)
# ---------------------------------------------------------------------------

_motores: Dict[Tuple[str, str], Tuple[Tuple[float, ...], Any]] = {}
_motores_lock = threading.Lock()
_PICKLES = ("hourly_disaggregator.pkl", "special_days_disaggregator.pkl")


def cargar_motor(ucp: str, models_root: Optional[Path] = None):
    """
    `HourlyDisaggregationEngine` como lo carga el endpoint (models/{ucp}, si no models/). Cacheado por
    UCP y directorio; se recarga si cambia la fecha de modificación de los .pkl (p. ej. tras /retrain),
    para no divergir del endpoint, que carga el motor en cada request. Seguro entre hilos.
    """
    from src.prediction.hourly import HourlyDisaggregationEngine

    raiz = Path(models_root) if models_root is not None else MODELS_ROOT
    models_dir = raiz / ucp
    if not models_dir.exists():
        models_dir = raiz
    clave = (ucp, str(models_dir))
    firma = tuple((models_dir / p).stat().st_mtime if (models_dir / p).exists() else -1.0 for p in _PICKLES)
    with _motores_lock:
        cacheado = _motores.get(clave)
        if cacheado is None or cacheado[0] != firma:
            motor = HourlyDisaggregationEngine(auto_load=True, models_dir=str(models_dir), ucp=ucp)
            _motores[clave] = cacheado = (firma, motor)
        return cacheado[1]


def verificar_motor(motor, ucp: str) -> None:
    """ValueError si el desagregador normal no está entrenado (el endpoint responde 404 en ese caso)."""
    if not motor.normal_disaggregator.is_fitted:
        raise ValueError(f"Modelo de desagregación normal no entrenado para UCP {ucp}")


def es_dia_especial(motor, fecha: pd.Timestamp, festivos: Optional[Iterable[str]] = None) -> bool:
    """
    `special_disaggregator.is_special_day` (solo si está entrenado, como el endpoint). Con `festivos`
    (ISO) se evita la API de festivos: 24-dic, 2-ene o festivo de la lista (misma regla, sin red).
    """
    esp = motor.special_disaggregator
    if not esp.is_fitted:
        return False
    if festivos is None:
        return bool(esp.is_special_day(fecha))
    festivos = festivos if isinstance(festivos, (set, frozenset)) else set(festivos)
    return fecha.strftime("%m-%d") in DIAS_MUY_ESPECIALES or fecha.strftime("%Y-%m-%d") in festivos


def senda_cluster(
    motor, fecha: str, festivos: Optional[Iterable[str]] = None
) -> Tuple[np.ndarray, int, str]:
    """(senda normalizada, cluster_id, método) igual que el endpoint."""
    ts = pd.to_datetime(fecha)
    if es_dia_especial(motor, ts, festivos):
        dis = motor.special_disaggregator
        cluster_id = dis.cluster_by_date.get(ts.strftime("%m-%d"), -1)
        if cluster_id == -1:
            senda = dis.average_holiday_profile
            if senda is None:  # el endpoint fallaría (500); aquí el día queda omitido
                raise ValueError("el desagregador de días especiales no tiene perfil promedio de festivos")
        else:
            senda = dis.cluster_profiles.loc[cluster_id].values
        return np.asarray(senda, dtype=float), int(cluster_id), "special"
    dis = motor.normal_disaggregator
    cluster_id = dis.cluster_by_dayofweek.get(ts.dayofweek, 0)
    return np.asarray(dis.cluster_profiles.loc[cluster_id].values, dtype=float), int(cluster_id), "normal"


def validar(
    predicciones: Sequence[float],
    ajustes: Sequence[float],
    senda: np.ndarray,
    cluster_id: int,
    metodo: str,
) -> Dict[str, Any]:
    """
    `HourlyAdjustmentValidator` con los parámetros del endpoint. Loguea por periodo como en el endpoint;
    el script de corrida histórica sube el nivel del logger "src" a WARNING.
    """
    from src.prediction.hourly.adjustment_validator import HourlyAdjustmentValidator

    validador = HourlyAdjustmentValidator(default_tolerance=TOLERANCIA_ENDPOINT, max_iterations=MAX_ITER_ENDPOINT)
    return validador.validate_adjustments(
        predicciones_actuales=list(predicciones), ajustes_solicitados=list(ajustes),
        senda_referencia=senda, cluster_id=cluster_id, method=metodo,
    )


def validar_payload(motor, payload: Dict[str, Any], festivos: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Equivalente local de POST /validate-hourly-adjustments (respuesta con la misma forma)."""
    verificar_motor(motor, payload.get("ucp"))
    senda, cluster_id, metodo = senda_cluster(motor, payload["fecha"], festivos)
    return validar(payload["predicciones_actuales"], payload["ajustes_solicitados"], senda, cluster_id, metodo)


# ---------------------------------------------------------------------------
# ASR de un día (con variantes de diagnóstico)
# ---------------------------------------------------------------------------

@dataclass
class ResultadoASR:
    fecha: str
    objetivo: List[float]            # pred_redondeada + ajustes (lo que pide el ASR antes del validador)
    final: List[float]               # salida del validador (o el objetivo si validador=False)
    ajustes: List[int]
    cluster_id: Optional[int] = None
    metodo: Optional[str] = None
    periodos_modificados: int = 0
    h_fechas: List[str] = field(default_factory=list)  # h1, h2, h3 que usa el front (orden por fecha)
    desvi2_fechas: List[str] = field(default_factory=list)  # par (a, b) de desvi2 = a − b


def asr_dia(
    ucp: str,
    fecha: str,
    pred: Sequence[Optional[float]],
    seleccion_clic: Sequence[Historico],
    *,
    motor=None,
    validador: bool = True,
    congelar_desvi2: bool = True,
    festivos: Optional[Iterable[str]] = None,
    senda_fn: Optional[Callable[[str], Tuple[np.ndarray, int, str]]] = None,
) -> ResultadoASR:
    """
    ASR de un día tal como lo hace Sphere + el endpoint. `validador=False` devuelve el objetivo sin
    el validador. `senda_fn(fecha)` permite inyectar la senda (tests o variantes); si no, se usa
    `motor` (o `cargar_motor(ucp)`).
    """
    payload = payload_asr(ucp, fecha, pred, seleccion_clic, congelar_desvi2)
    objetivo = [p + a for p, a in zip(payload["predicciones_actuales"], payload["ajustes_solicitados"])]
    hs = ordenar_por_fecha(seleccion_clic)
    congelado = orden_congelado(seleccion_clic) if congelar_desvi2 else hs
    res = ResultadoASR(
        fecha=fecha, objetivo=objetivo, final=list(objetivo), ajustes=payload["ajustes_solicitados"],
        h_fechas=[h.fecha for h in hs[:3]], desvi2_fechas=[congelado[1].fecha, congelado[2].fecha],
    )
    if not validador:
        return res
    if senda_fn is not None:
        senda, cluster_id, metodo = senda_fn(fecha)
    else:
        senda, cluster_id, metodo = senda_cluster(motor or cargar_motor(ucp), fecha, festivos)
    out = validar(payload["predicciones_actuales"], payload["ajustes_solicitados"], senda, cluster_id, metodo)
    res.final = [float(v) for v in out["valores_ajustados"]]
    res.cluster_id, res.metodo = cluster_id, metodo
    res.periodos_modificados = int(out["metadata"]["periodos_modificados"])
    return res


# ---------------------------------------------------------------------------
# Escenarios de selección de históricos (para el backtest)
# ---------------------------------------------------------------------------

ESCENARIOS = ("anual", "anual_alineado", "anual_desfase", "anual_clic_asc", "recientes")


def fechas_escenario(
    escenario: str, d: date, corte: date, desplazamiento_ref: int = -364
) -> List[date]:
    """
    Históricos que elegiría el analista para el día `d` (en ORDEN DE CLIC), con corte `corte`:
    - `anual`: h1 = ancla (último mismo día de semana ≤ corte), h2 = d − 364, h3 = ancla − 364;
      clic de arriba abajo en la lista (fecha descendente) → orden de clic = orden por fecha.
    - `anual_alineado`: igual con el desplazamiento de la semana de referencia de la mesa
      (`desplazamiento_ref`, días; alinea festivos homólogos).
    - `anual_desfase`: supone h1 = d − 7 para el par del año anterior (h3 = h2 − 7) aunque el
      ancla real sea anterior (p. ej. el domingo con corte el sábado).
    - `anual_clic_asc`: selección de `anual` marcada de abajo arriba (fecha ascendente) → desvi2
      congelado = h2 − h1.
    - `recientes`: los 3 últimos días reales del mismo día de semana (modo "similares").
    """
    from src.adjustments import data

    ancla = data.ancla_mismo_dow(d, corte)
    if escenario == "anual":
        sel = [ancla, d - timedelta(days=364), ancla - timedelta(days=364)]
    elif escenario == "anual_alineado":
        sel = [ancla, d + timedelta(days=desplazamiento_ref), ancla + timedelta(days=desplazamiento_ref)]
    elif escenario == "anual_desfase":
        sel = [ancla, d - timedelta(days=364), d - timedelta(days=371)]
    elif escenario == "anual_clic_asc":
        sel = [ancla - timedelta(days=364), d - timedelta(days=364), ancla]
    elif escenario == "recientes":
        sel = [ancla, ancla - timedelta(days=7), ancla - timedelta(days=14)]
    else:
        raise ValueError(f"escenario desconocido {escenario!r}; válidos: {ESCENARIOS}")
    if escenario != "anual_clic_asc":
        sel = sorted(sel, reverse=True)  # clic de arriba abajo = fecha descendente
    return sel
