"""
Tipos compartidos del módulo de ajustes por pasos.

Contrato fijo (ver docs/work/ajustes/PLAN.md, "Contratos entre módulos").
Todos los dataclasses exponen `to_dict()` (JSON-serializable, fechas como
"YYYY-MM-DD") y `from_dict()` para que el Store los persista.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

Curves = Dict[str, List[float]]  # {"YYYY-MM-DD": [P1..P24]}; tras from_dict puede traer None (= periodo inválido)


class StepState(str, Enum):
    APLICADO = "aplicado"
    OMITIDO = "omitido"
    EDITADO = "editado"


# ---------------------------------------------------------------------------
# Helpers de serialización
# ---------------------------------------------------------------------------

def to_jsonable(value: Any) -> Any:
    """Convierte recursivamente numpy/pandas/fechas a tipos JSON nativos."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    return value


def parse_date(value: Any) -> Optional[date]:
    """Acepta date/datetime/Timestamp/"YYYY-MM-DD" y devuelve date (o None)."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _valor_json(v: Any) -> Optional[float]:
    """float finito, o None si es None / NaN / ±inf / no numérico (JSON-safe)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def curves_to_dict(curves: Curves) -> Dict[str, List[Optional[float]]]:
    """Curvas JSON-safe: NaN/±inf → None (los validadores lo tratan como periodo inválido)."""
    return {str(k)[:10]: [_valor_json(v) for v in (vals or [])] for k, vals in (curves or {}).items()}


def curves_from_dict(data: Optional[Dict[str, Any]]) -> Curves:
    """Inversa de `curves_to_dict`: deja `None` donde no hay valor finito."""
    return {str(k)[:10]: [_valor_json(v) for v in (vals or [])] for k, vals in (data or {}).items()}


# ---------------------------------------------------------------------------
# Validación de curvas (única para pipeline, control, approval_blockers y router)
# ---------------------------------------------------------------------------

N_PERIODOS = 24
MOTIVOS_CURVA: Dict[str, str] = {
    "PERIODOS_FALTANTES": "Días sin los 24 periodos completos",
    "VALORES_NO_FINITOS": "Días con valores nulos, no numéricos o infinitos",
    "VALORES_NEGATIVOS": "Días con valores negativos",
}


def validar_curva(valores: Any) -> Optional[str]:
    """
    Motivo por el que una curva diaria no es válida, o None si lo es.

    El motivo es uno de los códigos de `MOTIVOS_CURVA`:
    - "PERIODOS_FALTANTES": None, no es una secuencia plana o no trae 24 periodos.
    - "VALORES_NO_FINITOS": algún periodo es None, NaN, ±inf o no numérico.
    - "VALORES_NEGATIVOS": algún periodo < 0.
    Nunca lanza excepción.
    """
    if valores is None or isinstance(valores, (str, bytes, dict)):
        return "PERIODOS_FALTANTES"
    try:
        n = len(valores)
    except TypeError:
        return "PERIODOS_FALTANTES"
    if n != N_PERIODOS:
        return "PERIODOS_FALTANTES"
    try:
        arr = np.asarray(valores, dtype=float)
    except (TypeError, ValueError):
        return "VALORES_NO_FINITOS"
    if arr.shape != (N_PERIODOS,):
        return "PERIODOS_FALTANTES"
    if not np.isfinite(arr).all():
        return "VALORES_NO_FINITOS"
    if (arr < 0).any():
        return "VALORES_NEGATIVOS"
    return None


def validar_curvas_editadas(curvas: Curves, fechas_permitidas: Iterable[str]) -> None:
    """
    Valida curvas manuales del analista; lanza ValueError con un mensaje claro
    (el router lo convierte en 422). Exige al menos un día, fechas dentro de
    `fechas_permitidas` y curvas válidas según `validar_curva`.
    """
    if not curvas:
        raise ValueError("No llegaron curvas editadas")
    permitidas = {str(f)[:10] for f in fechas_permitidas}
    for fecha, valores in curvas.items():
        f = str(fecha)[:10]
        if f not in permitidas:
            rango = f"{min(permitidas)}..{max(permitidas)}" if permitidas else "vacío"
            raise ValueError(f"Fecha editada {f} fuera del rango permitido ({rango})")
        motivo = validar_curva(valores)
        if motivo is not None:
            raise ValueError(f"{f}: curva inválida ({motivo}: {MOTIVOS_CURVA[motivo]})")


# ---------------------------------------------------------------------------
# Dataclasses del contrato
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    nivel: str  # "info" | "warning" | "bloqueo"
    codigo: str  # ej. "TENDENCIA_FUERA_BANDA", "SIN_EVIDENCIA_NIVEL", "PERIODOS_FALTANTES"
    mensaje: str
    fechas: List[str] = field(default_factory=list)
    detalle: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nivel": self.nivel,
            "codigo": self.codigo,
            "mensaje": self.mensaje,
            "fechas": [str(f)[:10] for f in self.fechas],
            "detalle": to_jsonable(self.detalle),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Alert":
        return cls(
            nivel=data["nivel"],
            codigo=data["codigo"],
            mensaje=data.get("mensaje", ""),
            fechas=list(data.get("fechas") or []),
            detalle=dict(data.get("detalle") or {}),
        )


@dataclass
class StepResult:
    curves: Curves
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    alerts: List[Alert] = field(default_factory=list)
    # Paso `editado`: curva CALCULADA (pre-edición) de cada día editado;
    # `curves` trae la editada. Vacío en pasos aplicados/omitidos.
    curvas_calculadas: Curves = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "curves": curves_to_dict(self.curves),
            "diagnostics": to_jsonable(self.diagnostics),
            "alerts": [a.to_dict() for a in self.alerts],
            "curvas_calculadas": curves_to_dict(self.curvas_calculadas),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepResult":
        return cls(
            curves=curves_from_dict(data.get("curves")),
            diagnostics=dict(data.get("diagnostics") or {}),
            alerts=[Alert.from_dict(a) for a in data.get("alerts") or []],
            curvas_calculadas=curves_from_dict(data.get("curvas_calculadas")),
        )


@dataclass
class RunContext:
    ucp: str
    semana_inicio: date  # lunes de la semana objetivo
    fecha_emision: date
    fecha_corte: date  # último dato real <= fecha_emision - 1
    historico: pd.DataFrame  # columnas: fecha (datetime64), P1..P24, TOTAL; solo fechas <= fecha_corte
    festivos: List[str]  # "YYYY-MM-DD"
    semana_ref_inicio: Optional[date] = None  # la fija el paso 1
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_historico: bool = False) -> Dict[str, Any]:
        """
        Serializa el contexto. El histórico (miles de filas) se omite por
        defecto: se reconstruye desde datos.csv con `build_context`.
        """
        data: Dict[str, Any] = {
            "ucp": self.ucp,
            "semana_inicio": self.semana_inicio.isoformat(),
            "fecha_emision": self.fecha_emision.isoformat(),
            "fecha_corte": self.fecha_corte.isoformat(),
            "festivos": [str(f)[:10] for f in self.festivos],
            "semana_ref_inicio": (
                self.semana_ref_inicio.isoformat() if self.semana_ref_inicio else None
            ),
            "extras": to_jsonable(self.extras),
        }
        if include_historico:
            df = self.historico.copy()
            df["fecha"] = pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d")
            data["historico"] = to_jsonable(df.to_dict(orient="records"))
        return data

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], historico: Optional[pd.DataFrame] = None
    ) -> "RunContext":
        if historico is None:
            rows = data.get("historico") or []
            historico = pd.DataFrame(rows)
            if "fecha" in historico.columns:
                historico["fecha"] = pd.to_datetime(historico["fecha"])
        return cls(
            ucp=data["ucp"],
            semana_inicio=parse_date(data["semana_inicio"]),
            fecha_emision=parse_date(data["fecha_emision"]),
            fecha_corte=parse_date(data["fecha_corte"]),
            historico=historico,
            festivos=list(data.get("festivos") or []),
            semana_ref_inicio=parse_date(data.get("semana_ref_inicio")),
            extras=dict(data.get("extras") or {}),
        )


@dataclass
class StepRecord:  # lo que se persiste por paso
    orden: int
    paso: str  # "base" | "referencia" | "nivel" | "forma" | "especiales" | "control"
    estado: StepState
    params: Dict[str, Any]
    result: StepResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "orden": int(self.orden),
            "paso": self.paso,
            "estado": StepState(self.estado).value,
            "params": to_jsonable(self.params),
            "result": self.result.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepRecord":
        return cls(
            orden=int(data["orden"]),
            paso=data["paso"],
            estado=StepState(data["estado"]),
            params=dict(data.get("params") or {}),
            result=StepResult.from_dict(data.get("result") or {}),
        )
