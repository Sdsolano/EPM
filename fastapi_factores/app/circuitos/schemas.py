"""
Pydantic schemas para el endpoint de circuitos.
"""
from typing import Dict, List
from pydantic import BaseModel, Field, validator


class MedidaCircuitoIn(BaseModel):
    """
    Una medida individual de un circuito.

    Attributes:
        codigo_rpm: Código del RPM
        circuito: Identificador del circuito (e.g., "CIRCUITO_A")
        ucp: UCP asociado (e.g., "PRIMEGRID")
        fecha: Fecha en formato YYYY-MM-DD
        flujo: Tipo de flujo (e.g., "AE", "AS")
        p1-p24: Valores de potencia para los 24 períodos
    """
    codigo_rpm: str
    circuito: str
    ucp: str
    fecha: str
    flujo: str

    # 24 períodos
    p1: float
    p2: float
    p3: float
    p4: float
    p5: float
    p6: float
    p7: float
    p8: float
    p9: float
    p10: float
    p11: float
    p12: float
    p13: float
    p14: float
    p15: float
    p16: float
    p17: float
    p18: float
    p19: float
    p20: float
    p21: float
    p22: float
    p23: float
    p24: float

    @validator('fecha')
    def validate_fecha(cls, v):
        """Valida formato de fecha YYYY-MM-DD."""
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("fecha debe estar en formato YYYY-MM-DD")
        return v


class CurvasTipicasCircuitosRequest(BaseModel):
    """
    Request para calcular curvas típicas por circuito.

    Attributes:
        medidas: Lista de medidas con información de circuito
        n_max: Número máximo de curvas típicas a seleccionar por circuito
    """
    medidas: List[MedidaCircuitoIn] = Field(
        ...,
        description="Lista de medidas con circuito, fecha y periodos"
    )
    n_max: int = Field(
        8,
        ge=1,
        le=100,
        description="Número máximo de curvas típicas por circuito"
    )


class CurvaSeleccionadaOut(BaseModel):
    """
    Una curva típica seleccionada.

    Attributes:
        codigo_rpm: Código del RPM de origen
        fecha: Fecha de la medida
        periodos: Diccionario con valores p1-p24
    """
    codigo_rpm: str
    fecha: str
    periodos: Dict[str, float]


class ResultadoCircuitoOut(BaseModel):
    """
    Resultado de procesamiento para un circuito.

    Attributes:
        curvas_seleccionadas: Lista de curvas típicas seleccionadas
        promedio: Promedio aritmético de las curvas seleccionadas (p1-p24)
        pesos: Pesos normalizados (suma = 1.0) calculados como periodo/suma_total
        n_curvas: Número de curvas seleccionadas (puede ser < n_max si hay pocas medidas)
    """
    curvas_seleccionadas: List[CurvaSeleccionadaOut]
    promedio: Dict[str, float]
    pesos: Dict[str, float]
    n_curvas: int


class CurvasTipicasCircuitosResponse(BaseModel):
    """
    Respuesta completa del endpoint.

    Attributes:
        ok: Indica éxito de la operación
        circuitos: Diccionario con resultados por circuito
    """
    ok: bool
    circuitos: Dict[str, ResultadoCircuitoOut]
