"""
Router de endpoints para procesamiento de circuitos.
"""
from fastapi import APIRouter, HTTPException
import logging

from app.circuitos.schemas import (
    CurvasTipicasCircuitosRequest,
    CurvasTipicasCircuitosResponse,
    ResultadoCircuitoOut,
    CurvaSeleccionadaOut
)
from app.circuitos.service import procesar_curvas_tipicas_por_circuito

# Configurar logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/factores/calculos", tags=["circuitos"])


@router.post(
    "/curvas-tipicas-circuitos",
    response_model=CurvasTipicasCircuitosResponse,
    summary="Procesar curvas típicas por circuito",
    description="""
    Procesa curvas típicas agrupadas por circuito sin consultar base de datos.

    **Flujo de procesamiento**:
    1. Agrupa medidas por campo `circuito`
    2. Para cada circuito:
       - Aplica filtrado de outliers por IQR
       - Selecciona hasta N curvas más típicas (forma + nivel)
       - Calcula promedio aritmético de las curvas seleccionadas
       - Calcula pesos normalizados (suma = 1.0)

    **Input**: JSON con lista de medidas (código RPM, circuito, fecha, periodos)
    **Output**: Resultados agrupados por circuito con curvas, promedio y pesos

    **Nota**: Este endpoint NO realiza consultas a base de datos. Todos los datos
    deben proporcionarse en el cuerpo de la request.
    """
)
def calcular_curvas_tipicas_circuitos(payload: CurvasTipicasCircuitosRequest):
    """
    Endpoint para procesar curvas típicas por circuito.

    Args:
        payload: Request con medidas y n_max

    Returns:
        Response con resultados por circuito

    Raises:
        HTTPException 400: Error de validación
        HTTPException 500: Error interno de procesamiento
    """
    try:
        logger.info(
            f"Procesando curvas típicas por circuito: "
            f"n_medidas={len(payload.medidas)}, n_max={payload.n_max}"
        )

        # Validaciones
        if not payload.medidas:
            raise ValueError("medidas no puede estar vacío")

        if payload.n_max < 1:
            raise ValueError("n_max debe ser mayor o igual a 1")

        # Contar circuitos únicos
        circuitos_unicos = set(m.circuito for m in payload.medidas)
        logger.info(f"Circuitos detectados: {len(circuitos_unicos)} ({', '.join(sorted(circuitos_unicos))})")

        # Convertir Pydantic models a dicts para procesamiento
        medidas_dict = [m.dict() for m in payload.medidas]

        # Procesar
        resultados = procesar_curvas_tipicas_por_circuito(medidas_dict, payload.n_max)

        # Convertir resultados a Pydantic models para response
        circuitos_response = {}
        for circuito, resultado in resultados.items():
            curvas_seleccionadas = [
                CurvaSeleccionadaOut(**curva)
                for curva in resultado["curvas_seleccionadas"]
            ]

            circuitos_response[circuito] = ResultadoCircuitoOut(
                curvas_seleccionadas=curvas_seleccionadas,
                promedio=resultado["promedio"],
                pesos=resultado["pesos"],
                n_curvas=resultado["n_curvas"]
            )

        # Logging de resumen
        for circuito, resultado in resultados.items():
            suma_pesos = sum(resultado["pesos"].values())
            logger.info(
                f"Circuito {circuito}: {resultado['n_curvas']} curvas, "
                f"suma_pesos={suma_pesos:.10f}"
            )
            if abs(suma_pesos - 1.0) > 1e-6:
                logger.warning(
                    f"Circuito {circuito}: suma de pesos {suma_pesos:.10f} != 1.0"
                )

        logger.info(f"Procesamiento exitoso: {len(resultados)} circuitos procesados")

        return CurvasTipicasCircuitosResponse(
            ok=True,
            circuitos=circuitos_response
        )

    except ValueError as e:
        logger.error(f"Error de validación: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error interno: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error interno en procesamiento de circuitos: {str(e)}"
        )


@router.get(
    "/circuitos/health",
    summary="Health check del servicio de circuitos",
    description="Verifica que el servicio de circuitos esté funcionando correctamente."
)
def health_check_circuitos():
    """Health check endpoint."""
    return {
        "ok": True,
        "service": "circuitos",
        "version": "1.0.0",
        "endpoint": "/factores/calculos/curvas-tipicas-circuitos"
    }
