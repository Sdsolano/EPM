"""
Servicio de cálculos para curvas típicas por circuito.

Este módulo NO realiza consultas a base de datos. Todo el procesamiento
se realiza sobre datos recibidos en la request.
"""
from typing import List, Dict, Any
import numpy as np

# Importar funciones reutilizables del servicio de cálculos existente
from app.services.calculos_service import (
    _seleccionar_curvas_tipicas,
    PERIODOS_COLUMNAS,
    PRECISION_DECIMALES
)


def _convertir_medida_a_curva(medida: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convierte una medida de circuito al formato esperado por _seleccionar_curvas_tipicas.

    Args:
        medida: Diccionario con codigo_rpm, circuito, fecha, flujo, p1-p24

    Returns:
        Diccionario con formato {barra, fecha, periodos}
        Usa 'codigo_rpm' como 'barra' para compatibilidad con funciones existentes
    """
    periodos = {f"p{i}": float(medida[f"p{i}"]) for i in range(1, 25)}

    return {
        "barra": medida["codigo_rpm"],  # Usar codigo_rpm como identificador
        "fecha": medida["fecha"],
        "periodos": periodos
    }


def _agrupar_medidas_por_circuito(medidas: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Agrupa medidas por circuito.

    Args:
        medidas: Lista de medidas con campo 'circuito'

    Returns:
        Diccionario {circuito: [lista de medidas]}
    """
    grupos = {}
    for medida in medidas:
        circuito = medida["circuito"]
        if circuito not in grupos:
            grupos[circuito] = []
        grupos[circuito].append(medida)

    return grupos


def _calcular_promedio_curvas(curvas: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calcula el promedio aritmético de las curvas seleccionadas.

    Args:
        curvas: Lista de curvas con formato {barra, fecha, periodos}

    Returns:
        Diccionario {p1: promedio, p2: promedio, ..., p24: promedio}
    """
    if not curvas:
        return {f"p{i}": 0.0 for i in range(1, 25)}

    # Extraer todos los periodos en matriz (n_curvas x 24)
    matriz = []
    for curva in curvas:
        periodos = curva.get("periodos", {})
        fila = [float(periodos.get(f"p{i}", 0)) for i in range(1, 25)]
        matriz.append(fila)

    # Calcular promedio por columna (por período)
    matriz_np = np.array(matriz)
    promedios = np.mean(matriz_np, axis=0)

    # Convertir a diccionario y redondear
    resultado = {
        f"p{i}": round(float(promedios[i-1]), PRECISION_DECIMALES)
        for i in range(1, 25)
    }

    return resultado


def _calcular_pesos_normalizados(promedio: Dict[str, float]) -> Dict[str, float]:
    """
    Calcula pesos normalizados: cada período / suma_total.
    Garantiza que sum(pesos) = 1.0 exactamente.

    Args:
        promedio: Diccionario {p1: valor, ..., p24: valor}

    Returns:
        Diccionario {p1: peso, ..., p24: peso} donde sum(pesos) = 1.0

    Algoritmo:
        1. Calcular suma total de todos los periodos
        2. Dividir cada periodo por la suma → peso_i = periodo_i / suma_total
        3. Redondear a PRECISION_DECIMALES
        4. Ajustar el valor máximo para garantizar suma exacta = 1.0
    """
    # Paso 1: Calcular suma total
    valores = [promedio[f"p{i}"] for i in range(1, 25)]
    suma_total = sum(valores)

    # Evitar división por cero
    if abs(suma_total) < 1e-10:
        # Si suma es 0, distribuir uniformemente
        return {f"p{i}": round(1.0 / 24, PRECISION_DECIMALES) for i in range(1, 25)}

    # Paso 2: Calcular pesos iniciales (normalizar)
    pesos = {}
    for i in range(1, 25):
        peso = promedio[f"p{i}"] / suma_total
        pesos[f"p{i}"] = round(peso, PRECISION_DECIMALES)

    # Paso 3: Calcular diferencia con 1.0 (causada por redondeo)
    suma_pesos = sum(pesos.values())
    diferencia = 1.0 - suma_pesos

    # Paso 4: Ajustar el valor máximo para que suma sea exactamente 1.0
    if abs(diferencia) > 1e-10:  # Solo ajustar si hay diferencia significativa
        # Encontrar el período con mayor peso
        periodo_max = max(pesos, key=pesos.get)
        # Aplicar ajuste
        pesos[periodo_max] = round(pesos[periodo_max] + diferencia, PRECISION_DECIMALES)

    return pesos


def procesar_curvas_tipicas_por_circuito(
    medidas: List[Dict[str, Any]],
    n_max: int
) -> Dict[str, Dict[str, Any]]:
    """
    Procesa curvas típicas agrupadas por circuito.

    Para cada circuito:
        1. Convierte medidas al formato esperado por _seleccionar_curvas_tipicas
        2. Selecciona hasta n_max curvas más típicas (usa algoritmo existente)
        3. Calcula promedio aritmético de las curvas seleccionadas
        4. Calcula pesos normalizados (suma = 1.0)

    Args:
        medidas: Lista de medidas con campos: codigo_rpm, circuito, fecha, flujo, p1-p24
        n_max: Número máximo de curvas típicas a seleccionar por circuito

    Returns:
        Diccionario {
            "CIRCUITO_A": {
                "curvas_seleccionadas": [...],
                "promedio": {p1: ..., p24: ...},
                "pesos": {p1: ..., p24: ...},
                "n_curvas": 8
            },
            ...
        }
    """
    # Agrupar medidas por circuito
    grupos = _agrupar_medidas_por_circuito(medidas)

    resultados = {}

    for circuito, medidas_circuito in grupos.items():
        # Convertir medidas al formato de curvas
        curvas = [_convertir_medida_a_curva(m) for m in medidas_circuito]

        # Seleccionar curvas típicas usando algoritmo existente
        # Esta función ya filtra outliers por IQR y selecciona las más centrales
        curvas_tipicas = _seleccionar_curvas_tipicas(curvas, n_max)

        if not curvas_tipicas:
            # Si no hay curvas, devolver valores vacíos
            resultados[circuito] = {
                "curvas_seleccionadas": [],
                "promedio": {f"p{i}": 0.0 for i in range(1, 25)},
                "pesos": {f"p{i}": round(1.0 / 24, PRECISION_DECIMALES) for i in range(1, 25)},
                "n_curvas": 0
            }
            continue

        # Calcular promedio de curvas seleccionadas
        promedio = _calcular_promedio_curvas(curvas_tipicas)

        # Calcular pesos normalizados
        pesos = _calcular_pesos_normalizados(promedio)

        # Formatear curvas seleccionadas para respuesta
        curvas_formateadas = []
        for curva in curvas_tipicas:
            curvas_formateadas.append({
                "codigo_rpm": curva["barra"],  # Recuperar codigo_rpm original
                "fecha": curva["fecha"],
                "periodos": curva["periodos"]
            })

        resultados[circuito] = {
            "curvas_seleccionadas": curvas_formateadas,
            "promedio": promedio,
            "pesos": pesos,
            "n_curvas": len(curvas_tipicas)
        }

    return resultados
