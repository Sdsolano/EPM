"""
Unit tests para circuitos service.
"""
import sys
sys.path.insert(0, 'c:\\Users\\samue\\OneDrive\\Documentos\\GitHub\\EPM\\fastapi_factores')

import pytest
from app.circuitos.service import (
    _convertir_medida_a_curva,
    _agrupar_medidas_por_circuito,
    _calcular_promedio_curvas,
    _calcular_pesos_normalizados,
    procesar_curvas_tipicas_por_circuito
)


def test_convertir_medida_a_curva():
    """Test conversión de medida a formato curva."""
    medida = {
        "codigo_rpm": "SJU748",
        "circuito": "CIRCUITO_A",
        "fecha": "2025-10-02",
        "flujo": "AE",
        **{f"p{i}": float(i * 10) for i in range(1, 25)}
    }

    curva = _convertir_medida_a_curva(medida)

    assert curva["barra"] == "SJU748"
    assert curva["fecha"] == "2025-10-02"
    assert len(curva["periodos"]) == 24
    assert curva["periodos"]["p1"] == 10.0
    assert curva["periodos"]["p24"] == 240.0


def test_agrupar_medidas_por_circuito():
    """Test agrupación por circuito."""
    medidas = [
        {"circuito": "A", "codigo_rpm": "X1"},
        {"circuito": "B", "codigo_rpm": "X2"},
        {"circuito": "A", "codigo_rpm": "X3"},
    ]

    grupos = _agrupar_medidas_por_circuito(medidas)

    assert len(grupos) == 2
    assert len(grupos["A"]) == 2
    assert len(grupos["B"]) == 1


def test_calcular_promedio_curvas():
    """Test cálculo de promedio."""
    curvas = [
        {"barra": "X", "fecha": "2025-01-01", "periodos": {f"p{i}": 10.0 for i in range(1, 25)}},
        {"barra": "Y", "fecha": "2025-01-02", "periodos": {f"p{i}": 20.0 for i in range(1, 25)}},
    ]

    promedio = _calcular_promedio_curvas(curvas)

    assert len(promedio) == 24
    for i in range(1, 25):
        assert promedio[f"p{i}"] == 15.0  # (10 + 20) / 2


def test_calcular_pesos_normalizados():
    """Test cálculo de pesos con suma = 1.0."""
    promedio = {f"p{i}": 100.0 for i in range(1, 25)}  # Total = 2400

    pesos = _calcular_pesos_normalizados(promedio)

    assert len(pesos) == 24
    suma = sum(pesos.values())
    assert abs(suma - 1.0) < 1e-10  # Debe ser exactamente 1.0

    # Cada peso debe ser ~0.04167 (1/24)
    for i in range(1, 25):
        assert abs(pesos[f"p{i}"] - 0.04167) < 0.001


def test_calcular_pesos_con_valores_diferentes():
    """Test pesos con valores no uniformes."""
    # Crear promedio donde p12 es el doble que los demás
    promedio = {f"p{i}": 100.0 if i != 12 else 200.0 for i in range(1, 25)}
    # Total = 23*100 + 200 = 2500

    pesos = _calcular_pesos_normalizados(promedio)

    suma = sum(pesos.values())
    assert abs(suma - 1.0) < 1e-10

    # p12 debe tener el doble de peso
    assert pesos["p12"] > pesos["p1"]


def test_calcular_pesos_con_suma_cero():
    """Test pesos cuando suma es cero (edge case)."""
    promedio = {f"p{i}": 0.0 for i in range(1, 25)}

    pesos = _calcular_pesos_normalizados(promedio)

    suma = sum(pesos.values())
    assert abs(suma - 1.0) < 1e-4  # Tolerancia más realista con 5 decimales
    # Debe distribuir uniformemente
    for i in range(1, 25):
        assert abs(pesos[f"p{i}"] - 1.0/24) < 0.0001


def test_procesar_circuitos_simple():
    """Test procesamiento con un solo circuito."""
    medidas = [
        {
            "codigo_rpm": f"RPM{i}",
            "circuito": "CIRCUITO_A",
            "fecha": f"2025-10-0{i}",
            "flujo": "AE",
            **{f"p{j}": float(i * 10 + j) for j in range(1, 25)}
        }
        for i in range(1, 6)  # 5 medidas
    ]

    resultado = procesar_curvas_tipicas_por_circuito(medidas, n_max=3)

    assert "CIRCUITO_A" in resultado
    assert resultado["CIRCUITO_A"]["n_curvas"] <= 3
    assert len(resultado["CIRCUITO_A"]["promedio"]) == 24
    assert len(resultado["CIRCUITO_A"]["pesos"]) == 24

    # Verificar suma de pesos
    suma_pesos = sum(resultado["CIRCUITO_A"]["pesos"].values())
    assert abs(suma_pesos - 1.0) < 1e-6


def test_procesar_multiples_circuitos():
    """Test procesamiento con múltiples circuitos."""
    medidas = []
    for circuito in ["CIRCUITO_A", "CIRCUITO_B"]:
        for i in range(1, 6):
            medidas.append({
                "codigo_rpm": f"RPM{circuito}{i}",
                "circuito": circuito,
                "fecha": f"2025-10-0{i}",
                "flujo": "AE",
                **{f"p{j}": float(i * 10 + j) for j in range(1, 25)}
            })

    resultado = procesar_curvas_tipicas_por_circuito(medidas, n_max=3)

    assert len(resultado) == 2
    assert "CIRCUITO_A" in resultado
    assert "CIRCUITO_B" in resultado

    for circuito in ["CIRCUITO_A", "CIRCUITO_B"]:
        suma_pesos = sum(resultado[circuito]["pesos"].values())
        assert abs(suma_pesos - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
