"""
Integration tests para circuitos endpoint.
"""
import sys
sys.path.insert(0, 'c:\\Users\\samue\\OneDrive\\Documentos\\GitHub\\EPM\\fastapi_factores')

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/factores/calculos/circuitos/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["service"] == "circuitos"


def test_endpoint_with_single_circuito():
    """Test endpoint con un solo circuito."""
    payload = {
        "medidas": [
            {
                "codigo_rpm": f"RPM{i}",
                "circuito": "CIRCUITO_A",
                "ucp": "PRIMEGRID",
                "fecha": f"2025-10-0{i}",
                "flujo": "AE",
                **{f"p{j}": i * 10.0 + j for j in range(1, 25)}
            }
            for i in range(1, 6)
        ],
        "n_max": 3
    }

    response = client.post("/factores/calculos/curvas-tipicas-circuitos", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["ok"] is True
    assert "CIRCUITO_A" in data["circuitos"]

    circuito_a = data["circuitos"]["CIRCUITO_A"]
    assert circuito_a["n_curvas"] <= 3
    assert len(circuito_a["curvas_seleccionadas"]) <= 3
    assert len(circuito_a["promedio"]) == 24
    assert len(circuito_a["pesos"]) == 24

    # Verificar suma de pesos
    suma = sum(circuito_a["pesos"].values())
    assert abs(suma - 1.0) < 1e-6


def test_endpoint_with_multiple_circuitos():
    """Test endpoint con múltiples circuitos."""
    medidas = []
    for circuito in ["CIRCUITO_A", "CIRCUITO_B", "CIRCUITO_C"]:
        for i in range(1, 6):
            medidas.append({
                "codigo_rpm": f"RPM_{circuito}_{i}",
                "circuito": circuito,
                "ucp": "PRIMEGRID",
                "fecha": f"2025-10-0{i}",
                "flujo": "AE",
                **{f"p{j}": i * 10.0 + j for j in range(1, 25)}
            })

    payload = {"medidas": medidas, "n_max": 3}

    response = client.post("/factores/calculos/curvas-tipicas-circuitos", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert len(data["circuitos"]) == 3

    for circuito in ["CIRCUITO_A", "CIRCUITO_B", "CIRCUITO_C"]:
        assert circuito in data["circuitos"]
        suma = sum(data["circuitos"][circuito]["pesos"].values())
        assert abs(suma - 1.0) < 1e-6


def test_endpoint_validation_empty_medidas():
    """Test validación con medidas vacías."""
    payload = {"medidas": [], "n_max": 3}

    response = client.post("/factores/calculos/curvas-tipicas-circuitos", json=payload)
    assert response.status_code == 400


def test_endpoint_validation_invalid_n_max():
    """Test validación con n_max inválido."""
    payload = {
        "medidas": [
            {
                "codigo_rpm": "RPM1",
                "circuito": "A",
                "ucp": "PRIMEGRID",
                "fecha": "2025-10-01",
                "flujo": "AE",
                **{f"p{i}": 10.0 for i in range(1, 25)}
            }
        ],
        "n_max": 0  # Inválido
    }

    response = client.post("/factores/calculos/curvas-tipicas-circuitos", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_endpoint_with_n_max_greater_than_available():
    """Test cuando n_max > curvas disponibles."""
    payload = {
        "medidas": [
            {
                "codigo_rpm": "RPM1",
                "circuito": "CIRCUITO_A",
                "ucp": "PRIMEGRID",
                "fecha": "2025-10-01",
                "flujo": "AE",
                **{f"p{i}": 10.0 for i in range(1, 25)}
            },
            {
                "codigo_rpm": "RPM2",
                "circuito": "CIRCUITO_A",
                "ucp": "PRIMEGRID",
                "fecha": "2025-10-02",
                "flujo": "AE",
                **{f"p{i}": 20.0 for i in range(1, 25)}
            }
        ],
        "n_max": 10  # Más que las 2 disponibles
    }

    response = client.post("/factores/calculos/curvas-tipicas-circuitos", json=payload)
    assert response.status_code == 200

    data = response.json()
    # Debe devolver solo 2 curvas
    assert data["circuitos"]["CIRCUITO_A"]["n_curvas"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
