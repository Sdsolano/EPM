"""
Test manual del endpoint usando TestClient.
"""
import sys
sys.path.insert(0, 'c:\\Users\\samue\\OneDrive\\Documentos\\GitHub\\EPM\\fastapi_factores')

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Test 1: Health check
print("=" * 80)
print("Test 1: Health Check")
print("=" * 80)
response = client.get("/factores/calculos/circuitos/health")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

# Test 2: Endpoint con datos de ejemplo
print("\n" + "=" * 80)
print("Test 2: Endpoint con datos de ejemplo")
print("=" * 80)

payload = {
    "medidas": [
        {
            "codigo_rpm": f"RPM{i}",
            "circuito": "CIRCUITO_TEST",
            "ucp": "PRIMEGRID",
            "fecha": f"2025-10-0{i}",
            "flujo": "AE",
            **{f"p{j}": float(i * 10 + j) for j in range(1, 25)}
        }
        for i in range(1, 6)
    ],
    "n_max": 3
}

response = client.post("/factores/calculos/curvas-tipicas-circuitos", json=payload)
print(f"Status: {response.status_code}")
data = response.json()
print(f"OK: {data['ok']}")
print(f"Circuitos procesados: {list(data['circuitos'].keys())}")

circuito_test = data['circuitos']['CIRCUITO_TEST']
print(f"\nCIRCUITO_TEST:")
print(f"  Curvas seleccionadas: {circuito_test['n_curvas']}")
print(f"  Suma de pesos: {sum(circuito_test['pesos'].values()):.15f}")

# Verificar suma exacta
suma = sum(circuito_test['pesos'].values())
if abs(suma - 1.0) < 1e-10:
    print(f"  [OK] Suma de pesos = 1.0")
else:
    print(f"  [WARN] Suma de pesos = {suma:.15f}")

print("\n" + "=" * 80)
print("TESTS COMPLETADOS EXITOSAMENTE")
print("=" * 80)
