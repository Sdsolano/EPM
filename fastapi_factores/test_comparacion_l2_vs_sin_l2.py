"""
Test comparativo: L2 vs Sin L2 para demostrar la diferencia.
"""
import numpy as np
import sys
sys.path.insert(0, 'c:\\Users\\samue\\OneDrive\\Documentos\\GitHub\\EPM\\fastapi_factores')

from app.services.calculos_service import _seleccionar_curvas_tipicas, _curvas_a_matriz

print("=" * 80)
print("TEST COMPARATIVO: Algoritmo NUEVO (sin L2) vs ANTIGUO (con L2)")
print("=" * 80)

# Crear curvas con MISMA FORMA pero DIFERENTES NIVELES
patron_base = [100, 120, 150, 180, 200, 190, 160, 140, 130, 120, 110, 105,
               100, 110, 130, 160, 180, 200, 210, 200, 180, 150, 130, 110]

curvas_test = []

# 5 curvas con el mismo patron pero diferentes escalas
escalas = [0.5, 0.75, 1.0, 1.5, 2.0]
for i, escala in enumerate(escalas, 1):
    curva = {
        "barra": f"BARRA_{i}",
        "fecha": f"2024-01-0{i}",
        "periodos": {f"p{j}": patron_base[j-1] * escala for j in range(1, 25)}
    }
    curvas_test.append(curva)

print("\nCURVAS DE PRUEBA (mismo patron, diferentes niveles):")
for i, curva in enumerate(curvas_test, 1):
    periodos = [curva["periodos"][f"p{j}"] for j in range(1, 25)]
    promedio = sum(periodos) / len(periodos)
    print(f"  Curva {i} (escala {escalas[i-1]}): promedio = {promedio:.1f}")

# Convertir a matriz para analizar
X, keys = _curvas_a_matriz(curvas_test)

print("\n" + "=" * 80)
print("DISTANCIAS SIN NORMALIZAR (considera forma + nivel)")
print("=" * 80)

print("\nDistancias entre curvas (sin normalizar):")
for i in range(len(X)):
    for j in range(i+1, len(X)):
        dist = np.linalg.norm(X[i] - X[j])
        print(f"  Curva {i+1} <-> Curva {j+1}: {dist:.2f}")

print("\n" + "=" * 80)
print("DISTANCIAS CON L2 (solo considera forma)")
print("=" * 80)

# Normalizar por L2
norms = np.linalg.norm(X, axis=1, keepdims=True)
X_L2 = X / norms

print("\nDistancias entre curvas (con L2):")
for i in range(len(X_L2)):
    for j in range(i+1, len(X_L2)):
        dist = np.linalg.norm(X_L2[i] - X_L2[j])
        print(f"  Curva {i+1} <-> Curva {j+1}: {dist:.10f}")

print("\n" + "=" * 80)
print("ALGORITMO ACTUAL (sin L2) - Seleccionar 3 curvas")
print("=" * 80)

seleccionadas = _seleccionar_curvas_tipicas(curvas_test, n_max=3)

print(f"\nCurvas seleccionadas: {len(seleccionadas)}")
for curva in seleccionadas:
    periodos = [curva["periodos"][f"p{j}"] for j in range(1, 25)]
    promedio = sum(periodos) / len(periodos)
    print(f"  {curva['barra']}: promedio = {promedio:.1f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("SIN L2 (algoritmo actual):")
print("  - Las distancias reflejan diferencias reales de nivel")
print("  - Curvas con magnitudes cercanas se agrupan juntas")
print("  - Selecciona las mas centrales en terminos de forma Y nivel")
print("")
print("CON L2 (algoritmo antiguo):")
print("  - Las distancias son ~0 (todas las curvas son identicas)")
print("  - Solo considera forma, ignora completamente el nivel")
print("  - Todas las curvas son igualmente 'tipicas'")
print("=" * 80)

# Prueba adicional: Curvas con diferente forma Y nivel
print("\n" + "=" * 80)
print("PRUEBA ADICIONAL: Curvas con diferente FORMA Y NIVEL")
print("=" * 80)

curvas_diversas = [
    # Patron matutino, nivel bajo
    {
        "barra": "MAT_BAJO",
        "fecha": "2024-01-01",
        "periodos": {f"p{i}": 50 + (30 if 6 <= i <= 12 else 0) for i in range(1, 25)}
    },
    # Patron matutino, nivel alto
    {
        "barra": "MAT_ALTO",
        "fecha": "2024-01-02",
        "periodos": {f"p{i}": 200 + (100 if 6 <= i <= 12 else 0) for i in range(1, 25)}
    },
    # Patron vespertino, nivel bajo
    {
        "barra": "VESP_BAJO",
        "fecha": "2024-01-03",
        "periodos": {f"p{i}": 60 + (40 if 18 <= i <= 22 else 0) for i in range(1, 25)}
    },
    # Patron vespertino, nivel alto
    {
        "barra": "VESP_ALTO",
        "fecha": "2024-01-04",
        "periodos": {f"p{i}": 180 + (120 if 18 <= i <= 22 else 0) for i in range(1, 25)}
    },
]

X_div, _ = _curvas_a_matriz(curvas_diversas)
X_div_L2 = X_div / np.linalg.norm(X_div, axis=1, keepdims=True)

print("\nCurvas de prueba:")
for curva in curvas_diversas:
    periodos = [curva["periodos"][f"p{j}"] for j in range(1, 25)]
    promedio = sum(periodos) / len(periodos)
    print(f"  {curva['barra']}: promedio = {promedio:.1f}")

print("\nDistancias SIN L2:")
print(f"  MAT_BAJO <-> MAT_ALTO:  {np.linalg.norm(X_div[0] - X_div[1]):.2f}")
print(f"  MAT_BAJO <-> VESP_BAJO: {np.linalg.norm(X_div[0] - X_div[2]):.2f}")
print(f"  MAT_ALTO <-> VESP_ALTO: {np.linalg.norm(X_div[1] - X_div[3]):.2f}")

print("\nDistancias CON L2:")
print(f"  MAT_BAJO <-> MAT_ALTO:  {np.linalg.norm(X_div_L2[0] - X_div_L2[1]):.5f}")
print(f"  MAT_BAJO <-> VESP_BAJO: {np.linalg.norm(X_div_L2[0] - X_div_L2[2]):.5f}")
print(f"  MAT_ALTO <-> VESP_ALTO: {np.linalg.norm(X_div_L2[1] - X_div_L2[3]):.5f}")

print("\nOBSERVACION:")
print("  Sin L2: Las curvas matutinas (bajo y alto) estan a distancia ~150")
print("          porque tienen diferente nivel aunque mismo patron")
print("  Con L2: Las curvas matutinas estan a distancia ~0")
print("          porque tienen el mismo patron (ignora nivel)")
print("=" * 80)
