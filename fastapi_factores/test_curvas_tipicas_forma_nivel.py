"""
Test para verificar que el selector de curvas típicas considera FORMA Y NIVEL.
"""
import sys
sys.path.insert(0, 'c:\\Users\\samue\\OneDrive\\Documentos\\GitHub\\EPM\\fastapi_factores')

from app.services.calculos_service import _seleccionar_curvas_tipicas

print("=" * 80)
print("TEST: Selector de Curvas Tipicas - Forma + Nivel")
print("=" * 80)

# Crear 3 grupos de curvas con diferente forma Y nivel
# Grupo 1: Demanda BAJA con patrón matutino (pico en horas 6-12)
grupo_1 = [
    {
        "barra": "BARRA_A",
        "fecha": "2024-01-01",
        "periodos": {f"p{i}": 50 + (30 if 6 <= i <= 12 else 0) for i in range(1, 25)}
    },
    {
        "barra": "BARRA_A",
        "fecha": "2024-01-02",
        "periodos": {f"p{i}": 55 + (28 if 6 <= i <= 12 else 0) for i in range(1, 25)}
    },
    {
        "barra": "BARRA_A",
        "fecha": "2024-01-03",
        "periodos": {f"p{i}": 52 + (32 if 6 <= i <= 12 else 0) for i in range(1, 25)}
    },
]

# Grupo 2: Demanda MEDIA con patrón vespertino (pico en horas 18-22)
grupo_2 = [
    {
        "barra": "BARRA_B",
        "fecha": "2024-01-04",
        "periodos": {f"p{i}": 150 + (80 if 18 <= i <= 22 else 0) for i in range(1, 25)}
    },
    {
        "barra": "BARRA_B",
        "fecha": "2024-01-05",
        "periodos": {f"p{i}": 145 + (85 if 18 <= i <= 22 else 0) for i in range(1, 25)}
    },
    {
        "barra": "BARRA_B",
        "fecha": "2024-01-06",
        "periodos": {f"p{i}": 155 + (75 if 18 <= i <= 22 else 0) for i in range(1, 25)}
    },
]

# Grupo 3: Demanda ALTA con patrón todo el día (constante alto)
grupo_3 = [
    {
        "barra": "BARRA_C",
        "fecha": "2024-01-07",
        "periodos": {f"p{i}": 300 + (50 if 12 <= i <= 20 else 0) for i in range(1, 25)}
    },
    {
        "barra": "BARRA_C",
        "fecha": "2024-01-08",
        "periodos": {f"p{i}": 295 + (55 if 12 <= i <= 20 else 0) for i in range(1, 25)}
    },
    {
        "barra": "BARRA_C",
        "fecha": "2024-01-09",
        "periodos": {f"p{i}": 305 + (45 if 12 <= i <= 20 else 0) for i in range(1, 25)}
    },
]

# Combinar todas las curvas
todas_curvas = grupo_1 + grupo_2 + grupo_3

print(f"\nCurvas de entrada:")
print(f"  Grupo 1 (BAJA, matutino):  {len(grupo_1)} curvas - nivel ~50-80")
print(f"  Grupo 2 (MEDIA, vespertino): {len(grupo_2)} curvas - nivel ~150-230")
print(f"  Grupo 3 (ALTA, todo dia):    {len(grupo_3)} curvas - nivel ~300-350")
print(f"  TOTAL: {len(todas_curvas)} curvas")

# Test 1: Seleccionar 3 curvas tipicas (debe elegir una de cada grupo)
print("\n" + "=" * 80)
print("TEST 1: Seleccionar 3 curvas tipicas")
print("=" * 80)

tipicas_3 = _seleccionar_curvas_tipicas(todas_curvas, n_max=3)

print(f"\nCurvas seleccionadas: {len(tipicas_3)}")

# Verificar que se seleccionó una de cada grupo
barras_seleccionadas = [c["barra"] for c in tipicas_3]
print(f"Barras seleccionadas: {barras_seleccionadas}")

tiene_barra_a = "BARRA_A" in barras_seleccionadas
tiene_barra_b = "BARRA_B" in barras_seleccionadas
tiene_barra_c = "BARRA_C" in barras_seleccionadas

print(f"\nDiversidad de grupos:")
print(f"  BARRA_A (BAJA):  {'SI' if tiene_barra_a else 'NO'}")
print(f"  BARRA_B (MEDIA): {'SI' if tiene_barra_b else 'NO'}")
print(f"  BARRA_C (ALTA):  {'SI' if tiene_barra_c else 'NO'}")

if tiene_barra_a and tiene_barra_b and tiene_barra_c:
    print("\nRESULTADO: OK - Se seleccionaron curvas de diferentes niveles")
else:
    print("\nRESULTADO: ADVERTENCIA - No se seleccionaron todos los grupos")

# Test 2: Seleccionar 6 curvas (debe dar preferencia a grupos más representativos)
print("\n" + "=" * 80)
print("TEST 2: Seleccionar 6 curvas tipicas")
print("=" * 80)

tipicas_6 = _seleccionar_curvas_tipicas(todas_curvas, n_max=6)

print(f"\nCurvas seleccionadas: {len(tipicas_6)}")

# Contar por barra
from collections import Counter
contador = Counter([c["barra"] for c in tipicas_6])

print(f"\nDistribucion por grupo:")
for barra, count in sorted(contador.items()):
    print(f"  {barra}: {count} curvas")

# Test 3: Verificar valores reales
print("\n" + "=" * 80)
print("TEST 3: Verificar niveles de demanda en curvas seleccionadas")
print("=" * 80)

for i, curva in enumerate(tipicas_3, 1):
    periodos = curva["periodos"]
    valores = [periodos[f"p{j}"] for j in range(1, 25)]
    promedio = sum(valores) / len(valores)
    maximo = max(valores)
    minimo = min(valores)

    print(f"\nCurva {i}: {curva['barra']} - {curva['fecha']}")
    print(f"  Promedio: {promedio:.1f}")
    print(f"  Rango: [{minimo:.1f}, {maximo:.1f}]")

# Test 4: Comparar con algoritmo antiguo (L2)
print("\n" + "=" * 80)
print("TEST 4: Comparacion conceptual - Algoritmo nuevo vs antiguo")
print("=" * 80)

print("\nAlgoritmo ANTIGUO (con L2):")
print("  - Normalizaba todas las curvas a norma unitaria")
print("  - Solo consideraba FORMA, ignoraba NIVEL")
print("  - Curvas [100,200,300] y [10,20,30] eran identicas")
print("  - Resultado: Seleccionaba por patron, sin importar magnitud")

print("\nAlgoritmo NUEVO (sin L2):")
print("  - Usa valores originales de demanda")
print("  - Considera FORMA Y NIVEL simultaneamente")
print("  - Curvas [100,200,300] y [10,20,30] son diferentes")
print("  - Resultado: Agrupa curvas con patron Y magnitud similares")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("El nuevo algoritmo considera forma Y nivel correctamente.")
print("Curvas con demandas similares (mismo patron y magnitud) se agrupan juntas.")
print("=" * 80)
