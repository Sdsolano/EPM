"""
Análisis del selector de curvas típicas actual vs. lo que debería hacer.
"""
import numpy as np

print("=" * 80)
print("ANÁLISIS: Selector de Curvas Típicas")
print("=" * 80)

# Simular 3 curvas con MISMA FORMA pero DIFERENTE NIVEL
curva_1 = np.array([100, 150, 200, 180, 160, 140, 120, 110])
curva_2 = np.array([200, 300, 400, 360, 320, 280, 240, 220])  # Curva 1 × 2
curva_3 = np.array([50, 75, 100, 90, 80, 70, 60, 55])         # Curva 1 × 0.5

print("\n1. CURVAS ORIGINALES (misma forma, diferente nivel):")
print(f"   Curva 1: {curva_1}")
print(f"   Curva 2: {curva_2} (Curva 1 × 2)")
print(f"   Curva 3: {curva_3} (Curva 1 × 0.5)")

# Normalizar por L2 (como hace el código actual)
norm_1 = np.linalg.norm(curva_1)
norm_2 = np.linalg.norm(curva_2)
norm_3 = np.linalg.norm(curva_3)

curva_1_L2 = curva_1 / norm_1
curva_2_L2 = curva_2 / norm_2
curva_3_L2 = curva_3 / norm_3

print(f"\n2. DESPUÉS DE NORMALIZACIÓN L2:")
print(f"   Norma Curva 1: {norm_1:.2f}")
print(f"   Norma Curva 2: {norm_2:.2f}")
print(f"   Norma Curva 3: {norm_3:.2f}")
print(f"\n   Curva 1 normalizada: {curva_1_L2}")
print(f"   Curva 2 normalizada: {curva_2_L2}")
print(f"   Curva 3 normalizada: {curva_3_L2}")

# Calcular distancias después de L2
dist_1_2_L2 = np.linalg.norm(curva_1_L2 - curva_2_L2)
dist_1_3_L2 = np.linalg.norm(curva_1_L2 - curva_3_L2)

print(f"\n3. DISTANCIAS DESPUÉS DE L2:")
print(f"   Distancia(Curva1, Curva2): {dist_1_2_L2:.10f}")
print(f"   Distancia(Curva1, Curva3): {dist_1_3_L2:.10f}")
print(f"\n   PROBLEMA: Las distancias son ~0 aunque los niveles son muy diferentes!")

# Distancias SIN normalización (para comparar forma + nivel)
dist_1_2_raw = np.linalg.norm(curva_1 - curva_2)
dist_1_3_raw = np.linalg.norm(curva_1 - curva_3)

print(f"\n4. DISTANCIAS SIN NORMALIZACIÓN (forma + nivel):")
print(f"   Distancia(Curva1, Curva2): {dist_1_2_raw:.2f}")
print(f"   Distancia(Curva1, Curva3): {dist_1_3_raw:.2f}")
print(f"\n   OK - Ahora si refleja las diferencias de nivel")

print("\n" + "=" * 80)
print("CONCLUSIÓN:")
print("=" * 80)
print("• El algoritmo actual (con L2) SOLO compara FORMA, ignora NIVEL")
print("• Curvas con misma forma pero diferente magnitud son consideradas idénticas")
print("• Si necesitas forma + nivel, NO debes normalizar por L2")
print("=" * 80)

# Ahora probemos curvas con diferente FORMA
print("\n" + "=" * 80)
print("PRUEBA 2: Curvas con DIFERENTE FORMA")
print("=" * 80)

curva_a = np.array([100, 150, 200, 180, 160, 140, 120, 110])  # Pico en medio
curva_b = np.array([110, 120, 140, 160, 180, 200, 150, 100])  # Pico al final

print(f"\nCurva A: {curva_a} (pico en medio)")
print(f"Curva B: {curva_b} (pico al final)")

# Normalizar por L2
curva_a_L2 = curva_a / np.linalg.norm(curva_a)
curva_b_L2 = curva_b / np.linalg.norm(curva_b)

dist_ab_L2 = np.linalg.norm(curva_a_L2 - curva_b_L2)
dist_ab_raw = np.linalg.norm(curva_a - curva_b)

print(f"\nDistancia con L2: {dist_ab_L2:.5f}")
print(f"Distancia sin L2: {dist_ab_raw:.2f}")
print("\nOK - L2 si detecta diferencias de FORMA correctamente")

print("\n" + "=" * 80)
