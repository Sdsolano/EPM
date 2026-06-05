"""
Test para verificar FDA con datos de 2 meses (caso real reportado).
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'c:\\Users\\samue\\OneDrive\\Documentos\\GitHub\\EPM\\fastapi_factores')

from app.services.calculos_service import _calcular_fda_normalizado, PERIODOS_COLUMNAS

# Simular 60 días (2 meses) con datos realistas de demanda eléctrica
np.random.seed(42)
n_dias = 60

# Generar curvas de demanda con patrón diario realista
data = {}
for i, col in enumerate(PERIODOS_COLUMNAS):
    hora = i + 1
    # Patrón de demanda típico: bajo en madrugada, alto mediodía y tarde
    if 1 <= hora <= 6:  # Madrugada: baja demanda
        base = np.random.uniform(50, 100, n_dias)
    elif 7 <= hora <= 12:  # Mañana: demanda creciente
        base = np.random.uniform(150, 250, n_dias)
    elif 13 <= hora <= 18:  # Tarde: demanda alta
        base = np.random.uniform(200, 300, n_dias)
    else:  # Noche: demanda decreciente
        base = np.random.uniform(100, 200, n_dias)

    # Agregar variabilidad
    data[col] = base + np.random.normal(0, 10, n_dias)

df = pd.DataFrame(data)

print("=" * 80)
print("TEST FDA: 2 MESES DE DATOS (60 días)")
print("=" * 80)
print(f"\nDataFrame de entrada: {n_dias} curvas (días) x 24 períodos")
print(f"Shape: {df.shape}")
print("\nPrimeras 5 curvas:")
print(df.head())
print("\nEstadísticas:")
print(df.describe())

# Aplicar FDA
print("\n" + "=" * 80)
print("Aplicando algoritmo FDA...")
print("=" * 80)

df_fda = _calcular_fda_normalizado(df)

print("\nResultado FDA (primeras 5 curvas):")
print(df_fda.head())

# VERIFICACIÓN CRÍTICA: Cada período debe sumar exactamente 1.0
print("\n" + "=" * 80)
print("VERIFICACION CRITICA: Suma por período")
print("=" * 80)

sumas = df_fda[PERIODOS_COLUMNAS].sum()
errores = []
max_error = 0.0

for col in PERIODOS_COLUMNAS:
    suma = sumas[col]
    error = abs(1.0 - suma)
    max_error = max(max_error, error)

    es_correcto = error < 1e-10
    if not es_correcto:
        errores.append((col, suma, error))

    status = "OK" if es_correcto else "ERROR"
    print(f"{col}: {suma:.15f} | Error: {error:.2e} | {status}")

# Resultado final
print("\n" + "=" * 80)
print("RESULTADO FINAL")
print("=" * 80)

if len(errores) == 0:
    print(f"EXITO! Todos los {len(PERIODOS_COLUMNAS)} periodos suman exactamente 1.0")
    print(f"Error maximo encontrado: {max_error:.2e}")
    print("\nEl problema reportado de ±0.01 esta RESUELTO")
else:
    print(f"ERROR: {len(errores)} periodos no suman 1.0:")
    for col, suma, error in errores:
        print(f"  {col}: suma={suma:.15f}, error={error:.5f}")
    print("\nEl problema persiste")

# Suma total debe ser 24.0
suma_total = sumas.sum()
print(f"\nSuma total: {suma_total:.15f} (esperado: 24.0)")
print(f"Diferencia con 24.0: {abs(24.0 - suma_total):.2e}")

# Verificar que cada fila sea válida (valores entre 0 y 1)
print("\n" + "=" * 80)
print("VALIDACION ADICIONAL")
print("=" * 80)

min_valor = df_fda[PERIODOS_COLUMNAS].min().min()
max_valor = df_fda[PERIODOS_COLUMNAS].max().max()
print(f"Valor minimo en FDA: {min_valor:.10f}")
print(f"Valor maximo en FDA: {max_valor:.10f}")
print(f"Todos los valores entre 0 y 1: {'SI' if 0 <= min_valor and max_valor <= 1 else 'NO'}")

# Verificar precisión decimal
print(f"\nPrecision de decimales: 5")
print(f"Todos los valores tienen max 5 decimales: ", end="")
todo_5_decimales = True
for col in PERIODOS_COLUMNAS:
    for val in df_fda[col]:
        # Verificar que al redondear a 5 decimales no cambie
        if abs(val - round(val, 5)) > 1e-10:
            todo_5_decimales = False
            break
    if not todo_5_decimales:
        break
print("SI" if todo_5_decimales else "NO")

print("\n" + "=" * 80)
