"""
Test para verificar que el FDA suma exactamente 1.0 después del fix.
"""
import pandas as pd
import sys
sys.path.insert(0, 'c:\\Users\\samue\\OneDrive\\Documentos\\GitHub\\EPM\\fastapi_factores')

from app.services.calculos_service import _calcular_fda_normalizado, PERIODOS_COLUMNAS

# Crear datos de prueba con valores que causan problemas de redondeo
# Simulamos 3 curvas con valores que al redondear causan problemas
data = {
    'p1': [100.0, 200.0, 150.0],
    'p2': [150.0, 250.0, 175.0],
    'p3': [200.0, 300.0, 225.0],
    'p4': [175.0, 275.0, 200.0],
    'p5': [125.0, 225.0, 160.0],
    'p6': [110.0, 210.0, 155.0],
    'p7': [105.0, 205.0, 152.0],
    'p8': [120.0, 220.0, 165.0],
    'p9': [140.0, 240.0, 180.0],
    'p10': [160.0, 260.0, 195.0],
    'p11': [180.0, 280.0, 210.0],
    'p12': [190.0, 290.0, 220.0],
    'p13': [195.0, 295.0, 222.0],
    'p14': [185.0, 285.0, 215.0],
    'p15': [170.0, 270.0, 205.0],
    'p16': [155.0, 255.0, 190.0],
    'p17': [145.0, 245.0, 185.0],
    'p18': [135.0, 235.0, 172.0],
    'p19': [130.0, 230.0, 168.0],
    'p20': [115.0, 215.0, 158.0],
    'p21': [108.0, 208.0, 153.0],
    'p22': [103.0, 203.0, 151.0],
    'p23': [98.0, 198.0, 148.0],
    'p24': [95.0, 195.0, 145.0],
}

df = pd.DataFrame(data)

print("=" * 80)
print("TEST: Verificación de suma FDA = 1.0")
print("=" * 80)
print(f"\nDataFrame de entrada (shape: {df.shape}):")
print(df.head())

# Aplicar FDA
df_fda = _calcular_fda_normalizado(df)

print("\n" + "=" * 80)
print("RESULTADOS FDA:")
print("=" * 80)
print(df_fda.head())

# Verificar sumas por período
print("\n" + "=" * 80)
print("VERIFICACIÓN: Suma por período (debe ser 1.0 para todos)")
print("=" * 80)

sumas = df_fda[PERIODOS_COLUMNAS].sum()
errores = []
todo_correcto = True

for col in PERIODOS_COLUMNAS:
    suma = sumas[col]
    diferencia = abs(1.0 - suma)
    es_correcto = diferencia < 1e-10

    if not es_correcto:
        todo_correcto = False
        errores.append((col, suma, diferencia))

    status = "OK" if es_correcto else "ERROR"
    print(f"{col}: {suma:.15f} | Diferencia: {diferencia:.2e} | {status}")

print("\n" + "=" * 80)
if todo_correcto:
    print("EXITO! Todos los periodos suman exactamente 1.0")
else:
    print(f"ERROR: {len(errores)} periodos no suman 1.0:")
    for col, suma, dif in errores:
        print(f"  {col}: suma={suma}, diferencia={dif}")
print("=" * 80)

# Verificar suma total
suma_total = df_fda[PERIODOS_COLUMNAS].sum().sum()
print(f"\nSuma total de todos los valores: {suma_total:.10f}")
print(f"Esperado: {len(PERIODOS_COLUMNAS)}.0 (24.0)")
print(f"Diferencia: {abs(24.0 - suma_total):.2e}")

if abs(24.0 - suma_total) < 1e-9:
    print("Suma total correcta")
else:
    print("Suma total incorrecta")
