"""
Test de integración con datos reales de PRIMEGRID.csv.
"""
import sys
sys.path.insert(0, 'c:\\Users\\samue\\OneDrive\\Documentos\\GitHub\\EPM\\fastapi_factores')

import csv
from app.circuitos.service import procesar_curvas_tipicas_por_circuito

print("=" * 80)
print("TEST: Curvas Típicas por Circuito con PRIMEGRID.csv")
print("=" * 80)

# Leer PRIMEGRID.csv
csv_path = "c:\\Users\\samue\\OneDrive\\Documentos\\GitHub\\EPM\\fastapi_factores\\PRIMEGRID.csv"

medidas = []
with open(csv_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Mapear UCP a circuito (para este test)
        # En producción, circuito vendría en los datos
        codigo_rpm = row['Cod. RPM']
        fecha = row['Fecha']
        flujo = row['Flujo']

        # Crear circuito basado en código RPM (simplificación para test)
        if codigo_rpm.startswith('SJU'):
            circuito = 'CIRCUITO_SJU'
        elif codigo_rpm.startswith('ZMB'):
            circuito = 'CIRCUITO_ZMB'
        elif codigo_rpm.startswith('EPAG'):
            circuito = 'CIRCUITO_EPAG'
        elif codigo_rpm.startswith('PROG'):
            circuito = 'CIRCUITO_PROG'
        elif codigo_rpm.startswith('BOLG'):
            circuito = 'CIRCUITO_BOLG'
        elif codigo_rpm.startswith('BYCG'):
            circuito = 'CIRCUITO_BYCG'
        elif codigo_rpm.startswith('GBTG'):
            circuito = 'CIRCUITO_GBTG'
        elif codigo_rpm.startswith('ZMBG'):
            circuito = 'CIRCUITO_ZMBG'
        elif codigo_rpm.startswith('AYAG'):
            circuito = 'CIRCUITO_AYAG'
        elif codigo_rpm.startswith('BOLT'):
            circuito = 'CIRCUITO_BOLT'
        else:
            circuito = 'CIRCUITO_OTROS'

        # Convertir fecha DD/MM/YYYY a YYYY-MM-DD
        partes = fecha.split('/')
        if len(partes) == 3:
            dia = partes[0].zfill(2)
            mes = partes[1].zfill(2)
            anio = partes[2]
            fecha_fmt = f"{anio}-{mes}-{dia}"
        else:
            continue

        medida = {
            "codigo_rpm": codigo_rpm,
            "circuito": circuito,
            "ucp": "PRIMEGRID",
            "fecha": fecha_fmt,
            "flujo": flujo,
        }

        # Agregar periodos
        for i in range(1, 25):
            try:
                valor = float(row[f'P{i}'])
                medida[f'p{i}'] = valor
            except (ValueError, KeyError):
                medida[f'p{i}'] = 0.0

        medidas.append(medida)

print(f"\nMedidas cargadas: {len(medidas)}")

# Contar por circuito
from collections import Counter
contador = Counter(m['circuito'] for m in medidas)
print(f"\nDistribución por circuito:")
for circuito, count in sorted(contador.items()):
    print(f"  {circuito}: {count} medidas")

# Procesar con n_max = 8
print(f"\n{'=' * 80}")
print("Procesando con n_max = 8")
print('=' * 80)

resultados = procesar_curvas_tipicas_por_circuito(medidas, n_max=8)

print(f"\nCircuitos procesados: {len(resultados)}")

for circuito, resultado in sorted(resultados.items()):
    print(f"\n{circuito}:")
    print(f"  Curvas seleccionadas: {resultado['n_curvas']}")

    # Verificar suma de pesos
    suma_pesos = sum(resultado['pesos'].values())
    print(f"  Suma de pesos: {suma_pesos:.15f}")

    if abs(suma_pesos - 1.0) < 1e-10:
        print(f"  [OK] Suma de pesos = 1.0")
    else:
        print(f"  [WARN] Suma de pesos != 1.0 (diferencia: {abs(suma_pesos - 1.0):.15f})")

    # Mostrar primeras curvas seleccionadas
    print(f"  Curvas seleccionadas:")
    for i, curva in enumerate(resultado['curvas_seleccionadas'][:3], 1):
        print(f"    {i}. {curva['codigo_rpm']} - {curva['fecha']}")
    if resultado['n_curvas'] > 3:
        print(f"    ... y {resultado['n_curvas'] - 3} más")

    # Mostrar muestra de promedio y pesos
    print(f"  Muestra de promedio y pesos:")
    for i in [1, 6, 12, 18, 24]:
        promedio_val = resultado['promedio'][f'p{i}']
        peso_val = resultado['pesos'][f'p{i}']
        print(f"    p{i:2d}: promedio={promedio_val:8.5f}, peso={peso_val:.7f}")

print(f"\n{'=' * 80}")
print("Test completado")
print('=' * 80)
