"""
Script de prueba para verificar la conversión automática de clima horario a diario
"""
import sys
from pathlib import Path

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.connectors import WeatherDataConnector

# Probar conversión
print("="*80)
print("PROBANDO CONVERSIÓN DE CLIMA HORARIO A DIARIO")
print("="*80)

weather_connector = WeatherDataConnector({'path': 'data/raw/clima.csv'})

# Leer datos SIN filtros de fecha primero para ver qué hay
df_weather_all = weather_connector.read_data()
print(f"\nDatos completos: {len(df_weather_all)} dias")
print(f"Rango: {df_weather_all['FECHA'].min()} a {df_weather_all['FECHA'].max()}")

# Ahora filtrar por el rango de datos de demanda
df_weather = weather_connector.read_data(start_date='2017-01-01')

print("\nRESULTADOS:")
print(f"  Registros diarios generados: {len(df_weather)}")
print(f"  Rango de fechas: {df_weather['FECHA'].min().date()} a {df_weather['FECHA'].max().date()}")
print(f"\nColumnas generadas:")
for col in sorted(df_weather.columns):
    print(f"    - {col}")

print(f"\n📈 Muestra de datos (primeros 5 días):")
print(df_weather.head())

print(f"\n📊 Estadísticas de temperatura:")
print(df_weather[['temp_mean', 'temp_min', 'temp_max']].describe())

print("\n" + "="*80)
print("✅ CONVERSIÓN EXITOSA")
print("="*80)
