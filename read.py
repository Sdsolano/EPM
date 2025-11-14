import pandas as pd

ruta = "/Users/pablo/epm/a50f5d9785250195ea4ef2cb78efad38.csv"

# Cargar datos
data1 = pd.read_csv(ruta)
# Lista de días laborales
def clear_info_power(data):
    data=data[['UCP', 'VARIABLE', 'FECHA', 'Clasificador interno', 'TIPO DIA', 'P1',
       'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12',
       'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22',
       'P23', 'P24', 'TOTAL']]
    laborales = ["LUNES", "MARTES", "MIERCOLES", "MIÉRCOLES", "JUEVES", "VIERNES"]

    # Clasificar los días
    data["TIPO DIA"] = data["TIPO DIA"].apply(
        lambda x: "LABORAL" if str(x).upper() in laborales else "FESTIVO"
    )

    # 🧹 Eliminar filas con cualquier valor NaN
    data = data.dropna()
    data.to_csv("data_cleaned_power.csv", index=False)
    return data



def clear_info_weather(data):
    data=data[['dt', 'dt_iso', 'timezone', 'city_name', 'lat', 'lon', 'temp',
       'visibility', 'dew_point', 'feels_like', 'temp_min', 'temp_max',
       'pressure', 'humidity', 'wind_speed',
       'wind_deg', 'wind_gust', 'rain_1h', 'rain_3h',
       'clouds_all', 'weather_id', 'weather_main', 'weather_description',
       'weather_icon']]
    
    # 🧹 Eliminar filas con cualquier valor NaN
    data = data.dropna(subset=['dt', 'dt_iso', 'timezone', 'city_name', 'lat', 'lon', 'temp',
       'visibility', 'dew_point', 'feels_like', 'temp_min', 'temp_max',
       'pressure', 'humidity', 'wind_speed',
       'wind_deg', 'clouds_all', 'weather_id', 'weather_main', 'weather_description',
       'weather_icon'])
    data.to_csv("data_cleaned_weather.csv", index=False)
    return data

