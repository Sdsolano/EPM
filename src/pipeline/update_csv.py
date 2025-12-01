import pandas as pd
import json
from datetime import datetime, timedelta
import requests
# --- 1. JSON de entrada ---
data_json = {
    "success": True,
    "message": "Historicos de Cartagena entontrados",
    "data": [
        {
            "fecha": "2025-11-04T05:00:00.000Z",
            "p1": 426.4, "p2": 413.6, "p3": 403, "p4": 390.3, "p5": 382.7, "p6": 370.8,
            "p7": 356.1, "p8": 366, "p9": 387.7, "p10": 405.9, "p11": 415.7, "p12": 419.4,
            "p13": 419.1, "p14": 421.3, "p15": 423.2, "p16": 419.3, "p17": 408, "p18": 398.9,
            "p19": 411.3, "p20": 425.7, "p21": 426.9, "p22": 426.4, "p23": 430.8, "p24": 426.6,
            "observacion": ""
        },
        {
            "fecha": "2025-11-05T05:00:00.000Z",
            "p1": 414.9, "p2": 400.7, "p3": 385.6, "p4": 380, "p5": 370.4, "p6": 357.9,
            "p7": 338.7, "p8": 339, "p9": 347.8, "p10": 350.7, "p11": 360.5, "p12": 364.1,
            "p13": 363.4, "p14": 369, "p15": 370.1, "p16": 357.2, "p17": 358.2, "p18": 359.6,
            "p19": 377.5, "p20": 402.4, "p21": 404.7, "p22": 409.3, "p23": 411.6, "p24": 406.7,
            "observacion": ""
        },
        {
            "fecha": "2025-11-06T05:00:00.000Z",
            "p1": 394.6, "p2": 381.3, "p3": 373.8, "p4": 366, "p5": 360.2, "p6": 351.1,
            "p7": 340, "p8": 334.6, "p9": 339.4, "p10": 345, "p11": 353.3, "p12": 358.6,
            "p13": 359.8, "p14": 364.4, "p15": 370.6, "p16": 359.7, "p17": 342.6, "p18": 337,
            "p19": 374.7, "p20": 408.5, "p21": 414.9, "p22": 420.4, "p23": 424.9, "p24": 412,
            "observacion": ""
        }
    ]
}

# --- 2. Convertir JSON a DataFrame ---
def json_to_csv_power(data_json,ucp_name,variable="Demanda_Real",clasificador="NORMAL"):
    archivo='../../data/raw/datos.csv'
    df = pd.DataFrame(data_json["data"])
    
    # --- 3. Convertir fecha a YYYY-MM-DD ---
    df["FECHA"] = pd.to_datetime(df["fecha"]).dt.date
    df.drop(columns=["fecha"], inplace=True)

    # --- 4. Renombrar p1..p24 a P1..P24 ---
    df.rename(columns={f"p{i}": f"P{i}" for i in range(1, 25)}, inplace=True)

    # --- 5. Calcular TOTAL ---
    cols_p = [f"P{i}" for i in range(1, 25)]
    df["TOTAL"] = df[cols_p].sum(axis=1)

    # --- 6. Crear columnas que no estaban en el JSON ---
    df["UCP"] = ucp_name
    df["VARIABLE"] = variable
    df["Clasificador interno"] = clasificador

    # --- 7. Obtener día de la semana en español ---
    # Esto devuelve nombres como: Monday→lunes, Tuesday→martes, etc.
    df["TIPO DIA"] = pd.to_datetime(df["FECHA"]).dt.day_name(locale="es_ES")

    # --- 8. Reordenar columnas como tu CSV ---
    final_cols = ['UCP', 'VARIABLE', 'FECHA', 'Clasificador interno', 'TIPO DIA'] + cols_p + ["TOTAL"]
    df2 = df[final_cols]
    #df2 el generado del json, se le pega a df1 que es el historico original
    df1=pd.read_csv(archivo)
    df_final = pd.concat([df1, df2], axis=0, ignore_index=True)

    # --- 7. Guardar CSV ---
    df_final.to_csv(archivo, index=False)

    print("CSV generado correctamente.")

# df = pd.read_csv('../../data/raw/clima.csv')
# print(df.head())
def regresar_nuevo_csv(ucp):
    df=pd.read_csv('../../data/raw/datos.csv')
    ultima_fecha=df['FECHA'].max()  
    base_url = "http://localhost:8000"
    url = f"{base_url}/cargarPeriodosxUCPDesdeFecha/{ucp}/{ultima_fecha}"
    response = requests.get(url)
    json_to_csv_power(response.json(),ucp_name=ucp)





response_json = {
    "success": 'true',
    "message": "las variables climaticas de Atlantico entontradas",
    "data": [
        {
            "fecha": "2019-06-29T05:00:00.000Z",
            "p1_t": 33.4,
            "p2_t": 34.5,
            "p3_t": 35.9,
            "p4_t": 37.5,
            "p5_t": 38.3,
            "p6_t": 38.3,
            "p7_t": 36.9,
            "p8_t": 36,
            "p9_t": 34,
            "p10_t": 31.6,
            "p11_t": 30.4,
            "p12_t": 31.2,
            "p13_t": 30.2,
            "p14_t": 29.5,
            "p15_t": 30.4,
            "p16_t": 29.1,
            "p17_t": 29.5,
            "p18_t": 29.9,
            "p19_t": 29.2,
            "p20_t": 29.3,
            "p21_t": 28.2,
            "p22_t": 28.6,
            "p23_t": 29.9,
            "p24_t": 31.2,
            "p1_h": 74,
            "p2_h": 69,
            "p3_h": 62,
            "p4_h": 58,
            "p5_h": 55,
            "p6_h": 55,
            "p7_h": 62,
            "p8_h": 62,
            "p9_h": 69,
            "p10_h": 78,
            "p11_h": 78,
            "p12_h": 78,
            "p13_h": 73,
            "p14_h": 78,
            "p15_h": 83,
            "p16_h": 78,
            "p17_h": 78,
            "p18_h": 78,
            "p19_h": 73,
            "p20_h": 78,
            "p21_h": 83,
            "p22_h": 83,
            "p23_h": 83,
            "p24_h": 78,
            "p1_v": 18.5,
            "p2_v": 24.1,
            "p3_v": 20.4,
            "p4_v": 18.5,
            "p5_v": 16.7,
            "p6_v": 14.8,
            "p7_v": 25.9,
            "p8_v": 29.6,
            "p9_v": 22.2,
            "p10_v": 20.4,
            "p11_v": 25.9,
            "p12_v": 16.7,
            "p13_v": 14.8,
            "p14_v": 16.7,
            "p15_v": 18.5,
            "p16_v": 20.4,
            "p17_v": 16.7,
            "p18_v": 13,
            "p19_v": 14.8,
            "p20_v": 18.5,
            "p21_v": 18.5,
            "p22_v": 14.8,
            "p23_v": 7.4,
            "p24_v": 11.1,
            "ucp": "Atlantico",
            "festivo": 'null',
            "p1_i": 0,
            "p2_i": 0,
            "p3_i": 0,
            "p4_i": 0,
            "p5_i": 0,
            "p6_i": 0,
            "p7_i": 0,
            "p8_i": 0,
            "p9_i": 0,
            "p10_i": 0,
            "p11_i": 0,
            "p12_i": 0,
            "p13_i": 0,
            "p14_i": 0,
            "p15_i": 0,
            "p16_i": 0,
            "p17_i": 0,
            "p18_i": 0,
            "p19_i": 0,
            "p20_i": 0,
            "p21_i": 0,
            "p22_i": 0,
            "p23_i": 0,
            "p24_i": 0
        },
        {
            "fecha": "2019-06-30T05:00:00.000Z",
            "p1_t": 33.4,
            "p2_t": 34.5,
            "p3_t": 35.9,
            "p4_t": 37.5,
            "p5_t": 38.3,
            "p6_t": 38.3,
            "p7_t": 36.9,
            "p8_t": 36,
            "p9_t": 34,
            "p10_t": 31.6,
            "p11_t": 30.4,
            "p12_t": 31.2,
            "p13_t": 30.2,
            "p14_t": 29.5,
            "p15_t": 30.4,
            "p16_t": 29.1,
            "p17_t": 29.5,
            "p18_t": 29.9,
            "p19_t": 29.2,
            "p20_t": 29.3,
            "p21_t": 28.2,
            "p22_t": 28.6,
            "p23_t": 29.9,
            "p24_t": 31.2,
            "p1_h": 74,
            "p2_h": 69,
            "p3_h": 62,
            "p4_h": 58,
            "p5_h": 55,
            "p6_h": 55,
            "p7_h": 62,
            "p8_h": 62,
            "p9_h": 69,
            "p10_h": 78,
            "p11_h": 78,
            "p12_h": 78,
            "p13_h": 73,
            "p14_h": 78,
            "p15_h": 83,
            "p16_h": 78,
            "p17_h": 78,
            "p18_h": 78,
            "p19_h": 73,
            "p20_h": 78,
            "p21_h": 83,
            "p22_h": 83,
            "p23_h": 83,
            "p24_h": 78,
            "p1_v": 18.5,
            "p2_v": 24.1,
            "p3_v": 20.4,
            "p4_v": 18.5,
            "p5_v": 16.7,
            "p6_v": 14.8,
            "p7_v": 25.9,
            "p8_v": 29.6,
            "p9_v": 22.2,
            "p10_v": 20.4,
            "p11_v": 25.9,
            "p12_v": 16.7,
            "p13_v": 14.8,
            "p14_v": 16.7,
            "p15_v": 18.5,
            "p16_v": 20.4,
            "p17_v": 16.7,
            "p18_v": 13,
            "p19_v": 14.8,
            "p20_v": 18.5,
            "p21_v": 18.5,
            "p22_v": 14.8,
            "p23_v": 7.4,
            "p24_v": 11.1,
            "ucp": "Atlantico",
            "festivo": 'null',
            "p1_i": 0,
            "p2_i": 0,
            "p3_i": 0,
            "p4_i": 0,
            "p5_i": 0,
            "p6_i": 0,
            "p7_i": 0,
            "p8_i": 0,
            "p9_i": 0,
            "p10_i": 0,
            "p11_i": 0,
            "p12_i": 0,
            "p13_i": 0,
            "p14_i": 0,
            "p15_i": 0,
            "p16_i": 0,
            "p17_i": 0,
            "p18_i": 0,
            "p19_i": 0,
            "p20_i": 0,
            "p21_i": 0,
            "p22_i": 0,
            "p23_i": 0,
            "p24_i": 0
        },
        {
            "fecha": "2019-07-01T05:00:00.000Z",
            "p1_t": 33.4,
            "p2_t": 34.5,
            "p3_t": 35.9,
            "p4_t": 37.5,
            "p5_t": 38.3,
            "p6_t": 38.3,
            "p7_t": 36.9,
            "p8_t": 36,
            "p9_t": 34,
            "p10_t": 31.6,
            "p11_t": 30.4,
            "p12_t": 31.2,
            "p13_t": 30.2,
            "p14_t": 29.5,
            "p15_t": 30.4,
            "p16_t": 29.1,
            "p17_t": 29.5,
            "p18_t": 29.9,
            "p19_t": 29.2,
            "p20_t": 29.3,
            "p21_t": 28.2,
            "p22_t": 28.6,
            "p23_t": 29.9,
            "p24_t": 31.2,
            "p1_h": 74,
            "p2_h": 69,
            "p3_h": 62,
            "p4_h": 58,
            "p5_h": 55,
            "p6_h": 55,
            "p7_h": 62,
            "p8_h": 62,
            "p9_h": 69,
            "p10_h": 78,
            "p11_h": 78,
            "p12_h": 78,
            "p13_h": 73,
            "p14_h": 78,
            "p15_h": 83,
            "p16_h": 78,
            "p17_h": 78,
            "p18_h": 78,
            "p19_h": 73,
            "p20_h": 78,
            "p21_h": 83,
            "p22_h": 83,
            "p23_h": 83,
            "p24_h": 78,
            "p1_v": 18.5,
            "p2_v": 24.1,
            "p3_v": 20.4,
            "p4_v": 18.5,
            "p5_v": 16.7,
            "p6_v": 14.8,
            "p7_v": 25.9,
            "p8_v": 29.6,
            "p9_v": 22.2,
            "p10_v": 20.4,
            "p11_v": 25.9,
            "p12_v": 16.7,
            "p13_v": 14.8,
            "p14_v": 16.7,
            "p15_v": 18.5,
            "p16_v": 20.4,
            "p17_v": 16.7,
            "p18_v": 13,
            "p19_v": 14.8,
            "p20_v": 18.5,
            "p21_v": 18.5,
            "p22_v": 14.8,
            "p23_v": 7.4,
            "p24_v": 11.1,
            "ucp": "Atlantico",
            "festivo": 'null',
            "p1_i": 0,
            "p2_i": 0,
            "p3_i": 0,
            "p4_i": 0,
            "p5_i": 0,
            "p6_i": 0,
            "p7_i": 0,
            "p8_i": 0,
            "p9_i": 0,
            "p10_i": 0,
            "p11_i": 0,
            "p12_i": 0,
            "p13_i": 0,
            "p14_i": 0,
            "p15_i": 0,
            "p16_i": 0,
            "p17_i": 0,
            "p18_i": 0,
            "p19_i": 0,
            "p20_i": 0,
            "p21_i": 0,
            "p22_i": 0,
            "p23_i": 0,
            "p24_i": 0
        },
   ]
}
# df viene de: df = pd.DataFrame(response_json["data"])



def regresar_nuevo_csv_clima(response_json):
    ruta='../../data/raw/clima2.csv'
    df = pd.DataFrame(response_json["data"])
    df["fecha"] = pd.to_datetime(df["fecha"])
    filas = []
    for _, row in df.iterrows():
        fecha = row["fecha"]
        
        for p in range(1, 25):
            
            fila = {
                "fecha": fecha.strftime("%Y-%m-%d"),
                "periodo": p,
                "p_t": row[f"p{p}_t"],
                "p_h": row[f"p{p}_h"],
                "p_v": row[f"p{p}_v"],
                "p_i": row[f"p{p}_i"]
            }
            filas.append(fila)
    df_final = pd.DataFrame(filas)
    df_inicial= pd.read_csv(ruta)
    print(df_inicial)
    print(df_final)
    df_concat= pd.concat([df_final,df_final],axis=0, ignore_index=True)
    df_concat.to_csv(ruta, index=False)


def req_clima_api(ucp,fecha_fin):
    df=pd.read_csv('../../data/raw/clima2.csv')
    fecha_inicio=df['fecha'].max()  
    base_url = "http://localhost:8000"
    url = f"{base_url}/cargarClimaXUCPDesdeHasta/{ucp}/{fecha_inicio}/{fecha_fin}"
    print(fecha_inicio)
    # response = requests.get(url)
    # regresar_nuevo_csv_clima(response.json())

#req_clima_api('Atlantico','2025-11-06')
df=pd.read_csv('../../data/raw/clima.csv')

df["dt_iso"] = (
    df["dt_iso"]
    .str.replace(" UTC", "", regex=False)
    .pipe(pd.to_datetime, format="%Y-%m-%d %H:%M:%S %z")
)

# Crear columnas necesarias
df["fecha"] = df["dt_iso"].dt.date
df["hora"] = df["dt_iso"].dt.hour
df["periodo"] = (df["hora"] + 1) 
df_final = df[[
    "fecha",
    "periodo",
    "wind_speed",
    "rain_1h",
    "temp",
    "humidity"
    
]]
df_renamed = df_final.rename(columns={
    "wind_speed": "p_v",
    "rain_1h": "p_i",
    "temp": "p_t",
    "humidity": "p_h"
})
df_renamed=df_renamed.fillna(0)
df_renamed.to_csv('../../data/raw/clima_new.csv', index=False)