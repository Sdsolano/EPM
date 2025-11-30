import pandas as pd
import json
from datetime import datetime, timedelta
# --- 1. JSON de entrada ---
data_json = {
    "success": True,
    "message": "Historicos de Cartagena entontrados",
    "data": [
        {
            "fecha": "2019-06-29T05:00:00.000Z",
            "p1": 426.4, "p2": 413.6, "p3": 403, "p4": 390.3, "p5": 382.7, "p6": 370.8,
            "p7": 356.1, "p8": 366, "p9": 387.7, "p10": 405.9, "p11": 415.7, "p12": 419.4,
            "p13": 419.1, "p14": 421.3, "p15": 423.2, "p16": 419.3, "p17": 408, "p18": 398.9,
            "p19": 411.3, "p20": 425.7, "p21": 426.9, "p22": 426.4, "p23": 430.8, "p24": 426.6,
            "observacion": ""
        },
        {
            "fecha": "2019-06-30T05:00:00.000Z",
            "p1": 414.9, "p2": 400.7, "p3": 385.6, "p4": 380, "p5": 370.4, "p6": 357.9,
            "p7": 338.7, "p8": 339, "p9": 347.8, "p10": 350.7, "p11": 360.5, "p12": 364.1,
            "p13": 363.4, "p14": 369, "p15": 370.1, "p16": 357.2, "p17": 358.2, "p18": 359.6,
            "p19": 377.5, "p20": 402.4, "p21": 404.7, "p22": 409.3, "p23": 411.6, "p24": 406.7,
            "observacion": ""
        },
        {
            "fecha": "2019-07-01T05:00:00.000Z",
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
    df = df[final_cols]

    # --- 7. Guardar CSV ---
    df.to_csv("../../data/raw/historicos_cartagena.csv", index=False)

    print("CSV generado correctamente.")

#df=pd.read_csv('/Users/pablo/Documents/GitHub/EPM/string/data_with_features_latest.csv')
df=pd.read_csv('../../data/raw/datos.csv')
ultima_fecha=df['FECHA'].max()
print(ultima_fecha)   