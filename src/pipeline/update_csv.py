import pandas as pd
import json
from datetime import datetime, timedelta
import requests
import os
from pathlib import Path
#-- 1. Función para convertir JSON a CSV y guardarlo ---
def json_to_csv_power(data_json,ucp_name,archivo,variable="Demanda_Real",clasificador="NORMAL"):
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
    df["TIPO DIA"] = pd.to_datetime(df["FECHA"]).dt.day_name(locale="es_ES.utf8")

    # --- 8. Reordenar columnas como tu CSV ---
    final_cols = ['UCP', 'VARIABLE', 'FECHA', 'Clasificador interno', 'TIPO DIA'] + cols_p + ["TOTAL"]
    df2 = df[final_cols]
    #df2 el generado del json, se le pega a df1 que es el historico original
    df1=pd.read_csv(archivo)
    df1["FECHA"] = pd.to_datetime(df1["FECHA"]).dt.date
    df_final = pd.concat([df1, df2], axis=0, ignore_index=True)
    df_final.drop_duplicates(subset=['FECHA'],inplace=True)
    # --- 7. Guardar CSV ---
    df_final.to_csv(archivo, index=False)

    print("CSV generado correctamente.")
#-- 2. Función para solicitar datos y generar CSV ---
def regresar_nuevo_csv(ucp):
    

    path = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / ucp / "datos.csv"

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df=pd.DataFrame([],columns=["UCP","VARIABLE","FECHA","Clasificador interno","TIPO DIA"]+[f'P{i}' for i in range(1,25)]+["TOTAL"])
        df.to_csv(path, index=False)
        base_url = "http://localhost:3000"
        url = f"{base_url}/api/v1/admin/configuracion-interna/cargarPeriodosxUCPDesdeFecha/{ucp}/2005-11-06"
        response = requests.get(url)
        print(response.json())
        json_to_csv_power(response.json(),ucp,path)
    else:
        df=pd.read_csv(path)
        fecha_inicio=df['FECHA'].max()
        if not fecha_inicio or pd.isna(fecha_inicio):
            fecha_inicio='2005-11-06'
        base_url = "http://localhost:3000"
        url = f"{base_url}/api/v1/admin/configuracion-interna/cargarPeriodosxUCPDesdeFecha/{ucp}/{fecha_inicio}"
        response = requests.get(url)
        print(response.json())
        json_to_csv_power(response.json(),ucp,path)
#-- 3. Función para solicitar datos climáticos, pasar de json a CSV y guardarlo---
def regresar_nuevo_csv_clima(response_json,ruta):
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
    df_concat= pd.concat([df_inicial,df_final],axis=0, ignore_index=True)
    df_concat.drop_duplicates(inplace=True)
    df_concat.to_csv(ruta, index=False)
#-- 4. Función para solicitar datos climáticos y generar CSV ---
def req_clima_api(ucp):
    path = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / ucp / "clima_new.csv"

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df=pd.DataFrame(columns=["fecha","periodo","p_t","p_h","p_v","p_i"])
        df.to_csv(path, index=False)
        base_url = "http://localhost:3000"
        url = f"{base_url}/api/v1/admin/configuracion-interna/cargarVariablesClimaticasxUCPDesdeFecha/{ucp}/2005-11-06"
        response = requests.get(url)
        response_json = response.json()
        print(response_json)
        # Un mercado nuevo puede no tener clima ingestado todavía (ej. sin
        # ciudad asociada en jano-proxy) — sin esto, un mercado sin clima
        # tumbaba TODO el flujo (hourly y diario) con KeyError: 'data' acá.
        # Se deja el CSV vacío con headers (ya creado arriba) y el
        # pipeline sigue sin variables climáticas en vez de fallar.
        if response_json.get("data"):
            regresar_nuevo_csv_clima(response_json, path)
        else:
            print(f"Sin datos climáticos para {ucp} — se continúa sin clima.")
    else:
        df=pd.read_csv(path)
        fecha_inicio=df['fecha'].max()
        if not fecha_inicio or pd.isna(fecha_inicio):
            fecha_inicio='2005-11-06'
        base_url = "http://localhost:3000"
        url = f"{base_url}/api/v1/admin/configuracion-interna/cargarVariablesClimaticasxUCPDesdeFecha/{ucp}/{fecha_inicio}"
        response = requests.get(url)
        response_json = response.json()
        print(response_json)
        if response_json.get("data"):
            regresar_nuevo_csv_clima(response_json, path)
        else:
            print(f"Sin datos climáticos nuevos para {ucp} — se continúa con lo que ya había.")



#todo el proceso
def full_update_csv(ucp):
    regresar_nuevo_csv(ucp)
    req_clima_api(ucp)
    print("Proceso de actualización de CSV completado.")
# Descomentar para actualizar manualmente
# full_update_csv('Atlantico')

#-- 5. Temporalidad diaria: histórico es fecha/total (sin p1..p24) ---
# El módulo "Pronóstico Diario" (endpoint /predict-daily) reentrena con
# mercados que solo reportan un total por día (ver issue "Desarrollar
# módulo pronostico en la Temporalidad Diaria"), no con periodos horarios.
# json_to_csv_power (arriba) no sirve para esto: calcula TOTAL sumando
# P1..P24, que acá no existen. Se guarda en un CSV aparte
# (datos_diario.csv, no datos.csv) para no mezclar con un eventual
# histórico horario real del mismo UCP.
def json_to_csv_power_diario(data_json, ucp_name, archivo, variable="Demanda_Real", clasificador="NORMAL"):
    df = pd.DataFrame(data_json["data"])
    df["FECHA"] = pd.to_datetime(df["fecha"]).dt.date
    df["TOTAL"] = pd.to_numeric(df["total"])
    df.drop(columns=["fecha", "total"], inplace=True)

    # P1..P24 no existen para datos diarios. Se llenan en 0 (no NaN): la
    # limpieza del pipeline (cleaning.py) descarta filas con NaN en esas
    # columnas, así que NaN borraría todo el histórico. El modelo diario
    # entrena sobre TOTAL, no sobre los periodos, así que esto no afecta
    # la predicción — solo deja sin efecto las features de pico horario
    # (P8/P12/P18/P20), que ya se saltan solas si la columna no aporta.
    for i in range(1, 25):
        df[f"P{i}"] = 0.0
    cols_p = [f"P{i}" for i in range(1, 25)]

    df["UCP"] = ucp_name
    df["VARIABLE"] = variable
    df["Clasificador interno"] = clasificador

    # Igual que json_to_csv_power: si tipo_dia viene vacío, se recalcula
    # desde la fecha para no dejar el campo crítico en null.
    dia_semana = pd.to_datetime(df["FECHA"]).dt.day_name(locale="es_ES.utf8")
    if "tipo_dia" in df.columns:
        df["TIPO DIA"] = df["tipo_dia"].fillna(dia_semana)
        df.drop(columns=["tipo_dia"], inplace=True)
    else:
        df["TIPO DIA"] = dia_semana

    final_cols = ['UCP', 'VARIABLE', 'FECHA', 'Clasificador interno', 'TIPO DIA'] + cols_p + ["TOTAL"]
    df2 = df[final_cols]

    df1 = pd.read_csv(archivo)
    df1["FECHA"] = pd.to_datetime(df1["FECHA"]).dt.date
    df_final = pd.concat([df1, df2], axis=0, ignore_index=True)
    df_final.drop_duplicates(subset=['FECHA'], inplace=True)
    df_final.to_csv(archivo, index=False)

    print("CSV diario generado correctamente.")

def regresar_nuevo_csv_diario(ucp):
    path = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / ucp / "datos_diario.csv"

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame([], columns=["UCP", "VARIABLE", "FECHA", "Clasificador interno", "TIPO DIA"] + [f'P{i}' for i in range(1, 25)] + ["TOTAL"])
        df.to_csv(path, index=False)
        base_url = "http://localhost:3000"
        url = f"{base_url}/api/v1/admin/configuracion-interna/cargarDemandaDiariaPorUCPDesdeFecha/{ucp}/2005-11-06"
        response = requests.get(url)
        print(response.json())
        json_to_csv_power_diario(response.json(), ucp, path)
    else:
        df = pd.read_csv(path)
        fecha_inicio = df['FECHA'].max()
        if not fecha_inicio or pd.isna(fecha_inicio):
            fecha_inicio = '2005-11-06'
        base_url = "http://localhost:3000"
        url = f"{base_url}/api/v1/admin/configuracion-interna/cargarDemandaDiariaPorUCPDesdeFecha/{ucp}/{fecha_inicio}"
        response = requests.get(url)
        print(response.json())
        json_to_csv_power_diario(response.json(), ucp, path)

def full_update_csv_diario(ucp):
    regresar_nuevo_csv_diario(ucp)
    req_clima_api(ucp)
    print("Proceso de actualización de CSV diario completado.")