
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import dias 

def predecir(total_predicho, fecha,k=35,path="/Users/pablo/Documents/GitHub/EPM/data/features/data_with_features_latest.csv"):
    df = pd.read_csv(path)
    df["fecha"] = pd.to_datetime(df["FECHA"])
    periodos = [f"P{i}" for i in range(1, 25)]

    # Matriz día × 24 periodos
    X = df[periodos].values

    # ============================================================
    # 2. Normalizar cada día
    # ============================================================
    suma_dia = X.sum(axis=1).reshape(-1, 1)
    X_norm = X / suma_dia

    # ============================================================
    # 3. Clustering con KMeans
    # ============================================================
   
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=20)
    labels = kmeans.fit_predict(X_norm)
    df["cluster"] = labels

    # ============================================================
    # 4. Curvas base por cluster (normalizadas)
    # ============================================================
    curvas_base = (
        df.groupby("cluster")[periodos]
        .mean()
        .div(df.groupby("cluster")[periodos].mean().sum(axis=1), axis=0)
    )

    # ============================================================
    # 5. Cluster típico por día de la semana
    #    Lunes = 0 ... Domingo = 6
    # ============================================================
    cluster_por_dow = (
        df.groupby(df["fecha"].dt.dayofweek)["cluster"]
        .agg(lambda x: x.mode()[0])
    )


    fecha = pd.to_datetime(fecha)
    dow = fecha.dayofweek
    cluster = cluster_por_dow[dow]
    perfil = curvas_base.loc[cluster].values
    return perfil * total_predicho




# Cargar datos reales para comparar
# df_real = pd.read_csv("/Users/pablo/Documents/GitHub/EPM/data/features/data_with_features_latest.csv")
# df_real["fecha"] = pd.to_datetime(df_real["FECHA"])

# Elegir la fecha real que quieres comparar
fecha_objetivo = "2023-03-25"
def full_predict(total_predicho, fecha):
    dias_comprobacion = ['12-25', '07-20', '06-10',
                     '05-13', '12-08', '01-01', '05-01', '08-07']
    if fecha[5:] in dias_comprobacion:
        return dias.predecir_dia_importante(total_predicho, fecha)

    else:
        return predecir(total_predicho, fecha)
