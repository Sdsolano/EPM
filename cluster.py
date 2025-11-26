# ============================================================
# MODELO COMPLETO PARA PREDICIÓN HORARIA (P1–P24)
# USANDO TOTAL + FECHA (día de la semana)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


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



