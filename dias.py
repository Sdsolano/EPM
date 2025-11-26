import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime

def predecir_dia_importante(total_predicho, fecha, k=15,
                            path="/Users/pablo/Documents/GitHub/EPM/data/features/data_with_features_latest.csv"):
    dias_importantes = [
    '01-01', '05-01', '06-03', '08-07', '11-04', '12-25',
    '01-08', '03-29', '07-20', '03-25', '07-01', '08-19',
    '10-14', '11-11', '03-28', '12-08'
    ]

    periodos = [f"P{i}" for i in range(1, 25)]
    df = pd.read_csv(path)
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df["mmdd"] = df["FECHA"].dt.strftime("%m-%d")

    # Filtrar solo días importantes
    df_f = df[df["mmdd"].isin(dias_importantes)]

    # Perfil diario (ya tienes P1–P24)
    df_dias = df_f.groupby("FECHA")[periodos].mean()

    # Normalizar
    X = df_dias.values
    totales = X.sum(axis=1).reshape(-1, 1)
    X_norm = X / totales

    # Clustering
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=20)
    labels = kmeans.fit_predict(X_norm)
    df_dias["cluster"] = labels

    # Curvas base normalizadas
    curvas_base = (
        df_dias.groupby("cluster")[periodos]
        .mean()
        .div(df_dias.groupby("cluster")[periodos].mean().sum(axis=1), axis=0)
    )

    # Cluster típico según mm-dd (similar a cluster_por_dow)
    df_dias["mmdd"] = df_dias.index.strftime("%m-%d")
    cluster_por_mmdd = (
        df_dias.groupby("mmdd")["cluster"]
        .agg(lambda x: x.mode()[0])
    )

    fecha = pd.to_datetime(fecha)
    mmdd = fecha.strftime("%m-%d")

    if mmdd not in cluster_por_mmdd:
        # Día no está en los importantes
        return None

    cluster = cluster_por_mmdd[mmdd]

    perfil = curvas_base.loc[cluster].values

    return perfil * total_predicho

dias_comprobacion = ['12-25', '07-20', '06-10',
                     '05-13', '12-08', '01-01', '05-01', '08-07']
fecha='2024-12-25'
if fecha[5:] in dias_comprobacion:
    print(predecir_dia_importante(1000, fecha))

