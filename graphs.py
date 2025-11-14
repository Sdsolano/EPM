import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- Configuración ---
st.set_page_config(page_title="Análisis con Clustering y PCA", layout="wide")
st.title("📊 Análisis interactivo y Clustering por Tipo de Día")

# --- Carga de datos ---
df = pd.read_csv("datos.csv")

horas = [f'P{i}' for i in range(1, 25)]

# --- Filtros interactivos ---
tipos_sel = st.multiselect("Selecciona Tipo de Día", sorted(df['TIPO DIA'].unique()), default=sorted(df['TIPO DIA'].unique()))
vars_sel = st.multiselect("Selecciona Variable", sorted(df['VARIABLE'].unique()), default=sorted(df['VARIABLE'].unique()))

df_filtrado = df[df['TIPO DIA'].isin(tipos_sel) & df['VARIABLE'].isin(vars_sel)]

# --- Gráficos descriptivos ---
promedio_horas = df_filtrado.groupby('TIPO DIA')[horas].mean().reset_index()
df_melt = promedio_horas.melt(id_vars='TIPO DIA', var_name='Hora', value_name='Valor')

fig1 = px.line(df_melt, x='Hora', y='Valor', color='TIPO DIA', markers=True,
               title='Promedio horario por Tipo de Día')
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.bar(df_filtrado, x='TIPO DIA', y='TOTAL', color='TIPO DIA', barmode='group',
              title='Promedio TOTAL por Tipo de Día')
st.plotly_chart(fig2, use_container_width=True)


# --- 🧠 Análisis de Clustering ---
st.subheader("🧩 Agrupamiento con K-Means y PCA")

# Escalamos los datos horarios (P1-P24)
X = df_filtrado[horas].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reducimos a 2D con PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

df_filtrado['PCA1'] = pca_result[:, 0]
df_filtrado['PCA2'] = pca_result[:, 1]

# K-Means (selección de número de clusters)
n_clusters = st.slider("Selecciona número de clusters (K)", 2, 10, 3)
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
df_filtrado['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualización de clusters
fig3 = px.scatter(
    df_filtrado,
    x='PCA1', y='PCA2',
    color=df_filtrado['Cluster'].astype(str),
    hover_data=['TIPO DIA', 'VARIABLE', 'TOTAL'],
    title=f"Clusters detectados con K-Means (k={n_clusters}) en espacio PCA",
)
st.plotly_chart(fig3, use_container_width=True)

# --- Información adicional ---
st.write("🔍 **Varianza explicada por PCA:**")
st.write(f"- Componente 1: {pca.explained_variance_ratio_[0]*100:.2f}%")
st.write(f"- Componente 2: {pca.explained_variance_ratio_[1]*100:.2f}%")

st.markdown("""
> 💡 **Interpretación:**
> - Los puntos más cercanos representan días con patrones horarios similares.  
> - Los clusters de color muestran grupos naturales según el consumo o variable.  
> - Si el PCA1 explica una gran parte (>50%), significa que la mayoría de la variabilidad se captura en el primer eje.
""")
# --- 🗓️ Análisis temporal por semana ---
st.subheader("📆 Evolución semanal por Tipo de Día")

# Asegurar que FECHA esté en formato datetime
df_filtrado['FECHA'] = pd.to_datetime(df_filtrado['FECHA'], errors='coerce')

# Crear columna de número de semana
df_filtrado['SEMANA'] = df_filtrado['FECHA'].dt.isocalendar().week
df_filtrado['AÑO'] = df_filtrado['FECHA'].dt.year

# Promedio semanal por tipo de día
promedio_semanal = (
    df_filtrado.groupby(['AÑO', 'SEMANA', 'TIPO DIA'])['TOTAL']
    .mean()
    .reset_index()
)

fig4 = px.line(
    promedio_semanal,
    x='SEMANA', y='TOTAL', color='TIPO DIA',
    facet_col='AÑO',
    markers=True,
    title='Variación semanal del TOTAL por Tipo de Día',
)
st.plotly_chart(fig4, use_container_width=True)

# --- 🔥 Heatmap horario por semana ---
st.subheader("🔥 Mapa de calor semanal por Tipo de Día (promedio por hora)")

tipo_dia_heat = st.selectbox(
    "Selecciona un Tipo de Día para analizar",
    sorted(df_filtrado['TIPO DIA'].unique())
)

df_tipo = df_filtrado[df_filtrado['TIPO DIA'] == tipo_dia_heat]
horas = [f'P{i}' for i in range(1, 25)]

heatmap_data = (
    df_tipo.groupby(['AÑO', 'SEMANA'])[horas].mean().reset_index()
    .melt(id_vars=['AÑO', 'SEMANA'], var_name='Hora', value_name='Valor')
)

fig5 = px.imshow(
    heatmap_data.pivot_table(index=['AÑO', 'SEMANA'], columns='Hora', values='Valor'),
    aspect='auto',
    color_continuous_scale='Viridis',
    title=f"Intensidad horaria a lo largo de las semanas ({tipo_dia_heat})",
)
st.plotly_chart(fig5, use_container_width=True)

# --- 🎢 Comparación entre semanas ---
st.subheader("📊 Comparación del perfil horario entre semanas")

semanas_disponibles = sorted(df_tipo['SEMANA'].unique())
semanas_sel = st.multiselect(
    "Selecciona semanas a comparar",
    semanas_disponibles,
    default=semanas_disponibles[:3]
)

comparacion = (
    df_tipo[df_tipo['SEMANA'].isin(semanas_sel)]
    .groupby('SEMANA')[horas]
    .mean()
    .reset_index()
    .melt(id_vars='SEMANA', var_name='Hora', value_name='Valor')
)

fig6 = px.line(
    comparacion,
    x='Hora', y='Valor', color='SEMANA', markers=True,
    title=f"Comparación de perfiles horarios ({tipo_dia_heat}) entre semanas seleccionadas"
)
st.plotly_chart(fig6, use_container_width=True)
