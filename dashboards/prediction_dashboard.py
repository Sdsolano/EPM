"""
Dashboard de Predicción: Próximos 30 Días con Desagregación Horaria

Dashboard Streamlit para generar y visualizar predicciones de demanda
para los próximos 30 días con desagregación horaria automática.

Uso:
    streamlit run dashboards/prediction_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction.forecaster import ForecastPipeline
from src.prediction.hourly import HourlyDisaggregationEngine
from src.config.settings import FEATURES_DATA_DIR

# Configurar logging
logging.basicConfig(level=logging.WARNING)

# Configuración de página
st.set_page_config(
    page_title="Predicción 30 Días - EPM",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .future-badge {
        background-color: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_prediction_pipeline():
    """Carga el pipeline de predicción"""
    try:
        pipeline = ForecastPipeline(
            model_path='models/trained/xgboost_20251120_161937.joblib',
            historical_data_path='data/features/data_with_features_latest.csv',
            enable_hourly_disaggregation=True
        )
        return pipeline, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_hourly_engine():
    """Carga motor de desagregación horaria"""
    try:
        engine = HourlyDisaggregationEngine(auto_load=True)
        return engine, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=3600)  # Cache por 1 hora
def generate_predictions(n_days=30):
    """Genera predicciones para los próximos N días"""
    pipeline, error = load_prediction_pipeline()

    if pipeline is None:
        return None, error

    try:
        predictions = pipeline.predict_next_n_days(n_days=n_days)
        return predictions, None
    except Exception as e:
        return None, str(e)


def plot_daily_predictions(predictions_df):
    """Gráfica de predicciones diarias"""
    fig = go.Figure()

    # Línea de predicción
    fig.add_trace(go.Scatter(
        x=predictions_df['fecha'],
        y=predictions_df['demanda_predicha'],
        mode='lines+markers',
        name='Demanda Predicha',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Demanda: %{y:,.2f} MW<extra></extra>'
    ))

    # Colorear fines de semana y festivos
    for idx, row in predictions_df.iterrows():
        if row['is_festivo']:
            fig.add_vrect(
                x0=row['fecha'], x1=row['fecha'] + timedelta(days=1),
                fillcolor="red", opacity=0.1,
                layer="below", line_width=0,
            )
        elif row['is_weekend']:
            fig.add_vrect(
                x0=row['fecha'], x1=row['fecha'] + timedelta(days=1),
                fillcolor="gray", opacity=0.05,
                layer="below", line_width=0,
            )

    fig.update_layout(
        title="<b>Predicción de Demanda Energética - Próximos 30 Días</b>",
        xaxis_title="Fecha",
        yaxis_title="Demanda (MW)",
        hovermode='x unified',
        height=500,
        showlegend=True
    )

    return fig


def plot_hourly_heatmap(predictions_df):
    """Mapa de calor de demanda horaria"""
    period_cols = [f'P{i}' for i in range(1, 25)]

    # Verificar si existen columnas de períodos
    if not all(col in predictions_df.columns for col in period_cols):
        return None

    # Crear matriz de horas × días
    hourly_matrix = predictions_df[period_cols].values.T

    # Fechas para el eje X
    dates = predictions_df['fecha'].dt.strftime('%Y-%m-%d').tolist()

    fig = go.Figure(data=go.Heatmap(
        z=hourly_matrix,
        x=dates,
        y=[f'{i:02d}:00' for i in range(24)],
        colorscale='YlOrRd',
        colorbar=dict(title="MW"),
        hovertemplate='<b>Fecha: %{x}</b><br>Hora: %{y}<br>Demanda: %{z:,.2f} MW<extra></extra>'
    ))

    fig.update_layout(
        title="<b>Patrón de Demanda Horaria - Próximos 30 Días</b>",
        xaxis_title="Fecha",
        yaxis_title="Hora del Día",
        height=600,
        xaxis=dict(tickangle=-45)
    )

    return fig


def plot_weekly_comparison(predictions_df):
    """Comparación por semana"""
    predictions_df['week'] = predictions_df['fecha'].dt.isocalendar().week

    weekly_stats = predictions_df.groupby('week').agg({
        'demanda_predicha': ['mean', 'min', 'max'],
        'fecha': 'first'
    }).reset_index()

    weekly_stats.columns = ['week', 'mean', 'min', 'max', 'start_date']

    fig = go.Figure()

    # Área entre min y max
    fig.add_trace(go.Scatter(
        x=weekly_stats['start_date'],
        y=weekly_stats['max'],
        fill=None,
        mode='lines',
        line_color='lightblue',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=weekly_stats['start_date'],
        y=weekly_stats['min'],
        fill='tonexty',
        mode='lines',
        line_color='lightblue',
        name='Rango Min-Max'
    ))

    # Línea promedio
    fig.add_trace(go.Scatter(
        x=weekly_stats['start_date'],
        y=weekly_stats['mean'],
        mode='lines+markers',
        name='Demanda Promedio',
        line=dict(color='blue', width=3),
        marker=dict(size=10)
    ))

    fig.update_layout(
        title="<b>Demanda Semanal Promedio</b>",
        xaxis_title="Semana",
        yaxis_title="Demanda (MW)",
        height=400
    )

    return fig


def plot_day_detail(predictions_df, selected_date):
    """Detalle horario de un día específico"""
    day_data = predictions_df[predictions_df['fecha'] == pd.to_datetime(selected_date)]

    if len(day_data) == 0:
        return None

    period_cols = [f'P{i}' for i in range(1, 25)]

    if not all(col in day_data.columns for col in period_cols):
        st.warning("No hay datos de desagregación horaria disponibles")
        return None

    hourly_values = day_data[period_cols].values[0]
    hours = list(range(24))

    fig = go.Figure()

    # Barras de demanda horaria
    fig.add_trace(go.Bar(
        x=hours,
        y=hourly_values,
        name='Demanda',
        marker_color='steelblue',
        hovertemplate='<b>Hora %{x}:00</b><br>Demanda: %{y:,.2f} MW<extra></extra>'
    ))

    # Línea de promedio
    avg_hourly = hourly_values.mean()
    fig.add_hline(
        y=avg_hourly,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Promedio: {avg_hourly:.2f} MW"
    )

    day_info = day_data.iloc[0]

    fig.update_layout(
        title=f"<b>Demanda Horaria - {selected_date}</b><br>"
              f"<sub>Total: {day_info['demanda_predicha']:,.2f} MW | "
              f"Tipo: {day_info.get('day_type', 'N/A')}</sub>",
        xaxis_title="Hora del Día",
        yaxis_title="Demanda (MW)",
        height=500,
        xaxis=dict(tickmode='linear', dtick=1)
    )

    return fig


def main():
    """Aplicación principal"""

    # Header
    st.markdown('<p class="main-header">⚡ Predicción de Demanda Energética - EPM</p>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: #e7f3ff; border-radius: 0.5rem; margin-bottom: 2rem;'>
        <h3 style='color: #1f77b4; margin: 0;'>🔮 Sistema de Pronóstico Automatizado</h3>
        <p style='margin: 0.5rem 0 0 0;'>Predicciones con desagregación horaria automática usando clustering inteligente</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("⚙️ Configuración")

    # Número de días a predecir
    n_days = st.sidebar.slider(
        "Días a predecir:",
        min_value=7,
        max_value=60,
        value=30,
        step=1
    )

    # Botón de generación
    if st.sidebar.button("🚀 Generar Predicciones", type="primary", use_container_width=True):
        st.cache_data.clear()  # Limpiar cache para regenerar

    # Información del sistema
    with st.sidebar.expander("ℹ️ Información del Sistema", expanded=False):
        st.write("**Modelo Principal:**")
        st.write("- XGBoost optimizado")
        st.write("- MAPE histórico: 0.45%")
        st.write("- 63 features predictivas")

        st.write("\n**Desagregación Horaria:**")
        st.write("- 35 clusters (días normales)")
        st.write("- 15 clusters (días especiales)")
        st.write("- Validación suma = total")

    # Estado del sistema
    engine, error = load_hourly_engine()

    if engine:
        status = engine.get_engine_status()
        with st.sidebar.expander("📊 Estado Desagregación", expanded=False):
            st.write(f"Normal: {'✅' if status['normal_disaggregator']['fitted'] else '❌'}")
            st.write(f"Especial: {'✅' if status['special_disaggregator']['fitted'] else '❌'}")

    # Generar predicciones
    with st.spinner(f"🔮 Generando predicciones para {n_days} días..."):
        predictions_df, error = generate_predictions(n_days)

    if predictions_df is None:
        st.error(f"❌ Error al generar predicciones: {error}")
        st.info("Verifique que el modelo esté entrenado:\n\n```bash\npython scripts/train_models.py\n```")
        st.stop()

    # Información de las predicciones
    fecha_inicio = predictions_df['fecha'].min()
    fecha_fin = predictions_df['fecha'].max()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>📅 Período de Predicción</h4>
            <p><b>{fecha_inicio.strftime('%Y-%m-%d')}</b> a <b>{fecha_fin.strftime('%Y-%m-%d')}</b></p>
            <span class='future-badge'>FUTURO</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>📊 Demanda Promedio</h4>
            <p style='font-size: 2rem; margin: 0; color: #1f77b4;'><b>{predictions_df['demanda_predicha'].mean():,.0f}</b> MW</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        dias_festivos = int(predictions_df.get('is_festivo', pd.Series([0]*len(predictions_df))).sum())
        st.markdown(f"""
        <div class='metric-card'>
            <h4>🎉 Días Especiales</h4>
            <p><b>{dias_festivos}</b> festivos<br><b>{int(predictions_df.get('is_weekend', pd.Series([0]*len(predictions_df))).sum())}</b> fin de semana</p>
        </div>
        """, unsafe_allow_html=True)

    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Vista General",
        "🔥 Mapa de Calor Horario",
        "📅 Detalle por Día",
        "✅ Validación con Históricos",
        "📊 Estadísticas"
    ])

    # ==================== TAB 1: VISTA GENERAL ====================
    with tab1:
        st.header("Predicción de Demanda Diaria")

        fig_daily = plot_daily_predictions(predictions_df)
        st.plotly_chart(fig_daily, use_container_width=True)

        # Comparación semanal
        st.subheader("Demanda por Semana")
        fig_weekly = plot_weekly_comparison(predictions_df)
        st.plotly_chart(fig_weekly, use_container_width=True)

        # Métricas adicionales
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Demanda Mínima", f"{predictions_df['demanda_predicha'].min():,.0f} MW")

        with col2:
            st.metric("Demanda Máxima", f"{predictions_df['demanda_predicha'].max():,.0f} MW")

        with col3:
            variacion = predictions_df['demanda_predicha'].std()
            st.metric("Desv. Estándar", f"{variacion:,.0f} MW")

        with col4:
            total_mensual = predictions_df['demanda_predicha'].sum()
            st.metric("Total Acumulado", f"{total_mensual:,.0f} MWh")

    # ==================== TAB 2: MAPA DE CALOR ====================
    with tab2:
        st.header("Patrón de Demanda Horaria (24h)")

        period_cols = [f'P{i}' for i in range(1, 25)]
        if all(col in predictions_df.columns for col in period_cols):
            fig_heatmap = plot_hourly_heatmap(predictions_df)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)

                # Estadísticas por hora
                st.subheader("Demanda Promedio por Hora del Día")

                hourly_avg = predictions_df[period_cols].mean()

                fig_hourly_avg = go.Figure()
                fig_hourly_avg.add_trace(go.Bar(
                    x=list(range(24)),
                    y=hourly_avg.values,
                    marker_color='teal',
                    text=[f'{v:.0f}' for v in hourly_avg.values],
                    textposition='outside'
                ))

                fig_hourly_avg.update_layout(
                    title="Demanda Promedio por Período Horario",
                    xaxis_title="Hora del Día",
                    yaxis_title="Demanda Promedio (MW)",
                    height=400
                )

                st.plotly_chart(fig_hourly_avg, use_container_width=True)
        else:
            st.warning("⚠️ Desagregación horaria no disponible")
            st.info("Para habilitar desagregación horaria:\n\n```bash\npython scripts/train_hourly_disaggregation.py\n```")

    # ==================== TAB 3: DETALLE POR DÍA ====================
    with tab3:
        st.header("Detalle Horario por Día")

        # Selector de fecha
        selected_date = st.date_input(
            "Seleccione una fecha:",
            value=fecha_inicio.date(),
            min_value=fecha_inicio.date(),
            max_value=fecha_fin.date()
        )

        fig_day = plot_day_detail(predictions_df, selected_date)

        if fig_day:
            st.plotly_chart(fig_day, use_container_width=True)

            # Tabla de datos horarios
            with st.expander("📋 Ver Datos Horarios Detallados"):
                day_data = predictions_df[predictions_df['fecha'] == pd.to_datetime(selected_date)]

                if len(day_data) > 0 and all(f'P{i}' in day_data.columns for i in range(1, 25)):
                    period_cols = [f'P{i}' for i in range(1, 25)]
                    hourly_values = day_data[period_cols].values[0]

                    df_hourly = pd.DataFrame({
                        'Hora': [f'{i:02d}:00 - {i+1:02d}:00' for i in range(24)],
                        'Demanda (MW)': hourly_values,
                        '% del Total': (hourly_values / hourly_values.sum() * 100)
                    })

                    st.dataframe(
                        df_hourly.style.format({
                            'Demanda (MW)': '{:,.2f}',
                            '% del Total': '{:.2f}%'
                        }),
                        use_container_width=True
                    )
        else:
            st.warning("No hay datos de desagregación horaria para esta fecha")

    # ==================== TAB 4: VALIDACIÓN CON HISTÓRICOS ====================
    with tab4:
        st.header("✅ Validación: Predicciones vs Datos Reales")

        st.info("📌 Esta sección compara las predicciones del sistema con datos históricos reales para validar la precisión.")

        # Cargar datos históricos
        @st.cache_data
        def load_historical_data():
            data_path = Path(FEATURES_DATA_DIR) / "data_with_features_latest.csv"
            if not data_path.exists():
                return None
            df = pd.read_csv(data_path)
            df['FECHA'] = pd.to_datetime(df['FECHA'])
            return df

        df_historico = load_historical_data()

        if df_historico is None:
            st.error("❌ No se encontraron datos históricos para validación")
        else:
            st.success(f"✅ {len(df_historico)} días históricos disponibles")

            # Selector de rango de validación
            available_dates = df_historico['FECHA'].dt.date.tolist()

            col1, col2 = st.columns(2)

            with col1:
                validation_start = st.date_input(
                    "Fecha inicial:",
                    value=available_dates[-30] if len(available_dates) >= 30 else available_dates[0],
                    min_value=min(available_dates),
                    max_value=max(available_dates)
                )

            with col2:
                validation_end = st.date_input(
                    "Fecha final:",
                    value=available_dates[-1],
                    min_value=validation_start,
                    max_value=max(available_dates)
                )

            if st.button("🔍 Ejecutar Validación", type="primary"):
                with st.spinner("Validando predicciones..."):
                    # Filtrar datos históricos
                    mask = (df_historico['FECHA'].dt.date >= validation_start) & \
                           (df_historico['FECHA'].dt.date <= validation_end)
                    df_validation = df_historico[mask].copy()

                    if len(df_validation) == 0:
                        st.warning("No hay datos en el rango seleccionado")
                    else:
                        # Cargar motor de desagregación
                        engine, error = load_hourly_engine()

                        if engine is None:
                            st.error(f"❌ Motor de desagregación no disponible: {error}")
                        else:
                            results = []
                            period_cols = [f'P{i}' for i in range(1, 25)]

                            progress_bar = st.progress(0)

                            for idx, row in df_validation.iterrows():
                                fecha = row['FECHA']
                                real_total = row['TOTAL'] if 'TOTAL' in row else row[period_cols].sum()
                                real_hourly = row[period_cols].values

                                # Predecir con el sistema
                                try:
                                    result = engine.predict_hourly(fecha, real_total)
                                    pred_hourly = result['hourly']

                                    # Calcular errores
                                    error_abs = np.abs(pred_hourly - real_hourly)
                                    error_pct = (pred_hourly - real_hourly) / real_hourly * 100

                                    results.append({
                                        'fecha': fecha,
                                        'real_total': real_total,
                                        'pred_total': result['total_daily'],
                                        'method': result['method'],
                                        'day_type': result['day_type'],
                                        'mae': np.mean(error_abs),
                                        'rmse': np.sqrt(np.mean((pred_hourly - real_hourly)**2)),
                                        'mape': np.mean(np.abs(error_pct)),
                                        'max_error': np.max(error_abs),
                                        'validation_ok': result['validation']['is_valid']
                                    })
                                except Exception as e:
                                    st.error(f"Error en {fecha}: {e}")

                                progress_bar.progress((idx + 1) / len(df_validation))

                            progress_bar.empty()

                            if results:
                                df_results = pd.DataFrame(results)

                                # Métricas globales
                                st.subheader("📊 Métricas Globales de Precisión")

                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    mae_avg = df_results['mae'].mean()
                                    st.metric(
                                        "MAE Promedio",
                                        f"{mae_avg:.2f} MW",
                                        delta=f"{'✅' if mae_avg < 5 else '⚠️'}"
                                    )

                                with col2:
                                    rmse_avg = df_results['rmse'].mean()
                                    st.metric(
                                        "RMSE Promedio",
                                        f"{rmse_avg:.2f} MW",
                                        delta=f"{'✅' if rmse_avg < 10 else '⚠️'}"
                                    )

                                with col3:
                                    mape_avg = df_results['mape'].mean()
                                    mape_status = "✅ EXCELENTE" if mape_avg < 2 else \
                                                 "✅ BUENO" if mape_avg < 5 else "⚠️ ACEPTABLE"
                                    st.metric(
                                        "MAPE Promedio",
                                        f"{mape_avg:.2f}%",
                                        delta=mape_status
                                    )

                                with col4:
                                    validation_ok = (df_results['validation_ok'].sum() / len(df_results)) * 100
                                    st.metric(
                                        "Validación OK",
                                        f"{validation_ok:.1f}%",
                                        delta="✅" if validation_ok > 95 else "⚠️"
                                    )

                                # Gráfica de comparación temporal
                                st.subheader("📈 MAPE a lo Largo del Tiempo")

                                fig_temporal = go.Figure()

                                fig_temporal.add_trace(go.Scatter(
                                    x=df_results['fecha'],
                                    y=df_results['mape'],
                                    mode='lines+markers',
                                    name='MAPE Diario',
                                    line=dict(color='steelblue', width=2),
                                    marker=dict(size=6)
                                ))

                                # Línea de objetivo regulatorio
                                fig_temporal.add_hline(
                                    y=5,
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text="Objetivo Regulatorio: 5%"
                                )

                                # Línea de promedio
                                fig_temporal.add_hline(
                                    y=mape_avg,
                                    line_dash="dot",
                                    line_color="green",
                                    annotation_text=f"Promedio: {mape_avg:.2f}%"
                                )

                                fig_temporal.update_layout(
                                    title="MAPE Diario - Validación Histórica",
                                    xaxis_title="Fecha",
                                    yaxis_title="MAPE (%)",
                                    height=400,
                                    hovermode='x unified'
                                )

                                st.plotly_chart(fig_temporal, use_container_width=True)

                                # Comparación por tipo de día
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.subheader("📊 MAPE por Tipo de Día")

                                    fig_box = px.box(
                                        df_results,
                                        x='day_type',
                                        y='mape',
                                        color='day_type',
                                        title="Distribución de MAPE por Tipo de Día",
                                        labels={'day_type': 'Tipo de Día', 'mape': 'MAPE (%)'}
                                    )
                                    st.plotly_chart(fig_box, use_container_width=True)

                                with col2:
                                    st.subheader("📊 MAPE por Método")

                                    fig_method = px.box(
                                        df_results,
                                        x='method',
                                        y='mape',
                                        color='method',
                                        title="Distribución de MAPE por Método de Desagregación",
                                        labels={'method': 'Método', 'mape': 'MAPE (%)'}
                                    )
                                    st.plotly_chart(fig_method, use_container_width=True)

                                # Tabla resumen
                                st.subheader("📋 Resumen Estadístico por Categoría")

                                summary_by_type = df_results.groupby('day_type')['mape'].agg([
                                    ('Media (%)', 'mean'),
                                    ('Mínimo (%)', 'min'),
                                    ('Máximo (%)', 'max'),
                                    ('Desv. Std (%)', 'std'),
                                    ('Días', 'count')
                                ]).round(2)

                                st.dataframe(summary_by_type, use_container_width=True)

                                # Días con mayor error
                                st.subheader("⚠️ Días con Mayor Error (Top 10)")

                                top_errors = df_results.nlargest(10, 'mape')[
                                    ['fecha', 'day_type', 'method', 'mape', 'mae', 'rmse']
                                ].copy()

                                top_errors['fecha'] = top_errors['fecha'].dt.strftime('%Y-%m-%d')

                                st.dataframe(
                                    top_errors.style.format({
                                        'mape': '{:.2f}%',
                                        'mae': '{:.2f} MW',
                                        'rmse': '{:.2f} MW'
                                    }),
                                    use_container_width=True
                                )

                                # Comparación visual de un día específico
                                st.subheader("🔍 Detalle de Día Específico")

                                selected_validation_date = st.selectbox(
                                    "Seleccione un día para ver detalle:",
                                    options=df_results['fecha'].dt.strftime('%Y-%m-%d').tolist()
                                )

                                if selected_validation_date:
                                    sel_date = pd.to_datetime(selected_validation_date)
                                    day_row = df_validation[df_validation['FECHA'] == sel_date].iloc[0]

                                    real_hourly = day_row[period_cols].values
                                    real_total = day_row['TOTAL'] if 'TOTAL' in day_row else real_hourly.sum()

                                    result_sel = engine.predict_hourly(sel_date, real_total)
                                    pred_hourly = result_sel['hourly']

                                    # Gráfica comparativa
                                    fig_compare = go.Figure()

                                    hours = list(range(24))

                                    fig_compare.add_trace(go.Scatter(
                                        x=hours,
                                        y=real_hourly,
                                        mode='lines+markers',
                                        name='Real',
                                        line=dict(color='blue', width=3),
                                        marker=dict(size=8)
                                    ))

                                    fig_compare.add_trace(go.Scatter(
                                        x=hours,
                                        y=pred_hourly,
                                        mode='lines+markers',
                                        name='Predicción',
                                        line=dict(color='red', width=3, dash='dash'),
                                        marker=dict(size=6, symbol='square')
                                    ))

                                    error_abs = np.abs(pred_hourly - real_hourly)
                                    mape_day = np.mean(np.abs((pred_hourly - real_hourly) / real_hourly * 100))

                                    fig_compare.update_layout(
                                        title=f"Comparación Detallada: {selected_validation_date}<br>"
                                              f"<sub>MAPE: {mape_day:.2f}% | MAE: {np.mean(error_abs):.2f} MW</sub>",
                                        xaxis_title="Hora del Día",
                                        yaxis_title="Demanda (MW)",
                                        height=500,
                                        hovermode='x unified'
                                    )

                                    st.plotly_chart(fig_compare, use_container_width=True)

    # ==================== TAB 5: ESTADÍSTICAS ====================
    with tab5:
        st.header("Estadísticas y Análisis")

        col1, col2 = st.columns(2)

        with col1:
            # Distribución de demanda
            fig_dist = px.histogram(
                predictions_df,
                x='demanda_predicha',
                nbins=30,
                title="Distribución de Demanda Predicha",
                labels={'demanda_predicha': 'Demanda (MW)'},
                color_discrete_sequence=['steelblue']
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            # Box plot por día de la semana
            predictions_df['dia_semana'] = predictions_df['fecha'].dt.day_name()

            fig_box = px.box(
                predictions_df,
                x='dia_semana',
                y='demanda_predicha',
                title="Demanda por Día de la Semana",
                labels={'dia_semana': 'Día', 'demanda_predicha': 'Demanda (MW)'},
                color='dia_semana'
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # Tabla resumen
        st.subheader("Resumen Estadístico")

        summary = predictions_df['demanda_predicha'].describe()

        summary_df = pd.DataFrame({
            'Métrica': ['Media', 'Desv. Estándar', 'Mínimo', '25%', 'Mediana', '75%', 'Máximo'],
            'Valor (MW)': [
                summary['mean'],
                summary['std'],
                summary['min'],
                summary['25%'],
                summary['50%'],
                summary['75%'],
                summary['max']
            ]
        })

        st.dataframe(
            summary_df.style.format({'Valor (MW)': '{:,.2f}'}),
            use_container_width=True
        )

        # Descargar datos
        st.subheader("💾 Exportar Datos")

        csv = predictions_df.to_csv(index=False)

        st.download_button(
            label="📥 Descargar Predicciones (CSV)",
            data=csv,
            file_name=f"predicciones_epm_{fecha_inicio.strftime('%Y%m%d')}_{fecha_fin.strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
