"""
Dashboard de Comparación: Predicción Horaria vs Real (30 Días)

Dashboard para comparar predicciones de 30 días con desagregación horaria (24 períodos)
contra los datos históricos reales.

Uso:
    streamlit run dashboards/hourly_comparison_dashboard.py
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

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction.hourly import HourlyDisaggregationEngine
from src.config.settings import FEATURES_DATA_DIR

# Configuración
st.set_page_config(
    page_title="Comparación Horaria 30 Días - EPM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-good { border-left: 4px solid #28a745; }
    .metric-warning { border-left: 4px solid #ffc107; }
    .metric-bad { border-left: 4px solid #dc3545; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_historical_data():
    """Carga datos históricos"""
    data_path = Path(FEATURES_DATA_DIR) / "data_with_features_latest.csv"
    if not data_path.exists():
        return None
    df = pd.read_csv(data_path)
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    return df


@st.cache_resource
def load_engine():
    """Carga motor de desagregación"""
    try:
        engine = HourlyDisaggregationEngine(auto_load=True)
        return engine, None
    except Exception as e:
        return None, str(e)


def predict_30_days_hourly(engine, df_historico, start_date, n_days=30):
    """
    Predice 30 días con desagregación horaria completa

    Returns:
        df_predictions: DataFrame con predicciones (fecha, P1-P24)
        df_real: DataFrame con datos reales (fecha, P1-P24)
    """
    # Filtrar datos reales
    end_date = start_date + timedelta(days=n_days-1)
    mask = (df_historico['FECHA'].dt.date >= start_date) & \
           (df_historico['FECHA'].dt.date <= end_date)
    df_real = df_historico[mask].copy()

    if len(df_real) == 0:
        return None, None

    # Generar predicciones
    period_cols = [f'P{i}' for i in range(1, 25)]
    predictions = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for counter, (idx, row) in enumerate(df_real.iterrows()):
        fecha = row['FECHA']
        real_total = row['TOTAL'] if 'TOTAL' in row else row[period_cols].sum()

        status_text.text(f"Prediciendo {fecha.strftime('%Y-%m-%d')} ({counter+1}/{len(df_real)})...")

        # Predecir con el motor
        result = engine.predict_hourly(fecha, real_total)
        pred_hourly = result['hourly']

        pred_row = {
            'FECHA': fecha,
            'TOTAL': real_total,
            'method': result['method'],
            'day_type': result['day_type'],
        }
        pred_row.update({f'P{i}': pred_hourly[i-1] for i in range(1, 25)})

        predictions.append(pred_row)

        progress_bar.progress((counter + 1) / len(df_real))

    progress_bar.empty()
    status_text.empty()

    df_predictions = pd.DataFrame(predictions)

    # Seleccionar solo columnas necesarias del real
    cols_needed = ['FECHA', 'TOTAL'] + period_cols
    df_real = df_real[cols_needed].copy()

    return df_predictions, df_real


def calculate_metrics(df_pred, df_real):
    """Calcula métricas de comparación"""
    period_cols = [f'P{i}' for i in range(1, 25)]

    metrics = []

    for idx in range(len(df_pred)):
        fecha = df_pred.iloc[idx]['FECHA']

        pred_hourly = df_pred.iloc[idx][period_cols].values
        real_hourly = df_real.iloc[idx][period_cols].values

        # Errores
        error_abs = np.abs(pred_hourly - real_hourly)
        error_pct = (pred_hourly - real_hourly) / real_hourly * 100

        metrics.append({
            'fecha': fecha,
            'day_type': df_pred.iloc[idx]['day_type'],
            'method': df_pred.iloc[idx]['method'],
            'mae': np.mean(error_abs),
            'rmse': np.sqrt(np.mean((pred_hourly - real_hourly)**2)),
            'mape': np.mean(np.abs(error_pct)),
            'max_error': np.max(error_abs),
            'max_error_pct': np.max(np.abs(error_pct)),
            'periods_over_5pct': np.sum(np.abs(error_pct) > 5)
        })

    return pd.DataFrame(metrics)


def plot_heatmap_comparison(df_pred, df_real, title_prefix=""):
    """Crea mapa de calor comparativo"""
    period_cols = [f'P{i}' for i in range(1, 25)]

    # Matrices
    pred_matrix = df_pred[period_cols].values.T
    real_matrix = df_real[period_cols].values.T
    error_matrix = pred_matrix - real_matrix
    error_pct_matrix = (error_matrix / real_matrix) * 100

    # Fechas
    dates = df_pred['FECHA'].dt.strftime('%Y-%m-%d').tolist()
    hours = [f'{i:02d}:00' for i in range(24)]

    # Crear subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Predicción (MW)',
            'Real (MW)',
            'Error Absoluto (MW)',
            'Error Porcentual (%)'
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    # 1. Predicción
    fig.add_trace(
        go.Heatmap(
            z=pred_matrix,
            x=dates,
            y=hours,
            colorscale='YlOrRd',
            name='Predicción',
            showscale=True,
            colorbar=dict(x=0.46, len=0.4, y=0.77)
        ),
        row=1, col=1
    )

    # 2. Real
    fig.add_trace(
        go.Heatmap(
            z=real_matrix,
            x=dates,
            y=hours,
            colorscale='YlOrRd',
            name='Real',
            showscale=True,
            colorbar=dict(x=1.02, len=0.4, y=0.77)
        ),
        row=1, col=2
    )

    # 3. Error Absoluto
    fig.add_trace(
        go.Heatmap(
            z=np.abs(error_matrix),
            x=dates,
            y=hours,
            colorscale='Reds',
            name='Error Abs',
            showscale=True,
            colorbar=dict(x=0.46, len=0.4, y=0.23)
        ),
        row=2, col=1
    )

    # 4. Error Porcentual
    fig.add_trace(
        go.Heatmap(
            z=error_pct_matrix,
            x=dates,
            y=hours,
            colorscale='RdBu_r',
            zmid=0,
            name='Error %',
            showscale=True,
            colorbar=dict(x=1.02, len=0.4, y=0.23)
        ),
        row=2, col=2
    )

    fig.update_xaxes(tickangle=-45)

    fig.update_layout(
        title=f"<b>{title_prefix}Comparación Horaria: Predicción vs Real (30 Días)</b>",
        height=800,
        showlegend=False
    )

    return fig


def plot_daily_comparison(df_metrics):
    """Gráfica de métricas diarias"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('MAPE Diario (%)', 'MAE Diario (MW)'),
        vertical_spacing=0.15
    )

    # MAPE
    colors_mape = ['green' if m < 2 else 'orange' if m < 5 else 'red'
                   for m in df_metrics['mape']]

    fig.add_trace(
        go.Bar(
            x=df_metrics['fecha'],
            y=df_metrics['mape'],
            name='MAPE',
            marker_color=colors_mape,
            showlegend=False
        ),
        row=1, col=1
    )

    fig.add_hline(y=5, line_dash="dash", line_color="red",
                 annotation_text="Objetivo: 5%", row=1, col=1)

    fig.add_hline(y=df_metrics['mape'].mean(), line_dash="dot",
                 line_color="blue",
                 annotation_text=f"Promedio: {df_metrics['mape'].mean():.2f}%",
                 row=1, col=1)

    # MAE
    fig.add_trace(
        go.Bar(
            x=df_metrics['fecha'],
            y=df_metrics['mae'],
            name='MAE',
            marker_color='steelblue',
            showlegend=False
        ),
        row=2, col=1
    )

    fig.add_hline(y=df_metrics['mae'].mean(), line_dash="dot",
                 line_color="green",
                 annotation_text=f"Promedio: {df_metrics['mae'].mean():.2f} MW",
                 row=2, col=1)

    fig.update_xaxes(tickangle=-45)
    fig.update_layout(height=600, showlegend=False)

    return fig


def plot_hourly_average_comparison(df_pred, df_real):
    """Compara promedio por hora del día"""
    period_cols = [f'P{i}' for i in range(1, 25)]

    pred_avg = df_pred[period_cols].mean()
    real_avg = df_real[period_cols].mean()

    fig = go.Figure()

    hours = list(range(24))

    fig.add_trace(go.Scatter(
        x=hours,
        y=real_avg.values,
        mode='lines+markers',
        name='Real Promedio',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=hours,
        y=pred_avg.values,
        mode='lines+markers',
        name='Predicción Promedio',
        line=dict(color='red', width=3, dash='dash'),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title="Patrón Promedio de Demanda Horaria (30 días)",
        xaxis_title="Hora del Día",
        yaxis_title="Demanda Promedio (MW)",
        height=400
    )

    return fig


def main():
    """Aplicación principal"""

    st.markdown('<p class="main-header">📊 Comparación Horaria: 30 Días Predichos vs Reales</p>',
                unsafe_allow_html=True)

    st.info("📌 Este dashboard compara **30 días completos** con **24 períodos horarios** cada uno: "
            "Predicción vs Datos Reales")

    # Sidebar
    st.sidebar.title("⚙️ Configuración")

    # Cargar datos
    df_historico = load_historical_data()

    if df_historico is None:
        st.error("❌ No se encontraron datos históricos")
        st.stop()

    st.sidebar.success(f"✅ {len(df_historico)} días disponibles")

    # Cargar motor
    engine, error = load_engine()

    if engine is None:
        st.error(f"❌ Error al cargar motor: {error}")
        st.info("Entrene primero:\n```bash\npython scripts/train_hourly_disaggregation.py\n```")
        st.stop()

    st.sidebar.success("✅ Motor de desagregación cargado")

    # Selector de fecha
    available_dates = df_historico['FECHA'].dt.date.tolist()

    st.sidebar.subheader("📅 Selección de Período")

    # Por defecto: últimos 30 días
    default_start = available_dates[-30] if len(available_dates) >= 30 else available_dates[0]

    start_date = st.sidebar.date_input(
        "Fecha inicial (30 días desde aquí):",
        value=default_start,
        min_value=min(available_dates),
        max_value=max(available_dates) - timedelta(days=29)
    )

    n_days = st.sidebar.slider("Número de días:", 7, 60, 30)

    # Botón de generación
    if st.sidebar.button("🚀 Generar Comparación", type="primary", use_container_width=True):
        with st.spinner(f"Generando predicciones para {n_days} días..."):
            df_pred, df_real = predict_30_days_hourly(engine, df_historico, start_date, n_days)

            if df_pred is None:
                st.error("No hay datos suficientes para el período seleccionado")
                st.stop()

            # Guardar en session_state
            st.session_state['df_pred'] = df_pred
            st.session_state['df_real'] = df_real
            st.session_state['start_date'] = start_date
            st.session_state['n_days'] = n_days

    # Verificar si hay datos
    if 'df_pred' not in st.session_state:
        st.warning("👆 Seleccione un período y haga click en 'Generar Comparación'")
        st.stop()

    # Obtener datos de session_state
    df_pred = st.session_state['df_pred']
    df_real = st.session_state['df_real']
    start_date = st.session_state['start_date']
    n_days = st.session_state['n_days']

    # Calcular métricas
    df_metrics = calculate_metrics(df_pred, df_real)

    # Header con métricas globales
    st.subheader(f"📊 Período: {start_date} → {start_date + timedelta(days=n_days-1)}")

    col1, col2, col3, col4, col5 = st.columns(5)

    mae_avg = df_metrics['mae'].mean()
    rmse_avg = df_metrics['rmse'].mean()
    mape_avg = df_metrics['mape'].mean()

    with col1:
        metric_class = "metric-good" if mape_avg < 2 else "metric-warning" if mape_avg < 5 else "metric-bad"
        st.markdown(f'<div class="{metric_class}" style="padding: 1rem; border-radius: 0.5rem;">',
                   unsafe_allow_html=True)
        st.metric("MAPE Promedio", f"{mape_avg:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.metric("MAE Promedio", f"{mae_avg:.2f} MW")

    with col3:
        st.metric("RMSE Promedio", f"{rmse_avg:.2f} MW")

    with col4:
        st.metric("Error Máximo", f"{df_metrics['max_error'].max():.2f} MW")

    with col5:
        periods_over_5 = df_metrics['periods_over_5pct'].sum()
        total_periods = n_days * 24
        pct_over_5 = (periods_over_5 / total_periods) * 100
        st.metric("Períodos > 5%", f"{periods_over_5}/{total_periods} ({pct_over_5:.1f}%)")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔥 Mapas de Calor",
        "📈 Métricas Diarias",
        "⏰ Promedio Horario",
        "📋 Datos Detallados"
    ])

    # TAB 1: Mapas de Calor
    with tab1:
        st.header("Comparación Visual: Mapas de Calor")

        fig_heatmap = plot_heatmap_comparison(df_pred, df_real)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown("""
        **Interpretación:**
        - 🟨🟥 **Predicción y Real**: Intensidad de demanda (más rojo = mayor demanda)
        - 🟥 **Error Absoluto**: Diferencia en MW (más rojo = mayor error)
        - 🔵🔴 **Error %**: Azul = subestimación, Rojo = sobreestimación
        """)

    # TAB 2: Métricas Diarias
    with tab2:
        st.header("Evolución de Métricas Día por Día")

        fig_daily = plot_daily_comparison(df_metrics)
        st.plotly_chart(fig_daily, use_container_width=True)

        # Tabla de métricas
        st.subheader("📊 Métricas por Día")

        df_metrics_display = df_metrics.copy()
        df_metrics_display['fecha'] = df_metrics_display['fecha'].dt.strftime('%Y-%m-%d')

        st.dataframe(
            df_metrics_display.style.format({
                'mape': '{:.2f}%',
                'mae': '{:.2f} MW',
                'rmse': '{:.2f} MW',
                'max_error': '{:.2f} MW',
                'max_error_pct': '{:.2f}%'
            }),
            use_container_width=True
        )

        # Estadísticas por tipo de día
        st.subheader("📋 Resumen por Tipo de Día")

        summary = df_metrics.groupby('day_type').agg({
            'mape': ['mean', 'min', 'max', 'std'],
            'mae': ['mean', 'min', 'max'],
            'fecha': 'count'
        }).round(2)

        summary.columns = ['_'.join(col) for col in summary.columns]
        summary = summary.rename(columns={'fecha_count': 'Días'})

        st.dataframe(summary, use_container_width=True)

    # TAB 3: Promedio Horario
    with tab3:
        st.header("Patrón Promedio de Demanda Horaria")

        fig_hourly = plot_hourly_average_comparison(df_pred, df_real)
        st.plotly_chart(fig_hourly, use_container_width=True)

        # Comparación por tipo de día
        st.subheader("Comparación por Tipo de Día")

        col1, col2 = st.columns(2)

        period_cols = [f'P{i}' for i in range(1, 25)]

        with col1:
            # MAPE por tipo de día
            fig_box = px.box(
                df_metrics,
                x='day_type',
                y='mape',
                color='day_type',
                title="MAPE por Tipo de Día"
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with col2:
            # MAPE por método
            fig_method = px.box(
                df_metrics,
                x='method',
                y='mape',
                color='method',
                title="MAPE por Método de Desagregación"
            )
            st.plotly_chart(fig_method, use_container_width=True)

    # TAB 4: Datos Detallados
    with tab4:
        st.header("Datos Completos")

        # Selector de día
        selected_date = st.selectbox(
            "Seleccione un día para ver detalle:",
            options=df_pred['FECHA'].dt.strftime('%Y-%m-%d').tolist()
        )

        if selected_date:
            sel_date = pd.to_datetime(selected_date)

            pred_row = df_pred[df_pred['FECHA'] == sel_date].iloc[0]
            real_row = df_real[df_real['FECHA'] == sel_date].iloc[0]

            period_cols = [f'P{i}' for i in range(1, 25)]

            pred_hourly = pred_row[period_cols].values
            real_hourly = real_row[period_cols].values

            # Gráfica comparativa del día
            fig_day = go.Figure()

            hours = list(range(24))

            fig_day.add_trace(go.Scatter(
                x=hours,
                y=real_hourly,
                mode='lines+markers',
                name='Real',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))

            fig_day.add_trace(go.Scatter(
                x=hours,
                y=pred_hourly,
                mode='lines+markers',
                name='Predicción',
                line=dict(color='red', width=3, dash='dash'),
                marker=dict(size=6)
            ))

            error_abs = np.abs(pred_hourly - real_hourly)
            mape_day = np.mean(np.abs((pred_hourly - real_hourly) / real_hourly * 100))

            fig_day.update_layout(
                title=f"Comparación Detallada: {selected_date}<br>"
                      f"<sub>MAPE: {mape_day:.2f}% | MAE: {np.mean(error_abs):.2f} MW</sub>",
                xaxis_title="Hora del Día",
                yaxis_title="Demanda (MW)",
                height=500
            )

            st.plotly_chart(fig_day, use_container_width=True)

            # Tabla comparativa
            st.subheader("Tabla de Comparación Horaria")

            df_comparison = pd.DataFrame({
                'Hora': [f'{i:02d}:00-{i+1:02d}:00' for i in range(24)],
                'Real (MW)': real_hourly,
                'Predicción (MW)': pred_hourly,
                'Error (MW)': pred_hourly - real_hourly,
                'Error (%)': (pred_hourly - real_hourly) / real_hourly * 100
            })

            st.dataframe(
                df_comparison.style.format({
                    'Real (MW)': '{:.2f}',
                    'Predicción (MW)': '{:.2f}',
                    'Error (MW)': '{:+.2f}',
                    'Error (%)': '{:+.2f}%'
                }).background_gradient(subset=['Error (%)'], cmap='RdYlGn_r', vmin=-10, vmax=10),
                use_container_width=True
            )

        # Exportar datos
        st.subheader("💾 Exportar Datos")

        col1, col2 = st.columns(2)

        with col1:
            csv_pred = df_pred.to_csv(index=False)
            st.download_button(
                "📥 Descargar Predicciones",
                csv_pred,
                f"predicciones_{start_date}.csv",
                "text/csv",
                use_container_width=True
            )

        with col2:
            csv_metrics = df_metrics.to_csv(index=False)
            st.download_button(
                "📥 Descargar Métricas",
                csv_metrics,
                f"metricas_{start_date}.csv",
                "text/csv",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
