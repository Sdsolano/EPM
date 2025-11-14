"""
Dashboard Interactivo - Resultados del Modelo Prototipo
Sistema de Pronóstico de Demanda Energética - EPM
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(
    page_title="Resultados Modelo Prototipo - EPM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #667eea;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #718096;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📊 Resultados del Modelo Prototipo</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sistema de Pronóstico de Demanda Energética - EPM</div>', unsafe_allow_html=True)

# Cargar datos
@st.cache_data
def load_data():
    base_path = Path("data/features")

    # Cargar resumen
    with open(base_path / "prototype_summary.json", 'r') as f:
        summary = json.load(f)

    # Cargar predicciones
    predictions = pd.read_csv(base_path / "prototype_predictions.csv")

    # Cargar predicciones de todos los modelos
    try:
        all_models_predictions = pd.read_csv(base_path / "prototype_all_models_predictions.csv")
    except:
        all_models_predictions = None

    # Cargar feature importance si existe
    try:
        feature_importance = pd.read_csv(base_path / "feature_importance_prototype.csv")
    except:
        feature_importance = None

    return summary, predictions, all_models_predictions, feature_importance

try:
    summary, predictions, all_models_predictions, feature_importance = load_data()
except FileNotFoundError:
    st.error("❌ No se encontraron los archivos de resultados. Ejecuta primero: `python prototype_model.py`")
    st.stop()

# Sidebar con información clave
with st.sidebar:
    st.markdown("### 🎯 Información del Modelo")
    st.metric("Modelo Ganador", summary['best_model'])
    st.metric("MAPE Test", f"{summary['test_mape']:.2f}%")
    st.metric("R² Score", f"{summary['test_r2']:.3f}")

    st.markdown("---")
    st.markdown("### 📊 Dataset")
    st.write(f"**Train:** {summary['train_size']} registros")
    st.write(f"**Test:** {summary['test_size']} registros")
    st.write(f"**Features:** {summary['total_features']}")

    st.markdown("---")
    st.markdown("### ✅ Cumplimiento")
    if summary['cumple_objetivo_5pct']:
        st.success("✅ Cumple MAPE < 5%")
        st.write(f"**{(5 / summary['test_mape']):.1f}x mejor** que el objetivo")
    else:
        st.warning("⚠️ No cumple MAPE < 5%")

# Métricas principales
st.markdown("## 📈 Métricas Principales")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="MAPE (Test)",
        value=f"{summary['test_mape']:.2f}%",
        delta=f"{5 - summary['test_mape']:.2f}% mejor que objetivo",
        delta_color="normal"
    )

with col2:
    st.metric(
        label="MAE (Test)",
        value=f"{summary['test_mae']:.2f}",
        help="Mean Absolute Error"
    )

with col3:
    st.metric(
        label="RMSE (Test)",
        value=f"{summary['test_rmse']:.2f}",
        help="Root Mean Squared Error"
    )

with col4:
    st.metric(
        label="R² Score",
        value=f"{summary['test_r2']:.3f}",
        help="Coeficiente de determinación"
    )

# Validación Cruzada
st.markdown("## 🔄 Validación Cruzada Temporal")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="MAPE Promedio (CV)",
        value=f"{summary['cv_mape_mean']:.2f}%"
    )

with col2:
    st.metric(
        label="Desviación Estándar",
        value=f"{summary['cv_mape_std']:.2f}%"
    )



# Comparación de Modelos
if all_models_predictions is not None:
    st.markdown("## 🔬 Comparación de los 3 Modelos")

    # Calcular métricas para cada modelo
    models_comparison = []

    for col in all_models_predictions.columns:
        if col.startswith('pred_'):
            model_name = col.replace('pred_', '').replace('_', ' ').title()
            pred_values = all_models_predictions[col]
            actual_values = all_models_predictions['actual']

            mae = np.mean(np.abs(actual_values - pred_values))
            mape = np.mean(np.abs((actual_values - pred_values) / actual_values)) * 100
            rmse = np.sqrt(np.mean((actual_values - pred_values) ** 2))
            r2 = 1 - (np.sum((actual_values - pred_values) ** 2) / np.sum((actual_values - np.mean(actual_values)) ** 2))

            models_comparison.append({
                'Modelo': model_name,
                'MAPE (%)': mape,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2
            })

    comparison_df = pd.DataFrame(models_comparison)

    # Tabla de comparación
    st.markdown("### Métricas Comparativas")

    # Formatear para visualización
    comparison_display = comparison_df.copy()
    comparison_display['MAPE (%)'] = comparison_display['MAPE (%)'].apply(lambda x: f"{x:.2f}%")
    comparison_display['MAE'] = comparison_display['MAE'].apply(lambda x: f"{x:.2f}")
    comparison_display['RMSE'] = comparison_display['RMSE'].apply(lambda x: f"{x:.2f}")
    comparison_display['R²'] = comparison_display['R²'].apply(lambda x: f"{x:.3f}")

    st.dataframe(comparison_display, hide_index=True, use_container_width=True)

    # Gráficos de comparación
    col1, col2 = st.columns(2)

    with col1:
        # Comparación de MAPE
        fig_mape_comp = px.bar(
            comparison_df,
            x='Modelo',
            y='MAPE (%)',
            title='MAPE por Modelo',
            color='MAPE (%)',
            color_continuous_scale='RdYlGn_r',
            text='MAPE (%)'
        )
        fig_mape_comp.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_mape_comp.add_hline(y=5, line_dash="dash", line_color="red",
                                annotation_text="Objetivo: 5%", annotation_position="right")
        fig_mape_comp.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_mape_comp, use_container_width=True)

    with col2:
        # Comparación de R²
        fig_r2_comp = px.bar(
            comparison_df,
            x='Modelo',
            y='R²',
            title='R² por Modelo',
            color='R²',
            color_continuous_scale='Viridis',
            text='R²'
        )
        fig_r2_comp.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_r2_comp.update_layout(showlegend=False, height=400, yaxis_range=[0, 1])
        st.plotly_chart(fig_r2_comp, use_container_width=True)

    # Gráfico comparativo mensual de los 3 modelos
    st.markdown("### Predicciones Mensuales: Comparación de Modelos")

    # Agregar fechas
    all_models_with_date = all_models_predictions.copy()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=len(all_models_with_date)-1)
    all_models_with_date['fecha'] = pd.date_range(start=start_date, periods=len(all_models_with_date), freq='D')
    all_models_with_date['year_month'] = all_models_with_date['fecha'].dt.to_period('M')

    # Agregación mensual
    monthly_comparison = all_models_with_date.groupby('year_month').agg({
        'actual': 'mean',
        'pred_linear_regression': 'mean',
        'pred_random_forest': 'mean',
        'pred_gradient_boosting': 'mean'
    }).reset_index()

    monthly_comparison['year_month_str'] = monthly_comparison['year_month'].astype(str)

    fig_models_monthly = go.Figure()

    # Real
    fig_models_monthly.add_trace(go.Scatter(
        x=monthly_comparison['year_month_str'],
        y=monthly_comparison['actual'],
        mode='lines+markers',
        name='Real',
        line=dict(color='white', width=3),
        marker=dict(size=8)
    ))

    # Linear Regression
    fig_models_monthly.add_trace(go.Scatter(
        x=monthly_comparison['year_month_str'],
        y=monthly_comparison['pred_linear_regression'],
        mode='lines+markers',
        name='Linear Regression',
        line=dict(color='#667eea', width=2, dash='dash'),
        marker=dict(size=6)
    ))

    # Random Forest
    fig_models_monthly.add_trace(go.Scatter(
        x=monthly_comparison['year_month_str'],
        y=monthly_comparison['pred_random_forest'],
        mode='lines+markers',
        name='Random Forest',
        line=dict(color='#48bb78', width=2, dash='dot'),
        marker=dict(size=6)
    ))

    # Gradient Boosting
    fig_models_monthly.add_trace(go.Scatter(
        x=monthly_comparison['year_month_str'],
        y=monthly_comparison['pred_gradient_boosting'],
        mode='lines+markers',
        name='Gradient Boosting',
        line=dict(color='#f56565', width=2, dash='dashdot'),
        marker=dict(size=6)
    ))

    fig_models_monthly.update_layout(
        title='Comparación Mensual: Real vs 3 Modelos',
        xaxis_title='Mes',
        yaxis_title='Demanda Promedio',
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_models_monthly, use_container_width=True)

# Análisis de Errores
st.markdown("## 📊 Análisis de Errores (Modelo Ganador)")

col1, col2 = st.columns(2)

# Distribución de errores porcentuales
with col1:
    st.markdown("### Distribución de Errores Porcentuales")

    error_bins = [
        ("< 1%", (predictions['error_pct'] < 1).sum()),
        ("1-3%", ((predictions['error_pct'] >= 1) & (predictions['error_pct'] < 3)).sum()),
        ("3-5%", ((predictions['error_pct'] >= 3) & (predictions['error_pct'] < 5)).sum()),
        ("> 5%", (predictions['error_pct'] >= 5).sum())
    ]

    error_df = pd.DataFrame(error_bins, columns=['Rango', 'Cantidad'])
    error_df['Porcentaje'] = (error_df['Cantidad'] / len(predictions) * 100).round(1)

    fig_bars = px.bar(
        error_df,
        x='Rango',
        y='Cantidad',
        text='Porcentaje',
        title='',
        color='Cantidad',
        color_continuous_scale='Viridis'
    )
    fig_bars.update_traces(texttemplate='%{text}%', textposition='outside')
    fig_bars.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_bars, use_container_width=True)

# Estadísticas de error
with col2:
    st.markdown("### Estadísticas de Error")

    stats = {
        'Métrica': ['Promedio', 'Mediana', 'Mínimo', 'Máximo', 'Desv. Estándar'],
        'Valor': [
            f"{predictions['error'].mean():.2f}",
            f"{predictions['error'].median():.2f}",
            f"{predictions['error'].min():.2f}",
            f"{predictions['error'].max():.2f}",
            f"{predictions['error'].std():.2f}"
        ]
    }

    st.dataframe(pd.DataFrame(stats), hide_index=True, use_container_width=True)

    # Mensaje de cumplimiento
    dias_con_error_mayor_5 = (predictions['error_pct'] > 5).sum()
    pct_dias_error_5 = (dias_con_error_mayor_5 / len(predictions)) * 100


# Gráfico: Predicciones vs Reales
st.markdown("## 📉 Predicciones vs Valores Reales")

# Agregar columna de fecha simulada (asumiendo que es secuencial)
predictions_with_date = predictions.copy()
# Simulamos fechas para el test set (últimos 644 días desde hoy)
end_date = datetime.now()
start_date = end_date - timedelta(days=len(predictions_with_date)-1)
predictions_with_date['fecha'] = pd.date_range(start=start_date, periods=len(predictions_with_date), freq='D')

# Agregar año-mes para agregación
predictions_with_date['year_month'] = predictions_with_date['fecha'].dt.to_period('M')

# Tabs para diferentes vistas
tab1, tab2, tab3 = st.tabs(["📊 Vista Mensual", "📅 Vista Diaria (Últimos 60 días)", "📈 Vista Completa"])

with tab1:
    st.markdown("### Demanda Mensual Promedio: Real vs Predicción")

    # Agregación mensual
    monthly_data = predictions_with_date.groupby('year_month').agg({
        'actual': 'mean',
        'predicted': 'mean',
        'error': 'mean',
        'error_pct': 'mean'
    }).reset_index()

    monthly_data['year_month_str'] = monthly_data['year_month'].astype(str)

    fig_monthly = go.Figure()

    fig_monthly.add_trace(go.Bar(
        x=monthly_data['year_month_str'],
        y=monthly_data['actual'],
        name='Demanda Real',
        marker_color='#667eea',
        opacity=0.8
    ))

    fig_monthly.add_trace(go.Bar(
        x=monthly_data['year_month_str'],
        y=monthly_data['predicted'],
        name='Demanda Predicha',
        marker_color='#f56565',
        opacity=0.6
    ))

    fig_monthly.update_layout(
        title='Comparación Mensual de Demanda',
        xaxis_title='Mes',
        yaxis_title='Demanda Promedio',
        barmode='group',
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig_monthly, use_container_width=True)

    # Tabla de errores mensuales
    st.markdown("#### Errores Mensuales")
    monthly_errors = monthly_data[['year_month_str', 'error', 'error_pct']].copy()
    monthly_errors.columns = ['Mes', 'Error Promedio', 'Error % Promedio']
    monthly_errors['Error % Promedio'] = monthly_errors['Error % Promedio'].apply(lambda x: f"{x:.2f}%")
    monthly_errors['Error Promedio'] = monthly_errors['Error Promedio'].apply(lambda x: f"{x:.2f}")

    st.dataframe(monthly_errors, hide_index=True, use_container_width=True)

with tab2:
    st.markdown("### Últimos 60 días del Test Set")

    # Últimos 60 días
    last_60 = predictions_with_date.tail(60).copy()

    fig_daily = go.Figure()

    fig_daily.add_trace(go.Scatter(
        x=last_60['fecha'],
        y=last_60['actual'],
        mode='lines+markers',
        name='Real',
        line=dict(color='#667eea', width=2),
        marker=dict(size=6)
    ))

    fig_daily.add_trace(go.Scatter(
        x=last_60['fecha'],
        y=last_60['predicted'],
        mode='lines+markers',
        name='Predicción',
        line=dict(color='#f56565', width=2, dash='dash'),
        marker=dict(size=6)
    ))

    fig_daily.update_layout(
        title='',
        xaxis_title='Fecha',
        yaxis_title='Demanda Total',
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig_daily, use_container_width=True)

with tab3:
    st.markdown("### Todos los datos del Test Set")

    # Slider para seleccionar rango
    col1, col2 = st.columns(2)
    with col1:
        start_idx = st.slider("Día inicio", 0, len(predictions_with_date)-50, 0)
    with col2:
        end_idx = st.slider("Día fin", start_idx+10, len(predictions_with_date), min(start_idx+100, len(predictions_with_date)))

    selected_data = predictions_with_date.iloc[start_idx:end_idx]

    fig_full = go.Figure()

    fig_full.add_trace(go.Scatter(
        x=selected_data['fecha'],
        y=selected_data['actual'],
        mode='lines',
        name='Real',
        line=dict(color='#667eea', width=1.5),
    ))

    fig_full.add_trace(go.Scatter(
        x=selected_data['fecha'],
        y=selected_data['predicted'],
        mode='lines',
        name='Predicción',
        line=dict(color='#f56565', width=1.5, dash='dash'),
    ))

    fig_full.update_layout(
        title=f'Días {start_idx} a {end_idx}',
        xaxis_title='Fecha',
        yaxis_title='Demanda Total',
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig_full, use_container_width=True)

# Scatter plot: Actual vs Predicted
st.markdown("## 🎯 Correlación: Predicho vs Real")

fig_scatter = px.scatter(
    predictions,
    x='actual',
    y='predicted',
    color='error_pct',
    color_continuous_scale='RdYlGn_r',
    title='',
    labels={'actual': 'Valor Real', 'predicted': 'Valor Predicho', 'error_pct': 'Error %'},
    hover_data=['error']
)

# Línea diagonal perfecta (y=x)
max_val = max(predictions['actual'].max(), predictions['predicted'].max())
min_val = min(predictions['actual'].min(), predictions['predicted'].min())

fig_scatter.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode='lines',
    name='Predicción Perfecta',
    line=dict(color='gray', dash='dash')
))

fig_scatter.update_layout(height=500)
st.plotly_chart(fig_scatter, use_container_width=True)

# Distribución de errores
st.markdown("## 📊 Distribución de Errores")

col1, col2 = st.columns(2)

with col1:
    fig_hist_error = px.histogram(
        predictions,
        x='error',
        nbins=50,
        title='Distribución de Errores Absolutos',
        labels={'error': 'Error Absoluto'},
        color_discrete_sequence=['#667eea']
    )
    fig_hist_error.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_hist_error, use_container_width=True)

with col2:
    fig_hist_error_pct = px.histogram(
        predictions,
        x='error_pct',
        nbins=50,
        title='Distribución de Errores Porcentuales',
        labels={'error_pct': 'Error %'},
        color_discrete_sequence=['#764ba2']
    )
    fig_hist_error_pct.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_hist_error_pct, use_container_width=True)

# Feature Importance
if feature_importance is not None:
    st.markdown("## 🔍 Top 20 Features Más Importantes")

    top_features = feature_importance.head(20)

    fig_importance = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title='',
        color='importance',
        color_continuous_scale='Viridis',
        labels={'importance': 'Importancia', 'feature': 'Feature'}
    )
    fig_importance.update_layout(height=600, showlegend=False)
    fig_importance.update_yaxes(autorange="reversed")

    st.plotly_chart(fig_importance, use_container_width=True)

    # Tabla de features
    with st.expander("Ver tabla completa de features"):
        st.dataframe(feature_importance, use_container_width=True, height=400)

# Análisis temporal de errores
st.markdown("## ⏰ Evolución Temporal del Error")

predictions_temporal = predictions.copy()
predictions_temporal['dia'] = range(len(predictions_temporal))

fig_temporal = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Error Absoluto en el Tiempo', 'Error Porcentual en el Tiempo'),
    vertical_spacing=0.12
)

fig_temporal.add_trace(
    go.Scatter(
        x=predictions_temporal['dia'],
        y=predictions_temporal['error'],
        mode='lines',
        name='Error Absoluto',
        line=dict(color='#667eea', width=1)
    ),
    row=1, col=1
)

fig_temporal.add_trace(
    go.Scatter(
        x=predictions_temporal['dia'],
        y=predictions_temporal['error_pct'],
        mode='lines',
        name='Error %',
        line=dict(color='#f56565', width=1)
    ),
    row=2, col=1
)

# Línea de referencia en 5%
fig_temporal.add_hline(y=5, line_dash="dash", line_color="red", row=2, col=1,
                       annotation_text="Objetivo: 5%", annotation_position="right")

fig_temporal.update_xaxes(title_text="Día del Test Set", row=2, col=1)
fig_temporal.update_yaxes(title_text="Error Absoluto", row=1, col=1)
fig_temporal.update_yaxes(title_text="Error %", row=2, col=1)

fig_temporal.update_layout(height=600, showlegend=False)

st.plotly_chart(fig_temporal, use_container_width=True)



# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; font-size: 0.9em;">
    <strong>Sistema de Pronóstico Automatizado de Demanda Energética</strong><br>
    EPM - Empresas Públicas de Medellín | Noviembre 2024<br>
    Modelo Prototipo - Validación de Features
</div>
""", unsafe_allow_html=True)
