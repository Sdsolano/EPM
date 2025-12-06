"""
Dashboard de Predicción: Próximos 30 Días con Desagregación Horaria

Dashboard Streamlit para visualizar predicciones de demanda
para los próximos 30 días con desagregación horaria automática.

MODO AUTOMÁTICO: No requiere interacción, todo con valores por defecto.

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

# ============================================================================
# CONFIGURACIÓN POR DEFECTO (SIN INTERACCIÓN)
# ============================================================================
DEFAULT_UCP = "Atlantico"
DEFAULT_N_DAYS = 30
DEFAULT_VALIDATION_DAYS = 30  # Últimos 30 días para validación

# Configuración de página
st.set_page_config(
    page_title="Predicción 30 Días - EPM",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    .ucp-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# FUNCIONES DE CARGA
# ============================================================================

def get_available_ucps():
    """Obtiene lista de UCPs disponibles con modelos entrenados"""
    models_base = Path('models')
    ucps = []
    if models_base.exists():
        for ucp_dir in models_base.iterdir():
            if ucp_dir.is_dir():
                champion_path = ucp_dir / 'registry' / 'champion_model.joblib'
                trained_dir = ucp_dir / 'trained'
                if champion_path.exists() or (trained_dir.exists() and list(trained_dir.glob('*.joblib'))):
                    ucps.append(ucp_dir.name)
    return sorted(ucps) if ucps else [DEFAULT_UCP]


def get_model_path_for_ucp(ucp: str) -> tuple:
    """Obtiene el path del modelo para un UCP específico"""
    champion_path = Path(f'models/{ucp}/registry/champion_model.joblib')
    if champion_path.exists():
        return str(champion_path), None
    
    trained_dir = Path(f'models/{ucp}/trained')
    if trained_dir.exists():
        model_files = sorted(trained_dir.glob('*.joblib'), key=lambda p: p.stat().st_mtime, reverse=True)
        if model_files:
            return str(model_files[0]), None
    
    return None, f"No se encontró modelo para {ucp}"


def get_data_path_for_ucp(ucp: str) -> tuple:
    """Obtiene el path de datos para un UCP específico"""
    ucp_data_path = Path(f'data/features/{ucp}/data_with_features_latest.csv')
    if ucp_data_path.exists():
        return str(ucp_data_path), None
    
    generic_path = Path('data/features/data_with_features_latest.csv')
    if generic_path.exists():
        return str(generic_path), f"Usando datos genéricos"
    
    return None, f"No hay datos para {ucp}"


@st.cache_resource
def load_prediction_pipeline(ucp: str):
    """Carga el pipeline de predicción para un UCP específico"""
    try:
        model_path, model_error = get_model_path_for_ucp(ucp)
        if model_path is None:
            return None, model_error
        
        data_path, _ = get_data_path_for_ucp(ucp)
        if data_path is None:
            return None, "No hay datos históricos"
        
        climate_path = f'data/raw/{ucp}/clima_new.csv'
        if not Path(climate_path).exists():
            climate_path = 'data/raw/clima_new.csv'
        
        pipeline = ForecastPipeline(
            model_path=model_path,
            historical_data_path=data_path,
            enable_hourly_disaggregation=True,
            raw_climate_path=climate_path,
            ucp=ucp
        )
        return pipeline, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_hourly_engine(ucp: str):
    """Carga motor de desagregación horaria para un UCP específico"""
    try:
        models_dir = f'models/{ucp}'
        engine = HourlyDisaggregationEngine(auto_load=True, models_dir=models_dir)
        return engine, None
    except Exception as e:
        try:
            engine = HourlyDisaggregationEngine(auto_load=True)
            return engine, "Usando desagregador genérico"
        except Exception as e2:
            return None, str(e2)


@st.cache_data(ttl=3600)
def generate_predictions(ucp: str, n_days: int):
    """Genera predicciones para los próximos N días"""
    pipeline, error = load_prediction_pipeline(ucp)
    if pipeline is None:
        return None, error
    try:
        predictions = pipeline.predict_next_n_days(n_days=n_days)
        return predictions, None
    except Exception as e:
        return None, str(e)


@st.cache_data
def load_historical_data(ucp: str):
    """Carga datos históricos para un UCP"""
    ucp_path = Path(f'data/features/{ucp}/data_with_features_latest.csv')
    if ucp_path.exists():
        df = pd.read_csv(ucp_path)
        df['FECHA'] = pd.to_datetime(df['FECHA'])
        return df, None
    
    generic_path = Path(FEATURES_DATA_DIR) / "data_with_features_latest.csv"
    if generic_path.exists():
        df = pd.read_csv(generic_path)
        df['FECHA'] = pd.to_datetime(df['FECHA'])
        return df, "Usando datos genéricos"
    
    return None, "No hay datos históricos"


@st.cache_data
def run_validation(ucp: str, n_days: int):
    """Ejecuta validación contra históricos automáticamente"""
    df_historico, _ = load_historical_data(ucp)
    if df_historico is None:
        return None, "No hay datos históricos"
    
    engine, _ = load_hourly_engine(ucp)
    if engine is None:
        return None, "Motor de desagregación no disponible"
    
    # Últimos N días
    available_dates = df_historico['FECHA'].sort_values()
    if len(available_dates) < n_days:
        n_days = len(available_dates)
    
    validation_end = available_dates.iloc[-1]
    validation_start = validation_end - timedelta(days=n_days - 1)
    
    mask = (df_historico['FECHA'] >= validation_start) & (df_historico['FECHA'] <= validation_end)
    df_validation = df_historico[mask].copy()
    
    if len(df_validation) == 0:
        return None, "No hay datos en el rango"
    
    results = []
    period_cols = [f'P{i}' for i in range(1, 25)]
    
    for i, (_, row) in enumerate(df_validation.iterrows()):
        fecha = row['FECHA']
        real_total = row['TOTAL'] if 'TOTAL' in row else row[period_cols].sum()
        real_hourly = row[period_cols].values
        
        try:
            result = engine.predict_hourly(fecha, real_total)
            pred_hourly = result['hourly']
            
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
        except Exception:
            pass  # Ignorar errores individuales
    
    if not results:
        return None, "No se pudieron procesar los datos"
    
    return pd.DataFrame(results), None


@st.cache_data
def run_model_backtest(ucp: str):
    """
    Ejecuta backtest del modelo: predice sobre datos históricos
    y compara con valores reales (train y validation split)
    """
    import joblib
    
    df_historico, _ = load_historical_data(ucp)
    if df_historico is None:
        return None, None, "No hay datos históricos"
    
    model_path, _ = get_model_path_for_ucp(ucp)
    if model_path is None:
        return None, None, "No hay modelo entrenado"
    
    try:
        # Cargar modelo
        model_dict = joblib.load(model_path)
        model = model_dict['model'] if isinstance(model_dict, dict) else model_dict
        feature_names = model_dict.get('feature_names', None) if isinstance(model_dict, dict) else None
        
        # Preparar datos
        df = df_historico.copy()
        df = df.sort_values('FECHA').reset_index(drop=True)
        
        # Identificar columnas de features (excluir fecha, total, períodos)
        exclude_cols = ['FECHA', 'fecha', 'TOTAL', 'demanda_total'] + [f'P{i}' for i in range(1, 25)]
        
        if feature_names:
            # Usar las features del modelo
            available_features = [f for f in feature_names if f in df.columns]
            X = df[available_features].fillna(0)
        else:
            # Inferir features
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            X = df[feature_cols].fillna(0)
        
        # Target
        target_col = 'TOTAL' if 'TOTAL' in df.columns else 'demanda_total'
        y = df[target_col].values
        
        # Predecir
        y_pred = model.predict(X)
        
        # Split 80/20 (mismo que entrenamiento)
        split_idx = int(len(df) * 0.8)
        
        # Datos de entrenamiento
        df_train = pd.DataFrame({
            'fecha': df['FECHA'].iloc[:split_idx],
            'real': y[:split_idx],
            'prediccion': y_pred[:split_idx]
        })
        df_train['error_pct'] = np.abs(df_train['prediccion'] - df_train['real']) / df_train['real'] * 100
        
        # Datos de validación
        df_val = pd.DataFrame({
            'fecha': df['FECHA'].iloc[split_idx:],
            'real': y[split_idx:],
            'prediccion': y_pred[split_idx:]
        })
        df_val['error_pct'] = np.abs(df_val['prediccion'] - df_val['real']) / df_val['real'] * 100
        
        return df_train, df_val, None
        
    except Exception as e:
        return None, None, str(e)


# ============================================================================
# FUNCIONES DE GRÁFICAS
# ============================================================================

def plot_daily_predictions(predictions_df):
    """Gráfica de predicciones diarias"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=predictions_df['fecha'],
        y=predictions_df['demanda_predicha'],
        mode='lines+markers',
        name='Demanda Predicha',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Demanda: %{y:,.2f} MW<extra></extra>'
    ))
    
    for idx, row in predictions_df.iterrows():
        if row.get('is_festivo', False):
            fig.add_vrect(
                x0=row['fecha'], x1=row['fecha'] + timedelta(days=1),
                fillcolor="red", opacity=0.1, layer="below", line_width=0
            )
        elif row.get('is_weekend', False):
            fig.add_vrect(
                x0=row['fecha'], x1=row['fecha'] + timedelta(days=1),
                fillcolor="gray", opacity=0.05, layer="below", line_width=0
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
    if not all(col in predictions_df.columns for col in period_cols):
        return None
    
    hourly_matrix = predictions_df[period_cols].values.T
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
    predictions_df = predictions_df.copy()
    predictions_df['week'] = predictions_df['fecha'].dt.isocalendar().week
    
    weekly_stats = predictions_df.groupby('week').agg({
        'demanda_predicha': ['mean', 'min', 'max'],
        'fecha': 'first'
    }).reset_index()
    weekly_stats.columns = ['week', 'mean', 'min', 'max', 'start_date']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=weekly_stats['start_date'], y=weekly_stats['max'],
        fill=None, mode='lines', line_color='lightblue', showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=weekly_stats['start_date'], y=weekly_stats['min'],
        fill='tonexty', mode='lines', line_color='lightblue', name='Rango Min-Max'
    ))
    fig.add_trace(go.Scatter(
        x=weekly_stats['start_date'], y=weekly_stats['mean'],
        mode='lines+markers', name='Demanda Promedio',
        line=dict(color='blue', width=3), marker=dict(size=10)
    ))
    
    fig.update_layout(
        title="<b>Demanda Semanal Promedio</b>",
        xaxis_title="Semana", yaxis_title="Demanda (MW)", height=400
    )
    return fig


def plot_hourly_average(predictions_df):
    """Demanda promedio por hora"""
    period_cols = [f'P{i}' for i in range(1, 25)]
    if not all(col in predictions_df.columns for col in period_cols):
        return None
    
    hourly_avg = predictions_df[period_cols].mean()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(24)),
        y=hourly_avg.values,
        marker_color='teal',
        text=[f'{v:.0f}' for v in hourly_avg.values],
        textposition='outside'
    ))
    fig.update_layout(
        title="<b>Demanda Promedio por Período Horario</b>",
        xaxis_title="Hora del Día",
        yaxis_title="Demanda Promedio (MW)",
        height=400
    )
    return fig


def plot_validation_mape(df_results):
    """Gráfica de MAPE temporal"""
    mape_avg = df_results['mape'].mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_results['fecha'], y=df_results['mape'],
        mode='lines+markers', name='MAPE Diario',
        line=dict(color='steelblue', width=2), marker=dict(size=6)
    ))
    fig.add_hline(y=5, line_dash="dash", line_color="red",
                  annotation_text="Objetivo Regulatorio: 5%")
    fig.add_hline(y=mape_avg, line_dash="dot", line_color="green",
                  annotation_text=f"Promedio: {mape_avg:.2f}%")
    
    fig.update_layout(
        title="<b>MAPE Diario - Validación Histórica</b>",
        xaxis_title="Fecha", yaxis_title="MAPE (%)",
        height=400, hovermode='x unified'
    )
    return fig


def plot_real_vs_predicted(df_data, title: str, color_real: str = 'blue', color_pred: str = 'red'):
    """Gráfica de Real vs Predicción"""
    fig = go.Figure()
    
    # Línea de valores reales
    fig.add_trace(go.Scatter(
        x=df_data['fecha'],
        y=df_data['real'],
        mode='lines',
        name='Real (Histórico)',
        line=dict(color=color_real, width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Real: %{y:,.0f} MW<extra></extra>'
    ))
    
    # Línea de predicciones
    fig.add_trace(go.Scatter(
        x=df_data['fecha'],
        y=df_data['prediccion'],
        mode='lines',
        name='Predicción (Modelo)',
        line=dict(color=color_pred, width=2, dash='dash'),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Predicción: %{y:,.0f} MW<extra></extra>'
    ))
    
    # Calcular métricas
    mape = df_data['error_pct'].mean()
    r2 = 1 - (np.sum((df_data['real'] - df_data['prediccion'])**2) / 
              np.sum((df_data['real'] - df_data['real'].mean())**2))
    
    fig.update_layout(
        title=f"<b>{title}</b><br><sub>MAPE: {mape:.2f}% | R²: {r2:.4f}</sub>",
        xaxis_title="Fecha",
        yaxis_title="Demanda (MW)",
        height=450,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    return fig, mape, r2


def plot_scatter_real_vs_pred(df_data, title: str):
    """Scatter plot de Real vs Predicción con línea diagonal"""
    fig = go.Figure()
    
    # Puntos
    fig.add_trace(go.Scatter(
        x=df_data['real'],
        y=df_data['prediccion'],
        mode='markers',
        name='Datos',
        marker=dict(color='steelblue', size=6, opacity=0.6),
        hovertemplate='Real: %{x:,.0f} MW<br>Pred: %{y:,.0f} MW<extra></extra>'
    ))
    
    # Línea diagonal (predicción perfecta)
    min_val = min(df_data['real'].min(), df_data['prediccion'].min())
    max_val = max(df_data['real'].max(), df_data['prediccion'].max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Predicción Perfecta',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Calcular R²
    r2 = 1 - (np.sum((df_data['real'] - df_data['prediccion'])**2) / 
              np.sum((df_data['real'] - df_data['real'].mean())**2))
    
    fig.update_layout(
        title=f"<b>{title}</b><br><sub>R²: {r2:.4f}</sub>",
        xaxis_title="Demanda Real (MW)",
        yaxis_title="Demanda Predicha (MW)",
        height=400,
        showlegend=True
    )
    return fig


def plot_error_distribution(df_data, title: str):
    """Histograma de distribución de errores"""
    errors = df_data['prediccion'] - df_data['real']
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=50,
        name='Error',
        marker_color='steelblue',
        opacity=0.7
    ))
    
    # Línea vertical en 0
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
    
    # Estadísticas
    mean_error = errors.mean()
    std_error = errors.std()
    
    fig.update_layout(
        title=f"<b>{title}</b><br><sub>Media: {mean_error:,.0f} MW | Std: {std_error:,.0f} MW</sub>",
        xaxis_title="Error (Predicción - Real) [MW]",
        yaxis_title="Frecuencia",
        height=350
    )
    return fig


# ============================================================================
# APLICACIÓN PRINCIPAL (AUTOMÁTICA)
# ============================================================================

def main():
    """Aplicación principal - MODO AUTOMÁTICO"""
    
    # Usar UCP por defecto
    selected_ucp = DEFAULT_UCP
    n_days = DEFAULT_N_DAYS
    
    # Header
    st.markdown('<p class="main-header">⚡ Predicción de Demanda Energética - EPM</p>',
                unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem; background-color: #e7f3ff; border-radius: 0.5rem; margin-bottom: 2rem;'>
        <h3 style='color: #1f77b4; margin: 0;'>🔮 Sistema de Pronóstico Automatizado</h3>
        <p style='margin: 0.5rem 0 0 0;'>
            <span class='ucp-badge'>{selected_ucp}</span> | 
            Predicciones con desagregación horaria automática
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar modelo
    model_path, model_error = get_model_path_for_ucp(selected_ucp)
    if model_path is None:
        st.error(f"❌ {model_error}")
        st.info("Ejecute primero la API con force_retrain=true para entrenar el modelo")
        st.stop()
    
    # Generar predicciones automáticamente
    with st.spinner(f"🔮 Generando predicciones para {n_days} días..."):
        predictions_df, error = generate_predictions(selected_ucp, n_days)
    
    if predictions_df is None:
        st.error(f"❌ Error al generar predicciones: {error}")
        st.stop()
    
    # Información de las predicciones
    fecha_inicio = predictions_df['fecha'].min()
    fecha_fin = predictions_df['fecha'].max()
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📅 Período", f"{fecha_inicio.strftime('%d/%m')} - {fecha_fin.strftime('%d/%m/%Y')}")
    with col2:
        st.metric("📊 Demanda Promedio", f"{predictions_df['demanda_predicha'].mean():,.0f} MW")
    with col3:
        dias_festivos = int(predictions_df.get('is_festivo', pd.Series([0]*len(predictions_df))).sum())
        st.metric("🎉 Festivos", f"{dias_festivos} días")
    with col4:
        dias_weekend = int(predictions_df.get('is_weekend', pd.Series([0]*len(predictions_df))).sum())
        st.metric("📆 Fin de Semana", f"{dias_weekend} días")
    
    # ==================== TABS ====================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Vista General",
        "🔥 Mapa de Calor Horario",
        "🎯 Modelo: Train vs Val",
        "✅ Validación Históricos",
        "📊 Estadísticas"
    ])
    
    # ==================== TAB 1: VISTA GENERAL ====================
    with tab1:
        st.plotly_chart(plot_daily_predictions(predictions_df), use_container_width=True)
        st.plotly_chart(plot_weekly_comparison(predictions_df), use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Demanda Mínima", f"{predictions_df['demanda_predicha'].min():,.0f} MW")
        with col2:
            st.metric("Demanda Máxima", f"{predictions_df['demanda_predicha'].max():,.0f} MW")
        with col3:
            st.metric("Desv. Estándar", f"{predictions_df['demanda_predicha'].std():,.0f} MW")
        with col4:
            st.metric("Total Acumulado", f"{predictions_df['demanda_predicha'].sum():,.0f} MWh")
    
    # ==================== TAB 2: MAPA DE CALOR ====================
    with tab2:
        fig_heatmap = plot_hourly_heatmap(predictions_df)
        if fig_heatmap:
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            fig_hourly = plot_hourly_average(predictions_df)
            if fig_hourly:
                st.plotly_chart(fig_hourly, use_container_width=True)
        else:
            st.warning("⚠️ Desagregación horaria no disponible")
    
    # ==================== TAB 3: MODELO TRAIN VS VALIDATION ====================
    with tab3:
        st.header("🎯 Desempeño del Modelo: Entrenamiento vs Validación")
        st.info("📌 Comparación de predicciones del modelo contra datos históricos reales (split 80% train / 20% validation)")
        
        with st.spinner("🔄 Ejecutando backtest del modelo..."):
            df_train, df_val, backtest_error = run_model_backtest(selected_ucp)
        
        if backtest_error:
            st.error(f"❌ Error en backtest: {backtest_error}")
        elif df_train is not None and df_val is not None:
            # Métricas resumen
            col1, col2, col3, col4 = st.columns(4)
            
            train_mape = df_train['error_pct'].mean()
            val_mape = df_val['error_pct'].mean()
            
            with col1:
                st.metric("📚 MAPE Entrenamiento", f"{train_mape:.2f}%",
                         delta="✅" if train_mape < 5 else "⚠️")
            with col2:
                st.metric("🧪 MAPE Validación", f"{val_mape:.2f}%",
                         delta="✅" if val_mape < 5 else "⚠️")
            with col3:
                st.metric("📊 Registros Train", f"{len(df_train):,}")
            with col4:
                st.metric("📊 Registros Val", f"{len(df_val):,}")
            
            # Separador
            st.markdown("---")
            
            # ====== GRÁFICA 1: HISTÓRICOS VS ENTRENAMIENTO ======
            st.subheader("📚 Históricos vs Entrenamiento (80% datos)")
            
            fig_train, mape_train, r2_train = plot_real_vs_predicted(
                df_train, 
                "Real vs Predicción - Período de Entrenamiento",
                color_real='#1f77b4',
                color_pred='#ff7f0e'
            )
            st.plotly_chart(fig_train, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fig_scatter_train = plot_scatter_real_vs_pred(df_train, "Scatter: Entrenamiento")
                st.plotly_chart(fig_scatter_train, use_container_width=True)
            with col2:
                fig_error_train = plot_error_distribution(df_train, "Distribución de Errores: Entrenamiento")
                st.plotly_chart(fig_error_train, use_container_width=True)
            
            # Separador
            st.markdown("---")
            
            # ====== GRÁFICA 2: HISTÓRICOS VS VALIDACIÓN ======
            st.subheader("🧪 Históricos vs Validación (20% datos)")
            
            fig_val, mape_val, r2_val = plot_real_vs_predicted(
                df_val,
                "Real vs Predicción - Período de Validación",
                color_real='#2ca02c',
                color_pred='#d62728'
            )
            st.plotly_chart(fig_val, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fig_scatter_val = plot_scatter_real_vs_pred(df_val, "Scatter: Validación")
                st.plotly_chart(fig_scatter_val, use_container_width=True)
            with col2:
                fig_error_val = plot_error_distribution(df_val, "Distribución de Errores: Validación")
                st.plotly_chart(fig_error_val, use_container_width=True)
            
            # Separador
            st.markdown("---")
            
            # ====== TABLA COMPARATIVA ======
            st.subheader("📋 Resumen Comparativo")
            
            summary_data = {
                'Métrica': ['MAPE (%)', 'R²', 'MAE (MW)', 'RMSE (MW)', 'Error Medio (MW)', 'Error Std (MW)', 'Registros'],
                'Entrenamiento': [
                    f"{train_mape:.2f}",
                    f"{r2_train:.4f}",
                    f"{np.mean(np.abs(df_train['prediccion'] - df_train['real'])):,.0f}",
                    f"{np.sqrt(np.mean((df_train['prediccion'] - df_train['real'])**2)):,.0f}",
                    f"{np.mean(df_train['prediccion'] - df_train['real']):,.0f}",
                    f"{np.std(df_train['prediccion'] - df_train['real']):,.0f}",
                    f"{len(df_train):,}"
                ],
                'Validación': [
                    f"{val_mape:.2f}",
                    f"{r2_val:.4f}",
                    f"{np.mean(np.abs(df_val['prediccion'] - df_val['real'])):,.0f}",
                    f"{np.sqrt(np.mean((df_val['prediccion'] - df_val['real'])**2)):,.0f}",
                    f"{np.mean(df_val['prediccion'] - df_val['real']):,.0f}",
                    f"{np.std(df_val['prediccion'] - df_val['real']):,.0f}",
                    f"{len(df_val):,}"
                ]
            }
            
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
            
            # Interpretación
            overfitting_ratio = val_mape / train_mape if train_mape > 0 else 0
            
            if overfitting_ratio < 1.5:
                st.success(f"✅ **Modelo bien generalizado**: El error de validación ({val_mape:.2f}%) es similar al de entrenamiento ({train_mape:.2f}%)")
            elif overfitting_ratio < 2.0:
                st.warning(f"⚠️ **Posible sobreajuste leve**: El error de validación ({val_mape:.2f}%) es {overfitting_ratio:.1f}x mayor que entrenamiento ({train_mape:.2f}%)")
            else:
                st.error(f"❌ **Sobreajuste detectado**: El error de validación ({val_mape:.2f}%) es {overfitting_ratio:.1f}x mayor que entrenamiento ({train_mape:.2f}%)")
        else:
            st.warning("⚠️ No se pudo ejecutar el backtest")
    
    # ==================== TAB 4: VALIDACIÓN HISTÓRICOS ====================
    with tab4:
        st.header(f"✅ Validación: Últimos {DEFAULT_VALIDATION_DAYS} Días")
        
        with st.spinner("🔍 Ejecutando validación automática..."):
            df_results, val_error = run_validation(selected_ucp, DEFAULT_VALIDATION_DAYS)
        
        if df_results is None:
            st.warning(f"⚠️ {val_error}")
        else:
            # Métricas globales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                mae_avg = df_results['mae'].mean()
                st.metric("MAE Promedio", f"{mae_avg:.2f} MW",
                         delta="✅" if mae_avg < 50 else "⚠️")
            with col2:
                rmse_avg = df_results['rmse'].mean()
                st.metric("RMSE Promedio", f"{rmse_avg:.2f} MW",
                         delta="✅" if rmse_avg < 100 else "⚠️")
            with col3:
                mape_avg = df_results['mape'].mean()
                status = "✅ EXCELENTE" if mape_avg < 2 else "✅ BUENO" if mape_avg < 5 else "⚠️"
                st.metric("MAPE Promedio", f"{mape_avg:.2f}%", delta=status)
            with col4:
                val_ok = (df_results['validation_ok'].sum() / len(df_results)) * 100
                st.metric("Validación OK", f"{val_ok:.1f}%",
                         delta="✅" if val_ok > 95 else "⚠️")
            
            # Gráficas
            st.plotly_chart(plot_validation_mape(df_results), use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_box = px.box(df_results, x='day_type', y='mape', color='day_type',
                                title="<b>MAPE por Tipo de Día</b>",
                                labels={'day_type': 'Tipo', 'mape': 'MAPE (%)'})
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                fig_method = px.box(df_results, x='method', y='mape', color='method',
                                   title="<b>MAPE por Método de Desagregación</b>",
                                   labels={'method': 'Método', 'mape': 'MAPE (%)'})
                st.plotly_chart(fig_method, use_container_width=True)
            
            # Tabla resumen
            st.subheader("📋 Resumen por Tipo de Día")
            summary = df_results.groupby('day_type')['mape'].agg([
                ('Media (%)', 'mean'),
                ('Mín (%)', 'min'),
                ('Máx (%)', 'max'),
                ('Días', 'count')
            ]).round(2)
            st.dataframe(summary, use_container_width=True)
    
    # ==================== TAB 5: ESTADÍSTICAS ====================
    with tab5:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_dist = px.histogram(
                predictions_df, x='demanda_predicha', nbins=30,
                title="<b>Distribución de Demanda Predicha</b>",
                labels={'demanda_predicha': 'Demanda (MW)'},
                color_discrete_sequence=['steelblue']
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            predictions_df_copy = predictions_df.copy()
            predictions_df_copy['dia_semana'] = predictions_df_copy['fecha'].dt.day_name()
            fig_box = px.box(
                predictions_df_copy, x='dia_semana', y='demanda_predicha',
                title="<b>Demanda por Día de la Semana</b>",
                labels={'dia_semana': 'Día', 'demanda_predicha': 'Demanda (MW)'},
                color='dia_semana'
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Tabla resumen
        st.subheader("📋 Resumen Estadístico")
        summary = predictions_df['demanda_predicha'].describe()
        summary_df = pd.DataFrame({
            'Métrica': ['Media', 'Desv. Std', 'Mínimo', '25%', 'Mediana', '75%', 'Máximo'],
            'Valor (MW)': [summary['mean'], summary['std'], summary['min'],
                          summary['25%'], summary['50%'], summary['75%'], summary['max']]
        })
        st.dataframe(summary_df.style.format({'Valor (MW)': '{:,.2f}'}), use_container_width=True)
        
        # Descargar
        st.subheader("💾 Exportar Datos")
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Predicciones (CSV)",
            data=csv,
            file_name=f"predicciones_{selected_ucp}_{fecha_inicio.strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
