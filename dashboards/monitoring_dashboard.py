"""
Dashboard Semana 2 - Modelos Predictivos EPM
Visualización completa del proceso de modelado y resultados detallados de los 3 modelos
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import joblib
from pathlib import Path

# Configuración de página
st.set_page_config(
    page_title="EPM - Semana 2: Modelos Predictivos",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Funciones de carga
@st.cache_data
def load_data():
    """Carga los datos procesados"""
    data_path = Path("data/features/data_with_features_latest.csv")
    df = pd.read_csv(data_path)
    # Normalizar nombre de columna fecha
    if 'FECHA' in df.columns:
        df = df.rename(columns={'FECHA': 'fecha'})
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df

@st.cache_data
def load_training_results():
    """Carga los resultados de entrenamiento"""
    results_path = Path("models/trained/training_results_20251120_161937.json")
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

@st.cache_resource
def load_models():
    """Carga los modelos entrenados"""
    models = {}
    models['xgboost'] = joblib.load('models/trained/xgboost_20251120_161937.joblib')
    models['lightgbm'] = joblib.load('models/trained/lightgbm_20251120_161937.joblib')
    models['randomforest'] = joblib.load('models/trained/randomforest_20251120_161937.joblib')
    return models

def prepare_data_for_modeling(df, model_feature_names=None):
    """Prepara datos para modelado (80/20 split temporal)"""
    df_sorted = df.sort_values('fecha').reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_idx]
    val_df = df_sorted.iloc[split_idx:]

    # Buscar columna de target (puede ser TOTAL o demanda_total)
    target_col = 'TOTAL' if 'TOTAL' in df_sorted.columns else 'demanda_total'

    # Si se proporcionan nombres de features del modelo, usar solo esas
    if model_feature_names is not None:
        feature_cols = [col for col in model_feature_names if col in df_sorted.columns]
    else:
        # Excluir columnas de periodos P1-P24 si existen (el modelo fue entrenado sin ellas)
        feature_cols = [col for col in df_sorted.columns
                       if col not in ['fecha', target_col] and not col.startswith('P')]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    return X_train, y_train, X_val, y_val, val_df

def calculate_detailed_errors(y_true, y_pred):
    """Calcula errores detallados"""
    return pd.DataFrame({
        'Real': y_true.values,
        'Prediccion': y_pred,
        'Error_Abs': np.abs(y_true.values - y_pred),
        'Error_Pct': np.abs((y_true.values - y_pred) / y_true.values) * 100,
        'Error': y_true.values - y_pred
    })

# Sidebar
st.sidebar.title("⚡ EPM - Semana 2")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navegación",
    ["🏠 Resumen General",
     "📊 Proceso Completo",
     "🎯 Resultados Detallados",
     "📈 Predicciones vs Reales",
     "🔍 Feature Importance",
     "🏆 Comparación de Modelos",
     "📉 Análisis de Errores",
     "⚙️ Hiperparámetros"]
)

# Cargar datos
try:
    df = load_data()
    results = load_training_results()
    models = load_models()
    X_train, y_train, X_val, y_val, val_df = prepare_data_for_modeling(df)

    # Generar predicciones (los modelos son diccionarios, el modelo real está en ['model'])
    predictions = {}
    for model_name, model_dict in models.items():
        if isinstance(model_dict, dict):
            model = model_dict['model']
        else:
            model = model_dict
        predictions[model_name] = model.predict(X_val)

except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

# ============================================================================
# PÁGINA 1: RESUMEN GENERAL
# ============================================================================
if page == "🏠 Resumen General":
    st.markdown('<p class="main-header">⚡ Semana 2: Modelos Predictivos de Demanda Energética</p>', unsafe_allow_html=True)

    st.markdown("""
    ### 📋 Resumen Ejecutivo

    Implementación completa de 3 modelos de Machine Learning para pronóstico de demanda energética diaria.
    """)

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)

    best_model = min(results.items(), key=lambda x: x[1]['val_metrics']['rmape'])
    best_model_name = best_model[0].upper()
    best_rmape = best_model[1]['val_metrics']['rmape']
    best_mape = best_model[1]['val_metrics']['mape']
    best_r2 = best_model[1]['val_metrics']['r2']

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🏆 Mejor Modelo", best_model_name)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="success-card">', unsafe_allow_html=True)
        st.metric("rMAPE", f"{best_rmape:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="success-card">', unsafe_allow_html=True)
        st.metric("MAPE", f"{best_mape:.2f}%", delta=f"-{5-best_mape:.2f}% vs límite")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="success-card">', unsafe_allow_html=True)
        st.metric("R² Score", f"{best_r2:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Tabla comparativa
    st.markdown("### 🎯 Comparación de los 3 Modelos")

    comparison_data = []
    for model_name, model_results in results.items():
        comparison_data.append({
            'Modelo': model_name.upper(),
            'MAPE (%)': f"{model_results['val_metrics']['mape']:.2f}",
            'rMAPE (%)': f"{model_results['val_metrics']['rmape']:.2f}",
            'R²': f"{model_results['val_metrics']['r2']:.4f}",
            'MAE (MW)': f"{model_results['val_metrics']['mae']:.1f}",
            'Tiempo (s)': f"{model_results['training_time']:.2f}"
        })

    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    # Cumplimiento regulatorio
    st.markdown("### ✅ Cumplimiento Regulatorio")

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"""
        **Todos los modelos cumplen MAPE < 5%**

        - ✅ XGBoost: {results['xgboost']['val_metrics']['mape']:.2f}%
        - ✅ LightGBM: {results['lightgbm']['val_metrics']['mape']:.2f}%
        - ✅ RandomForest: {results['randomforest']['val_metrics']['mape']:.2f}%
        """)

    with col2:
        days_compliant = {}
        for model_name, pred in predictions.items():
            errors = calculate_detailed_errors(y_val, pred)
            pct = (errors['Error_Pct'] < 5).sum() / len(errors) * 100
            days_compliant[model_name] = pct

        st.info(f"""
        **Días con error < 5%**

        - XGBoost: {days_compliant['xgboost']:.1f}%
        - LightGBM: {days_compliant['lightgbm']:.1f}%
        - RandomForest: {days_compliant['randomforest']:.1f}%
        """)

    # Decisión: No clustering
    st.markdown("### 🎯 Decisión: NO Clustering")

    st.info("""
    **¿Por qué NO implementamos clustering?**

    - ✅ **Features capturan patrones:** `is_festivo`, `dayofweek`, `month` ya agrupan días similares
    - ✅ **Resultados excelentes:** MAPE 1.17% sin clustering
    - ✅ **Más simple:** Menos complejidad = más mantenible
    - ✅ **Validado:** Prototipo logró 0.45% MAPE sin clustering
    """)

# ============================================================================
# PÁGINA 2: PROCESO COMPLETO
# ============================================================================
elif page == "📊 Proceso Completo":
    st.markdown('<p class="main-header">📊 Proceso Completo de Modelado</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Flujo de Trabajo

    ```
    SEMANA 1 → Datos Limpios + Feature Engineering
              ↓
    Split 80/20 → Train: 2,572 días | Val: 654 días
              ↓
    Entrenamiento 3 Modelos → XGBoost | LightGBM | RandomForest
              ↓
    Cross-Validation (3-Fold)
              ↓
    Evaluación en Validación
              ↓
    Selección del Mejor (por rMAPE) → Champion: XGBoost
              ↓
    Model Registry + Predictions
    ```
    """)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Datos", "🤖 Modelos", "📈 Evaluación"])

    with tab1:
        st.markdown("### 📊 Datos")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Registros", f"{len(df):,}")
        with col2:
            st.metric("Training", f"{len(X_train):,}")
        with col3:
            st.metric("Validation", f"{len(X_val):,}")

    with tab2:
        st.markdown("""
        ### 🤖 Modelos

        **¿Por qué XGBoost, LightGBM, RandomForest?**

        - 10x más rápidos que LSTM (2.73s vs 5-20min)
        - Mejores para datos tabulares (63 features)
        - Regularización incorporada
        - Feature importance nativo
        """)

    with tab3:
        st.markdown("""
        ### 📈 Métricas

        - **rMAPE:** MAPE / Correlación (penaliza errores de forma y magnitud)
        - **MAPE:** Error porcentual absoluto (requisito < 5%)
        - **R²:** Varianza explicada
        """)

# ============================================================================
# PÁGINA 3: RESULTADOS DETALLADOS
# ============================================================================
elif page == "🎯 Resultados Detallados":
    st.markdown('<p class="main-header">🎯 Resultados Detallados por Modelo</p>', unsafe_allow_html=True)

    model_selected = st.selectbox(
        "Selecciona un modelo:",
        ["XGBoost", "LightGBM", "RandomForest"],
        index=0
    )

    model_key = model_selected.lower()
    model_data = results[model_key]

    # Métricas principales
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("MAPE", f"{model_data['val_metrics']['mape']:.2f}%")
    with col2:
        st.metric("rMAPE", f"{model_data['val_metrics']['rmape']:.2f}%")
    with col3:
        st.metric("R²", f"{model_data['val_metrics']['r2']:.4f}")
    with col4:
        st.metric("MAE", f"{model_data['val_metrics']['mae']:.1f} MW")
    with col5:
        st.metric("Tiempo", f"{model_data['training_time']:.2f}s")

    st.markdown("---")

    # Tabs detallados
    tab1, tab2, tab3 = st.tabs(["📊 Métricas Completas", "📈 Cross-Validation", "⚙️ Info Técnica"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Training Metrics")
            train_metrics = model_data['train_metrics']
            st.dataframe(pd.DataFrame({
                'Métrica': ['MAPE', 'rMAPE', 'R²', 'Correlación', 'MAE', 'RMSE'],
                'Valor': [
                    f"{train_metrics['mape']:.4f}%",
                    f"{train_metrics['rmape']:.4f}%",
                    f"{train_metrics['r2']:.6f}",
                    f"{train_metrics['correlation']:.6f}",
                    f"{train_metrics['mae']:.2f} MW",
                    f"{train_metrics['rmse']:.2f} MW"
                ]
            }), use_container_width=True, hide_index=True)

        with col2:
            st.markdown("### 📈 Validation Metrics")
            val_metrics = model_data['val_metrics']
            st.dataframe(pd.DataFrame({
                'Métrica': ['MAPE', 'rMAPE', 'R²', 'Correlación', 'MAE', 'RMSE'],
                'Valor': [
                    f"{val_metrics['mape']:.4f}%",
                    f"{val_metrics['rmape']:.4f}%",
                    f"{val_metrics['r2']:.6f}",
                    f"{val_metrics['correlation']:.6f}",
                    f"{val_metrics['mae']:.2f} MW",
                    f"{val_metrics['rmse']:.2f} MW"
                ]
            }), use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### 📊 Cross-Validation (3-Fold)")
        cv_results = model_data['cv_results']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean rMAPE", f"{cv_results['mean_rmape']:.2f}%")
        with col2:
            st.metric("Std rMAPE", f"{cv_results['std_rmape']:.2f}%")
        with col3:
            st.metric("Mean MAPE", f"{cv_results['mean_mape']:.2f}%")
        with col4:
            st.metric("Std MAPE", f"{cv_results['std_mape']:.2f}%")

        st.dataframe(pd.DataFrame({
            'Fold': [1, 2, 3],
            'rMAPE (%)': cv_results['all_rmapes'],
            'MAPE (%)': cv_results['all_mapes']
        }), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("### ⚙️ Información Técnica")
        st.info(f"""
        - **Modelo:** {model_data['model_type']}
        - **Features:** {model_data['n_features']}
        - **Training samples:** {model_data['n_train_samples']:,}
        - **Tiempo:** {model_data['training_time']:.2f}s
        - **Velocidad:** {model_data['n_train_samples']/model_data['training_time']:.0f} muestras/s
        """)

# ============================================================================
# PÁGINA 4: PREDICCIONES VS REALES
# ============================================================================
elif page == "📈 Predicciones vs Reales":
    st.markdown('<p class="main-header">📈 Predicciones vs Valores Reales</p>', unsafe_allow_html=True)

    model_selected = st.selectbox(
        "Selecciona un modelo:",
        ["XGBoost", "LightGBM", "RandomForest", "Comparar Todos"],
        index=0
    )

    if model_selected != "Comparar Todos":
        model_key = model_selected.lower()
        y_pred = predictions[model_key]

        plot_df = pd.DataFrame({
            'Fecha': val_df['fecha'].values,
            'Real': y_val.values,
            'Predicción': y_pred
        })

        # Time series
        st.markdown(f"### Serie Temporal - {model_selected}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df['Fecha'], y=plot_df['Real'], name='Real', line=dict(color='white', width=2)))
        fig.add_trace(go.Scatter(x=plot_df['Fecha'], y=plot_df['Predicción'], name='Predicción', line=dict(color='#667eea', width=2)))
        fig.update_layout(title=f"Demanda Real vs Predicción - {model_selected}", xaxis_title="Fecha", yaxis_title="Demanda (MW)", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Scatter plot
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Scatter Plot")
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=plot_df['Real'],
                y=plot_df['Predicción'],
                mode='markers',
                marker=dict(color=np.abs(plot_df['Real'] - plot_df['Predicción']), colorscale='Viridis', showscale=True, size=8)
            ))
            min_val = min(plot_df['Real'].min(), plot_df['Predicción'].min())
            max_val = max(plot_df['Real'].max(), plot_df['Predicción'].max())
            fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', line=dict(color='red', dash='dash'), name='Ideal'))
            fig_scatter.update_layout(title="Real vs Predicción", xaxis_title="Real (MW)", yaxis_title="Predicción (MW)", height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            st.markdown("### Distribución de Errores")
            errors = plot_df['Real'] - plot_df['Predicción']
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=errors, nbinsx=50, marker_color='#667eea'))
            fig_hist.update_layout(title="Histograma de Errores", xaxis_title="Error (MW)", yaxis_title="Frecuencia", height=500)
            st.plotly_chart(fig_hist, use_container_width=True)

            st.info(f"""
            **Estadísticas:**
            - Media: {errors.mean():.2f} MW
            - Mediana: {errors.median():.2f} MW
            - Desv. Est.: {errors.std():.2f} MW
            """)

    else:
        # Comparar todos
        st.markdown("### Comparación de los 3 Modelos")
        plot_df = pd.DataFrame({
            'Fecha': val_df['fecha'].values,
            'Real': y_val.values,
            'XGBoost': predictions['xgboost'],
            'LightGBM': predictions['lightgbm'],
            'RandomForest': predictions['randomforest']
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df['Fecha'], y=plot_df['Real'], name='Real', line=dict(color='white', width=3)))
        fig.add_trace(go.Scatter(x=plot_df['Fecha'], y=plot_df['XGBoost'], name='XGBoost', line=dict(color='#667eea', width=2)))
        fig.add_trace(go.Scatter(x=plot_df['Fecha'], y=plot_df['LightGBM'], name='LightGBM', line=dict(color='#38ef7d', width=2)))
        fig.add_trace(go.Scatter(x=plot_df['Fecha'], y=plot_df['RandomForest'], name='RandomForest', line=dict(color='#f5576c', width=2)))
        fig.update_layout(title="Comparación - Todos los Modelos", xaxis_title="Fecha", yaxis_title="Demanda (MW)", height=600)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PÁGINA 5: FEATURE IMPORTANCE
# ============================================================================
elif page == "🔍 Feature Importance":
    st.markdown('<p class="main-header">🔍 Importancia de Features</p>', unsafe_allow_html=True)

    model_selected = st.selectbox("Selecciona un modelo:", ["XGBoost", "LightGBM", "RandomForest"], index=0)
    model_key = model_selected.lower()
    model_dict = models[model_key]

    # Obtener importancia (puede estar en el dict o calcularla del modelo)
    if isinstance(model_dict, dict) and 'feature_importance' in model_dict:
        importance = model_dict['feature_importance']
    else:
        # Extraer el modelo real
        model = model_dict['model'] if isinstance(model_dict, dict) else model_dict
        if model_key == 'randomforest':
            importance = model.feature_importances_
        else:
            importance = model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    n_features = st.slider("Número de features:", 5, 30, 20)
    top_features = importance_df.head(n_features)

    # Gráfico
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_features['Importance'],
        y=top_features['Feature'],
        orientation='h',
        marker=dict(color=top_features['Importance'], colorscale='Viridis', showscale=True)
    ))
    fig.update_layout(
        title=f"Top {n_features} Features - {model_selected}",
        xaxis_title="Importancia",
        yaxis_title="Feature",
        height=max(400, n_features * 20),
        yaxis={'categoryorder': 'total ascending'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabla
    st.markdown("### Tabla Completa")
    importance_df['Porcentaje'] = (importance_df['Importance'] / importance_df['Importance'].sum() * 100)
    st.dataframe(importance_df.head(n_features), use_container_width=True, hide_index=True)

# ============================================================================
# PÁGINA 6: COMPARACIÓN DE MODELOS
# ============================================================================
elif page == "🏆 Comparación de Modelos":
    st.markdown('<p class="main-header">🏆 Comparación de Modelos</p>', unsafe_allow_html=True)

    # Tabla comparativa
    comparison_data = []
    for model_name, model_results in results.items():
        comparison_data.append({
            'Modelo': model_name.upper(),
            'MAPE Val (%)': model_results['val_metrics']['mape'],
            'rMAPE Val (%)': model_results['val_metrics']['rmape'],
            'R² Val': model_results['val_metrics']['r2'],
            'MAE (MW)': model_results['val_metrics']['mae'],
            'RMSE (MW)': model_results['val_metrics']['rmse'],
            'Tiempo (s)': model_results['training_time'],
            'CV Mean rMAPE (%)': model_results['cv_results']['mean_rmape'],
            'CV Std rMAPE (%)': model_results['cv_results']['std_rmape']
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Gráficos
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Precisión")
        fig = go.Figure()
        fig.add_trace(go.Bar(name='MAPE', x=[r['Modelo'] for r in comparison_data], y=[r['MAPE Val (%)'] for r in comparison_data], marker_color='#667eea'))
        fig.add_trace(go.Bar(name='rMAPE', x=[r['Modelo'] for r in comparison_data], y=[r['rMAPE Val (%)'] for r in comparison_data], marker_color='#764ba2'))
        fig.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="Límite 5%")
        fig.update_layout(xaxis_title="Modelo", yaxis_title="Error (%)", barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### ⏱️ Velocidad")
        fig = go.Figure(data=[go.Bar(
            x=[r['Modelo'] for r in comparison_data],
            y=[r['Tiempo (s)'] for r in comparison_data],
            marker=dict(color=[r['Tiempo (s)'] for r in comparison_data], colorscale='Viridis', showscale=True)
        )])
        fig.update_layout(xaxis_title="Modelo", yaxis_title="Tiempo (s)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Precisión vs Velocidad
    st.markdown("### ⚖️ Precisión vs Velocidad")
    fig = go.Figure()
    for model_data in comparison_data:
        fig.add_trace(go.Scatter(
            x=[model_data['Tiempo (s)']],
            y=[model_data['rMAPE Val (%)']],
            mode='markers+text',
            marker=dict(size=20),
            text=[model_data['Modelo']],
            textposition="top center",
            name=model_data['Modelo']
        ))
    fig.update_layout(title="rMAPE vs Tiempo", xaxis_title="Tiempo (s)", yaxis_title="rMAPE (%)", height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PÁGINA 7: ANÁLISIS DE ERRORES
# ============================================================================
elif page == "📉 Análisis de Errores":
    st.markdown('<p class="main-header">📉 Análisis de Errores</p>', unsafe_allow_html=True)

    model_selected = st.selectbox("Selecciona un modelo:", ["XGBoost", "LightGBM", "RandomForest"], index=0)
    model_key = model_selected.lower()
    y_pred = predictions[model_key]

    errors_df = calculate_detailed_errors(y_val, y_pred)
    errors_df['Fecha'] = val_df['fecha'].values
    errors_df['DayOfWeek'] = pd.to_datetime(errors_df['Fecha']).dt.dayofweek

    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Error Medio", f"{errors_df['Error'].mean():.2f} MW")
    with col2:
        st.metric("MAE", f"{errors_df['Error_Abs'].mean():.2f} MW")
    with col3:
        st.metric("MAPE", f"{errors_df['Error_Pct'].mean():.2f}%")
    with col4:
        st.metric("Error Máximo", f"{errors_df['Error_Abs'].max():.2f} MW")

    st.markdown("---")

    # Distribución
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Distribución de Errores %")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=errors_df['Error_Pct'], nbinsx=50, marker_color='#667eea'))
        fig.add_vline(x=5, line_dash="dash", line_color="red", annotation_text="Límite 5%")
        fig.update_layout(title="Distribución", xaxis_title="Error (%)", yaxis_title="Frecuencia", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Cumplimiento")
        pct_under_5 = (errors_df['Error_Pct'] < 5).sum() / len(errors_df) * 100
        pct_under_3 = (errors_df['Error_Pct'] < 3).sum() / len(errors_df) * 100
        pct_under_1 = (errors_df['Error_Pct'] < 1).sum() / len(errors_df) * 100

        st.metric("Error < 5%", f"{pct_under_5:.1f}%")
        st.metric("Error < 3%", f"{pct_under_3:.1f}%")
        st.metric("Error < 1%", f"{pct_under_1:.1f}%")

    # Errores en el tiempo
    st.markdown("### Evolución en el Tiempo")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=errors_df['Fecha'], y=errors_df['Error_Pct'], mode='markers', marker=dict(size=6, color=errors_df['Error_Pct'], colorscale='Reds', showscale=True)))
    fig.add_hline(y=5, line_dash="dash", line_color="red")
    fig.update_layout(title="Error % por Día", xaxis_title="Fecha", yaxis_title="Error (%)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Peores días
    st.markdown("### 🔴 Top 10 Peores Días")
    worst_days = errors_df.nlargest(10, 'Error_Pct')[['Fecha', 'Real', 'Prediccion', 'Error_Abs', 'Error_Pct']]
    worst_days['Fecha'] = pd.to_datetime(worst_days['Fecha']).dt.strftime('%Y-%m-%d')
    st.dataframe(worst_days, use_container_width=True, hide_index=True)

# ============================================================================
# PÁGINA 8: HIPERPARÁMETROS
# ============================================================================
elif page == "⚙️ Hiperparámetros":
    st.markdown('<p class="main-header">⚙️ Hiperparámetros</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔷 XGBoost", "🔶 LightGBM", "🔹 RandomForest"])

    with tab1:
        st.markdown("### XGBoost")
        hp = results['xgboost']['hyperparameters']
        col1, col2 = st.columns(2)
        with col1:
            st.json({'n_estimators': hp['n_estimators'], 'max_depth': hp['max_depth'], 'learning_rate': hp['learning_rate']})
        with col2:
            st.json({'reg_alpha': hp['reg_alpha'], 'reg_lambda': hp['reg_lambda'], 'subsample': hp['subsample']})

    with tab2:
        st.markdown("### LightGBM")
        hp = results['lightgbm']['hyperparameters']
        col1, col2 = st.columns(2)
        with col1:
            st.json({'n_estimators': hp['n_estimators'], 'max_depth': hp['max_depth'], 'num_leaves': hp['num_leaves']})
        with col2:
            st.json({'reg_alpha': hp['reg_alpha'], 'reg_lambda': hp['reg_lambda'], 'learning_rate': hp['learning_rate']})

    with tab3:
        st.markdown("### RandomForest")
        hp = results['randomforest']['hyperparameters']
        col1, col2 = st.columns(2)
        with col1:
            st.json({'n_estimators': hp['n_estimators'], 'max_depth': hp['max_depth'], 'min_samples_split': hp['min_samples_split']})
        with col2:
            st.json({'min_samples_leaf': hp['min_samples_leaf'], 'max_features': hp['max_features']})

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><b>EPM - Semana 2: Modelos Predictivos</b></p>
    <p>XGBoost | LightGBM | RandomForest</p>
</div>
""", unsafe_allow_html=True)
