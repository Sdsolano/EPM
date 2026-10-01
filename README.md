# Sistema de Pronóstico Automatizado de Demanda Energética - EPM

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sistema inteligente de pronóstico de demanda energética con capacidad de autoaprendizaje para el sistema de distribución de EPM en Antioquia, Colombia.

## 🎯 Descripción

Sistema de machine learning que automatiza el pronóstico de demanda energética, cumpliendo con el **Acuerdo CNO 1303 de 2020** y el **Proyecto de Resolución CREG 143 de 2021**.

### Características Principales

- ✅ **Pipeline Automatizado de Datos**: Lectura, limpieza y transformación automática
- ✅ **Feature Engineering Inteligente**: 63 features creadas automáticamente
- ✅ **Modelos de ML Optimizados**: XGBoost, LightGBM, RandomForest
- ✅ **Desagregación Horaria con Clustering**: K-Means dual (35 + 15 clusters)
- ✅ **Métrica rMAPE Innovadora**: Del paper de Universidad del Norte
- ✅ **Versionado de Modelos**: Registry completo con selección automática del campeón
- ✅ **Alta Precisión**: MAPE 0.45% diario + 1.61% horario
- ✅ **Dashboards Interactivos**: Visualización y validación con Streamlit

## 📊 Estado del Proyecto

| Fase | Componente | Estado | Avance |
|------|-----------|--------|--------|
| **Fase 1** | Pipeline Automatizado de Datos | ✅ Completada | 100% |
| **Fase 2** | Modelos Predictivos + Entrenamiento | ✅ Completada | 100% |
| **Fase 2.5** | Desagregación Horaria (Clustering) | ✅ Completada | 100% |
| **Fase 3** | Sistema de Validación y Dashboards | ✅ Completada | 100% |
| **Fase 4** | API Gateway + Monitoreo + Reentrenamiento | ⏸️ Pendiente | 10% |

## 🚀 Inicio Rápido

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/epm/forecast-system.git
cd forecast-system

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar en modo desarrollo
pip install -e .
```

### Ejecutar Pipeline Completo

```bash
# 1. Ejecutar pipeline de datos
python pipeline/orchestrator.py

# 2. Entrenar modelos de predicción diaria
python train_models.py

# 3. Entrenar sistema de desagregación horaria
python scripts/train_hourly_disaggregation.py

# 4. Generar predicciones (30 días con desagregación horaria)
python src/prediction/forecaster.py

# 5. Validar desagregación horaria
python scripts/validate_hourly_disaggregation.py
```

### Uso Programático

```python
from src.prediction.forecaster import ForecastPipeline
from src.prediction.hourly import HourlyDisaggregationEngine

# 1. Pipeline completo de predicción (incluye desagregación horaria automática)
pipeline = ForecastPipeline(
    model_path='models/trained/xgboost_20251120_161937.joblib',
    historical_data_path='data/features/data_with_features_latest.csv',
    enable_hourly_disaggregation=True
)

# Predecir próximos 30 días
predictions = pipeline.predict_next_n_days(n_days=30)

# 2. Usar desagregación horaria independiente
engine = HourlyDisaggregationEngine(auto_load=True)
result = engine.predict_hourly(date="2024-03-15", total_daily=31500.0)

print(f"Método usado: {result['method']}")  # 'normal' o 'special'
print(f"Suma válida: {result['validation']['is_valid']}")  # True
print(f"P1-P24: {result['hourly']}")  # Array de 24 valores
```

## 📁 Estructura del Proyecto

```
EPM/
├── src/                          # Código fuente
│   ├── pipeline/                 # Pipeline de datos (Fase 1)
│   ├── models/                   # Modelos ML (Fase 2)
│   ├── prediction/               # Sistema de predicción
│   │   ├── forecaster.py         # Pipeline de predicción
│   │   └── hourly/               # ✨ Desagregación horaria (NUEVO)
│   │       ├── calendar_utils.py      # Clasificador de días (holidays)
│   │       ├── hourly_disaggregator.py # Clustering días normales
│   │       ├── special_days.py        # Clustering días especiales
│   │       └── disaggregation_engine.py # Orquestador
│   ├── api/                      # API Gateway (Fase 4)
│   ├── monitoring/               # Monitoreo y reentrenamiento
│   └── config/                   # Configuración
│
├── scripts/                      # Scripts ejecutables
│   ├── train_hourly_disaggregation.py  # Entrenar clustering horario
│   └── validate_hourly_disaggregation.py # Validación interna del sistema
│
├── tests/                        # Tests unitarios
│   └── test_hourly_disaggregation.py  # Tests del sistema horario
│
├── dashboards/                   # Dashboards Streamlit interactivos
│   ├── hourly_comparison_dashboard.py  # Comparación 30d vs históricos
│   ├── hourly_validation_dashboard.py  # Validación retrospectiva
│   └── prediction_dashboard.py         # Predicciones futuras
│
├── notebooks/                    # Jupyter notebooks (exploración)
├── data/                         # Datos (gitignored)
│   ├── raw/                      # Datos originales
│   ├── processed/                # Datos procesados
│   ├── features/                 # Features engineering
│   └── predictions/              # Predicciones generadas
│
├── models/                       # Modelos entrenados (gitignored)
│   ├── trained/                  # Modelos de predicción diaria
│   ├── trained_diario/           # Modelo de nivel de /predict-daily (ver docs/MODELO_DIARIO_NIVEL.md)
│   ├── registry/                 # Model registry (campeón)
│   ├── hourly_disaggregator.pkl  # Clustering días normales
│   └── special_days_disaggregator.pkl # Clustering festivos
│
└── logs/                         # Logs del sistema
    ├── pipeline/                 # Logs de pipeline de datos
    ├── training/                 # Logs de entrenamiento
    └── validation/               # Reportes de validación
```

## 🧠 Modelos Implementados

### 1. **XGBoost** (Campeón)
- **MAPE**: 0.3-0.6%
- **rMAPE**: 3-5
- **R²**: 0.94-0.96
- Optimizado con Bayesian Optimization

### 2. **LightGBM**
- **MAPE**: 0.4-0.7%
- **rMAPE**: 3.5-5.5
- 10x más rápido que XGBoost

### 3. **Random Forest**
- **MAPE**: 0.8-1.5%
- **rMAPE**: 5-8
- Modelo robusto de fallback

## 📈 Resultados

### Métricas de Desempeño

| Métrica | Objetivo Regulatorio | Resultado Actual | Estado |
|---------|---------------------|------------------|--------|
| MAPE mensual | < 5% | **0.45%** | ✅ **11x mejor** |
| R² | > 0.85 | **0.946** | ✅ Excelente |
| Días con error < 5% | > 95% | **99.4%** | ✅ Superior |

### Features Creadas (63 total)

- **19 features de calendario**: Temporales + cíclicas (sin/cos)
- **25 features de demanda**: Lags + rolling statistics
- **25 features climáticas**: Temperatura, humedad, sensación térmica
- **4 features de estacionalidad**: Temporada lluviosa/seca
- **3 features de interacción**: Clima × calendario

## ⏰ Sistema de Desagregación Horaria

El sistema convierte pronósticos **diarios totales** en distribuciones **horarias (P1-P24)** usando clustering inteligente basado en K-Means.

### Arquitectura

```
Predicción Diaria (TOTAL)
    ↓
CalendarClassifier (holidays library)
    ↓
¿Es festivo/especial? → SÍ → SpecialDaysDisaggregator (15 clusters)
    ↓                          ↓
   NO                   Perfil Horario P1-P24
    ↓                          ↓
HourlyDisaggregator      Validación: sum(P1-P24) = TOTAL
(35 clusters)                  ↓
    ↓                    Predicción Horaria Lista
Perfil Horario P1-P24
```

### Características Técnicas

- ✅ **Clustering Dual K-Means**:
  - 35 clusters para días normales (laborales, fines de semana)
  - 15 clusters para días especiales (festivos colombianos)
- ✅ **Librería `holidays`**: Festivos de Colombia automáticos 2017-2030
- ✅ **Validación Matemática**: Garantiza `sum(P1-P24) == TOTAL_DIARIO` (error < 0.01 MWh)
- ✅ **Clasificación Inteligente**:
  - Tipo de día: Laboral / Festivo / Fin de semana
  - Temporada: Lluviosa / Seca (clima Antioquia)
- ✅ **Precisión Validada**: MAPE 1.61% en 60 días de prueba
- ✅ **Production-Ready**: Modelos serializados, logging, tests completos

### Métricas de Validación (60 días)

| Métrica | Valor | Estado |
|---------|-------|--------|
| **MAPE Global** | 1.61% | ✅ Excelente |
| **MAE** | 19.57 MW | ✅ Bajo error |
| **RMSE** | 23.42 MW | ✅ Consistente |
| **Validación Suma** | 100% válido | ✅ Perfecto |
| **Días Laborales** | MAPE 1.39% | ✅ Superior |
| **Fines de Semana** | MAPE 2.20% | ✅ Bueno |
| **Festivos** | MAPE 1.19% | ✅ Excelente |

### Uso Rápido

```python
from src.prediction.hourly import HourlyDisaggregationEngine

# Cargar sistema entrenado
engine = HourlyDisaggregationEngine(auto_load=True)

# Predecir distribución horaria
result = engine.predict_hourly(
    date="2024-03-15",
    total_daily=31500.0,
    validate=True
)

print(f"Método: {result['method']}")           # 'normal' o 'special'
print(f"P1-P24: {result['hourly']}")           # Array[24] con valores
print(f"Suma válida: {result['validation']['is_valid']}")  # True
print(f"Suma total: {result['validation']['sum']:.2f}")    # 31500.00
```

### Entrenar y Validar

```bash
# Entrenar modelos de clustering (3,226 días normales + 156 festivos)
python scripts/train_hourly_disaggregation.py

# Validar sistema contra históricos (genera reporte completo)
python scripts/validate_hourly_disaggregation.py --days 60

# Ejecutar tests unitarios
pytest tests/test_hourly_disaggregation.py -v
```

### Dashboards Interactivos

```bash
# Dashboard de comparación (30 días × 24 horas vs históricos)
streamlit run dashboards/hourly_comparison_dashboard.py

# Dashboard de validación retrospectiva
streamlit run dashboards/hourly_validation_dashboard.py

# Dashboard de predicciones futuras
streamlit run dashboards/prediction_dashboard.py
```

## 🔧 Configuración

Editar `src/config/settings.py` para ajustar:

- Rutas de directorios
- Umbrales de calidad de datos
- Parámetros de feature engineering
- Métricas regulatorias
- Horizontes de pronóstico

## 📚 Documentación

### Fases del Proyecto
- [Especificaciones del Proyecto](docs/proyecto_especificaciones.pdf) - PDF con requerimientos completos

### Reportes de Validación
- **Validación Horaria**: `logs/validation/validation_report.txt`
  - 60 días evaluados (Sep-Nov 2025)
  - MAPE global: 1.61%
  - Validación de suma: 100% perfecta
  - Desglose por tipo de día y método de clustering

### Datos de Salida
- **Predicciones**: `data/predictions/predictions_next_30_days.csv`
- **Features Engineering**: `data/features/data_with_features_latest.csv`
- **Logs del Sistema**: `logs/pipeline/`, `logs/training/`, `logs/validation/`

## 🧪 Testing

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Test específico de desagregación horaria
pytest tests/test_hourly_disaggregation.py -v

# Con coverage
pytest --cov=src tests/

# Tests críticos incluidos:
# - Validación suma(P1-P24) = TOTAL
# - Formato de salida (24 valores)
# - Clustering con diferentes n_clusters
# - Manejo de días especiales
```

## 📋 Requisitos Regulatorios

El sistema cumple con:

- **Acuerdo CNO 1303 de 2020**: Pronóstico de demanda para operadores de red
- **Proyecto CREG 143 de 2021**: Requisitos de precisión y granularidad

### Horizontes de Pronóstico

- **Mensual**: Actualización mensual, con un mes de antelación
- **Semanal**: Jueves antes de 12pm para semana siguiente
- **Diario**: 6am para día siguiente
- **Intradiario**: 3 actualizaciones al día

### Granularidades

- ✅ **Horaria (24 períodos)** - Implementada con clustering K-Means
  - MAPE: 1.61% (validado en 60 días)
  - Validación matemática: suma(P1-P24) = TOTAL
- ⏸️ **15 minutos (96 períodos)** - Pendiente (Fase 4)

## 🤝 Contribución

Este es un proyecto interno de EPM. Para contribuir:

1. Crear feature branch desde `development`
2. Implementar cambios con tests
3. Crear pull request con descripción detallada
4. Esperar revisión del equipo

## 📄 Licencia

Propiedad de **Empresas Públicas de Medellín (EPM)**

## 👥 Equipo

**Desarrollado para EPM - Empresas Públicas de Medellín**

---

## 🎓 Metodología Técnica

### Pipeline de Predicción Completo

1. **Ingesta de Datos**
   - Datos históricos de demanda (TOTAL + P1-P24)
   - Datos climáticos (temperatura, humedad, sensación térmica)
   - Calendario de festivos (librería `holidays`)

2. **Feature Engineering** (63 features)
   - 19 temporales: año, mes, día, día de semana, sin/cos
   - 25 de demanda: lags (1d, 7d, 14d) + rolling stats (7d, 14d, 28d)
   - 25 climáticas: temperatura, humedad, feels_like con lags
   - 4 estacionales: temporada lluviosa/seca
   - 3 de interacción: clima × calendario

3. **Predicción Diaria** (XGBoost)
   - Input: 63 features
   - Output: TOTAL_DIARIO
   - MAPE: 0.45%

4. **Desagregación Horaria** (K-Means Clustering)
   - Input: TOTAL_DIARIO + fecha
   - Clasificación: Laboral/Festivo/Fin_de_semana
   - Clustering: 35 clusters (normal) o 15 clusters (especial)
   - Output: P1-P24 (24 períodos horarios)
   - MAPE: 1.61%
   - Validación: sum(P1-P24) = TOTAL

5. **Validación y Monitoreo**
   - Validación retrospectiva vs datos históricos
   - Dashboards interactivos con Streamlit
   - Reportes automáticos con métricas detalladas

---

**Versión**: 2.0.0
**Última actualización**: Noviembre 2025

### Changelog

**v2.0.0** (Nov 2025)
- ✨ Sistema completo de desagregación horaria con clustering K-Means
- ✨ Integración con librería `holidays` para festivos colombianos
- ✨ 3 dashboards interactivos con Streamlit
- ✨ Script de validación interna automatizada
- ✨ Tests completos del sistema horario
- 🎯 MAPE horario: 1.61% (validado en 60 días)
- 🎯 Validación matemática: 100% suma correcta

**v1.0.0** (Nov 2024)
- ✅ Pipeline automatizado de datos
- ✅ Modelos ML (XGBoost, LightGBM, RandomForest)
- ✅ Feature engineering (63 features)
- ✅ Model registry con versionado
- 🎯 MAPE diario: 0.45%
