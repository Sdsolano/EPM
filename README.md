# Sistema de Pronóstico Automatizado de Demanda Energética - EPM

Sistema inteligente de pronóstico de demanda energética con capacidad de autoaprendizaje para el sistema de distribución de EPM en Antioquia, Colombia.

## Estado Actual: Fase 1 Completada ✓ + Prototipo Validado ✓

**Pipeline Automatizado de Datos** - Implementado y funcional
**Modelo Prototipo** - MAPE 0.45% (11x mejor que objetivo de 5%)

## Estructura del Proyecto

```
EPM/
├── config.py                    # Configuración central del sistema
├── pipeline/                    # Pipeline automatizado de datos (Fase 1)
│   ├── __init__.py
│   ├── data_connectors.py      # Conectores para lectura automática de datos
│   ├── data_cleaning.py        # Sistema de limpieza y validación automática
│   ├── feature_engineering.py  # Feature engineering automático
│   ├── monitoring.py           # Sistema de logging y monitoreo
│   └── orchestrator.py         # Orquestador principal del pipeline
├── data/                       # Datos procesados
│   ├── raw/                    # Datos crudos (input)
│   ├── processed/              # Datos limpios
│   └── features/               # Datos con features (listos para modelos)
├── logs/                       # Logs de ejecución y reportes
└── models/                     # Modelos entrenados (Fase 2)
```

## Uso Rápido

### Ejecutar Pipeline Completo

```python
from pipeline.orchestrator import run_automated_pipeline

# Ejecutar pipeline con datos de 2017 en adelante
df_features, report = run_automated_pipeline(
    power_data_path='datos.csv',
    weather_data_path='data_cleaned_weather.csv',
    start_date='2017-01-01'
)

print(f"Datos procesados: {len(df_features)} registros")
print(f"Features creadas: {report['data_summary']['features_created']}")
```

### Usar Componentes Individuales

```python
# 1. Conectores de datos
from pipeline.data_connectors import PowerDataConnector

connector = PowerDataConnector({'path': 'datos.csv'})
df = connector.read_latest_data(days_back=30)

# 2. Limpieza de datos
from pipeline.data_cleaning import clean_power_data

df_clean, quality_report = clean_power_data(df)
print(quality_report.summary())

# 3. Feature Engineering
from pipeline.feature_engineering import create_features

df_features, summary = create_features(df_clean, weather_df)
```

## Características Implementadas (Fase 1)

### 1. Conectores Automatizados
- Lectura automática desde CSV (extensible a API/BD)
- Filtrado por rango de fechas
- Validación de conexión
- Logging de lectura

### 2. Limpieza y Validación Automática
- Validación de esquema
- Conversión automática de tipos
- Clasificación de días (LABORAL/FESTIVO)
- Detección de valores faltantes
- Detección de outliers (método IQR + desviaciones estándar)
- Validación de consistencia de datos
- Reportes de calidad estructurados

### 3. Feature Engineering Automático

**Features de Calendario (19 features):**
- Componentes temporales: año, mes, día, semana, trimestre
- Indicadores: fin de semana, festivo, inicio/fin de mes
- Features cíclicas: sin/cos para día de semana, mes, día del año

**Features de Demanda Histórica (25 features):**
- Lags: 1, 7, 14 días
- Rolling statistics: media, std, min, max (ventanas de 7, 14, 28 días)
- Tasa de cambio día a día
- Lags de períodos horarios clave (P8, P12, P18, P20)

**Features de Estacionalidad (4 features):**
- Temporada (lluviosa/seca para Colombia)
- Indicadores de meses especiales (enero, diciembre)
- Semana del mes

**Features Climáticas (25 features):**
- Agregaciones diarias: temperatura, humedad, sensación térmica
- Estadísticas: media, min, max, std
- Lags de variables climáticas
- Interacciones clima-calendario

**Features de Interacción (3 features):**
- día_semana × festivo
- mes × festivo
- fin_semana × mes

**Total: 63 features creadas automáticamente**

### 4. Sistema de Logging y Monitoreo
- Logging estructurado con múltiples niveles
- Tracking de ejecución del pipeline por etapas
- Detección y registro de alertas
- Monitoreo de calidad de datos
- Reportes en formato JSON
- Métricas de tiempo de ejecución

## Datos de Salida

El pipeline genera automáticamente:

1. **power_clean_{timestamp}.csv** - Datos de demanda limpios
2. **weather_clean_{timestamp}.csv** - Datos meteorológicos limpios
3. **data_with_features_{timestamp}.csv** - Dataset completo con todas las features
4. **data_with_features_latest.csv** - Última versión (fácil acceso)
5. **Reportes JSON** en `logs/` con métricas de ejecución

## Configuración

Edita `config.py` para ajustar:
- Rutas de directorios
- Umbrales de calidad de datos
- Ventanas de rolling statistics
- Lags de variables
- Métricas regulatorias (MAPE, desviaciones)

## Validación con Modelo Prototipo ✓

Se validó que las features creadas son altamente efectivas:

**Resultados del Prototipo:**
- **MAPE: 0.45%** - 11x mejor que el objetivo de 5%
- **R²: 0.938** - Excelente ajuste del modelo
- **99.4% de días** con error < 5%
- **Validación cruzada:** MAPE promedio 0.77%

Ver dashboard interactivo:
```bash
streamlit run prototype_dashboard.py
```

Ver reporte completo: [PROTOTIPO_RESULTADOS.md](PROTOTIPO_RESULTADOS.md)

## Próximos Pasos - Fase 2

- [ ] Desarrollo de 3 modelos predictivos de ML
- [ ] Sistema de entrenamiento automático
- [ ] Versionado y gestión de modelos
- [ ] Sistema de evaluación y selección automática
- [ ] Backtesting automático

## Requisitos

```
pandas
numpy
scikit-learn
```

## Ejecución del Pipeline

```bash
python pipeline/orchestrator.py
```

## Cumplimiento Regulatorio

El sistema está diseñado para cumplir con:
- **Acuerdo CNO 1303 de 2020**
- **Proyecto de resolución CREG 143 de 2021**

Métricas objetivo:
- MAPE mensual < 5%
- Desviaciones diarias < 5%
- Desviaciones horarias < 60 conteos/mes

## Versión

**v1.0.0** - Fase 1 Completada (Noviembre 2024)

---

**Desarrollado para EPM - Empresas Públicas de Medellín**
