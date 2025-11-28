# Sistema de Desagregación Horaria

## Descripción

El sistema de desagregación horaria convierte pronósticos de demanda **diaria total** en distribuciones **horarias (P1-P24)** utilizando técnicas de clustering sobre patrones históricos.

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│           PREDICCIÓN DIARIA (Modelo ML)                     │
│  Input: 63 features → Output: TOTAL_DÍA (ej: 1500 MWh)     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │  CLASIFICACIÓN DE DÍA          │
         │  (holidays library + patterns) │
         │  - Laboral / Festivo / Fin de  │
         │    semana                      │
         │  - Temporada lluviosa/seca     │
         └───────────┬───────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌────────────────────┐  ┌────────────────────┐
│  DÍAS NORMALES     │  │  DÍAS ESPECIALES   │
│  (35 clusters)     │  │  (15 clusters)     │
│  K-Means sobre     │  │  K-Means sobre     │
│  patrones          │  │  festivos          │
│  normalizados      │  │  históricos        │
└────────┬───────────┘  └────────┬───────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  DISTRIBUCIÓN HORARIA  │
         │  P1-P24 (suma = total) │
         └───────────────────────┘
```

## Componentes

### 1. **CalendarClassifier** ([src/prediction/hourly/calendar_utils.py](src/prediction/hourly/calendar_utils.py))

Utiliza la librería `holidays` para gestionar festivos de Colombia automáticamente.

**Características:**
- ✅ Detección automática de festivos colombianos
- ✅ Clasificación: laboral, festivo, fin de semana
- ✅ Temporadas: lluviosa (Abr-May, Sep-Nov) vs seca
- ✅ Información completa de cada día

**Ejemplo:**
```python
from src.prediction.hourly import CalendarClassifier

classifier = CalendarClassifier()
info = classifier.get_full_classification(pd.to_datetime("2024-12-25"))

print(info)
# {
#   'fecha': Timestamp('2024-12-25'),
#   'dia_semana': 'Miércoles',
#   'tipo_dia': 'festivo',
#   'es_festivo': True,
#   'nombre_festivo': 'Navidad',
#   'temporada': 'seca',
#   ...
# }
```

### 2. **HourlyDisaggregator** ([src/prediction/hourly/hourly_disaggregator.py](src/prediction/hourly/hourly_disaggregator.py))

Desagregador para días **normales** (laborales, fines de semana).

**Funcionamiento:**
1. Normaliza patrones históricos (cada día suma 1.0)
2. Clustering K-Means (35 clusters por defecto)
3. Identifica cluster típico por día de la semana
4. Escala perfil normalizado × total diario

**Ejemplo:**
```python
from src.prediction.hourly.hourly_disaggregator import HourlyDisaggregator

# Entrenar
disaggregator = HourlyDisaggregator(n_clusters=35)
disaggregator.fit(df_historico, date_column='FECHA')

# Predecir
hourly = disaggregator.predict_hourly_profile(
    date=pd.to_datetime("2024-03-15"),
    total_daily=1500.0
)

print(hourly.sum())  # 1500.0 (exacto)
```

### 3. **SpecialDaysDisaggregator** ([src/prediction/hourly/special_days.py](src/prediction/hourly/special_days.py))

Desagregador para días **especiales** (festivos con patrones únicos).

**Características:**
- Filtra solo días festivos del histórico
- Agrupa por fecha (mm-dd) para consolidar múltiples años
- Clusters específicos para cada festivo

**Ejemplo:**
```python
from src.prediction.hourly.special_days import SpecialDaysDisaggregator

disaggregator = SpecialDaysDisaggregator(n_clusters=15)
disaggregator.fit(df_historico)

# Ver festivos conocidos
print(disaggregator.get_special_days_list())

# Predecir Navidad
hourly = disaggregator.predict_hourly_profile(
    date=pd.to_datetime("2024-12-25"),
    total_daily=1100.0
)
```

### 4. **HourlyDisaggregationEngine** ([src/prediction/hourly/disaggregation_engine.py](src/prediction/hourly/disaggregation_engine.py))

**Motor unificado** que orquesta todo el sistema.

**Flujo:**
1. Recibe fecha + total diario
2. Clasifica el día (calendario + festivos)
3. Selecciona desagregador apropiado
4. Genera predicción horaria
5. Valida que suma = total

**Ejemplo:**
```python
from src.prediction.hourly import HourlyDisaggregationEngine

# Cargar modelos entrenados
engine = HourlyDisaggregationEngine(auto_load=True)

# Predecir
result = engine.predict_hourly(
    date="2024-03-15",
    total_daily=1500.0
)

print(result)
# {
#   'date': ...,
#   'total_daily': 1500.0,
#   'hourly': array([...]),  # 24 valores
#   'method': 'normal',
#   'day_type': 'laboral',
#   'validation': {
#       'sum': 1500.0,
#       'is_valid': True,
#       'difference': 0.0
#   }
# }
```

## Integración con Forecaster

El sistema está integrado automáticamente en [src/prediction/forecaster.py](src/prediction/forecaster.py):

```python
from src.prediction.forecaster import ForecastPipeline

# Inicializar pipeline con desagregación horaria
pipeline = ForecastPipeline(
    model_path='models/trained/xgboost_latest.joblib',
    enable_hourly_disaggregation=True  # ← Habilita desagregación
)

# Predecir 30 días (ahora incluye P1-P24)
predictions = pipeline.predict_next_n_days(n_days=30)

print(predictions.columns)
# ['fecha', 'demanda_predicha', 'is_festivo', 'is_weekend',
#  'P1', 'P2', ..., 'P24']  ← Períodos horarios incluidos
```

## Entrenamiento

### Entrenar modelos de desagregación:

```bash
python scripts/train_hourly_disaggregation.py
```

Esto genera:
- `models/hourly_disaggregator.pkl` (días normales)
- `models/special_days_disaggregator.pkl` (días especiales)

### Entrenar desde código:

```python
from src.prediction.hourly import HourlyDisaggregationEngine

engine = HourlyDisaggregationEngine(auto_load=False)
engine.train_all(
    data_path='data/features/data_with_features_latest.csv',
    n_clusters_normal=35,
    n_clusters_special=15,
    save=True
)
```

## Testing

Ejecutar tests completos:

```bash
pytest tests/test_hourly_disaggregation.py -v
```

**Tests incluyen:**
- ✅ Validación crítica: `suma(P1-P24) == TOTAL_DIARIO`
- ✅ Clasificación de días (festivos, fines de semana)
- ✅ Clustering normal vs especial
- ✅ Guardado/carga de modelos
- ✅ Predicciones batch
- ✅ Integración con forecaster

## Validación Crítica

**REGLA FUNDAMENTAL:** La suma de los 24 períodos horarios **DEBE** ser igual al total diario.

```python
# Validación automática en cada predicción
result = engine.predict_hourly("2024-03-15", 1500.0)

assert result['validation']['is_valid']  # True
assert abs(result['hourly'].sum() - 1500.0) < 0.01  # Tolerancia 0.01 MWh
```

## Ventajas del Sistema

### ✅ Uso de Librerías Profesionales

- **`holidays`**: Festivos de Colombia automáticos (no hardcoded)
- **`scikit-learn`**: Clustering robusto y optimizado
- **`pandas`**: Manejo eficiente de fechas y temporadas

### ✅ Arquitectura Modular

- Calendario separado de clustering
- Días normales vs especiales en módulos independientes
- Fácil extensión a granularidad de 15 minutos

### ✅ Validación Rigurosa

- Tests automáticos para suma = total
- Validación en cada predicción
- Detección de anomalías

### ✅ Producción-Ready

- Modelos versionados y guardables
- Logging completo
- Manejo de errores (fallback a placeholders)

## Próximos Pasos

### Desagregación a 15 Minutos

Para cumplir requisito regulatorio de granularidad de 15 min:

```python
# TODO: Implementar en Fase 4
class FifteenMinuteDisaggregator:
    """Desagrega períodos horarios en intervalos de 15 min"""

    def disaggregate_hour_to_15min(self, hourly_value: float, hour: int) -> np.ndarray:
        """Retorna 4 valores de 15 min que suman hourly_value"""
        pass
```

### API Endpoints (Fase 4)

```python
# Ejemplo de endpoint
POST /predict
{
  "horizon": "daily",
  "granularity": "hourly",  # o "15min"
  "start_date": "2024-12-01",
  "days": 30
}

Response:
{
  "predictions": [
    {
      "date": "2024-12-01",
      "total_daily": 1500.5,
      "hourly_breakdown": {"P1": 45.2, "P2": 40.1, ...},
      "method": "normal",
      "validation": {"is_valid": true, "sum": 1500.5}
    }
  ]
}
```

## Referencias

- Especificaciones del proyecto: [docs/proyecto_especificaciones.pdf](docs/proyecto_especificaciones.pdf)
- Tests: [tests/test_hourly_disaggregation.py](tests/test_hourly_disaggregation.py)
- Clustering original: [notebooks/cluster.py](notebooks/cluster.py) (deprecado)

---

**Versión:** 1.0.0
**Fecha:** Noviembre 2024
**Estado:** ✅ Completado e integrado
