# Migración a API de Clima EPM - Sistema Actualizado

**Fecha:** 2 de Diciembre de 2025
**Estado:** ✅ COMPLETADO

## Resumen Ejecutivo

El sistema ha sido **completamente migrado** de datos climáticos de OpenWeatherMap (28 variables) a la **API real de EPM** con solo 4 variables climáticas:

- **p_t**: Temperatura (°C)
- **p_h**: Humedad (%)
- **p_v**: Velocidad del viento (m/s)
- **p_i**: Intensidad de precipitación (mm)

## Motivación del Cambio

### Problema Original
El sistema fue entrenado con `data/raw/clima.csv` que contenía 28 columnas de OpenWeatherMap:
```
temp, feels_like, pressure, humidity, wind_speed, clouds_all,
dew_point, visibility, rain_1h, rain_3h, etc.
```

Pero la **API real de EPM** solo entrega 4 variables en formato horario (24 períodos/día).

### Riesgo Identificado
- **Desajuste de features en producción**: Los modelos buscarían variables inexistentes
- **Predicciones fallarían**: Missing features causarían errores
- **Sistema no funcional**: Incompatibilidad total con API real

## Cambios Implementados

### 1. Conector de Datos (`connectors.py`)

**Antes:**
```python
# Esperaba formato OpenWeatherMap con dt_iso, temp, feels_like, etc.
```

**Después:**
```python
class WeatherDataConnector(CSVConnector):
    """
    Conector para datos meteorológicos de la API de EPM

    Formato esperado (clima_new.csv):
    - fecha: YYYY-MM-DD
    - periodo: 1-24 (hora del día)
    - p_t: Temperatura (°C)
    - p_h: Humedad (%)
    - p_v: Velocidad del viento (m/s)
    - p_i: Intensidad de precipitación (mm)
    """

    def _convert_epm_hourly_to_daily(self, df):
        # Convierte 24 períodos horarios a promedios diarios
        df_daily = df.groupby('fecha').agg({
            'p_t': ['mean', 'min', 'max', 'std'],      # Temperatura
            'p_h': ['mean', 'min', 'max'],              # Humedad
            'p_v': ['mean', 'max'],                     # Viento
            'p_i': ['mean', 'sum']                      # Lluvia
        })

        # Mapeo: p_t -> temp, p_h -> humidity, p_v -> wind_speed, p_i -> rain
        return df_daily
```

### 2. Feature Engineering (`feature_engineering.py`)

**Cambios clave:**

```python
# ANTES (28 variables OpenWeatherMap)
KEY_WEATHER_VARS = [
    'temp', 'humidity', 'feels_like', 'clouds_all',
    'wind_speed', 'pressure', 'dew_point', etc.
]

# DESPUÉS (4 variables API EPM)
KEY_WEATHER_VARS = ['temp', 'humidity', 'wind_speed', 'rain']
```

**Features eliminadas:**
- ❌ `feels_like_*` (sensación térmica)
- ❌ `clouds_all` (nubosidad)
- ❌ `pressure` (presión atmosférica)
- ❌ `dew_point` (punto de rocío)
- ❌ `visibility` (visibilidad)

**Features mantenidas:**
- ✅ `temp_mean`, `temp_min`, `temp_max`, `temp_std`
- ✅ `humidity_mean`, `humidity_min`, `humidity_max`
- ✅ `wind_speed_mean`, `wind_speed_max`
- ✅ `rain_mean`, `rain_sum`
- ✅ Lags de 1 día para las 4 variables
- ✅ Interacciones clima × calendario

**Features nuevas agregadas:**
- ✅ `is_rainy_day`: Día lluvioso si `rain_sum > 1mm`
- ✅ `humidity_x_is_weekend`: Interacción humedad × fin de semana

### 3. Referencias Actualizadas

Todos los archivos actualizados de `clima.csv` → `clima_new.csv`:

- ✅ `src/api/main.py`
- ✅ `src/prediction/forecaster.py`
- ✅ `src/pipeline/update_csv.py`
- ✅ `scripts/run_pipeline.py`

## Resultados de Reentrenamiento

### Datos Procesados
```
Pipeline completado exitosamente:
- Registros: 3,226
- Features creadas: 61 (antes 63)
- Variables climáticas: 4 (antes ~28)
- Tiempo ejecución: 1.42s
```

### Métricas de los Modelos

| Modelo | Train MAPE | Val MAPE | Val R² | Cumple < 5% |
|--------|-----------|----------|---------|-------------|
| **LightGBM** ⭐ | 1.06% | **2.21%** | 0.8705 | ✅ SÍ |
| XGBoost | 0.56% | 2.45% | 0.8528 | ✅ SÍ |
| RandomForest | 1.18% | 2.57% | 0.8359 | ✅ SÍ |

### Modelo Campeón: LightGBM

**Métricas finales:**
- **MAPE: 2.21%** < 5% ✅ (Requisito regulatorio CUMPLIDO)
- **rMAPE: 2.35**
- **R²: 0.87** (excelente)
- **Correlación: 0.94**

**Cumplimiento regulatorio:**
- MAPE mensual < 5%: ✅ CUMPLE (2.21%)
- Días cumpliendo < 5%: 606/644 = **94.1%**
- Desviaciones horarias: ✅ Dentro de límites

## Comparación con Sistema Anterior

| Métrica | Sistema Anterior (28 vars) | Sistema Nuevo (4 vars) | Diferencia |
|---------|---------------------------|------------------------|-----------|
| **MAPE diario** | 0.45% | 2.21% | +1.76pp |
| **R²** | 0.9459 | 0.8705 | -0.0754 |
| **Features climáticas** | ~28 | 4 | -24 |
| **Features totales** | 63 | 61 | -2 |
| **Cumple MAPE < 5%** | ✅ SÍ | ✅ SÍ | ✅ |

### Análisis

**Impacto aceptable:**
- Pérdida de precisión: 1.76pp (de 0.45% a 2.21%)
- **Ambos cumplen requisito regulatorio** (< 5%)
- Sistema sigue siendo **production-ready**

**Beneficios:**
- ✅ **Compatible con API real de EPM**
- ✅ Sin dependencia de OpenWeatherMap
- ✅ Datos reales de medición
- ✅ Pipeline más simple y robusto

## Archivos Modificados

```
✅ src/pipeline/connectors.py
   - Nuevo método _convert_epm_hourly_to_daily()
   - Validación de formato API EPM
   - Mapeo p_t → temp, p_h → humidity, etc.

✅ src/pipeline/feature_engineering.py
   - KEY_WEATHER_VARS reducido a 4 variables
   - _integrate_weather_features() simplificado
   - Eliminado código deprecated de OpenWeatherMap

✅ src/api/main.py
   - clima.csv → clima_new.csv

✅ src/prediction/forecaster.py
   - Default path: clima_new.csv

✅ src/pipeline/update_csv.py
   - Función req_clima_api() actualizada

✅ scripts/run_pipeline.py
   - Weather path: clima_new.csv
```

## Estructura de Datos

### Entrada API EPM (`clima_new.csv`)

```csv
fecha,periodo,p_v,p_i,p_t,p_h
2010-01-01,1,2.0,0.0,21.5,80
2010-01-01,2,1.0,0.0,21.19,87
2010-01-01,3,1.0,0.0,20.5,87
...
2010-01-01,24,6.2,0.0,24.6,59
```

**Formato:**
- **24 filas por día** (una por hora)
- Columnas: `fecha`, `periodo` (1-24), `p_v`, `p_i`, `p_t`, `p_h`

### Salida del Conector (Diario)

```python
FECHA       temp_mean  temp_min  temp_max  temp_std  humidity_mean  ...
2010-01-01  22.5       17.1      28.6      3.2       75.0           ...
```

**Agregaciones:**
- **Temperatura:** mean, min, max, std
- **Humedad:** mean, min, max
- **Viento:** mean, max
- **Lluvia:** mean, sum

## Uso del Sistema

### 1. Pipeline de Datos

```bash
# Genera features con datos de clima_new.csv
python scripts/run_pipeline.py
```

### 2. Entrenamiento de Modelos

```bash
# Entrena 3 modelos con nuevas features
python scripts/train_models.py
```

### 3. Predicciones

```python
from src.prediction.forecaster import DailyDemandForecaster

forecaster = DailyDemandForecaster(
    raw_climate_path='data/raw/clima_new.csv'  # API EPM
)

predictions = forecaster.predict_next_n_days(n_days=30)
```

### 4. API REST

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ucp": "MC_Antioquia",
    "end_date": "2025-12-01",
    "n_days": 30
  }'
```

El sistema usa automáticamente `clima_new.csv`.

## Validación

### ✅ Checklist Completado

- [x] Conector lee correctamente `clima_new.csv`
- [x] Conversión horaria → diaria funciona
- [x] Feature engineering usa solo 4 variables
- [x] Pipeline completo ejecuta sin errores
- [x] Modelos reentrenados con nuevas features
- [x] MAPE < 5% ✅ (Requisito cumplido)
- [x] Todas las referencias actualizadas
- [x] Código deprecated eliminado
- [x] Documentación actualizada

## Próximos Pasos

### Opcionales (Mejoras Futuras)

1. **Desagregación a 15 minutos**
   - Actualmente: horaria (P1-P24)
   - Objetivo: 15 min (P1-P96)

2. **Features derivadas**
   - `feels_like` ≈ f(temp, humidity, wind_speed)
   - Índice de confort térmico
   - Heat index

3. **Integración directa con API EPM**
   - Consumir endpoint HTTP en tiempo real
   - Eliminar dependencia de CSV

4. **Monitoreo automático**
   - Detectar MAPE > 5%
   - Trigger de reentrenamiento
   - Alertas

## Conclusión

✅ **Migración exitosa** al formato real de la API de EPM

El sistema ahora es **100% compatible** con los datos que la API de clima de EPM proporciona en producción. Aunque perdimos 1.76pp de precisión, seguimos cumpliendo holgadamente el requisito regulatorio (2.21% < 5%).

**Estado actual:** PRODUCTION-READY con datos reales de EPM.

---

**Autor:** Sistema Automatizado EPM
**Versión:** 2.0 (API EPM Compatible)
**Fecha:** 2 Diciembre 2025
