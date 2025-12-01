# 📊 Reporte de Desempeño del Sistema EPM

Este directorio contiene el reporte completo de desempeño del **Sistema de Pronóstico Automatizado de Demanda Energética** para EPM.

---

## 📁 Archivos Generados

### 1. Reportes Estáticos

#### `reporte_desempeno.html`
- Reporte HTML interactivo con visualizaciones embebidas
- Incluye todas las métricas y gráficos
- Abre directamente en el navegador
- **Para abrir:** Doble clic en el archivo o:
  ```
  file:///c:/Users/samue/OneDrive/Documentos/GitHub/EPM/reports/performance/reporte_desempeno.html
  ```

#### `RESUMEN_DESEMPENO.md`
- Resumen ejecutivo en formato Markdown
- Incluye análisis detallado de resultados
- Conclusiones y recomendaciones
- Cumplimiento regulatorio

### 2. Visualizaciones (PNG)

#### `daily_model_performance.png`
Visualizaciones del modelo de predicción diaria:
- Barras de MAPE por conjunto (Train/Val/Test)
- R² por conjunto
- Scatter plot: Predicciones vs Real (Test)
- Histograma de distribución de errores

#### `hourly_disaggregation_performance.png`
Visualizaciones de desagregación horaria:
- MAPE por método (Normal vs Especial)
- Distribución de errores en desagregación
- Evaluación de 90 días históricos

---

## 🚀 Dashboard Interactivo Streamlit

### Aplicación: `app_reporte_desempeno.py`

Dashboard interactivo con visualizaciones dinámicas usando Plotly.

#### Características:
- ✅ Métricas en tiempo real
- ✅ Gráficos interactivos con Plotly
- ✅ Filtros y controles dinámicos
- ✅ Exportación de datos
- ✅ Diseño profesional y responsive

#### Para ejecutar:

```bash
# Desde el directorio raíz del proyecto
streamlit run app_reporte_desempeno.py
```

**URL:** http://localhost:8501

---

## 📈 Resultados Principales

### Modelo de Predicción Diaria

| Métrica | Train | Validation | **Test** |
|---------|-------|------------|----------|
| **MAPE (%)** | 0.56 | 0.48 | **2.21** ✅ |
| **rMAPE** | 0.56 | 0.48 | 2.33 |
| **R²** | 0.9959 | 0.9954 | 0.8747 |
| **MAE (MWh)** | 169.39 | 148.64 | 582.69 |

**✅ CUMPLE:** MAPE de 2.21% está **muy por debajo** del umbral regulatorio de 5%

### Desagregación Horaria

| Métrica | Valor | Estado |
|---------|-------|--------|
| **MAPE Global** | 1.57% | ✅ Excelente |
| **MAE** | 18.87 MW | ✅ Bajo |
| **Validación Suma** | 100.0% | ✅ Perfecto |

**Métodos Evaluados:**
- **Normal (días regulares):** 1.59% MAPE en 71 días
- **Especial (festivos):** 1.51% MAPE en 19 días

---

## 🎯 Cumplimiento Regulatorio

| Requisito | Meta | Resultado | Estado |
|-----------|------|-----------|--------|
| MAPE Mensual | < 5% | 2.21% | ✅ **CUMPLE** |
| R² | > 0.80 | 0.8747 | ✅ **CUMPLE** |
| Desagregación Horaria | Implementado | ✓ | ✅ **CUMPLE** |
| Validación Suma P1-P24 | > 95% | 100% | ✅ **CUMPLE** |

### Desempeño vs Umbral
- MAPE: **2.21%** vs 5% umbral
- **56% mejor** que el requisito regulatorio
- Margen de **2.79 puntos porcentuales**

---

## 🔧 Cómo Regenerar el Reporte

### Script de Generación Automática

```bash
# Generar todos los reportes y visualizaciones
python scripts/generate_performance_report.py
```

Este script:
1. ✅ Carga el modelo campeón
2. ✅ Evalúa en Train/Val/Test (60%/20%/20%)
3. ✅ Calcula todas las métricas
4. ✅ Genera visualizaciones PNG
5. ✅ Crea reporte HTML
6. ✅ Evalúa desagregación horaria

**Tiempo de ejecución:** ~30-60 segundos

---

## 📊 Secciones del Reporte

### 1. Modelo de Predicción Diaria
- Métricas por conjunto (Train/Val/Test)
- Curvas de MAPE
- Scatter plots (predicciones vs real)
- Distribución de errores
- Serie temporal (si hay fechas disponibles)

### 2. Desagregación Horaria
- Métricas globales (MAE, RMSE, MAPE)
- Comparación por método (Normal vs Especial)
- Validación de suma (P1-P24 = TOTAL)
- Distribución de errores por día

### 3. Cumplimiento Regulatorio
- Tabla de cumplimiento vs requisitos
- Estado de cada métrica
- Recomendaciones

---

## 🛠️ Dependencias

Las siguientes librerías son necesarias:

```txt
pandas
numpy
matplotlib
seaborn
plotly
streamlit
joblib
scipy
scikit-learn
```

Para instalar:
```bash
pip install pandas numpy matplotlib seaborn plotly streamlit joblib scipy scikit-learn
```

---

## 📝 Notas Técnicas

### Splits de Datos
- **Train:** 60% (primeros registros cronológicos)
- **Validation:** 20% (siguientes registros)
- **Test:** 20% (últimos registros)

### Features Excluidas
Para evitar data leakage en predicción recursiva:
- Lags de demanda total (1d, 7d, 14d)
- Lags de períodos horarios
- Variables de cambio diario

### Evaluación de Desagregación
- **Período:** Últimos 90 días históricos
- **Método:** Comparación directa predicción vs real
- **Clustering:** K-Means con 35 clusters (normal) y 15 (especial)

---

## 🎓 Interpretación de Métricas

### MAPE (Mean Absolute Percentage Error)
- Error porcentual promedio
- **< 5%:** Excelente
- **5-10%:** Bueno
- **> 10%:** Revisar

### rMAPE (Relative MAPE)
- MAPE dividido por correlación de Pearson
- Penaliza predicciones con baja correlación
- **Mejor métrica** que MAPE solo

### R² (Coeficiente de Determinación)
- Proporción de varianza explicada
- **> 0.9:** Excelente
- **0.7-0.9:** Bueno
- **< 0.7:** Revisar

### MAE (Mean Absolute Error)
- Error absoluto promedio en MWh
- Más interpretable que RMSE
- Sensible a outliers

---

## 📞 Soporte

Para preguntas sobre este reporte:

- **Proyecto:** Sistema de Pronóstico Automatizado EPM
- **Versión:** 1.0.0
- **Fecha:** Diciembre 2024

---

## 🚀 Próximos Pasos

1. ✅ **Aprobado para Producción:** El sistema cumple todos los requisitos
2. ⏭️ **Monitoreo en Tiempo Real:** Implementar dashboard de monitoreo continuo
3. ⏭️ **Reentrenamiento Automático:** Activar cuando MAPE > 5%
4. ⏭️ **Validación Prospectiva:** Evaluar en producción con predicciones futuras

---

**Generado automáticamente por el Sistema de Evaluación EPM**
*Empresa de Energía de Antioquia - 2024*
