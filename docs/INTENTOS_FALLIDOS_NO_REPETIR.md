# Intentos Fallidos - NO REPETIR

**Fecha**: 2025-11-30
**Contexto**: Intentos de mejorar predicciones horarias que EMPEORARON los resultados

---

## 🚫 Intento 1: Enhanced Features para Predicción Diaria

### Qué Hicimos
Agregamos 218 enhanced features al modelo de predicción diaria:
- Expanding means por día de semana (`dow_0_expanding_mean`, etc.)
- Expanding means por mes (`month_1_expanding_mean`, etc.)
- Patrones horarios históricos
- Features de interacción climática
- Features de temperatura agregada

**Archivo modificado**: `src/prediction/forecaster.py` - método `add_enhanced_features()`

### Resultado
❌ **EMPEORÓ la predicción diaria**:
- MAPE aumentó de 2.74% a 4.77%
- Predicciones bajaron de ~30,000 MW a ~24,000-29,000 MW

### Por Qué Falló
1. **Overfitting**: Demasiadas features (270 totales) para relativamente pocos datos
2. **Ruido**: Las features expandidas agregaban ruido en lugar de señal
3. **Modelo simple funciona mejor**: El modelo con 52 features básicas ya estaba bien optimizado

### Lección
✅ **NO agregar enhanced features al modelo de predicción diaria**
✅ **Mantener modelo simple con 52 features básicas**

---

## 🚫 Intento 2: ML Disaggregator para Desagregación Horaria

### Qué Hicimos
Implementamos un ML-based hourly disaggregator usando MultiOutputRegressor:
- Entrenado con 3 meses de datos (jul-sep 2025)
- 270 features (52 básicas + 218 enhanced)
- Predice proporciones para 24 horas simultáneamente

**Archivos creados**:
- `src/prediction/hourly/ml_disaggregator.py`
- `scripts/train_improved_hourly_disaggregation.py`
- `models/hourly_disaggregation_ml.joblib`

**Archivos modificados**:
- `src/prediction/forecaster.py` - para usar ML disaggregator

### Métricas de Entrenamiento (ENGAÑOSAS)
✅ MAPE: 2.77%
✅ Horas con error >5%: 2.86% (439/15,360 horas)

### Resultado en Producción
❌ **MUCHO PEOR que el clustering simple**:

| Métrica | Clustering (ANTES) | ML Disaggregator (DESPUÉS) |
|---------|-------------------|----------------------------|
| MAPE mensual | 2.75% | 2.83% (+0.08%) |
| Días error >5% | 10% | 16.7% (+6.7 pp) |
| Horas error >5% | **18.33%** | **31.11% (+12.78 pp)** |

### Por Qué Falló
1. **Overfitting severo**: Gap de 28 puntos entre train (2.86%) y producción (31.11%)
2. **Datos insuficientes**: Solo 3 meses de entrenamiento
3. **Cambio de distribución**: Octubre 2025 tuvo patrones diferentes a jul-sep
4. **Complejidad innecesaria**: Clustering simple generaliza mejor

### Evidencia del Problema
- Sábados de oct-2025: Sobreestimados en 5.5%
- Sábados de jul-sep: Subestimados en -2.04%
- El modelo no pudo adaptarse a la variación

### Lección
✅ **NO usar ML disaggregator con pocos datos**
✅ **Clustering simple es más robusto**
✅ **Métricas de training pueden ser engañosas**
✅ **Validar en MÚLTIPLES meses antes de desplegar**

---

## 🚫 Intento 3: Enfoque Híbrido (Básicas + Enhanced)

### Qué Hicimos
Como las enhanced features fallaron para predicción diaria pero el ML disaggregator las necesitaba:
- Predicción diaria: 52 features básicas
- Desagregación horaria: 270 features (básicas + enhanced)
- Generar enhanced features SOLO cuando se llama al ML disaggregator

**Archivo modificado**: `src/prediction/forecaster.py`

### Resultado
❌ **No resolvió el problema fundamental**:
- La predicción diaria mejoró (volvió a 2.74%)
- Pero el ML disaggregator seguía siendo malo (31.11% horas error)

### Lección
✅ **El problema era el ML disaggregator, no las features**
✅ **Agregar complejidad no arregla un modelo fundamentalmente mal entrenado**

---

## ✅ Qué SÍ Funciona (Estado Actual)

### Predicción Diaria
**52 features básicas**:
- 21 features de calendario (año, mes, día, día_semana, etc.)
- 22 features climáticas (temp, humidity, etc.)
- 4 features de estacionalidad
- 3 features de interacción
- 2 features temporales

**Resultado**: MAPE 2.74%, estable y confiable

### Desagregación Horaria
**Clustering simple** (`models/hourly_disaggregator.pkl`):
- Basado en patrones históricos promedio
- Clusters por: tipo_día (laborable/sábado/domingo/festivo) + mes
- Usa datos de múltiples años
- No requiere features complejas

**Resultado**: 18.33% horas con error >5% (MEJOR que ML: 31.11%)

---

## 📋 Checklist Antes de Implementar Nuevas "Mejoras"

Antes de intentar optimizar el sistema, verificar:

- [ ] ¿El modelo actual ya cumple los requisitos regulatorios?
- [ ] ¿Tenemos al menos 12 meses de datos para entrenar?
- [ ] ¿Validamos en al menos 3 meses diferentes de test?
- [ ] ¿Comparamos métricas en producción vs entrenamiento?
- [ ] ¿El nuevo método es significativamente mejor (>10% mejora)?
- [ ] ¿Entendemos POR QUÉ el nuevo método es mejor?
- [ ] ¿Tenemos un plan de rollback rápido?

**Si respondiste NO a alguna pregunta → NO implementar el cambio**

---

## 🎯 Requisitos Regulatorios (Recordatorio)

| Requisito | Objetivo | Estado Actual |
|-----------|----------|---------------|
| MAPE mensual | < 5% | ✅ 2.75% |
| Días error >5% | < 5% de días (~1.5 días/mes) | ❌ 10% (3 días) |
| Horas error >5% | < 60 horas/mes (8.33%) | ❌ 18.33% (132 hrs) |

**Cumplimos 1 de 3 requisitos**

---

## 💡 Posibles Mejoras Futuras (A Investigar con CUIDADO)

### 1. Mejorar Clustering Disaggregator
- Agregar features climáticas a los clusters
- Optimizar número de clusters (actualmente por tipo_día + mes)
- Validar si clustering por semana del año es mejor

**Riesgo**: Bajo (es mejora iterativa del método que funciona)
**Requisito**: Validar en al menos 3 meses

### 2. Ajustar Predicción Diaria para Sábados
- Los sábados son sistemáticamente sobreestimados
- Investigar si agregar un factor de corrección específico para sábados
- O entrenar un modelo separado para fines de semana

**Riesgo**: Medio (podría romper otros días)
**Requisito**: A/B testing en producción

### 3. Features Climáticas Mejoradas
- Actualmente usamos lag=1d de clima
- Investigar si promedios de 3-7 días mejoran
- O features de tendencia climática

**Riesgo**: Bajo (son pocas features)
**Requisito**: Validación cruzada exhaustiva

---

## 🔴 Lo Que NUNCA Hacer

1. ❌ **NO entrenar modelos ML complejos con < 12 meses de datos**
2. ❌ **NO confiar solo en métricas de training/validation**
3. ❌ **NO agregar >50 features nuevas de golpe**
4. ❌ **NO desplegar a producción sin validar en múltiples meses**
5. ❌ **NO asumir que "más complejo = mejor"**
6. ❌ **NO cambiar múltiples componentes simultáneamente**

---

## 📝 Conclusión

**Sistema actual (52 features + clustering)**:
- MAPE: 2.75% ✅
- Horas error >5%: 18.33% ❌

**Sistema con "mejoras" (270 features + ML disaggregator)**:
- MAPE: 2.83% ❌ (peor)
- Horas error >5%: 31.11% ❌❌ (MUCHO peor)

**Decisión correcta**: REVERTIR y mantener el sistema simple que funciona mejor.

---

**Última actualización**: 2025-11-30
**Estado**: ML Disaggregator DESHABILITADO (archivo renombrado a .DISABLED)
**Sistema activo**: 52 features básicas + clustering disaggregator
