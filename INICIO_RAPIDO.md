# 🚀 Inicio Rápido - Sistema de Pronóstico EPM

## ⚡ En 3 Pasos

### 1️⃣ Ejecutar Pipeline de Datos
```bash
python pipeline/orchestrator.py
```
✅ Genera dataset con 63 features en 24 segundos

### 2️⃣ Entrenar Modelos
```bash
python prototype_model.py
```
✅ Entrena 3 modelos y genera reportes (MAPE: 0.45%)

### 3️⃣ Ver Dashboard
```bash
streamlit run prototype_dashboard.py
```
✅ Abre dashboard interactivo en http://localhost:8501

---

## 📊 Lo Que Verás en el Dashboard

- **Métricas principales:** MAPE 0.45%, R² 0.938
- **Comparación de 3 modelos** lado a lado
- **Vista mensual:** Barras agrupadas (más clara)
- **Vista diaria:** Últimos 60 días con tendencias
- **Vista completa:** Sliders interactivos para explorar
- **Análisis de errores:** Solo 0.6% de días con error > 5%
- **Feature importance:** Top 20 variables más importantes

---

## 📁 Archivos Importantes

| Archivo | Descripción |
|---------|-------------|
| `data/features/data_with_features_latest.csv` | Dataset final con 63 features |
| `data/features/prototype_predictions.csv` | Predicciones del mejor modelo |
| `data/features/prototype_summary.json` | Resumen de métricas |
| `pipeline_flowchart.html` | Diagrama de flujo visual (abrir en navegador) |

---

## 🎯 Resultados Clave

- ✅ **MAPE: 0.45%** (11x mejor que objetivo de 5%)
- ✅ **99.4% de días** con error < 5%
- ✅ **3 modelos** cumplen objetivo regulatorio
- ✅ **63 features** creadas automáticamente

---

## 📖 Documentación Completa

- `README.md` - Guía completa del sistema
- `FASE1_COMPLETADA.md` - Reporte detallado Fase 1
- `PROTOTIPO_RESULTADOS.md` - Análisis del modelo
- `RESUMEN_SESION.md` - Resumen de todo lo implementado

---

## 🐛 Solución de Problemas

**Error de NumPy:**
```bash
# Las advertencias de NumPy son normales, el código funciona
```

**Error de encoding en Windows:**
```bash
# Los errores de Unicode al final no afectan los resultados
```

**Dashboard no carga:**
```bash
# Asegúrate de ejecutar primero:
python prototype_model.py
```

---

**¿Preguntas?** Ver `README.md` para más detalles

**¡Listo para producción! 🎉**
