# ⚠️ IMPORTANTE: Entrenamiento de Modelos

## ✅ **AMBOS SCRIPTS EXCLUYEN LAGS AUTOMÁTICAMENTE**

### **Scripts Disponibles (ambos correctos):**
```bash
python scripts/train_models.py           # ← Modificado para excluir lags
python scripts/train_models_no_lags.py   # ← Versión dedicada sin lags
```

### **API Endpoint con Force Retrain:**
```json
POST /predict
{
  "force_retrain": true  // ← También excluye lags automáticamente
}
```

**Todos los métodos de entrenamiento ahora excluyen lags por defecto.**

---

## 📋 **Por Qué:**

El sistema tiene un problema fundamental con features de lag en predicción recursiva:

- **Con lags**: Validación 1.33% MAPE, Producción ~20% MAPE (domingos)
- **Sin lags**: Validación 2.66% MAPE, Producción ~2.5% MAPE (domingos)

**Causa:** En predicción recursiva, los lags usan predicciones anteriores en lugar de valores reales, creando un ciclo vicioso de errores acumulados.

---

## 🔧 **Features Eliminados (13 total):**

- `total_lag_1d`, `total_lag_7d`, `total_lag_14d`
- `p8_lag_1d`, `p8_lag_7d`
- `p12_lag_1d`, `p12_lag_7d`
- `p18_lag_1d`, `p18_lag_7d`
- `p20_lag_1d`, `p20_lag_7d`
- `total_day_change`, `total_day_change_pct`

---

## ✅ **Features Usados (52 total):**

- Features temporales: `year`, `month`, `day`, `dayofweek`, `is_weekend`, `is_festivo`
- Features climáticos: temperatura, humedad, feels_like
- Rolling statistics: `rolling_mean_7d`, `rolling_std_7d` (más robustas que lags)
- Features de interacción: `month_x_festivo`, `dayofweek_x_festivo`

---

## 🔄 **Si Necesitas Re-entrenar:**

1. Ejecuta: `python scripts/train_models_no_lags.py`
2. El script automáticamente:
   - Elimina features de lag
   - Entrena 3 modelos (XGBoost, LightGBM, RandomForest)
   - Selecciona el mejor
   - Lo registra como campeón
3. Reinicia el API para usar el nuevo modelo

---

## 📊 **Resultados Esperados:**

- **Validación MAPE:** ~2.5-2.7%
- **Producción MAPE:** ~2.5-3.0%
- **Error en domingos:** ±2-3% (vs ±20% con lags)

---

## 🔒 **Mantener Consistencia:**

El forecaster (`src/prediction/forecaster.py`) SIGUE calculando lags por compatibilidad, pero el modelo los IGNORA automáticamente porque no están en su lista de `feature_names`.

Esto permite:
- ✅ Código estable sin modificaciones arriesgadas
- ✅ El modelo filtra las features que necesita
- ✅ Si en el futuro se encuentra una forma de usar lags correctamente, el código ya está

---

**Última actualización:** 2025-11-29  
**Modelo actual:** LightGBM sin lags (2.66% rMAPE)

