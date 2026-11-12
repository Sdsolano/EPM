# Cambios de Régimen en la Demanda Energética

Este documento registra cambios estructurales observados en los patrones de demanda energética que requieren features especiales en el modelo de predicción.

## 📊 Cambio de Régimen: 24 de Marzo 2026

### Descripción del Cambio
A partir del **24 de marzo de 2026**, se observó un cambio estructural significativo en el nivel base de demanda energética para el UCP Antioquia.

### Análisis Cuantitativo

#### Antes del 24 de marzo 2026:
- **Error promedio**: 1.93%
- **Sesgo**: +318 MWh (despreciable)
- **Rango de demanda**: 26,000 - 32,000 MWh

#### Desde el 24 de marzo 2026:
- **Error promedio sin ajuste**: 5.07%
- **Sesgo sistemático**: +1,644 MWh
- **Factor de incremento**: +5.34%
- **Rango de demanda**: 28,000 - 34,000 MWh

### Hipótesis de Causas
Posibles factores que explican este cambio:
1. Nueva política energética o tarifaria
2. Incremento en demanda industrial
3. Cambios estacionales atípicos
4. Nuevas cargas conectadas a la red
5. Cambios en patrones de consumo post-pandemia

### Solución Implementada

#### ⚠️ Feature ML (DESCARTADO - No Funcionó)
**Intentado:** Feature binario `is_high_demand_regime`
**Resultado:** Empeoró el error de 5.03% a 6.08% (+1.04%)
**Causa del fallo:** Desbalance de datos (9 años con regime=0 vs 2 meses con regime=1) causó aprendizaje inverso

#### ✅ Ajuste Post-Predicción Temporal (Implementado)
```python
# En forecaster.py, después de la predicción del modelo
REGIME_CHANGE_DATE = datetime(2026, 3, 24)
REGIME_ADJUSTMENT_FACTOR = 1.053  # +5.3%

if fecha >= REGIME_CHANGE_DATE and not es_semana_santa:
    demanda_pred = demanda_pred * REGIME_ADJUSTMENT_FACTOR
```

**Características:**
- ✅ Ajuste simple y directo de +5.3%
- ✅ No afecta Semana Santa ni días especiales (ya tienen ajuste propio)
- ✅ **TEMPORAL**: Se debe remover al reentrenar con suficientes datos (sept 2026)
- ⚠️ Requiere actualización manual si el régimen cambia

#### Archivos Modificados
- `src/prediction/forecaster.py`: Línea ~1141 (ajuste post-predicción temporal)

### Resultados Esperados

Con el ajuste post-predicción temporal:
- Error esperado: ~2-3% (reducción de 5.03% actual)
- Reducción de sesgo: de +1,644 MWh a ~500-700 MWh
- Mejora en días con error alto: de 8.57% máximo a ~4-5%

**Nota:** Esta es una solución temporal. El objetivo es acumular datos suficientes del nuevo régimen para que el modelo aprenda naturalmente el patrón sin necesidad de ajuste manual.

### Instrucciones para Remover el Ajuste Temporal

**Cuándo remover:** Cuando tengamos al menos 6-12 meses de datos del nuevo régimen (estimado: septiembre 2026)

**Pasos:**
1. Verificar que tienes datos desde marzo 24, 2026 hasta al menos septiembre 2026
2. Comentar o remover el bloque de código del ajuste temporal en `src/prediction/forecaster.py` (líneas ~1141-1162)
3. Reentrenar el modelo:
```bash
python scripts/train_model.py --ucp Antioquia
```
4. Validar que el MAPE < 2% sin el ajuste temporal
5. Si MAPE > 3%, mantener el ajuste y revisar en 3 meses más

### Monitoreo Continuo

**Indicadores a vigilar:**
1. **MAPE mensual**: Debe mantenerse < 2% después de reentrenamiento
2. **Sesgo**: Debe ser cercano a 0 (± 300 MWh)
3. **Días con error > 5%**: Debe ser < 5% de los días

**Alertas:**
- Si MAPE > 3% por más de 5 días consecutivos → Posible nuevo cambio de régimen
- Si sesgo > 1,000 MWh por más de 7 días → Revisar si hay otro cambio estructural

### Fecha de Revisión
**Próxima revisión programada**: 24 de junio 2026 (3 meses después del cambio)

**Responsable**: Equipo de Data Science - EPM

---

## 🔮 Futuras Extensiones

Si se observan cambios de régimen adicionales, considerar:

1. **Múltiples regímenes**: Agregar features `regime_id` (1, 2, 3, etc.)
2. **Transiciones suaves**: Usar features continuos en lugar de binarios
3. **Detección automática**: Implementar algoritmo de detección de cambios de régimen
4. **Features dinámicos**: Calcular el régimen actual basado en rolling averages

---

**Última actualización**: 2026-06-01
**Versión**: 1.0
