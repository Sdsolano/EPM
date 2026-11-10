# Log — Fix Modelo Base festivos (incidente 2026-11-02)

**Fecha:** 2026-11-10  
**Alcance:** Modelo Base / línea de referencia (`calculate_base_curves`)  
**UCP de evidencia:** Bolívar (CSV incidente); validación cruzada Cesar / Planeta  

---

## 1. Síntoma

Gráfico de monitoreo (`Flujo: AE = 1 + 1`, 01–03 nov 2026):

- **01-11 (domingo):** real y pronóstico alineados  
- **02-11 (lunes):** real baja (patrón festivo); pronóstico sube como lunes hábil  
- **03-11 (martes):** vuelve a comportarse como día normal  

Reentrenar el modelo IA **no corregía** el problema.

---

## 2. Diagnóstico

1. El gráfico usa el **Modelo Base**, no el campeón XGBoost/LightGBM.  
2. `2026-11-02` es **Todos los Santos** (primer lunes de noviembre, Ley Emiliani). Confirmado en API de festivos.  
3. El Método 1 del Modelo Base promediaba el **mismo mes-día** histórico (`02-nov` de años previos: sáb/dom/jue/mié…), no las ediciones previas del mismo festivo (`2025-11-03`, `2024-11-04`, …).  
4. En la práctica el 02-11 se comportaba como **lunes laboral** → sobreestimación.  
5. Hallazgo extra: festivos de Bolívar en API incompletos en 2021/2022/2024/2025; un matching solo por cercanía confundía con Día de la Raza (octubre).

---

## 3. Solución implementada

### Código

| Archivo | Rol |
|---------|-----|
| `src/prediction/holiday_homology.py` | Empareja festivos homólogos por **nombre** (calendario Colombia) + fallback; perfil horario con forma homóloga y nivel escalado por ratio festivo/domingo reciente |
| `src/api/main.py` | Método 1 de `calculate_base_curves` usa `average_holiday_hourly_profile` |
| `requirements.txt` | Dependencia `holidays>=0.40` |
| `tests/test_holiday_homology.py` | 11 tests (móviles, fijos, Pascua, API incompleta, etc.) |
| `scripts/validate_holiday_base_curve_fix.py` | Validación incidente + día regular + festivo fijo |

### Comportamiento

- **Días normales:** sin cambio (Método 2 intacto).  
- **Festivos:** se busca la misma fiesta en años anteriores; se ajusta el nivel con domingos recientes para no quedar bajos por crecimiento de demanda.

---

## 4. Resultados de validación

### Incidente 02-11-2026 (totales diarios)

| Mercado | Real | Después (fix) | Error |
|---------|------|------------------|-------|
| Bolívar | 11,887.7 | 11,898.1 | **+0.1%** |
| Cesar | 7,147.1 | 6,782.3 | −5.1% |
| Planeta | 972.4 | 1,014.5 | +4.3% |

Comparación Bolívar 02-11:

| Enfoque | Total | Error |
|---------|-------|-------|
| Real | 11,888 | — |
| Antes (como lunes hábil) | 12,984 | **+9.2%** |
| Legacy mes-día | 9,882 | −16.9% |
| Después (homólogos) | 11,898 | **+0.1%** |

### Regular / fijos

- Martes hábil 03-11 (Bolívar): ~−2.2% (misma lógica de siempre).  
- Navidad: homólogos `*-12-25` correctos.  
- Tests unitarios: **11/11 pass**.

### Artefactos visuales

- `incident/incidente_antes_despues.png` — real vs antes vs después  
- `incident/comparacion_ajuste_nov2026.csv` — series horarias 01–03 nov  
- Evidencia original: imagen WhatsApp + `csvs.zip`

---

## 5. Commits

| Hash | Mensaje |
|------|---------|
| `81a0417` | Add incident evidence for 2026-11-02 base-curve holiday miss. |
| `2d4bf3d` | Fix Modelo Base holiday curves with homologous festival matching. |
| *(este commit)* | Log del diagnóstico, fix y validación del incidente. |

---

## 6. Qué no se cambió / pendientes opcionales

- No se reentrenó ni alteró el modelo IA campeón.  
- El etiquetado `is_festivo` en feature engineering histórico (fallback `TIPO DIA` = fin de semana) sigue siendo un tema aparte si se quiere mejorar el modelo IA en festivos entre semana.  
- No se hizo backtest exhaustivo de todos los festivos de todos los UCP; la mejora es global en la **ruta de festivos** del Modelo Base, validada en incidente + tests estructurales.

---

## 7. Cómo revalidar

```bash
.venv/bin/python -m pytest tests/test_holiday_homology.py -v
.venv/bin/python scripts/validate_holiday_base_curve_fix.py
```
