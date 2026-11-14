# LOG B2

## Estado actual
- Hecho: PLAN.md; datos (`scripts/noche2_b2_datos.py`), lib (`scripts/noche2_b2_lib.py`), experimentos consolidados
  (`scripts/noche2_b2_exp.py` -> `resultados.json`, 25 s), módulo de producción `src/adjustments/correccion.py` (bandas empíricas,
  corregir_sesgo opt-in, corregir_clima opt-in experimental) con paridad EXACTA contra el experimento, tests `tests/test_adjustments_correccion.py` (22 pasan).
- LÍNEA BASE reproducida: Antioquia prueba base 2.44 -> final 2.33 (MAPE sem 1.52->1.37); Atlántico 5.86 -> 5.46 (4.57->4.41). Coincide con backtest_modelo_20261113.md.
- Veredictos (ver REPORTE.md): NADA cumple la regla para activarse por defecto. Sesgo: no. Mezcla/selección: mejora WAPE ~0.06-0.1 pp pero empeora MAPE semanal
  en alguna celda; clima: prometedor pero no calibrable en Antioquia y clima observado. Bandas: sí se recomiendan (no cambian el pronóstico), escala 1.2.
- Falta: REPORTE.md, suite completa final.
- Verificar: `<python> scripts/noche2_b2_datos.py --out <cache>` (1.5 min) y `<python> scripts/noche2_b2_exp.py --cache <cache> --out docs/work/ajustes/noche2/B2` (25 s);
  `<python> -m pytest tests/test_adjustments_correccion.py -q`.
- Archivos sin commitear (míos): scripts/noche2_b2_{datos,lib,exp}.py, src/adjustments/correccion.py, tests/test_adjustments_correccion.py, docs/work/ajustes/noche2/B2/*.
- Hallazgo de proceso: la primera versión del cache tomaba el domingo de brecha del bloque de la semana siguiente (pronóstico más informado = FUGA).
  Corregido: cada semana usa solo SU bloque de curvas (`_curvas_por_semana`). Cache en scratchpad (no versionado).

## Decisiones sin el usuario
(se irán anotando)
- La regla "mejora en las 4 celdas (UCP x partición) >= 0.05 pp y MAPE semanal <= +0.05 pp en las 4" se aplicó de forma estricta (ver `veredicto` en noche2_b2_exp.py).
- Hiperparámetros elegidos SIEMPRE en calibración, objetivo = suma de ΔWAPE relativo de ambas UCP, con restricción de MAPE semanal de calibración <= +0.05 pp.
- Cobertura de bandas: se excluyen las primeras 8 semanas del cache (sin historial de errores); `escala` = 1.2 elegida para ~80 % de cobertura media en calibración.
- Puente = día laboral adyacente a un festivo; clima = temperatura media diaria de `clima_new.csv` (observada).
- No se tocó el pipeline ni `_params.py`: ningún default cambia, corridas guardadas intactas.
