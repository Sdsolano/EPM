# LOG B1
## Estado actual: TERMINADO (sin commitear)
- Hecho: script + 2 CSV en src/adjustments/precision_data/; src/adjustments/insights.py; src/adjustments/router_insights.py; registro en src/api/main.py; tests/test_adjustments_insights.py (25) y tests/test_adjustments_insights_router.py (19); CONTRATO_B.md "Cambios" v1.1.
- Suite completa: 731 passed, 12 skipped, 1 failed (christmas, entorno aceptado).
- Verificar: `<python> -m pytest tests/test_adjustments_insights*.py -q -p no:cacheprovider`
## Decisiones tomadas sin el usuario
- patch.paso: "forma" si no esta omitido, si no "especiales"; ambos omitidos -> ajuste null.
- Nivel: objetivo = energia base * factor medido; si la corrida ya esta a <0.5% no se emite (evita doble conteo). Forma: conserva energia del dia.
- Evidencia: n>=6, |sesgo|>=0.5%, consistencia >=65%, mejora leave-one-out, muestra no mas vieja de 20 semanas.
- Festivos por ucp: libreria oficial; por run: los guardados en la corrida.
- Banda sin muestra: p*_pct null. kpis.alertas agrega "critico"; insights.nivel puede ser "critico".
- Extension B2: insights.sesgo_por_segmento / parametro proveedor_sesgo de recomendaciones().
