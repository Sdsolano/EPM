# PLAN_B — Pista B (backend/modelo), noche 2

Worktree `C:/Users/samue/OneDrive/Documentos/GitHub/EPM-wt-n2`, rama `feat/noche2-backend`. Python: `C:/Users/samue/OneDrive/Documentos/GitHub/EPM/.venv/Scripts/python.exe`.
Objetivo y principios: `noche2/PLAN.md`. Contrato de API: `noche2/CONTRATO_B.md` (fuente de verdad para el front).
Reglas de agentes: `docs/work/ajustes/REGLAS_AGENTES.md`. Curvas k1 de backtest: `C:/Users/samue/OneDrive/Documentos/GitHub/EPM/reports/ajustes/curvas_modelo_k1_{Antioquia,Atlantico}.csv` (52 semanas, cols ucp,k,semana_inicio,corte,origen,fecha,P1..P24; solo lectura).

## Partes
- **B1 — Insights (API)**: módulo `src/adjustments/insights.py` (lógica pura, testeable) + `src/adjustments/router_insights.py`
  (prefijo `/adjustments/insights`, incluir en `src/api/main.py` junto a `router_asr`) + datos `src/adjustments/precision_data/{ucp}_k1.csv`
  (copia compacta de las curvas k1: fecha + 24 columnas; script `scripts/generar_historial_pronosticos.py` que la genera; redondear a 1 decimal)
  + tests `tests/test_adjustments_insights*.py`. Dueño: agente B1 (puede partirse en B1a precisión/banda/series, B1b resumen/recomendaciones).
- **B2 — Mejora de pronósticos (investigación con backtest sin fuga)**: sandbox bajo `scratch/` o temp (no tocar data/ ni models/ reales), reutilizando
  `scripts/backtest_adjustments.py` y las curvas k1. Candidatas: (a) corrección de sesgo por tipo de día × hora con ventana móvil, (b) mezcla ponderada
  base+análogos+referencia con pesos por segmento, (c) efecto festivos/puentes/clima, (d) intervalos empíricos (alimenta la banda), (e) selección automática
  por segmento. Regla: solo se activa por defecto lo que mejora WAPE en calibración (60 % antiguo) Y en prueba (40 % reciente) en AMBAS UCP; lo demás se reporta.
  Resultado a `noche2/REPORTE_B.md` con tabla antes/después; si algo se activa, parámetro nuevo con default compatible y tests.
- **Revisión**: ronda 1 con 8 ángulos + regresión (sonnet) contra `git diff main...HEAD`.

## Orden
1. CONTRATO_B (hecho) → commit. 2. B1 y B2 en paralelo. 3. Integrar B2→B1 (recomendaciones por sesgo usan la corrección medida). 4. Revisión. 5. REPORTE_B.
