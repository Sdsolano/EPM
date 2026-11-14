# REPORTE_B — Pista B (backend/modelo), noche 2

Rama `feat/noche2-backend` (worktree `EPM-wt-n2`). Contrato para el front: `CONTRATO_B.md` (v1.5). Bitácora: `LOG_B.md`, sub-logs en `B1..B6/`.

## 1. Endpoints nuevos (solo lectura, prefijo `/adjustments/insights`)
`GET resumen?run_id=` · `GET precision?ucp=&semanas=&hasta=` · `GET recomendaciones?run_id=` · `GET banda?ucp=|run_id=` · `GET series?run_id=` (168 puntos base/ajustado/referencia/real/banda). Código: `src/adjustments/insights.py` (lógica pura), `router_insights.py`, datos `src/adjustments/precision_data/{Antioquia,Atlantico}_k1.csv` (foto del historial emitido por el modelo k1; refrescar con `scripts/generar_historial_pronosticos.py`). Registrado en `src/api/main.py`; no toca routers/pipeline existentes ni el modo demo. Tiempos con datos reales: 25-160 ms por endpoint (1a llamada hasta ~0.45 s por carga del historico).

## 2. Mejora de pronosticos (backtest sin fuga, modelo_k1, 52 sem/UCP, calibracion 31 / prueba 21)
Linea base (WAPE horario %): Antioquia prueba base 2.44 -> final del pipeline 2.33; Atlantico 5.86 -> 5.46 (coincide con `reports/ajustes/backtest_modelo_20261113.md`).
Regla: solo se activa lo que mejora en calibracion y prueba en ambas UCP (>=0.05 pp) sin empeorar MAPE semanal >0.05 pp. **Ninguna candidata cumplio; no se activo nada por defecto** (pipeline y defaults sin cambio; curvas identicas a main con base sintetica, hash igual).

| Candidata (elegida en calibracion) | dWAPE vs final (pp): Ant cal / Ant prueba / Atl cal / Atl prueba | Veredicto |
|---|---|---|
| Sesgo tipo de dia x hora | -0.12 / +0.01 / -0.05 / +0.18 | NO |
| Sesgo global x hora | -0.15 / 0.00 / +0.03 / +0.15 | NO |
| Sesgo dia de semana x hora | -0.05 / 0.00 / -0.07 / +0.20 | NO |
| Sesgo sobre la base (vs base) | -0.20 / 0.00 / -0.08 / +0.16 | NO |
| Mezcla ponderada base+final+analogos+referencia por tipo de dia | -0.17 / -0.07 / -0.09 / -0.03 (MAPE sem Ant prueba +0.07) | NO (energia semanal empeora) |
| Promedio 50/50 final y nivel por analogos | -0.16 / -0.06 / -0.03 / -0.09 (MAPE sem +0.05) | NO, por poco |
| Seleccion estatica / rodante por tipo de dia | MAPE sem empeora hasta +0.18 | NO |
| Festivo y puente | -0.12 / -0.01 / -0.06 / +0.14 | NO |
| Clima observado | Antioquia sin clima antes de 2025-12 (42 dias en calibracion) | NO CONCLUYENTE (correlacion residual-temperatura 0.35-0.45: mejor pista futura, con pronostico real de temperatura) |

Por que: el reentreno semanal y el paso nivel ya absorben el sesgo medio; la mejora en calibracion desaparece o se invierte en prueba (Atlantico cambio de regimen). La mezcla/analogos aporta casi todo por la energia diaria de analogos; es lo mas cercano a cumplir la regla (pendiente validar con ediciones reales del analista). Hallazgo metodologico: un primer cache tomaba el domingo de brecha de la semana siguiente (fuga); corregido y la linea base coincide con el reporte oficial.

**Entregado y medido (opt-in, no activado):** `src/adjustments/correccion.py` (`corregir_sesgo`, `corregir_clima`, bandas) con 22 tests.

## 3. Bandas de incertidumbre empiricas
Cuantiles p10-p90 del error relativo por tipo de dia x hora, escala 1.2 (elegida en calibracion). Cobertura medida sin fuga con la implementacion de la API (21 semanas de prueba, nominal 80 %): Antioquia 87.6 %, Atlantico 80.2 % (sin escala: 81.3 % / 73.2 %; ancho medio relativo 9.9 % / 20.1 %; minimos: festivo Antioquia 71 % con 7 dias). La API publica estas cifras en `cobertura_medida`. (El estudio B2 con log-error midio 91.3 % / 81.9 %; la API publica los numeros de SU implementacion.)

## 4. Recomendaciones: decision de honestidad
Las rondas 1-2 mostraron con backtest rolling que las recomendaciones de sesgo no mejoran fuera de muestra de forma consistente (filtro antiguo: Antioquia 3.08->2.70, Atlantico 5.55->7.53). Se entregan como **"tendencia reciente"**: `prioridad: baja`, `respaldo: "tendencia_reciente"`, `validado_fuera_de_muestra: false`, filtro estricto (>=8 dias, >=75 % de dias mejoran, >=0.3 pp de WAPE); con el filtro nuevo: Antioquia 14 de 364 dias (3.05->2.60), Atlantico 3 dias (4.57->6.55, sigue peor, de ahi la marca). Las recomendaciones accionables del sistema son las alertas reales del pipeline (festivos, eventos, clima, control). El `ajuste`/`patch` sigue disponible para aplicar a mano (probado: PATCH 200, `final_curves == curva_sugerida`). Con datos actuales casi nunca habra una recomendacion de tendencia emitida: el front no debe depender de que existan.

## 5. Revision
3 rondas multi-angulo + regresion (sonnet). R1: ALTO (honestidad de recomendaciones) + medios. R2: ALTO repetido en altitude (validacion en la misma ventana) + medios. R3: sin criticos/altos; medios/bajos corregidos en B6 y una verificacion estandar: suite 783 passed, 12 skipped, 1 failed (christmas, preexistente tambien en main); `import src.api.main` OK; pipeline con base sintetica identico a main. Aceptados sin cambio (ver `B6/LOG.md`): `list_runs` pesado con Postgres (no medido), sin cache de `validar_rolling` (~0.15 s), carga fria concurrente del historico, `precision()` con ucp rellenado por el router, `correccion.py` se queda en `src` (documentado opt-in).

## 6. Pendientes / riesgos
- Historial emitido estatico: Antioquia hasta 2026-06-28, Atlantico hasta 2026-05-03; precision/confianza dan `sin_datos` para semanas lejanas. Refrescar: regenerar curvas k1 y correr `scripts/generar_historial_pronosticos.py`, o persistir el pronostico base de cada corrida aprobada (pendiente).
- `lluvia_mm` en la alerta de control es suma de codigos de intensidad, no mm; la API no imprime esa cifra.
- Mezcla con analogos y clima con pronostico real: candidatas para la proxima sesion.
- La rama incluye commits de docs de la Pista A (solo docs) hechos en este worktree.
- Postgres `list_runs` no probado con DSN real.
