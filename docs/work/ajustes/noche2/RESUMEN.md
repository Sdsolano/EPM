# RESUMEN noche 2 (2026-11-14) — demo rediseñada y redesplegada

## Estado final
- **EPM `main`**: Pista B (backend de insights) + docs de ambas pistas + traspaso.
- **Sphere `test`**: `9ee2152` (Pista A: Resumen ejecutivo, flujo guiado 4 etapas, curvas, lenguaje de negocio, integración con el backend real).
- **Railway** (proyecto `epm-demo-ajustes`, servicio `app`): redesplegado con ambos. URL: https://app-production-d2ce6405.up.railway.app (acceso por HTTP Basic `demo:<DEMO_PASSWORD>`, variable del servicio).
- Verificado por HTTP en la nube: /health 200; front 200 con el bundle nuevo; /api/demo/ucps 401 sin clave y 200 con clave; `POST /adjustments/run` OK (Antioquia 2026-07-06 y Atlántico 2026-05-11, ~9 s) y los 5 endpoints `/adjustments/insights/*` 200 en 0.25–0.4 s.
- NO verificado visualmente en navegador contra Railway (la pantalla de acceso requiere la contraseña); la Pista A sí recorrió el flujo completo en navegador contra el backend real en local, con 77 capturas en `capturas_A/`.

## Qué cambió para el usuario/cliente
- Entrada = Resumen ejecutivo: semana en llano, pronóstico, confiabilidad ("se equivocaron en promedio 2.4 %"), qué revisar, recomendación.
- Todas las gráficas son curvas (24 h, 168 h semanales, energía diaria) con base / ajustado / referencia / real y banda de incertidumbre calibrada. Sin barras.
- Flujo guiado Ver → Revisar alertas → Ajustar → Aprobar, con lenguaje de negocio y glosario.

## Números (ver REPORTE_B.md)
- WAPE horario de prueba base→final: Antioquia 2.44→2.33, Atlántico 5.86→5.46; energía semanal MAPE 1.52→1.37 / 4.57→4.41.
- Ninguna candidata de mejora de pronóstico superó la regla (calibración Y holdout, ambas UCP): nada activado por defecto. `correccion.py` queda opt-in.
- Banda p10–p90: cobertura medida 87.6 % (Antioquia) y 80.2 % (Atlántico) para 80 % nominal.

## Qué decir al cliente
Mejoras de experiencia y de transparencia (confianza medida, bandas, recomendaciones explicadas), no un modelo "más preciso" esta noche: se probó y se descartó lo que no mejoraba. Pista a futuro: pronóstico de temperatura (señal 0.35–0.45) y validar mezcla con análogos en nivel (+0.05 pp, casi cumple).

## Limitaciones / pendientes
- Datos de la demo: la última demanda real es 2026-06-28 (Antioquia) y 2026-05-03 (Atlántico) y el backend limita la brecha a 90 días: solo se pueden calcular semanas hasta ~sep-2026 (Antioquia) y ~ago-2026 (Atlántico). Para fechas actuales hay que actualizar `demo_snapshot` con datos recientes (`scripts/demo_snapshot.py`) y regenerar `precision_data/*_k1.csv` (`scripts/generar_historial_pronosticos.py`).
- El botón "Aplicar el ajuste sugerido" casi no aparece con datos reales: ninguna recomendación de sesgo pasó el filtro estricto del backend.
- Las 2 corridas de prueba (borradores) quedaron en el Postgres de la demo.
- Alerta de clima: B corrigió la cifra de lluvia (ya no imprime mm falsos); la Pista A no la reverificó en UI.
- Aceptados sin corregir: ver REPORTE_A.md y B6/LOG.md. Sin push de Sphere `feat/noche2-front` a otras ramas salvo `test`.
- `railway up` exige el contexto mínimo (ver demo/ESTADO_DESPLIEGUE.md); la contraseña se lee con `railway variables --service app --kv`.
