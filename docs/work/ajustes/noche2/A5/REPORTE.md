# REPORTE A5 - integracion con backend real, pulido y capturas

Entorno: copia de trabajo en %TEMP%\epm_e2e (src/config de EPM-wt-n2 + data/models del checkout), uvicorn en 127.0.0.1:8765 con DEMO_MODE=1, DEMO_PASSWORD de prueba, DEMO_STATIC_DIR = dist de `npm run build:demo`, store en memoria (InMemoryStore via dependency_overrides; sin Postgres). Navegador: Playwright (Chromium). Copia, servidor y procesos eliminados al terminar.

## Integracion (flujo recorrido de punta a punta, con datos reales)
Acceso -> UCP y semana -> calculo real (11-15 s Antioquia, ~12 s Atlantico) -> Resumen ejecutivo con los 5 endpoints insights (200 en 20-140 ms) -> aplicar recomendacion (PATCH real) -> etapa Ajustar (pasos, curvas 168 h y 24 h con banda en Control) -> drawer de dias parecidos -> aprobar (POST approve real, v3). Los tipos de `ajustesInsights.d.ts` coinciden con las respuestas reales (resumen, precision, recomendaciones, banda v1.3, series): sin discrepancias de forma.

## Discrepancias halladas y corregidas en el FRONT
1. Semana por defecto en demo (lunes siguiente a hoy = 16 nov) cae a >90 dias del ultimo dato real: 422 en el primer clic. Ahora en demo, mientras la persona no elija semana, parte del lunes siguiente al ultimo dato real de la UCP (via /api/demo/reales). `useMesaAjustes.ts`.
2. Mensajes 422 crudos del backend ("Brecha demasiado grande ... fecha_corte") y "La semana ya tiene datos reales" se traducen a lenguaje de negocio (`humanizarDetalle` en providers/ajustes/index.ts).
3. Textos del backend con punto decimal ("230.5 GWh", "2.4 %", "26.3 °C") se localizan a coma es-CO y espacio sin salto (`localizarTexto` en insights.ts); el titulo ya no parte "230,5 / GWh".
4. Alertas del pipeline llegan tambien como recomendacion sin ajuste (id igual al aviso): salian dos veces y contaban doble ("Por revisar 2"). `sinDuplicadosDeAviso` (lenguaje.ts) en QueRevisar, Indicadores, insignia de etapa 2. "1 aviso y 0 ajustes" -> "1 aviso".
5. Resumen textual de las curvas y tablas: la diferencia de ENERGIA se mostraba en MW ("+3.358,7 MW"); ahora MWh/GWh (`formatoCambioEnergia`).
6. "-0,00 puntos / -0,0 %": cero negativo tras redondear; corregido en formatoPuntos, formatoPct, conSigno.
7. Riel del stepper decia "advertencia/nota": ahora "aviso/dato extra".
8. Drawer de dias parecidos: aviso de consola de recharts (width -1, graficas en drawer animando) y selects truncados a 768: initialDimension/minWidth en GraficaAnalogos y rejilla de filtros minmax(230px).

## Hallazgos del BACKEND (no tocados)
- B-A5-1 (medio) Alerta de clima: texto "lluvia de 126735 mm" (insights.py `CLIMA_OBSERVADO`: `lluvia_mm` es la suma de `p_i` de todas las filas del clima de la semana, no mm reales). Ejemplo: Antioquia 2026-07-06, id `clima-semana`. Tambien marca los 7 dias como "por revisar".
- B-A5-2 (bajo) `GET /adjustments/aprobada` devuelve 404 cuando no hay aprobada: cada seleccion de semana deja un "Failed to load resource 404" en consola del navegador (inevitable desde el front). Sugerido: 200 con null o HEAD.
- B-A5-3 (producto) En demo `POST /adjustments/run` rechaza semanas con datos reales (422 "ya tiene datos reales ... envie base_curvas") y semanas a >90 dias: no se puede mostrar "semana pasada con real completo" ni la serie `real` (siempre null). Solo alcanzable via /precision. Antioquia ultimo dato 2026-07-04; Atlantico 2026-05-07.
- B-A5-4 (producto) En 9 semanas probadas (Antioquia y Atlantico, mayo-sep 2026) NINGUNA recomendacion de sesgo paso el gating: solo salen alertas (`aplicable:false`). El "Aplicar el ajuste sugerido" del front se probo inyectando una recomendacion sintetica (jueves +2 %) contra el PATCH real: funciona (version 2, final_curves actualizado, energia +686 MWh).
- `kpis.recomendaciones.total` cuenta las alertas-recomendacion (ver 4).

## Estados y responsive
Carga (skeleton + "Calculando"), error (422 humanizado), vacio/degradacion por bloque (series/banda 503, precision 404, recomendaciones 422 con Reintentar; el resto de la pantalla sigue). Sin scroll horizontal de pagina a 320/375/768/1280 en acceso, bienvenida, error, resumen, alertas, ajustar, aprobar, aprobada. Sin barras en series temporales. Sin textos de desarrollo. Consola: 0 errores propios; solo los 404 de /aprobada (B-A5-2) y el 422 provocado a proposito.
Limitacion: pronostico "futuro" solamente (B-A5-3); la banda aparece en el paso Control y el resumen.

## Archivos tocados (Sphere-wt-n2, sin commit)
providers/ajustes/index.ts, providers/ajustes/insights.ts, views/pages/mesa_ajustes/{MesaAjustes.tsx, useMesaAjustes.ts, lenguaje.ts, utils.ts}, charts/{CurvaHoraria.tsx, series.ts}, components/ejecutivo/{Indicadores,QueRevisar,ResumenEjecutivo}.tsx, components/paso/helpers.ts, components/stepper/NodoPaso.tsx, components/analogos/{GraficaAnalogos.tsx, analogos.module.css}.

## Verificacion
tsc: 178 lineas = base (sin nuevas; compara ignorando CR del archivo base). eslint limpio en lo tocado. `npx vite build` y `npm run build:demo` OK.
Capturas: noche2/capturas_A/A5_*.png (acceso, bienvenida, error, carga, resumen 1280/768/375, aplicado, alertas 1280/768/375, ajustar forma semana 1280/768/375, dia 24 h 1280/768, control 1280/768, drawer 768/1280, aprobar 1280/768/375, dialogo, aprobada 1280/375, degradacion 1280/375).

## Pendientes
Revision visual formal (better-interface) la hace la Fase 4; no se probo teclado exhaustivo del tooltip; Indicadores usa el total del resumen si falla el bloque de recomendaciones (cuenta alerta doble en ese caso).
