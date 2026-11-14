# A3 - Resumen ejecutivo + flujo guiado (sub-plan)

Codigo: Sphere-wt-n2, `src/views/pages/mesa_ajustes/`. Lee PLAN_A.md, A1/A2 REPORTE, CONTRATO_B.md.

## Decisiones
- Etapas: estado local `vista` en MesaAjustes ("semana" | "alertas" | "ajustar"); etapa 4 (Aprobar) = vista "ajustar" con `pasoSeleccionado === "resumen"`. Asi se conserva intacto riel + DetallePaso + ResumenSemanal y `?corrida=`.
- Corrida para el resumen: los endpoints de insights exigen `run_id` guardado. El boton unico de bienvenida "Ver el pronostico de la semana" llama `crearCorrida()` (borrador guardado v1, sin vista previa). Si ya hay un borrador (localStorage/URL) se retoma solo; si hay aprobada se ofrece "Ver la semana aprobada".
- Aplicar recomendacion: `aplicarRecomendacion` / `aplicarTodasLasRecomendaciones` (providers) y luego `mesa.recargar()` para sincronizar el estado de la mesa con la version nueva.
- useInsights solo se activa en vistas "semana" y "alertas" (en "ajustar" se pasa null).
- Degradacion por bloque: cargando=esqueleto, error=mensaje+reintentar, no_disponible=una linea honesta.

## Archivos
components/ejecutivo/ (ResumenEjecutivo, EncabezadoSemana, Indicadores, TiraDias, QueRevisar, Confiabilidad, SiguientePaso, useAplicar, ejecutivo.module.css), components/flujo/ (BarraEtapas, Bienvenida, VistaAlertas, flujo.module.css), MesaAjustes.tsx, mesa.module.css, PanelSeleccion/seleccion/*, src/demo/*, lenguaje.ts (solo entradas).

## Estado
Ver REPORTE.md al terminar.
