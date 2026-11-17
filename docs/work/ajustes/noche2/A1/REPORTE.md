# A1 - Reporte (graficas de curvas)
Carpeta: Sphere-wt-n2/src/views/pages/mesa_ajustes/charts/ (sin commitear; lo hace el coordinador). Importar solo desde `charts/index.ts`.

## Componentes y props principales
- CurvaSemana168: semanaInicio, series{base,ajustado,referencia,real,calculada,banda{inferior,superior}} (168 valores, null=sin dato), dias[{fecha,tipo,etiqueta,editado,alerta}], destacados, nombres, ocultasIniciales, titulo, resumen("visible"|"sr"|"ninguno"), rotuloCambio, alturaMax, cargando/error/onReintentar/mensajeVacio/accionVacio.
- CurvaDia24: fecha, series (24 valores), modo("normal"|"paso"), dia, + mismos props. Modo paso: base="Paso anterior", ajustado="Resultado del paso".
- CurvaEnergiaDiaria: fechas(7), series (7 valores MWh), unidad("MWh"|"GWh"), nombres, resumen, estados. Linea/area con puntos, sin barras.
- CurvaMini: series[{valores,rol}], ariaLabel (obligatorio), alto, area, lineaCero, destacarUltimo, dominio.
- EsqueletoGrafica, EstadoVacioGrafica, ErrorGrafica, LeyendaSeries, TooltipCurvas, useReducedMotion, useEntradaUnica.
- series.ts: aplanarCurvas(curvas, fechas), fechasDeSemana, etiquetaDia/DiaLargo/Hora/FechaHora, formatoMW/MWh/GWh/Energia/Pct/Cambio (es-CO), dominioY, estadisticas, frasesResumen.

## Decisiones
- Gris de "base" = #6F7C75 (token --mesa-serie-previa, AA grafico) en vez de #9AA5A0 (2,5:1).
- Sin accessibilityLayer de recharts (foco dentro de role=img). Accesibilidad por aria-label + tabla resumen (y detalle por dia) + leyenda con aria-pressed.
- Animacion 600 ms ease-out solo en el primer montaje; apagada con prefers-reduced-motion; cambios de datos sin animar.
- Dominio Y estable (usa todas las series; ocultar una no reescala).
- Nota: NO usar `max-width:100%` en .recharts-wrapper/svg (colapsaba el SVG a ancho 0).

## Verificacion
tsc: sin lineas nuevas vs tsc_base_A.txt. eslint charts: limpio. vite build: OK. Consola: 0 errores. 320/375 px sin scroll horizontal de pagina.
Capturas: docs/work/ajustes/noche2/capturas_A/A1_previa_1280.png, A1_previa_768.png, A1_previa_375.png (pagina completa de Previa.tsx).
Previa: charts/__previa__/Previa.tsx (montaje temporal con entry propio; ya borrado).

## Pendientes / notas
- Sin pruebas unitarias de series.ts (puro, facil de probar).
- Tooltip solo con raton/tactil; sin navegacion por teclado punto a punto (el resumen y detalle por dia cubren la lectura sin raton).
- No se tocaron archivos fuera de charts/.
