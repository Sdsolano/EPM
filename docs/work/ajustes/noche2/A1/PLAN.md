# A1 - Biblioteca de graficas de curvas (sub-plan)
Codigo: Sphere-wt-n2/src/views/pages/mesa_ajustes/charts/ . Lee PLAN_A.md, CONTRATO_B.md, DESIGN.md.

## Decisiones
- recharts 3.x, sin dependencias nuevas. Todo es linea/area (type monotone); nada de barras.
- Un nucleo interno `CurvaHoraria` (X numerico 0..N-1) sirve a CurvaSemana168 (168 pts) y CurvaDia24 (24 pts).
- Colores literales en `colores.ts` (SVG no resuelve var() en todos los atributos de recharts); espejo de DESIGN.md.
- Accesibilidad: wrapper role="img" + aria-label generado; resumen textual (tabla por serie / por dia) visible o sr-only;
  leyenda = botones con aria-pressed (teclado). Sin accessibilityLayer de recharts (evita foco dentro de aria-hidden).
- Animacion: trazado izquierda-derecha 600 ms ease-out solo en la primera carga (useEntradaUnica), apagado con
  prefers-reduced-motion (useReducedMotion); cambios de datos sin animar.
- Responsive: altura proporcional al ancho (clamp), menos ticks en <480 px, Y sin etiquetas gigantes.
- Estados: EsqueletoGrafica, EstadoVacioGrafica, ErrorGrafica; las graficas aceptan cargando/error/onReintentar.

## Archivos
colores.ts, series.ts (utilidades puras), useReducedMotion.ts, useEntradaUnica.ts, charts.module.css,
LeyendaSeries.tsx, TooltipCurvas.tsx, EstadosGrafica.tsx (Esqueleto/Vacio/Error), CurvaHoraria.tsx (interno),
CurvaSemana168.tsx, CurvaDia24.tsx, CurvaEnergiaDiaria.tsx, CurvaMini.tsx, index.ts, __previa__/Previa.tsx.

## Terminado cuando
tsc sin lineas nuevas vs tsc_base_A.txt, eslint charts limpio, vite build OK, capturas 1280/768/375 sin errores de consola.
