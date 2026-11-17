# PLAN A (front) - Mesa de ajustes "wow" - sesion nocturna 2

Fuente de verdad de la Pista A. Plan general: `PLAN.md` (mismo directorio). Estado vivo: `LOG_A.md`. Cierre: `REPORTE_A.md`.
Codigo: `C:\Users\samue\OneDrive\Documentos\GitHub\Sphere-wt-n2` (rama `feat/noche2-front`), carpeta `src/views/pages/mesa_ajustes/`, providers `src/providers/ajustes/`, tipos `src/types/ajustes.d.ts`, demo `src/demo/`.
Backend EPM: `C:\Users\samue\OneDrive\Documentos\GitHub\EPM` (checkout principal: NO correr E2E que escriba en data/features) y worktree B `EPM-wt-n2`.

## Problema (feedback del usuario sobre la demo)
Nada intuitivo; graficas semanales en barras (deben ser curvas); no se entiende que hacer.

## Decisiones de diseno
- Libreria de graficos: `recharts` 3.x (ya en Sphere, la usa GraficaPaso). Sin dependencias nuevas.
- Todas las series como curvas suaves (`type="monotone"`), nunca barras: 24 h, 7x24=168 puntos, energia diaria (linea/area con puntos).
- Series: base (gris discontinua), ajustado (verde solido, protagonista), referencia (azul/teal fino discontinuo), real (azul oscuro punto+linea, si existe), banda de incertidumbre (area translucida si B la entrega; si no, no se dibuja y no se inventa).
- Flujo guiado de 4 etapas visibles en una barra superior: 1 Ver la semana (Resumen ejecutivo) -> 2 Revisar alertas -> 3 Ajustar (riel de pasos, con recomendacion de un clic) -> 4 Aprobar. El riel de 6 pasos existente se conserva como detalle de la etapa 3 (no se pierde funcionalidad).
- Pantalla de entrada "Resumen ejecutivo": semana y UCP, pronostico (energia semanal + curva 168 h), confiabilidad (error reciente por semana, del endpoint de precision de B), alertas (que revisar, ordenadas), recomendacion de un clic ("Aplicar ajuste sugerido" -> crea corrida con los pasos recomendados), siguiente accion clara.
- Lenguaje de negocio: glosario en un unico modulo (`lenguaje.ts`): ratio, alpha, ancla, analogos, banda, forma, nivel, referencia, tendencia explicados en llano, con ayuda contextual (tooltip/popover accesible). Cero textos de desarrollo/mock/"Fase"/"R2"/codigos crudos en pantalla (los codigos se traducen).
- Estados vacio/carga/error cuidados (skeleton con forma de la grafica, mensajes con siguiente accion, reintentar). Responsive (>=320 px sin scroll horizontal de pagina), accesible (foco, aria, resumen textual de cada grafica, contraste AA), movimiento sobrio (<=200 ms, solo respuesta; `prefers-reduced-motion` lo anula).
- Contrato de endpoints nuevos: lo publica la Pista B en `CONTRATO_B.md`. Hasta entonces el front define `src/types/ajustesInsights.d.ts` y un adaptador unico (`src/providers/ajustes/insights.ts`) con mock realista; cuando salga el contrato solo cambia el adaptador. Si un endpoint no existe/falla, el resumen degrada (oculta esa tarjeta con un mensaje honesto), nunca inventa numeros.
- Skills (los subagentes las invocan con la herramienta Skill): frontend-design, impeccable, emil-design-eng; revision: better-interface; movimiento: animate/review-animations. NO unslop.

## Partes (dueno = subagente) y criterio de terminado
Fase 1 (paralelo):
- A1 Graficas: `mesa_ajustes/charts/` (CurvaSemana168, CurvaDia24, CurvaEnergiaDiaria, leyenda, tooltip, resumen textual, esqueleto, `useReducedMotion`), utilidades de series. Sub-plan `A1/PLAN.md`. Listo: componentes con props tipadas, pagina de prueba sin datos reales, sin barras, tsc sin sumar errores.
- A2 Datos: tipos de insights, adaptador `insights.ts` con mock realista (semana, pronostico 168 h, banda, precision por semana, alertas priorizadas, recomendaciones por dia con explicacion), integracion en `providers/ajustes/index.ts` (mock/demo/real), hook `useResumenEjecutivo`, `lenguaje.ts` (glosario + traduccion de codigos de alerta). Sub-plan `A2/PLAN.md`.
Fase 2 (paralelo, tras A1+A2):
- A3 Resumen ejecutivo + flujo guiado: nueva pantalla de entrada, barra de etapas, `MesaAjustes.tsx` reordenado, recomendacion de un clic, aprobar. Sub-plan `A3/PLAN.md`.
- A4 Reemplazo de graficas y lenguaje en las vistas existentes: GraficaPaso (semana en barras -> curva 168 h; dia 24 h), EnergiaSemana, GraficaAnalogos, Diagnosticos/AccionesPaso/OpcionesCalculo con textos claros y ayuda. Sub-plan `A4/PLAN.md`.
Fase 3: A5 pulido (estados vacio/carga/error, responsive, movimiento, copy) + integracion con backend real de B; capturas.
Fase 4: revision (ronda 1 completa, 9 angulos + regresion, model sonnet; repetir solo si hay critico/alto), correccion, verificacion estandar, commit.

## Criterios globales
`npx tsc -b` no suma errores sobre la base (~178 preexistentes; se mide en la Fase 0 y se anota en LOG_A.md), `vite build` y `npm run build:demo` OK, 0 errores de consola en navegador, capturas de pantallas clave en `docs/work/ajustes/noche2/capturas_A/`, sin barras en ninguna serie temporal.

## Reglas
Commits locales en Sphere-wt-n2, espanol, `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`, sin push/amend/force. Docs de esta pista en el worktree EPM-wt-n2 (`git add -f`). Estado documentado antes de cada fase y tras cada paso en `LOG_A.md`.
