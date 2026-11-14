# R1 - angulo simplification (solo lectura, Sphere-wt-n2 8a6302c..HEAD)

Sin hallazgos critica/alta. Todo lo de abajo es media/baja (limpieza).

## Media
1. [CONFIRMED] src/views/pages/mesa_ajustes/charts/__previa__/Previa.tsx (147 lineas) y src/providers/ajustes/__verif__/verificar.ts (137): ninguno se importa desde produccion, ni esta referenciado en package.json/scripts; el propio comentario de Previa dice "montala temporalmente y borrala". Son artefactos de desarrollo commiteados en la rama. Fix: borrar ambos antes de mergear (o mover verificar.ts a un script fuera de src/ si se quiere conservar). Ojo: verificar.ts se compila con tsc si tsconfig incluye src (riesgo de romper build si cambia una firma).
2. [CONFIRMED] Props/ramas nunca usadas por ningun call site:
   - `ocultasIniciales` en CurvaSemana168, CurvaDia24, CurvaEnergiaDiaria, CurvaHoraria (CurvaHoraria.tsx:64,91,103): nadie la pasa.
   - `className` en CurvaHoraria/CurvaEnergiaDiaria/CurvaMini/CurvaSemana168/CurvaDia24: nadie la pasa.
   - CurvaMini: `lineaCero` (+ ReferenceLine, CurvaMini.tsx:33,60,87) y roles `secundaria`/`real` + `ESTILO_ROL` (solo se usa rol "principal" en Confiabilidad.tsx:32). `series` es array pero siempre hay una sola.
   - `resumen="visible"` y la tabla visible de ModoResumen: todos los usos pasan "sr" o el default; revisar si el modo "visible" se usa en algun lugar (GraficaPaso no lo pasa).
   Fix: eliminar las props/ramas o dejarlas solo si hay uso inmediato.

## Baja
3. [CONFIRMED] Exports/funciones sin ningun uso en el repo (solo se definen): `terminoGlosario` (lenguaje.ts:99), `formatoHoraInsight` (lenguaje.ts:349). Pre-existentes en utils.ts y ya sin uso: DESCRIPCION_PASO, ETIQUETA_NIVEL_ALERTA, ETIQUETA_RECOMENDACION, etiquetaCodigo, formatoPp (1 sola referencia = su definicion). Fix: borrar.
4. [CONFIRMED] `export` innecesario (solo uso interno del archivo): mensajeDeDetalle, humanizarDetalle (providers/ajustes/index.ts), motivoDeError, localizarTexto (insights.ts), cuerpoDePatch, OpcionesAplicar, ResultadoAplicar, ResultadoAplicarTodas (recomendaciones.ts), tipoDiaDe (mock/insights.ts), formatoGWh, OpcionesDominio, Estadisticas (series.ts), PropsNucleo (CurvaHoraria.tsx; ademas el comentario dice que no se exporta), TEXTO_CONFLICTO, GuiaPaso, EstadoAplicar, EstadoBloque, OpcionesInsights, y los *Props de DetallePaso/PanelSeleccion/ResumenSemanal/StepperPasos/AyudaTermino, MesaAjustesVista. Ensucia la API; quitar `export`.
5. [CONFIRMED] charts/index.ts hace `export * from "./series"` (barril completo) ademas de reexportar piezas de bajo uso (ESTILO_SERIE, NOMBRE_SERIE, useReducedMotion, useEntradaUnica, MuestraSerie, TooltipCurvas, LeyendaSeries solo los usan archivos hermanos que importan directo). Fix: exportar solo lo que consume el exterior (CurvaSemana168, CurvaDia24, CurvaEnergiaDiaria, CurvaMini, estados, fechasDeSemana y formatos usados).
6. [PLAUSIBLE] CurvaSemana168/CurvaDia24 son wrappers finos sobre CurvaHoraria con estadoAlterno duplicado; abstraccion aceptable, pero PropsCurvaSemana168 y PropsCurvaDia24 repiten 8 props identicas (extraer un tipo base).
7. Higiene: `AJUSTES_MOCK` y los `__mock*` solo los usa el propio mock (window hooks) - ok, no accionable. En el repo EPM hay `fastapi_factores/nul` sin trackear (artefacto de redireccion Windows `> nul`): borrar, no commitear.
