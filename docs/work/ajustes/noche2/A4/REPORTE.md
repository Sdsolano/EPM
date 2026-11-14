# A4 - Reporte (graficas en curvas + lenguaje de negocio en vistas existentes)
Codigo: Sphere-wt-n2 (rama feat/noche2-front), SIN commit (lo hace el coordinador).

## Cambios por archivo (mesa_ajustes/)
- components/paso/GraficaPaso.tsx: reescrito. Semana = CurvaSemana168 (resultado vs paso anterior, real, calculada, rango esperado solo en el paso Control); dia = CurvaDia24 modo "paso". Mantiene tabs (teclado) y la tabla accesible de los charts. 0 Bar/BarChart (grep limpio).
- components/DetallePaso.tsx: usa useInsights solo en el paso control (banda), textos de estado (vista previa, version, aprobada), guia por estado. Props fijas intactas.
- components/paso/guiaPaso.ts (nuevo): "que hace / que debes hacer" por paso, nota por estado, nombres tecnicos de parametros (solo en detalle secundario).
- EncabezadoPaso, AlertasPaso (traducirAlerta: titulo + explicacion + que hacer; codigo crudo nunca; mensaje del sistema en <details>), Diagnosticos (etiquetas de negocio, AyudaTermino en intensidad/ratio/ancla/forma/margen/tendencia, "Segun el historial de esta UCP, este ajuste suele ayudar/no ayudar" en vez de R2), AccionesPaso (parametros con nombre de negocio, deslizadores en %, botones "Aplicar el ajuste", "Omitir este paso", "Volver a usar este paso", "Buscar dias parecidos"), EditorCurvas ("Guardar mis cambios", errores con siguiente paso), helpers.ts (formatoPuntos, comoPorcentaje, mensaje de conflicto), paso.module.css.
- components/resumen/EnergiaSemana.tsx: nuevo CurvaEnergiaSemana (CurvaEnergiaDiaria: base/ajustado/referencia/real) + lectura de brecha por dia ("Se quedo corto 1,2 %"); textos de las tablas. ResumenSemanal.tsx la integra (useInsights para la referencia). AlertasVigentes, Historial, Aprobacion: traducirAlerta en vez de codigos, bloque "Esto es lo que vas a aprobar" (energia, avisos y justificacion, que pasa despues), confirmacion "Aprobar la semana", role=status en aprobada, resumen.module.css.
- components/analogos/GraficaAnalogos.tsx: lineas monotone, grises finos + promedio verde grueso + promedio de todos punteado + dia senalado azul, role=img + aria-label, tabla hora por hora alternativa. PanelAnalogos, FiltrosAnalogos, ListaCandidatos: "analogos" -> "dias parecidos", mensajes con siguiente accion. Corregido bug previo: el bloque "Mas filtros" siempre visible (.rejilla[hidden]).
- Fuera de zona (cambios minimos): utils.ts (etiquetas "analogos/homologo" -> llano), lenguaje.ts (un titulo de alerta), charts/CurvaEnergiaDiaria.tsx (ancho del eje Y en movil: cortaba "35.000").

## Decisiones
- Banda solo en el paso Control (final_curves); pasos intermedios sin banda. DetallePaso llama useInsights(corrida) solo ahi (5 GET en control); si A3 quiere evitarlo puede pasar insights por prop.
- Botones: se conserva "Crear corrida" (consistente con PanelSeleccion) explicado en la frase.
- Textos del vacio dicen "Calcular y guardar" (nombre del boton de A3).

## Verificacion
- tsc: sin lineas nuevas vs tsc_base_A.txt. eslint: limpio en components/paso, resumen, analogos, DetallePaso, ResumenSemanal, charts.
- Navegador (mock, vista propia temporal ya borrada): pasos Base/Nivel/Forma/Especiales/Control, semana y dia, banda en Control, drawer de dias parecidos, resumen/aprobacion, semana con alertas (7 dic 2026). 0 errores de consola; sin scroll horizontal de pagina a 375/768/1280.
- Capturas: noche2/capturas_A/A4_paso_dia_1280, A4_paso_nivel_alertas_1280, A4_analogos_1280, A4_resumen_energia_1280/768/375, A4_aprobacion_1280, A4_paso_semana_375 (.png).
- vite build: OK (1m32s). build:demo: OK (1m10s).
- better-interface: revision propia ligera sobre lo tocado (sin hallazgos HIGH: foco visible en summary, objetivos >=40 px, aria-live en acciones, tablas con scroll interno, reduced-motion heredado de charts).

## Pendientes
- Capturas a 768 de pasos y del drawer (solo 768 del resumen).
- Etiquetas de selects de filtros pueden truncarse en drawer estrecho (no verificado en 320).
- Rail del stepper (A3) aun dice "advertencia/nota".
- Sin prueba de teclado exhaustiva del tooltip de las curvas (limitacion de A1).
