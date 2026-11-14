# R1 - angulo removed-behavior (comportamiento eliminado o cambiado)

Diff revisado: `git diff 8a6302c..HEAD` (Sphere-wt-n2, feat/noche2-front). Solo lectura.

Resultado: sin hallazgos criticos ni altos. 3 medios, 3 bajos.

Verificado y SIN perdida: crear/ver corrida, historial/versiones (Historial + versiones siguen en la etapa Aprobar), `?corrida=` y localStorage (useMesaAjustes intacto salvo textos), analogos (buscar/elegir/aplicar a forma y nivel; grafica pasa a curvas + tabla alternativa), editor de curvas, omitir paso, aprobar con excepciones justificadas (fieldset + motivo obligatorio + dialogo), mock (se extiende con insights), perfil demo / Acceso / sesion (solo textos y document.title), R2 (SegmentosR2 conserva las 3 clasificaciones, solo cambia el texto), alertas IA (etiqueta "Sugerido por IA, no verificado" y eventos conservados), accesibilidad previa (tablist con flechas/Home/End, role=img en grafica de analogos, aria-live).

## Hallazgos

### 1. MEDIA - PLAUSIBLE - el enlace `?corrida=` puede perderse en demo (carrera con el salto a "semana siguiente al ultimo dato real")
`useMesaAjustes.ts:420-433` (efecto nuevo). La guardia solo mira `corridaRef.current`, que es null mientras `cargarCorrida` (URL) sigue en vuelo. Si `api.reales` responde antes que `api.corrida`, llama `setSemanaRef.current(destino)` -> `limpiarSeleccion()` (`:377-389`) que pone `idCargadoRef=null` y `fijarIdUrl(null)`: se borra el `?corrida=` y la corrida cargada puede descartarse (`:313`) o quedar con UCP/semana inconsistentes. Antes no existia ese efecto.
Arreglo: en la guardia agregar `|| idCargadoRef.current` (o `idUrl`), y no ejecutar el efecto mientras haya `idUrl` pendiente.

### 2. MEDIA - CONFIRMED - la validacion de "Opciones del calculo" ya no bloquea las acciones principales
`flujo/Bienvenida.tsx` (boton "Ver el pronostico de la semana", `disabled={mesa.loading || !mesa.ucp || mesa.cargandoUcps}`) y `StepperPasos.tsx:58` (`RielVacio`, ahora `crearCorrida()`). Antes "Calcular"/"Crear corrida" estaban en `PanelSeleccion` con `bloqueado = ... || !paramsValidos`. Ahora el boton primario ignora `paramsValidos` (estado local de PanelSeleccion) y el mensaje de error queda dentro de "Mas opciones" (`hidden` por defecto). `OpcionesCalculo.construir` descarta el valor invalido (y el alpha valido previo), asi que se calcula en silencio con los defaults.
Arreglo: subir `paramsValidos` a `mesa` (o al shell) y deshabilitar/avisar en Bienvenida y RielVacio; o mostrar el error fuera del bloque colapsado.

### 3. MEDIA - PLAUSIBLE - Resumen ejecutivo mezcla version vista y datos de la version actual
`MesaAjustes.tsx:271` `useInsights(vista === "ajustar" ? null : mesa.corrida)` usa la corrida actual, mientras `ResumenEjecutivo.tsx` pinta cabecera/estado con `mesa.corridaVista`. Si se abrio una version anterior desde Historial (etapa Aprobar) y luego se navega a "Ver la semana" o "Revisar", la cabecera dice "v1" pero KPIs, curva, dias y recomendaciones son de la version actual (`ResumenSemanal.tsx:34` y `DetallePaso.tsx:62` si lo evitan con `!mesa.versionVista`).
Arreglo: pasar `mesa.versionVista ? null : mesa.corrida` y mostrar un aviso "estas mirando la version N, resumen solo de la actual" con boton Volver a la actual.

### 4. BAJA - CONFIRMED - la fecha elegida en "Dia por dia" se ignora
`MesaAjustes.tsx:363-366` `onAbrirDia={(fecha) => { void fecha; verPaso("nivel"); }}`; TiraDias dice "Toca un dia para ajustarlo" pero abre el paso Nivel sin seleccionar el dia (DetallePaso tiene su propio estado `dia`) y aunque Nivel este omitido. Funcionalidad nueva, no regresion, pero promete algo que no hace.
Arreglo: pasar la fecha a DetallePaso (estado inicial) o ajustar el texto.

### 5. BAJA - CONFIRMED - se perdio la linea de estado con fecha de corte y fuente de la base
`EstadoCorrida.tsx` (y `LineaEstado` borrada de MesaAjustes.tsx) mostraba "corte <fecha>" y "base: pronostico emitido / modelo vX". Ahora solo se ve dentro del paso Base (`Diagnosticos`). Para el analista, saber si la base es emitida o del modelo era visible siempre.
Arreglo: reponerlo como dato secundario en el encabezado del resumen ejecutivo o en Confiabilidad.

### 6. BAJA - CONFIRMED - cambio de comportamiento: el boton del riel vacio ahora guarda una corrida
`StepperPasos.tsx:58` pasa de `mesa.calcular()` (vista previa sin guardar) a `mesa.crearCorrida()` (persiste un borrador y escribe localStorage). Coherente con la nueva Bienvenida, pero el texto "Ver el pronostico de la semana" no avisa que se guarda. Aceptable si es decision de producto; anotar.
