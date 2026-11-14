# R1 - Regresion (Sphere-wt-n2, feat/noche2-front, HEAD 0d6b132)

Resultado: SIN REGRESION. Cero hallazgos.

| Comprobacion | Resultado |
|---|---|
| tsc -b vs tsc_base_A.txt (sin numeros de linea) | 0 lineas nuevas (diff vacio, rc=0) |
| eslint mesa_ajustes, providers/ajustes, demo, types | 26 errores, todos `no-explicit-any` en 12 .d.ts preexistentes de src/types (carpetas, clima_mapa, config_ciudades_clima, convocatoria, cubrimiento, data.analize, factores, mpm, noticias, pronosticos, sesiones, valoracion), ninguno tocado por el diff. 0 errores en archivos del feature (ajustesInsights.d.ts limpio) |
| verificar.ts (A2) | 141 OK, 0 FALLA, rc=0 |
| grep `BarChart\|<Bar` en mesa_ajustes | 0 graficas de barras; unico match es `<BarraEtapas>` (stepper, falso positivo) |
| npm run build:demo | OK, rc=0 (1m29s; solo aviso de chunk >500 kB, preexistente) |
| npx vite build | OK, rc=0 (1m30s; mismo aviso) |
| git status tras builds | limpio |

## Happy path (razonado leyendo el diff)
- useMesaAjustes.ts: el diff es casi solo copy de mensajes + un efecto demo. Logica de calcular, crearCorrida, patchPaso, conflicto 409, versiones sin cambios.
- Efecto demo (semana por defecto): solo corre con PERFIL_DEMO, se detiene si hay corrida o si la persona eligio semana (semanaElegidaRef), y restaura la bandera tras fijar el valor por defecto. No afecta produccion.
- Aprobar (Aprobacion.tsx): bloqueado en vista previa y en version vieja, confirma con dialogo, manda motivo y excepciones_justificadas, maneja 409/422; el boton se deshabilita con bloqueos o carga.
- Historial: carga versiones al abrir, estado vacio/carga/error, "Ver" version via cargarVersion/verVersionActual.
- Crear corrida: todas las entradas (Bienvenida, Resumen, DetallePaso, PanelSeleccion, Analogos, Aprobacion) llaman mesa.crearCorrida(); el texto "Crear corrida" es coherente entre botones y mensajes.

Observacion menor (no regresion): "Crear corrida" sigue como etiqueta de boton mientras el resto del lenguaje habla de "semana guardada"; solo estilo.
