# R1 cross-file tracer (Pista A, noche 2)

Sin hallazgos criticos ni altos. Hay 5 medios y varios bajos. Los contratos de campos, unidades (MWh/MW, pp), nulls, rutas y cabeceras
coinciden con CONTRATO_B v1.5 y con `router_insights.py` / `insights.py`: ver "Verificado sin problemas" al final.

## Medios

1. **MEDIA / CONFIRMED. Se puede aplicar una recomendacion vieja y pisar una edicion manual.**
   `components/ejecutivo/useAplicar.ts:44` (`puedeAplicar`), `hooks/useInsights.ts` (`derivar`, rama `actualizando`).
   Al cambiar la version de la corrida, useInsights deja el dato anterior en pantalla (`actualizando: true`) hasta que llega el nuevo.
   `puedeAplicar` solo mira `esAplicable(corrida, rec)` y `ya_resueltas`, no la frescura. `curva_sugerida` es absoluta (dia completo).
   Flujo: Resumen -> Ajustar -> editar el sabado (PATCH forma, v N+1) -> volver al Resumen. Durante el refetch (la ruta de
   recomendaciones hace validacion rolling y puede tardar segundos) se muestra la recomendacion de la v N con "Aplicar" activo. Un clic
   manda `version_esperada = N+1` con la curva vieja: EPM lo acepta y sobrescribe la edicion manual en silencio.
   Arreglo: deshabilitar Aplicar / "Aplicar el ajuste sugerido" si `recomendaciones.actualizando` o
   `recomendaciones.data.version !== corrida.version`.

2. **MEDIA / CONFIRMED. "Por revisar" cuenta las tendencias recientes, que el backend v1.4 declara no pendientes.**
   `MesaAjustes.tsx:58-59` (`ajustesPropios`, `pendientes`), `ResumenEjecutivo.tsx` (prop a `Indicadores`), `Indicadores.tsx:27`.
   Cuentan todas las recomendaciones menos las duplicadas de aviso. Con `respaldo: "tendencia_reciente"` (baja, no marcan el dia como
   `revisar`, no empujan `siguiente_paso`) el front muestra "Por revisar 1: 1 ajuste sugerido" y el badge del stepper, mientras
   `siguiente_paso` dice "No hay nada pendiente de revisar: puedes aprobar la semana" y el dia sale "Sin novedad".
   Causa: `types/ajustesInsights.d.ts` no tiene `respaldo`, `validado_fuera_de_muestra`, `evidencia.ventana_seleccion`,
   `kpis.recomendaciones.del_sistema/tendencia_reciente` (v1.4); el front nunca distingue las dos clases.
   Arreglo: agregar los campos al tipo y contar como pendientes solo `respaldo === "alerta_del_sistema"` (o `kpis.recomendaciones.del_sistema`),
   mostrando las tendencias como "referencias" aparte. Tambien tipar `evidencia.n/sesgo_pct` como `number | null` (alertas envian null).

3. **MEDIA / CONFIRMED. Contradiccion entre la confianza del encabezado y el panel "Que tan confiable es".**
   `components/ejecutivo/Confiabilidad.tsx:23,30`. El historial de pronosticos es una foto estatica (Antioquia termina 2026-06-28,
   Atlantico 2026-05-03; CONTRATO_B v1.3/v1.5). `resumen.confianza` ya lo maneja (`sin_datos` o "de las N semanas hasta el <fecha>"),
   pero `precision` (sin `hasta`) devuelve las ultimas 12 semanas con datos y el front escribe fijo "En las ultimas {r.semanas}
   semanas el pronostico se equivoco ... X %". Con una semana objetivo de noviembre: encabezado "Sin historial" y panel "ultimas 12
   semanas, 2,4 %". Arreglo: usar `precision.hasta` para decir "hasta el <fecha>" cuando este lejos de la semana objetivo, o pedir
   `hasta` coherente con `resumen`.

4. **MEDIA / CONFIRMED. El mock diverge del contrato v1.4 y esconde el comportamiento real.**
   `providers/ajustes/mock/insights.ts:325-440, 512`.
   - Las recomendaciones de sesgo salen con prioridad alta/media, `impacto_esperado` "Reduce el error esperado de ~2,6 % a ~2,1 %" y
     explicacion "Sube el sabado en esa proporcion" (el backend real: baja, "no garantiza mejora", disclaimer). Sin `respaldo`.
   - Id de la alerta-recomendacion `alerta-${codigo}` vs id del aviso `alerta-${paso}-${codigo}`: `sinDuplicadosDeAviso` no las empareja y
     en el mock se cuentan y muestran dos veces (el backend real usa el mismo id: no pasa).
   - Consecuencia: el boton primario "Aplicar el ajuste sugerido" y su nota ("prioridad alta y media") solo aparecen con el mock; con
     EPM real nunca hay sugeridas (sesgo = baja, alertas con `ajuste: null`). Capturas y verificacion de A se hicieron contra un
     comportamiento que produccion no emite. Arreglo: alinear el mock a v1.4 (baja, `respaldo`, `impacto_esperado` honesto, mismo id de aviso)
     y decidir si el boton masivo se queda o se elimina.

5. **MEDIA / CONFIRMED. `onAbrirDia` ignora el dia.**
   `MesaAjustes.tsx` (`onAbrirDia={(fecha) => { void fecha; verPaso("nivel"); }}`). `TiraDias` anuncia "Ajustar este dia" (aria-label y nota "Toca un dia
   para ajustarlo"), pero siempre abre el paso Nivel sin dia seleccionado (DetallePaso arranca en la vista de semana/primer dia). Si Nivel
   esta omitido, el paso queda sin editor. Arreglo: llevar `fecha` a DetallePaso (estado en useMesaAjustes) o cambiar el texto a "Ir a ajustar".

## Bajos

6. **BAJA. Fallos transitorios se tratan como "no disponible" y sin Reintentar.** `providers/ajustes/insights.ts` (`motivoDeError`:
   `status === undefined`, 500/502/503/504 -> `no_disponible`) + `ejecutivo/Bloque.tsx:20`. Un timeout de 30 s, corte de red o un 500 real
   de EPM deja una nota "no esta disponible" sin boton; solo recargar la pagina lo repara. Arreglo: Reintentar tambien en `no_disponible`
   cuando `status` es `undefined` o 5xx distinto de 404/501.

7. **BAJA. Sobre-consulta.** `DetallePaso.tsx:97` (solo usa `banda`) y `ResumenSemanal.tsx:34` (solo usa `series`) llaman a `useInsights`,
   que dispara los 5 endpoints (incluye recomendaciones con validacion rolling y precision). En Ajustar -> Control/Resumen son 5 llamadas
   para usar 1. Arreglo: opcion `bloques` en `useInsights`.

8. **BAJA. Enums y campos sin guarda rompen toda la pantalla.** `Resumen`: `ETIQUETA_CONFIANZA[confianza.nivel].etiqueta` y
   `ETIQUETA_PRIORIDAD[rec.prioridad].etiqueta` sin `??`; `k.alertas.warning` con `kpis.alertas` no validado en `VALIDAR.resumen`.
   Un valor nuevo del backend produce TypeError (no hay error boundary). Arreglo: `?? generico` y validar `confianza`/`kpis.alertas`.

9. **BAJA. Textos que prometen mejora o contradicen el contrato honesto.** `lenguaje.ts` `ETIQUETA_PRIORIDAD` (alta: "ajuste con mas efecto
   esperado en el error"; baja: "Mejora pequena"; hoy sin uso, pero es copy muerto que contradice v1.4); `QueRevisar.tsx` rotula
   `impacto_esperado` como "Efecto esperado" aunque el texto es "no garantiza mejora" (usar "Que dicen los datos"). Nota "Aplica los
   ajustes de prioridad alta y media" idem (ver 4).

10. **BAJA. Avisos duplicados en "Revisar alertas".** `flujo/VistaAlertas.tsx`: "Avisos de la semana" (insights, derivados de las alertas)
    y "Avisos del calculo" (`alertasDeCorrida`) listan las mismas alertas dos veces con textos distintos.

11. **BAJA. Encabezado de version desalineado.** `ResumenEjecutivo.tsx` muestra `corridaVista` (version vieja si `versionVista`), pero los
    insights se piden con `mesa.corrida` (actual); `irEtapa` no limpia `versionVista`. Chip "v1" con titulo/KPIs de la v actual.

12. **BAJA. Nombre inconsistente de la accion de guardar.** "Crear corrida" (Aprobacion, DetallePaso, PanelAnalogos, Historial), "Guardar y ver
    el resumen" (ResumenEjecutivo), "Calcular y guardar" (PanelSeleccion); "corrida" es jerga que el feature intenta evitar.

## Verificado sin problemas
- Rutas: real `${BASE}/insights/*`, demo `/adjustments/insights/*` con `clienteDemo` (Basic); `semanas`/`hasta` opcionales; `banda` con `ucp`+`run_id`
  (EPM exige que coincidan: se manda `corrida.meta.ucp`, nombre EPM).
- Adaptador: `{success,data}` -> `ResultadoInsight`; `localizarTexto` solo toca decimal+unidad (ids, fechas, motivo del PATCH intactos).
- Unidades: energia MWh/dia, potencia MW, pp, hora 0..23 (`formatoRangoHoraInsight`), `p*_pct`; series 168 con null, `referencia` null o todo-null
  (los graficos descartan series sin valores), `energia_diaria.referencia` alineada por indice con la semana.
- Aplicar: `cuerpoDePatch` quita `usuario` y agrega `version_esperada`; el provider (real/demo) pone `usuario`; `aplicarTodas` encadena la version
  devuelta y para en el primer fallo; 409 recarga.
- Codigos de alerta del backend cubiertos por `TRADUCCION_ALERTA`; dedupe de avisos por id correcto contra el backend real.
