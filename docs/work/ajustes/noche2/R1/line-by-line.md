# R1 - line-by-line (Pista A, Sphere-wt-n2, `git diff 8a6302c..HEAD`)

Sin hallazgos criticos ni altos. Revisado: useInsights, useAplicar, recomendaciones.ts, insights.ts, MesaAjustes.tsx, useMesaAjustes.ts (diff), charts/* (series, CurvaHoraria, CurvaEnergiaDiaria, CurvaMini, hooks), ejecutivo/*, flujo/*, GraficaPaso, EnergiaSemana, Aprobacion, AccionesPaso.
Sin problemas: carreras en useInsights (cancelado + clave por corrida|version|intento + origen), 168 puntos/24 (normalizar recorta/rellena), fechas (aritmetica UTC en charts; dayjs local consistente en utils), limpieza de efectos (timers, ResizeObserver, listeners).

## Media

1. **MEDIA / CONFIRMED** - `components/ejecutivo/useAplicar.ts:42-44, 53-58, 62-79` + `hooks/useInsights.ts:derivar`.
   Tras aplicar (una o todas), `mesa.recargar()` sube la version y `useInsights` conserva las recomendaciones anteriores con `actualizando: true` hasta que llegan las nuevas. En esa ventana `puedeAplicar` y `sugeridas` salen de la lista vieja (`ya_resueltas` viejo), asi que "Aplicar" y "Aplicar el ajuste sugerido (N)" siguen habilitados (`ocupado` ya es false porque el estado paso a "listo"). Un segundo clic reaplica el mismo ajuste: version extra, historial duplicado, y mensaje "Listo" falso.
   Arreglo: `puedeAplicar` y `sugeridas` deben devolver false/[] mientras `recomendaciones.actualizando` (pasar el flag a useAplicar), o guardar localmente los ids aplicados hasta que llegue la version nueva.

2. **MEDIA / PLAUSIBLE** - `hooks/useInsights.ts:VALIDAR` (`providers/ajustes/insights.ts:68-74`) vs. consumo sin guardas.
   Las validaciones son superficiales: `series` no valida `dias`/`series.series` (CurvaPrincipal hace `series.dias.map`), `resumen` no valida `kpis.alertas`/`kpis.recomendaciones` (MesaAjustes.tsx:~68 y Indicadores leen `.alertas.warning` en render, fuera de cualquier boundary), `recomendaciones` no valida `ya_resueltas` (QueRevisar:214 `.length`), `precision` no valida `por_tipo_dia` (Confiabilidad `.length`). Si el backend omite un campo (version vieja, respuesta parcial) se cae toda la Mesa en vez de degradar ese bloque.
   Arreglo: ampliar VALIDAR a los campos que se recorren, o `?.`/`?? []` en los puntos de uso.

## Baja

3. **BAJA / CONFIRMED** - `MesaAjustes.tsx` (`onAbrirDia={(fecha) => { void fecha; verPaso("nivel"); }}`).
   La tira dice "Toca un dia para ajustarlo" y cada dia tiene aria-label "Ajustar este dia", pero la fecha se descarta: siempre abre el paso Nivel generico. Arreglo: pasar la fecha al detalle (dia seleccionado) o cambiar el texto/aria a "Ir a ajustar".

4. **BAJA / CONFIRMED** - `MesaAjustes.tsx` (`idPrevio`) + `useAplicar` estado.
   Al cambiar de UCP/semana/corrida solo se reinicia `vista`; `aplicar.estado` ("Listo: ..." con "Ver el cambio en la curva", o un error) persiste y aparece en la corrida nueva. Arreglo: `aplicar.descartar()` en el mismo bloque que `setVista("semana")`.

5. **BAJA / PLAUSIBLE** - `MesaAjustes.tsx` (`useInsights(... mesa.corrida)` vs `corridaVista`).
   Con `versionVista` activa (version anterior) y etapa "semana"/"alertas", el encabezado muestra meta/version de `corridaVista` pero indicadores, curva y avisos son de la version actual. Arreglo: pedir insights de la version vista o avisar "datos de la version actual".

6. **BAJA / CONFIRMED** - `components/ejecutivo/useAplicar.ts:54-57, 70`.
   El resultado de `mesa.recargar()` se ignora: si la recarga falla (red/404) se muestra "Listo..." y la pantalla queda en la version vieja. Arreglo: revisar `ok` y mostrar aviso de "no se pudo actualizar la semana".

7. **BAJA / PLAUSIBLE** - `components/paso/GraficaPaso.tsx:seriesSemana` (banda).
   `dias.flatMap(d => d.inferior/superior)` no normaliza cada dia a 24; si un dia trae otra longitud los dias siguientes se desplazan (CurvaHoraria solo recorta a 168 al final). Arreglo: `normalizar(d.inferior, 24)` por dia, como hace `aplanarCurvas`.

8. **BAJA / PLAUSIBLE** - `components/ejecutivo/Confiabilidad.tsx`.
   Etiqueta "de X % a Y %", `primera`/`ultima` y `destacarUltimo` asumen `semanas` en orden cronologico ascendente; el contrato no lo garantiza. Arreglo: ordenar por `semana_inicio` en el componente.
