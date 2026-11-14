# R1 - angulo altitude (arquitectura / nivel de abstraccion)

Diff revisado: `git diff 8a6302c..HEAD` en Sphere-wt-n2 (feat/noche2-front). Ningun hallazgo critico/alto.

## Media

1. **MEDIA, CONFIRMED** - `components/ejecutivo/useAplicar.ts:47-61,63-83` + `MesaAjustes.tsx:61-66`.
   `useAplicar` escribe por una via paralela a `useMesaAjustes`: llama a `aplicarRecomendacion` -> `api.patchPaso` directo y despues `mesa.recargar()`. No pasa por `patchPaso` de la mesa (sin `manejarConflicto`, sin `seleccionRef` anti-carrera, sin `accion/guardando`). Consecuencias: mientras se aplica, `mesa.accion` es null, asi que Aprobar, el editor de pasos y el stepper siguen habilitados (aprobar o editar en vuelo contra un PATCH en curso = 409 o version pisada). Ademas `irEtapa` llama `aplicar.descartar()` que vuelve a "reposo" mientras la promesa sigue viva: `ocupado` pasa a false y se puede lanzar un segundo "Aplicar" concurrente; al terminar el primero reescribe el estado a "listo" sobre otra etapa.
   Arreglo: exponer en `useMesaAjustes` una accion `aplicarRecomendaciones` (o hacer que `useAplicar` reciba `mesa.patchPaso`) para reutilizar guard de conflicto/seleccion y marcar `accion="guardando"`; y que `descartar` ignore (o cancele con un id de ejecucion) las promesas en vuelo.

2. **MEDIA, CONFIRMED** - formato numerico triplicado con comportamientos distintos: `utils.ts:325-345`, `charts/series.ts:79-144` y `components/paso/helpers.ts:40-50` (+ `lenguaje.ts:343` `formatoEnergiaAuto`).
   Mismos nombres exportados (`formatoMWh`, `formatoPct`, `SIN_DATO`, `formatoNumero`, `formatoMW`) con firmas y defaults distintos (`formatoMW` dec 0 vs 1; `formatoPct` con/sin `conSigno`). Este mismo diff arreglo el signo "-0,0 %" en `utils.formatoPct` redondeando antes (`Number(v.toFixed(dec))`), pero `charts/series.ts:108-112 formatoPct` sigue usando `v` sin redondear: una variacion de -0,04 % se pinta "−0,0 %" en tooltips/leyendas de las curvas y "0,0 %" en el resto. Ademas `charts/index.ts:36 export *` re-exporta todo `series` y colisiona conceptualmente con `utils`. Deuda: cualquier ajuste de formato hay que hacerlo en 3-4 sitios.
   Arreglo: una sola fuente (`utils.ts`), que `charts/series.ts` importe de ahi (o al reves) y borrar copias; al menos aplicar el mismo redondeo previo en `series.formatoPct`.

3. **MEDIA, CONFIRMED** - estado de navegacion duplicado: `MesaAjustes.tsx:44-71`.
   `vista` (useState local) + `mesa.pasoSeleccionado` (hook) describen juntos la etapa; `irEtapa` los sincroniza a mano (`seleccionarPaso("base")`, `seleccionarPaso("resumen")`) y el reset por cambio de corrida se hace con setState durante el render. Cualquier cambio externo de `pasoSeleccionado` (carga por `?corrida=`, versiones, `abrirAnalogos`) no mueve `vista`, y viceversa. Ademas `useInsights(vista === "ajustar" ? null : ...)` descarta/relanza los 5 pedidos cada vez que se vuelve de Ajustar a Semana/Alertas (5 requests por cambio de etapa) y el badge `pendientes` desaparece (null) en las etapas 3-4 porque los insights se apagan.
   Arreglo: subir `etapa` a `useMesaAjustes` (o a un reducer unico) y mantener `useInsights` activo (los datos ya se cachean por `claveRun`) o conservar el ultimo valor para el badge.

## Baja

4. **BAJA, CONFIRMED** - `MesaAjustes.tsx:136-139`: `onAbrirDia(fecha)` hace `void fecha; verPaso("nivel")`. La tira "Toca un dia para ajustarlo" (`ResumenEjecutivo.tsx:186`) promete abrir ese dia pero el dia se pierde (el detalle abre el paso nivel sin dia seleccionado). Prop publica muerta/enganosa. Arreglo: pasar la fecha al estado del detalle (vista de GraficaPaso/EditorCurvas) o quitar el parametro y cambiar el texto a "Toca un dia para ir a ajustar".

5. **BAJA, CONFIRMED** - logica de dominio repetida en capa de UI/shell: (a) `sinDuplicadosDeAviso(...)` se evalua en `MesaAjustes.tsx:57`, `ResumenEjecutivo.tsx:165` y `QueRevisar.tsx:187`, y el calculo de `pendientes` vive en el shell (`MesaAjustes.tsx:57-58`); (b) el criterio "sugeridas" esta duplicado entre `useAplicar.ts:26-30` (`sugeridasDe`) y `providers/ajustes/recomendaciones.ts:78-85` (`PRIORIDADES_AUTOMATICAS` + filtro), de modo que el contador del boton "(N)" y lo que realmente se aplica pueden divergir si se toca uno solo. Arreglo: derivar `ajustesPropios`/`pendientes` y `sugeridas` una vez (p. ej. en `useInsights`/`useAplicar`) y que el hook use la funcion del provider (exportar `candidatasAutomaticas`).

6. **BAJA, CONFIRMED** - `useMesaAjustes.ts:414-440`: logica especifica del perfil demo (consulta de `api.reales` 220 dias, mutacion de `semanaElegidaRef` con truco `setSemana` + reset a false) dentro del hook generico de 774 lineas, con `PERFIL_DEMO` importado en una vista. Mezcla capas (la regla "semana por defecto = lunes tras el ultimo dato real" es de dominio/provider) y deja un parpadeo de semana por defecto -> semana nueva. Arreglo: moverlo a un hook `useSemanaPorDefecto` o al provider demo (`ucps`/endpoint de "ultimo dato real").

7. **BAJA, CONFIRMED** - codigo de desarrollo dentro de `src/`: `charts/__previa__/Previa.tsx` (147 lineas, sin ruta ni importadores) y `providers/ajustes/__verif__/verificar.ts` (script esbuild) viajan en el arbol de fuentes y entran a typecheck/lint. Arreglo: moverlos fuera de `src` (p. ej. `dev/` o `scripts/`) o excluirlos en tsconfig/eslint.

8. **BAJA, PLAUSIBLE** - componentes grandes con reglas de dominio embebidas: `paso/Diagnosticos.tsx` (517 l), `paso/AccionesPaso.tsx` (437 l, `construirParams` valida/convierte parametros dentro del componente), `charts/CurvaHoraria.tsx` (505 l). `GraficaPaso.tsx:62-80` y `CurvaPrincipal.tsx:27-45` ensamblan cada uno a mano series/banda (168 puntos) y reglas de "dia con alerta/editado": mismo mapeo en dos sitios. Mantenimiento mas caro, no un bug. Arreglo: extraer `construirSeriesSemana(...)` a `charts/series.ts` o a un `paso/params.ts` testeable.

## Sin hallazgos
- Providers: contrato `ResultadoInsight` (nunca rechaza) bien separado de `ApiReturnAjustes`; carga perezosa mock/demo coherente. `useInsights` descarta respuestas tardias correctamente.
