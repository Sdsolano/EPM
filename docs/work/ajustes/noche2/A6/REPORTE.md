# A6 REPORTE - correcciones de la ronda 1 (Pista A, Sphere-wt-n2)

Sin commit (lo hace el coordinador). tsc sin lineas nuevas vs tsc_base_A.txt; eslint de lo tocado limpio; script A2 "TODO OK".

## Corregido
1. Aplicar (useAplicar.ts, MesaAjustes.tsx): ahora recibe el bloque `insights.recomendaciones`; solo opera si `!actualizando` y `data.version === corrida.version`
   (si no, `sugeridas=[]` y `puedeAplicar=false`). Guard `enCursoRef` contra doble clic; `descartar` no hace nada con una aplicacion en vuelo;
   reinicio (estado + generacion) al cambiar el id de la corrida; resultados de una aplicacion de otra corrida se ignoran. Si `mesa.recargar()` falla ya no dice "Listo" (error explicito).
2. `onAbrirDia`: DetallePaso acepta `diaInicial` (opcional); MesaAjustes guarda `diaAbierto` y abre el paso Nivel con ese dia seleccionado en la grafica y el editor (si el dia no pertenece al paso, cae a la vista Semana).
3. Con `versionVista`: useInsights recibe null; ResumenEjecutivo muestra "Estas viendo una version anterior (solo lectura)" + boton "Volver a la version actual"; VistaAlertas muestra la misma nota.
4. useMesaAjustes: el efecto demo tambien comprueba `idCargadoRef.current`.
5. `paramsValidos` sube a `mesa` (useMesaAjustes: `paramsValidos`/`setParamsValidos`); PanelSeleccion lo usa; Bienvenida y el boton del riel vacio se bloquean, y Bienvenida explica el motivo.
6. insights.ts VALIDAR ampliado (series.dias/energia_diaria, kpis.alertas/recomendaciones, confianza, siguiente_paso, ya_resueltas, por_tipo_dia, banda.dias); forma invalida -> no_disponible de ese bloque. `?.`/`?? []` en MesaAjustes, Indicadores, QueRevisar, Confiabilidad. Banda por dia normalizada a 24 en GraficaPaso (semana y dia). Confiabilidad: ordena semanas por fecha y dice "las N semanas hasta el <fecha>" si la confianza es sin_datos o el historial termina a >10 semanas de la objetivo. types/ajustesInsights.d.ts: `respaldo`, `validado_fuera_de_muestra`, `evidencia.ventana_seleccion`, `kpis.recomendaciones.del_sistema/tendencia_reciente`, `n/sesgo_pct` nullable. "Por revisar" y badge del riel cuentan con `ajustesPorRevisar` (lenguaje.ts: sin duplicados de aviso, sin ya resueltas, sin `respaldo === "tendencia_reciente"`). "Efecto esperado" solo si `validado_fuera_de_muestra === true`; si no, "Que dicen los datos"; chip "Tendencia reciente, sin validar". Reintentar tambien en bloques no_disponible transitorios (status undefined o 5xx salvo 501).
7. charts/series.ts `formatoPct` redondea antes del signo (sin "-0,0 %"); charts/index.ts ya no hace `export *` (solo utilidades de curvas, sin formateadores que choquen con utils). Hover primario unificado a #00552f (AccionesPaso, EditorCurvas, DetallePaso; tambien analogos/filtros.ts).
8. AlertasVigentes apila texto/boton a <=480 px y el aria-label empieza con el texto visible; CurvaHoraria muestra leyenda visible de las franjas (violeta dia con ajuste / ocre dia por revisar, solo las que aparecen); GraficaAnalogos usa colores de charts/colores.ts (aceptado #6F7C75, >=3:1) y su leyenda pasa a clases del CSS module; 11px -> 12px.
9. Borrado `charts/__previa__/`; `__verif__/verificar.ts` intacto. Accion de guardar unificada: "Preparar la semana para ajustar" (botones y textos de referencia; "Volver a preparar la semana" cuando ya hay corrida).

## No corregido / notas
- formatoMW: los defaults de series.ts (1 dec) y utils.ts (0 dec) no se unificaron (cambiaria cifras mostradas); solo se quito la re-exportacion del indice.
- Mock (cross-file 4, ids de aviso y prioridades alineadas a v1.4): fuera del alcance indicado; el boton "Aplicar el ajuste sugerido" sigue dependiendo de prioridades alta/media del mock. El coordinador debe decidirlo.
- Aceptados segun el encargo: bloques selectivos en useInsights, memoizaciones, formateadores duplicados restantes, props/exports sin uso, tablas a 320 px, componentes grandes.
- No se corrio build ni se verifico en navegador.
