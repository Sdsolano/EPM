# R1 - efficiency (Pista A, Sphere feat/noche2-front)

Sin hallazgos criticos ni altos. Cuatro medios/bajos.

## Media
1. CONFIRMED - peticiones de mas en useInsights (DetallePaso.tsx:97, ResumenSemanal.tsx:34, hooks/useInsights.ts:86-96).
   El hook siempre pide los 4 bloques por corrida (resumen, recomendaciones, series, banda) + precision. DetallePaso (paso control)
   solo usa `insights.banda`; ResumenSemanal solo usa `insights.series`. Cada montaje (el detalle tiene key=pasoSeleccionado, asi que
   se remonta al cambiar de paso) y cada cambio de version (cada patchPaso) repite las 5 peticiones, sin cache compartida entre
   instancias. Arreglo: opcion `bloques: ["banda"]` / `["series"]` en useInsights (solo se piden y se derivan esos; el resto "inactivo"),
   o cache a nivel de modulo/contexto por clave `runId|version|bloque`.

## Baja
2. CONFIRMED - CurvaHoraria.tsx:196-212, 309-325 (y analogo en CurvaEnergiaDiaria.tsx:~118): estadisticas, frasesResumen, aria-label y
   tablaDias (7 x slice de 24 + estadisticas) se recalculan en cada render sin useMemo. Re-renderiza por cada paso de ResizeObserver
   (arrastre de ventana, doble medicion con ResponsiveContainer) y por cada toggle de leyenda. Costo pequeno (168 puntos) pero evitable:
   envolver filasResumen/frases/etiquetaAria/tablaDias en useMemo([valores, nombres, comparacion...]).
3. CONFIRMED - identidades inestables que anulan memoizacion: `dataKey={(r)=>r.banda}` de <Area> (CurvaHoraria.tsx:403), objetos
   `dot`/`activeDot` inline, y en CurvaDia24.tsx:50-52 `fechas={[fecha]}` / `dias={[...]}` nuevos cada render (invalidan infoDia/marcasDia).
   Ademas CurvaEnergiaSemana pasa `series={{...}}` literal (EnergiaSemana.tsx:~254): `valores/filas/dominio` se recalculan cada render y,
   durante la animacion de entrada (~1 s), un re-render del padre con datos nuevos reinicia la animacion de recharts. Arreglo: dataKey
   como constante "banda" con la banda en campos planos (bandaLo/bandaHi) o funcion definida a nivel de modulo; useMemo en los literales.
4. PLAUSIBLE - useAncho + ResponsiveContainer miden por duplicado (dos ResizeObserver); el primer render usa ancho inicial 720
   (hooks.ts:35, CurvaHoraria.tsx:215) y en movil la altura salta (720*0.4 -> 360*0.7) tras el primer efecto: render extra y posible
   salto de layout. Arreglo: medir en useLayoutEffect, o derivar alto con CSS (aspect-ratio) y quitar useAncho.

## Revisado sin problema
- Listeners/timers: AyudaTermino (keydown con cleanup, timer limpiado), useEntradaUnica, useReducedMotion, ResizeObserver con disconnect: OK.
- Carreras en useInsights: bandera `cancelado` + clave por corrida/version: OK. No hay setInterval/polling.
- Mock/demo se cargan con import() dinamico (providers/ajustes/index.ts:196,226): no entran al bundle de produccion.
- Bundle: recharts ya lo usan ~20 paginas importadas de forma estatica en router.tsx; la Mesa no agrega una dependencia nueva, pero
  MesaAjustes se importa estatico (router.tsx:46) y suma ~8k lineas al chunk principal. Opcional: React.lazy de MesaAjustes como ya se hace con MesaAjustesDev.
- Hover en la grafica no re-renderiza CurvaHoraria (el tooltip vive en el store de recharts): OK.
