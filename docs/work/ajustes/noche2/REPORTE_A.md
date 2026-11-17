# REPORTE A (front) - Mesa de ajustes "wow" - noche 2

Rama `feat/noche2-front` (Sphere-wt-n2, desde origin/test 8a6302c). Sin push.

## Que se construyo
- **Resumen ejecutivo** (pantalla de entrada): semana, titulo, confianza en llano, 4 indicadores (demanda prevista protagonista, pico, error reciente, por revisar), curva semanal de 168 h (base/ajustado/referencia/real/banda), tira de 7 dias, "Que revisar" (alertas + recomendaciones con Aplicar y "Aplicar el ajuste sugerido"), "Que tan confiable es" (curva de error por semana), siguiente paso unico.
- **Flujo guiado** en 4 etapas (Ver la semana, Revisar alertas, Ajustar, Aprobar) con barra de etapas; el riel de 6 pasos queda dentro de "Ajustar"; bienvenida con un solo boton primario.
- **Curvas, no barras**: libreria `mesa_ajustes/charts/` (CurvaSemana168, CurvaDia24, CurvaEnergiaDiaria, CurvaMini, leyenda, tooltip, estados, resumen textual, reduced-motion). GraficaPaso, EnergiaSemana y GraficaAnalogos reemplazados. `grep BarChart|<Bar` = 0.
- **Lenguaje de negocio**: `lenguaje.ts` (glosario de 14 terminos, traduccion de 31 codigos de alerta), `AyudaTermino`, parametros con nombres de negocio, sin textos de desarrollo.
- **Datos**: tipos `ajustesInsights.d.ts`, adaptador `providers/ajustes/insights.ts` (real/demo/mock, degrada por bloque, nunca lanza), `recomendaciones.ts`, hook `useInsights`, mock realista.
- **Integracion con el backend real de B** (CONTRATO_B v1.5) probada en navegador desde una copia en TEMP (sin tocar `EPM\data\features`): formas coinciden; 7 discrepancias de front corregidas (semana por defecto en demo, mensajes 422 en llano, decimales, duplicados alerta/recomendacion, MW vs MWh, cero negativo, stepper/drawer).

## Commits (Sphere-wt-n2, feat/noche2-front)
f10c971 (graficas, tipos, provider, mock, lenguaje) - 696b23c (resumen, flujo, curvas en pasos, lenguaje) - 0d6b132 (integracion backend real) - commit de correcciones de la ronda 1 (ver `git log`).

## Verificacion
- `tsc -b`: 0 lineas nuevas sobre las 178 preexistentes (tsc_base_A.txt). eslint limpio en lo tocado. Script A2: TODO OK. `vite build` y `npm run build:demo` OK (~1m10-1m30 s).
- Navegador (real + mock): 0 errores de consola salvo los 404 de `GET /adjustments/aprobada` (backend, sin aprobada) y el 422 provocado a proposito. Sin scroll horizontal de pagina a 320/375/768/1280.
- Capturas: `noche2/capturas_A/` (A1 graficas, A3 flujo completo 320-1280, A4 pasos/analogos/aprobacion, A5 backend real: acceso, bienvenida, carga, error, resumen 1280/768/375, alertas, ajustar 168 h y 24 h, drawer a 768, aprobacion, degradacion).

## Revision ronda 1 (9 angulos + regresion, sonnet): 0 criticos/altos -> sin ronda 2
Corregidos inline (A6\REPORTE.md): doble aplicacion/recomendacion obsoleta, dia ignorado al abrir, version anterior mezclada con resumen, carrera del efecto demo, `paramsValidos` en bienvenida/riel, validacion de respuestas y degradacion, orden de semanas y frase de confianza, contador "Por revisar" (solo lo accionable), copy "Efecto esperado" solo con evidencia validada, -0,0 %, alerta apilada en movil + aria-label, leyenda de franjas, colores de GraficaAnalogos, nombre unico de la accion de guardar, borrado de `__previa__`.

### Aceptados (no corregidos)
- Bloques selectivos en `useInsights` (DetallePaso/ResumenSemanal piden 5 endpoints para usar 1): costo bajo, endpoints de 20-140 ms.
- Memoizaciones menores en charts, doble medicion de ancho: 168 puntos, costo despreciable.
- Formateadores es-CO duplicados con defaults distintos (`formatoMW` 1 vs 0 decimales), paletas hex en CSS, props/exports sin uso, componentes >400 lineas: deuda de mantenimiento, sin efecto visible.
- Tablas con scroll interno a 320 px; tooltip de curvas solo con raton (la lectura sin raton la cubren leyenda, tabla y detalle por dia).
- Mock no alineado a v1.4 (siempre ofrece recomendaciones alta/media; el backend real hoy casi no las emite): el boton "Aplicar el ajuste sugerido" solo aparecera con datos reales cuando haya recomendaciones aplicables. Probado con una recomendacion sintetica contra el PATCH real (+686 MWh).
- `__verif__/verificar.ts` se queda en src/ (script de A2; no entra al bundle).

## Hallazgos de backend (para B / orquestador)
- Alerta de clima con "lluvia 126735 mm" (Antioquia, sem. 6 jul); B reporto fix posterior (cb5a895), no reverificado por A.
- `GET /adjustments/aprobada` responde 404 sin aprobada (ruido en consola).
- En demo no se calculan semanas con real ni a >90 dias del ultimo dato: la serie `real` llega vacia; no se puede mostrar una semana pasada contra el real.
- En 9 semanas probadas ninguna recomendacion de sesgo paso el filtro: solo alertas.
- Historial de pronosticos es un snapshot estatico (Antioquia hasta 2026-06-28): la confianza dice "hasta <fecha>".

## Pendientes
Verificar visualmente en el despliegue de Railway tras integrar; reverificar la alerta de clima con el backend final; probar teclado en tooltips; opcional `React.lazy` de MesaAjustes. Decidir si "Crear corrida" interno debe seguir visible en algun texto.
