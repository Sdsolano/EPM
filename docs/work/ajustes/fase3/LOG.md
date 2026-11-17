# Fase 3 — LOG (Mesa de ajustes en Sphere)

## Estado actual (2026-11-13, coordinador 3 retoma: verifica P1–P4, build, integración, revisión)
- Retoma coordinador nuevo; ver pasos abajo.

## Estado previo (coordinador 2)
- El coordinador anterior fue detenido. Retoma un coordinador nuevo (mismo rol, autónomo).
- Hecho: F (foundation). En el worktree YA existen implementaciones de P1–P4 (no stubs): componentes de
  `components/` + subcarpetas `seleccion/ stepper/ paso/ resumen/ analogos/` (últimas escrituras 07:48–07:55).
  Los agentes P1–P4 del coordinador anterior no dejaron reporte; se verifica si siguen escribiendo.
- En curso: verificar estado real (tsc/vite/eslint) de P1–P4.
- Falta: integración, prueba en navegador con mock, ronda de revisión, correcciones, REPORTE.md, commit local.
- Archivos sin commitear: todo lo de la fase (ver `git -C ../Sphere-wt-mesa status`).
- Subagentes vivos lanzados por este coordinador: ninguno.
- Siguiente paso exacto: `npx tsc -b` (≤178) + `npx vite build` + eslint de `src/views/pages/mesa_ajustes`.

Snapshot resumible. Plan: `PLAN.md`. Diseño: `DESIGN.md`. Contrato backend: `CONTRATO_BACKEND.md`.

## Línea base (rama `feat/mesa-ajustes` = `origin/test` 57eb4fb, antes de tocar nada)
- `npm ci` OK.
- `npx eslint .`: **868 errores, 51 warnings** (preexistentes). De los archivos compartidos que se tocarán:
  `routes/router.tsx` y `utils/iconosMenu.ts` limpios; `context/MenuEstructuraContext.tsx` ya tiene 2 errores
  `react-refresh/only-export-components` (preexistentes, líneas 75-76).
- `npm run build` **YA FALLA en la base**: `tsc -b` da **178 errores TS** en 22 archivos preexistentes
  (Pronosticos.tsx 59, Dna.tsx 17, Indicator_board 14, ...). `vite build` solo pasa (exit 0).
  Criterio adoptado (decisión sin el usuario): no arreglar errores ajenos (tocaría otras pantallas); la
  fase cumple si `tsc -b` no suma errores (sigue en 178, ninguno en archivos nuevos/tocados) y `vite build` pasa.

## Decisiones tomadas sin el usuario
Ver `PLAN.md` § Decisiones (1–10). Añadidas:
- 11. `npm run build` roto en la base: criterio "no empeorar" (arriba).

## Progreso real
- [x] Lectura de reglas, contrato EPM (router.py, types.py, steps/*), convenciones Sphere.
- [x] PLAN.md, LOG.md, DESIGN.md.
- [x] F foundation (agente `ab2b0d5f2b47ab1db`, cortado una vez por límite de uso y reanudado). tsc 178 = base,
      eslint solo 2 preexistentes, vite build OK, Playwright con mock OK. Los subagentes no pueden escribir
      REPORT.md (restricción de la herramienta): el reporte queda aquí.
      Archivos (Sphere-wt-mesa/src): `types/ajustes.d.ts`, `vite-env.d.ts`, `constants/ajustes/index.ts`,
      `providers/ajustes/{index.ts,CONTRATO_BACKEND.md,mock/{index,simulador,fixtures,analogos}.ts}`,
      `views/pages/mesa_ajustes/{MesaAjustes.tsx,MesaAjustesDev.tsx,useMesaAjustes.ts,utils.ts,mesa.module.css}`,
      stubs `components/{PanelSeleccion,StepperPasos,DetallePaso,ResumenSemanal,BuscadorAnalogos}.tsx` (todos
      reciben `mesa: MesaAjustesState`; DetallePaso + `paso`,`tituloId`; ResumenSemanal + `tituloId`); aditivo en
      `routes/router.tsx`, `context/MenuEstructuraContext.tsx`, `utils/iconosMenu.ts`.
      Mock: `__mockBumpVersion(id)` (409), `__mockEscenarioWarning(id)`, `__mockEscenarioReconciliacionRota(id)`,
      `__mockConfig`, `__mockReiniciar` en `providers/ajustes/mock/index.ts`.
- [ ] P1–P4 lanzados en paralelo (2026-11-13). Reportes: los devuelven como texto; el coordinador los resume aquí.
- [ ] P1..P4 en paralelo.
- [ ] Integración + build + lint + prueba en navegador con mock.
- [ ] Ronda de revisión (8 ángulos + regresión + better-interface, sonnet).
- [ ] Correcciones + verificación.
- [ ] Commit local + REPORTE.md.

## [08:20] Coordinador 3: verificación P1–P4
- P1–P4 estaban implementados. Corregido `useId` sin usar en PanelAnalogos. `tsc -b` = 178 (base), `vite build` OK, eslint de mesa/providers/constants limpio.
- Siguiente: prueba en navegador (dev server VITE_AJUSTES_MOCK=true, /dev/mesa_ajustes), luego ronda 1 de revisión.

## [08:35] Prueba en navegador con mock OK
- Dev server (VITE_AJUSTES_MOCK=true, :5199) → /dev/mesa_ajustes: calcular vista previa, crear corrida, pasos Base/Nivel/Forma/Control/Resumen renderizan, drawer de análogos abre, 0 errores de consola.
- Siguiente: ronda 1 de revisión (9 revisores sonnet + regresión).

## [08:55] Ronda 1 de revisión cerrada (9 ángulos + regresión, sonnet)
Regresión: PASA (tsc 178, vite OK, eslint solo 2 preexistentes). Hay 2 hallazgos ALTOS reportados → corregir y repetir ronda completa.
ALTOS: (A1, better-interface, CONFIRMED) el Dialog de confirmación de Aprobacion.tsx va en portal fuera de `.mesa`: variables --mesa-* sin definir (botón "Aprobar semana" invisible, sin foco). (A2, efficiency, PLAUSIBLE) patchPaso/aprobar/cargarCorrida/cargarVersion/versiones en useMesaAjustes.ts no descartan respuestas tardías tras cambiar UCP/semana.
MEDIOS a corregir: M1 usarAnalogos hace merge y no permite quitar todos los análogos de un día (filtros.ts mapaParaEnviar + useMesaAjustes). M2 AccionesPaso numeroParam: String(0.125) → parsearNumero lo lee como 125 (usar coma). M3 aprobar 422: setCorrida directo → usar aplicarCorrida/refresco. M4 `calcular` pisa borrador abierto sin aviso (aceptar o aviso). M5 `{}` nuevo en DetallePaso:143 invalida useMemo; retorno del hook sin memo. M6 carrera retomar borrador local vs calcular. M7 doble mensaje 422 (aceptar). M8 graficas día sin alternativa tabular (aceptar/ <details>). Bajos: código muerto (normalizarCurva, festivosEn), exports, contraste serie previa, fuentes 11px, role=status, README VITE_AJUSTES_MOCK, TODO retiro menú, duplicación de helpers (aceptado salvo parsearNumero compartido).
Siguiente: agente de corrección, luego ronda 2 completa.

## [09:10] Correcciones de ronda 1 aplicadas (agente ab53bb50...): tokens.module.css compartido (Dialog), descartar respuestas tardías por selección, quitar análogos de un día, parsearNumero compartido, aplicarCorrida en 422, memo del hook, confirm en calcular con borrador, bajos (código muerto, role=status, contraste, details 24h, TODO menú, README). tsc 178, vite OK, eslint limpio (según el agente).
Siguiente: ronda 2 COMPLETA (9 ángulos + regresión, sonnet) por haber habido altos.

## [09:45] Ronda 2 cerrada
Regresión PASA (tsc 178, vite OK, eslint 2 preexistentes, navegador OK con Dialog de aprobación visible). ALTO nuevo (line-by-line CONFIRMED + efficiency + cross-file): el descarte por seleccionRef en cargarCorrida perdía en silencio `?corrida=<id>` si ucps() respondía antes. Corregido: no descartar cuando sel empieza por "|"; seleccionRef se asigna síncrono en setUcp/setSemana. tsc 178, eslint limpio.
Medios/bajos aceptados (una línea): menú inyectado afecta fallback de frecuentes (temporal, TODO), hex de hover/gráficas fijos, duplicación menor de helpers, cargarCorrida no llama limpiarSeleccion (ventana de ms, API valida), Aprobacion foco tras aprobar y Alert closeText (pendientes opcionales), AbortController no usado (respuestas descartadas), lógica de params de análogos en el hook.
Siguiente: ronda 3 completa (por el alto de la ronda 2).

## [10:20] FASE 3 TERMINADA — commit local en feat/mesa-ajustes (ver `git -C ../Sphere-wt-mesa log -1`)
Ronda 3 completa sin críticos/altos; regresión PASA. Cierre: quitado warningsVigentes (muerto), `ancla="analogos"` al activar el interruptor en Nivel, closeText "Cerrar". Verificación final: tsc 178 (0 en mesa), vite build OK, eslint limpio. Servidores dev de prueba cerrados.
REPORTE.md: el Write de subagentes lo bloquea; el contenido está en el reporte final del coordinador y en las entradas de ronda de este LOG (rondas 1-3, aceptados). Quedan: push a origin/test (orquestador: fetch, rebase origin/test, vite build, push feat/mesa-ajustes:test), backend intermedio debe poner `usuario`, módulo de menú en BD.
Aceptados: menú inyectado (frecuentes/permisos por perfil, temporal), hex fijos y duplicación menor de helpers, hook grande, sin AbortController, 3 GET repetidos por URL, foco tras aprobar, aria-disabled en Aprobar, exports innecesarios, `ucps()` caído deja "Buscando emitido…", README sin nota de /dev solo-DEV, mock pierde corridas al recargar (Map en memoria).
