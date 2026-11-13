# Fase 3 — LOG (Mesa de ajustes en Sphere)

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
