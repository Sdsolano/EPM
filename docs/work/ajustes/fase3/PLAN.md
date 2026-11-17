# Fase 3 — UI "Mesa de ajustes" en Sphere (sub-plan del coordinador)

Fuente de verdad de la fase. Progreso real en `LOG.md` (misma carpeta). Reporte final en `REPORTE.md`.
Contrato del backend intermedio en `CONTRATO_BACKEND.md` (y copia en Sphere `src/providers/ajustes/CONTRATO_BACKEND.md`).

## Objetivo
Pantalla nueva "Mesa de ajustes" en Sphere (ruta `/mesa_ajustes`, entrada de menú junto a Pronósticos) que
reemplaza en diseño al botón ASR: le da al analista el procedimiento paso a paso
(`base → referencia → nivel → forma → control → aprobación`) y deja que el sistema haga la mayor parte del
trabajo. Consume la API EPM `/adjustments/*` a través del backend intermedio (`admin/dashboard/ajustes/*`),
con un adaptador mock (`VITE_AJUSTES_MOCK=true`) para verla y probarla sin backend.

## Entorno
- Worktree Sphere: `C:/Users/samue/OneDrive/Documentos/GitHub/Sphere-wt-mesa`, rama `feat/mesa-ajustes`
  (desde `origin/test`). React 19 + Vite 7 + TS 5.9, MUI 7 (+ x-charts, x-date-pickers), CoreUI, recharts 3,
  dayjs, axios (`src/providers/axiosInstance.ts`). Fuente DM Sans. Verde marca `#009653` (`--primary`).
- Verificación: `npm run build` (tsc -b + vite build), `npx eslint <archivos tocados>` (línea base de lint
  del repo en `LOG.md`), prueba en navegador con mock (Playwright headless desde el scratchpad o panel).
- No tocar `../Sphere` ni `main` de Sphere. Commit local solo al final, por ruta explícita. Sin push.

## Convenciones de Sphere que se respetan
- URLs en `src/constants/<modulo>/index.ts` (`${HOST}${VERSION}admin/dashboard/...`), servicios en
  `src/providers/<modulo>/index.ts` como `const ApiX = () => {...}` devolviendo `ApiReturn<T>`
  (`{success:true,data}` | `{success:false,error}`), tipos en `src/types/<modulo>.d.ts`.
- Páginas en `src/views/pages/<modulo>/`. Rutas en `src/routes/router.tsx` (hijos de `MainLayout`,
  envueltos en `BlockSuperAdminRoute`). Menú desde backend (`MenuEstructuraContext`); iconos por ruta en
  `src/utils/iconosMenu.ts`.

## Decisiones (tomadas sin el usuario; justificación en LOG)
1. Capa de servicio en `src/providers/ajustes/` (convención de Sphere) en lugar de `src/services/`.
2. Menú: el backend gobierna los módulos por perfil. Mientras no exista el módulo en BD, el contexto de
   menú inyecta "Mesa de ajustes" (`/mesa_ajustes`) justo después de `/pronosticos` (misma área) a quien
   tenga Pronósticos; si el backend ya lo trae, no se duplica.
3. Ruta de desarrollo `/dev/mesa_ajustes` sin login, registrada SOLO si `import.meta.env.DEV` y
   `VITE_AJUSTES_MOCK === "true"` (en build de producción no existe). Sirve para probar con el mock.
4. Respuestas del backend intermedio con el envoltorio de Sphere: éxito `{success:true, data:<cuerpo EPM>}`;
   error con el MISMO status HTTP de EPM y `{success:false, message, detail:<detail EPM>}`.
5. El front no manda `usuario`; el backend lo pone desde la sesión (y sobrescribe si llega).
6. Pronóstico emitido: ruta `GET ajustes/pronostico-emitido` (backend lo arma desde sus datos) que da
   `base_curvas`, `base_ref`, `fecha_emision`, `modelo_version`; el front los reenvía en preview/run.
7. Curva real: ruta `GET ajustes/reales` (backend desde su histórico). Opcional; si falla, la gráfica no la muestra.
8. Retomar un borrador: EPM no lista corridas; el front guarda el id en la URL (`?corrida=<id>`) y en
   localStorage por (ucp, semana) como conveniencia. Listar borradores queda como pendiente de EPM.
9. R2 "sugerido vs automático": tipo `ModoPaso = "sugerido" | "automatico"` y mapa
   `modos_por_segmento` opcional; hoy todos "sugerido" (constante en front, sobrescribible por backend en
   `meta.extras.modos_paso` cuando exista).
10. Análogos: se consume el contrato fijo de `PLAN_NOCHE.md` tal cual. "Usar en forma" → PATCH `forma`
    con `{...params, fuente:"analogos", analogos:{fecha:[...]}}`; "usar en nivel" → PATCH `nivel` con
    `{...params, ancla:"analogos", analogos:{...}}`.

## Partes y dueños (archivos exclusivos)
Rutas relativas a `Sphere-wt-mesa/src/`.

### F — Foundation (serie, 1 agente)
- `types/ajustes.d.ts` — tipos exactos del contrato EPM + rutas intermedias.
- `constants/ajustes/index.ts` — URLs `admin/dashboard/ajustes/*`.
- `providers/ajustes/index.ts` — `ApiAjustes()`; si `VITE_AJUSTES_MOCK==="true"` delega en el mock.
- `providers/ajustes/mock/*` — fixtures realistas (Antioquia ~ 1500–2200 MW pico; Atlantico) y simulador en
  memoria que respeta versiones (409), estados de paso, recalculo de pasos siguientes, bloqueos, aprobación
  (422 con bloqueos / justificaciones inválidas), versiones y eventos, y análogos.
- `providers/ajustes/CONTRATO_BACKEND.md` (copia del de EPM).
- `views/pages/mesa_ajustes/MesaAjustes.tsx` (shell y layout), `useMesaAjustes.ts` (estado y acciones:
  calcular, crear corrida, cargar, patch con `version_esperada` y manejo de 409 recargando, aprobar),
  `utils.ts` (energía, pico, normalizar, formato es-CO, etiquetas de pasos/estados/alertas, lunes),
  `mesa.module.css` (tokens de la pantalla), y STUBS con props fijas en `components/`:
  `PanelSeleccion.tsx`, `StepperPasos.tsx`, `DetallePaso.tsx`, `ResumenSemanal.tsx`, `BuscadorAnalogos.tsx`.
- `routes/router.tsx` (ruta + ruta dev), `context/MenuEstructuraContext.tsx` (inyección),
  `utils/iconosMenu.ts` (icono), `vite-env`/`env.d.ts` si hace falta tipar `VITE_AJUSTES_MOCK`.

### P1 — Selección y stepper
`components/PanelSeleccion.tsx`, `components/StepperPasos.tsx` (+ subarchivos `components/seleccion/*`,
`components/stepper/*`).

### P2 — Detalle de paso
`components/DetallePaso.tsx` + `components/paso/*` (gráfica previa vs resultado vs real por día,
diagnósticos legibles por paso, alertas, acciones aplicar/omitir/editar params, tabla editable de curvas).

### P3 — Resumen, aprobación, historial
`components/ResumenSemanal.tsx` + `components/resumen/*`.

### P4 — Buscador de análogos
`components/BuscadorAnalogos.tsx` + `components/analogos/*`.

Nadie edita archivos de otra parte. Si una parte necesita algo de `useMesaAjustes.ts`/`utils.ts`/tipos, lo
pide en su REPORT y lo integra el coordinador.

## Skills por agente de implementación
`impeccable`, `frontend-design`, `emil-design-eng`, `design-engineering` (invocadas con Skill) y al final
`better-interface` sobre su parte (corregir HIGH y MEDIUM). NO `unslop`.

## Dirección visual (resumen; detalle en `DESIGN.md`)
Mesa de trabajo de analista: densa, numérica, calma. Base blanca/gris de Sphere, verde `#009653` solo para
acción primaria y estado "aplicado"; números tabulares; el elemento memorable es el riel de pasos con la
energía semanal de cada paso (cómo se mueve la semana paso a paso).

## Terminado
- Las 5 funciones del pedido operan contra el mock; build pasa; lint sin errores nuevos en lo tocado.
- Pronósticos y demás pantallas intactas (solo cambian router, menú e iconos, de forma aditiva).
- Ronda de revisión (8 ángulos + regresión + better-interface, todos `sonnet`) sin crítico/alto abierto.
- Commit local en `feat/mesa-ajustes`; `REPORTE.md` escrito.
