# F — Foundation (sub-plan)

Lee: `../PLAN.md` (parte F), `../DESIGN.md`, `../CONTRATO_BACKEND.md`, `../../PLAN_NOCHE.md` (análogos),
EPM `src/adjustments/router.py` + `steps/*.py` (mandan sobre los docs).

## Entregables (rutas bajo `Sphere-wt-mesa/src/`)
1. `types/ajustes.d.ts` — tipos del contrato (RunOut/StepOut/AlertOut/VersionInfo/VersionOut, requests sin
   `usuario`, diagnósticos y params por paso, análogos, rutas intermedias, `ApiReturnAjustes<T>` como unión
   discriminada).
2. `constants/ajustes/index.ts` — URLs `admin/dashboard/ajustes/*` con `encodeURIComponent`.
3. `providers/ajustes/index.ts` — `ApiAjustes()`; desenvuelve `{success,data}`; error con `status`,
   `message`, `detail`; timeouts 180 s (preview/run/patch base) / 30 s; delega en mock si `VITE_AJUSTES_MOCK`.
4. `providers/ajustes/mock/` — `fixtures.ts` (generador de curvas sintéticas, festivos), `simulador.ts`
   (pipeline base→referencia→nivel→forma→control, corridas en memoria, versiones, eventos, 409/404/422),
   `analogos.ts`, `index.ts` (misma interfaz que `ApiAjustes`, latencia 300–900 ms).
5. `providers/ajustes/CONTRATO_BACKEND.md` — copia literal.
6. `views/pages/mesa_ajustes/` — `MesaAjustes.tsx` (shell), `useMesaAjustes.ts` (estado+acciones),
   `utils.ts`, `mesa.module.css`, stubs en `components/` con props fijas (cada uno recibe `mesa:
   MesaAjustesState` + lo específico).
7. Integración aditiva: `routes/router.tsx` (ruta + ruta dev condicionada), `MenuEstructuraContext.tsx`
   (inyección del módulo), `utils/iconosMenu.ts`, `vite-env.d.ts` (tipo de `VITE_AJUSTES_MOCK`).

## Decisiones previstas
- `ApiReturnAjustes<T>` = `{success:true,data:T} | {success:false,error:ApiErrorAjustes}` (narrowing limpio).
- El mock se carga con `import()` dinámico solo cuando el flag está activo (no entra al bundle de prod).
- Semana por defecto: lunes siguiente a hoy. Mock: "hoy" del simulador = fecha real del sistema; reales
  existen hasta el día anterior a hoy.
- Escenarios del mock: festivo = semana que contiene un lunes festivo del calendario colombiano simplificado
  (lista fija de festivos 2024–2027); warning de control = editar curvas en `forma` o `nivel` > 1 %; atajo
  `__mockEscenarioWarning(id)`; 409 = `__mockBumpVersion(id)` o `__mockConfig({conflictoAleatorio:0.1})`.
- Las acciones del hook devuelven `ResultadoAccion` (`ok | conflicto | error`) para que la UI avise.

## Terminado
`tsc -b` sin errores en mis rutas, `vite build` pasa, eslint limpio en mis archivos (salvo 2 preexistentes del
contexto de menú), `/dev/mesa_ajustes` renderiza y "Calcular"/"Crear corrida" funcionan con el mock.
REPORT.md con props de stubs, API del hook, escenarios del mock, decisiones y verificación.
