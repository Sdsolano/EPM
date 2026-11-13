# REPORTE D2 — front demo (Sphere, rama `test`, commit 8a6302c)

## Cambios
- **Perfil demo**: `src/main.tsx` (bifurca por `VITE_DEMO`), `src/AppProduccion.tsx` (app normal, igual que antes), `src/demo/`
  (`DemoApp` router por hash, `Acceso` con nombre + contraseña, `DemoLayout` barra superior limpia con solo "Mesa de ajustes",
  `sesion.ts`, `cliente.ts`, `demo.module.css`), `src/constants/demo.ts`, `.env.demo`.
- **Provider**: `src/providers/ajustes/demo.ts` (EPM directo: `/adjustments/*`, `/api/demo/ucps|reales`); `index.ts` elige demo > mock > real.
  `usuario` se envía en run, PATCH y aprobar (demo y normal). Demo envía `aplicar_automatico: true`.
- **6 pasos**: `especiales` en tipos (`types/ajustes.d.ts`), utils (orden, etiquetas, editable/omitible), formulario
  (aplicar, α, evidencia), diagnóstico por día, mock del simulador, riel.
- **R2** por tipo de día (nivel/forma) y **alertas de eventos/elecciones IA** marcadas "Sugerido por IA, no verificado".
- Sin interruptor de pronóstico emitido en demo; texto de la pantalla vacía "seis pasos"; panel de opciones del cálculo ya no
  aparece siempre abierto.
- Menú demo: no hay inyección de módulo ni etiquetas mock/dev; la barra muestra solo "Mesa de ajustes" (los demás módulos de
  Sphere dependen del backend intermedio y no existen en la demo).

## Variables de build (Vite)
| Variable | Valor demo | Uso |
|---|---|---|
| `VITE_DEMO` | `true` (en `.env.demo`) | activa el perfil demo |
| `VITE_DEMO_API_BASE` | vacío (mismo origen) | prefijo de la API EPM, p. ej. `https://host` |
| `VITE_AJUSTES_MOCK` | `false` | sin mock |
| `VITE_BASE` | `/` (default) o `./` | base de assets (vite `base`) |

## Cómo construir y servir
```
cd Sphere && git checkout test && npm ci && npm run build:demo   # salida en dist/
```
`dist/` va a `DEMO_STATIC_DIR` (o `COPY` en la imagen). El backend debe: servir `GET /` -> `dist/index.html` y `/assets/*` sin
autenticación; proteger `/adjustments/*` y `/api/demo/*` con Basic (`DEMO_PASSWORD`) devolviendo 401 SIN `WWW-Authenticate`;
y exponer `GET /api/demo/ucps` y `GET /api/demo/reales` (ver `CONTRATO_AUTH.md`). Node usado: 24 (`/Users/sdsolano/.local/tools/node/bin`).

## Verificación
`tsc -b` = 178 errores (base 178; 0 en archivos tocados), `vite build` y `build:demo` OK, eslint de lo tocado limpio, Sphere no tiene
tests. Navegador contra un stub (acceso mala/buena clave, crear corrida, especiales, alertas IA, aprobar con `usuario`).

## No verificado
- Contra el backend real de D1 (faltan `/api/demo/ucps`, `/api/demo/reales` y el comportamiento exacto del 401 al momento de escribir esto).
- Con datos reales (curvas, "reales" en gráficas, tiempos de cálculo de hasta ~3 min con modelo) ni con OpenAI real.
- Buscador de análogos para el paso `especiales` con ancla `analogos` (el backend busca automáticamente; el front no deja elegir días).
- Móvil (<640 px) del marco y de la pantalla de acceso solo revisado a 800 px.
- Que `omitir` de segmentos `no_aplicar` (R2) lo haga el front: solo se informa; el backend los aplica igual.
