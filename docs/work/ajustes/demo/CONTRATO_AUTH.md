# Contrato front demo (Sphere) <-> backend demo (EPM FastAPI)

Autor: D2 (front). Revisado contra `PLAN_D0D1.md` de D1 (auth HTTP Basic con `DEMO_PASSWORD`, estáticos con
`DEMO_STATIC_DIR`). El front se adaptó a D1; este documento fija lo que el front asume y lo que le falta al backend.

## Autenticación (HTTP Basic, contraseña compartida `DEMO_PASSWORD`)
- El front tiene su propia pantalla de acceso (nombre + contraseña). NO hay endpoint de login: la contraseña se valida
  con una llamada real protegida (`GET /api/demo/ucps`) mandando `Authorization: Basic base64("demo:<contraseña>")`
  (usuario fijo `demo`; el backend debe ignorar el usuario). 200 = entra; 401 = "Contraseña incorrecta".
- La credencial se guarda en `sessionStorage` y se manda en TODAS las llamadas a `/adjustments/*` y `/api/demo/*`.
  Ante cualquier 401 el front la borra y vuelve a la pantalla de acceso.
- **REQUISITO para D1: las respuestas 401 de `/adjustments/*` y `/api/demo/*` NO deben llevar la cabecera
  `WWW-Authenticate`** (si la llevan, el navegador abre su diálogo nativo de contraseña encima de la pantalla de acceso).
  Mejor aún: el 401 de la demo sin esa cabecera en todas las rutas de API; solo `/health` queda exento.
- **REQUISITO para D1: `GET /` y los estáticos (`/assets/*`, `/vite.svg`, `*.js`, `*.css`, imágenes) deben ser públicos**
  (sin Basic): la pantalla de acceso tiene que poder cargarse antes de tener credencial. Si todo el sitio exigiera Basic,
  el navegador mostraría su diálogo nativo y la pantalla de acceso del front sobraría.
- Tipo de router del front: **hash** (`/#/acceso`, `/#/`): no necesita fallback SPA (basta `GET /` -> index.html).

## Rutas propias que el front necesita y que D1 aún no tiene (a cargo de D1)
Respuestas en JSON plano (**sin** envoltorio `{success,data}`), protegidas con Basic como el resto:
- `GET /api/demo/ucps` -> `[ { "ucp": "Antioquia", "nombre": "Antioquia" }, { "ucp": "Atlantico", "nombre": "Atlántico" } ]`
  (carpetas de `data/raw/` con `datos.csv` y modelo entrenado; `nombre` es el rótulo a mostrar).
- `GET /api/demo/reales?ucp=Antioquia&desde=2026-11-09&hasta=2026-11-22` -> `{ "YYYY-MM-DD": [24 MW], ... }` solo con los
  días que tengan 24 periodos observados en `datos.csv` (puede ser `{}`; el front lo usa para dibujar la serie real).
- No hay "pronóstico emitido": el front no lo pide (base = modelo/curvas del servidor; `base_curvas = null`).

## `/adjustments/*` tal cual EPM (el front ya lo cumple)
- `POST /adjustments/preview` y `/run` envían `aplicar_automatico: true`; `/run` y `PATCH .../steps/{paso}` envían `usuario`
  (nombre del analista); `POST .../approve` envía `usuario` (obligatorio) + `version_esperada` + `motivo` + `excepciones_justificadas`.
- Errores: el front lee `detail` (string, objeto con `mensaje`, o lista de errores de validación de pydantic). 409 = versión
  vieja (recarga), 422 = bloqueos/datos inválidos, 401 = sesión inválida.
- Pasos válidos en la URL: `base, referencia, nivel, forma, especiales, control`.
- `meta.extras.modos_paso`, `meta.extras.segmentos_paso`, alertas `EVENTOS_SEMANA` / `ELECCIONES_IA` (no verificadas) se muestran.

## Build del front
`npm ci && npm run build:demo` en Sphere (rama `test`) -> `dist/` (apuntar `DEMO_STATIC_DIR` allí). Variables: REPORTE_D2.md.
