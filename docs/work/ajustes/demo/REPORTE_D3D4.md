# Reporte D3 + D4 (despliegue y E2E local)

## Reconciliacion front/backend (tarea 1)
- Prefijo: el front D2 usa `/adjustments/*` (igual que el backend) y `/api/demo/*` solo para lo propio de la demo. Sin cambios en el front.
- `src/api/demo.py`: 401 SIN `WWW-Authenticate`; `GET /` y archivos reales del build son publicos; nuevos `GET /api/demo/ucps`
  (carpetas de data/raw con datos.csv y models/{ucp}; "Atlantico" se muestra "Atlántico") y `GET /api/demo/reales`
  (JSON plano {fecha: [24 MW]}, 404 UCP sin historico, 422 fechas/UCP invalidas).
- Hallazgo del E2E (corregido): el pronostico del modelo (`run_predict_flow`) llamaba a `full_update_csv` (API externa + escribe
  `data/raw`) aunque el router tuviera el refrescador apagado; fallaba por `locale es_ES.utf8` ausente (Mac y python-slim) y colgaba
  ~7 min en Atlantico. En demo ahora `full_update_csv` y `full_update_csv_diario` de main son no-op (salvo `DEMO_REFRESH_DATA=1`).
- La clave de OpenAI es `API_KEY` (nombre del codigo). Sin ella las alertas de eventos IA se omiten en silencio (corrida 201).

## D3 archivos
`Dockerfile` (2 etapas: venv de dependencias + runtime; copia src/, config/, `demo_snapshot/` a /app y `static_demo/`),
`.dockerignore`, `railway.json` (healthcheck /health), `docker-compose.yml` (app + postgres 16), `scripts/demo_build.sh`
(arma static_demo/ desde Sphere y demo_snapshot/), `README_DESPLIEGUE.md`. Decision: el build del front NO va dentro del Dockerfile
porque Sphere es otro repo; lo arma `demo_build.sh` antes. NO se pudo ejecutar Docker (no instalado): revision manual solamente.

## D4 E2E (uvicorn + Postgres 16 local temporal, DEMO_MODE=1, sin API_KEY)
- Migraciones al arrancar: 4 tablas `ajustes.*` creadas. `/health` 200.
- `/api/demo/ucps` sin clave 401 (sin WWW-Authenticate); con Basic 200 con 2 UCP. `/api/demo/reales` 200 (6 dias, 24 valores).
- Calcular `/adjustments/run` semana 2026-07-06: Antioquia 201 en 13.1 s; Atlantico 201 en 32.8 s; ambas con los 6 pasos
  (base, referencia, nivel, forma, especiales, control).
- Versiones (1), PATCH especiales omitido (v2), approve con `usuario` 200 y GET aprobada 200, `/adjustments/analogos` 200.
- Eventos con key: NO probado (no se llama a OpenAI real; cubierto por tests con cliente falso). Sin key: degrada sin error.
- Front: `npm run build:demo` OK (5.2 MB); busqueda de textos de desarrollo/avance en el bundle: ninguno. NO se hizo recorrido visual con
  navegador en esta pasada (solo verificado por D2 contra un stub).
- Apagados el servidor y Postgres temporal; BD y cluster borrados.

## Verificacion
- tsc -b Sphere: 178 errores (base 178). `vite build` OK.
- tests/test_demo_mode.py: 13 passed, 2 skipped (Postgres). Suite completa: 691 passed, 7 skipped, 4 failed (los 4 preexistentes permitidos). import src.api.main OK.

## Riesgos / pendientes del usuario
- El flujo del modelo ESCRIBE en `data/features/{ucp}/` (temp_*.csv y data_with_features_latest.csv). El E2E local los modifico y se
  restauraron con git checkout. En el contenedor es copia efimera de la imagen (no toca git ni el snapshot original); en local correr
  la demo sobre una copia del repo/snapshot, no sobre el checkout.
- Festivos: el flujo consulta `pronosticos.jmdatalabs.co` (festivos) si hay red; sin red depende del fallback del codigo (no probado offline).
- Faltan curvas `curvas_modelo_k1_*.csv`; la base es el modelo en CPU (13-33 s locales, puede ser mayor en Railway).
- Sin Docker ni Railway probados; el usuario ejecuta `scripts/demo_build.sh` y `railway up` (README_DESPLIEGUE.md).
- Tests de eventos con key real y revision visual en navegador pendientes.
