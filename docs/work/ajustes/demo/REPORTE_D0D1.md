# Reporte D0 + D1 (snapshot y backend demo)

## Variables de entorno (exactas)
| Variable | Obligatoria | Efecto |
|---|---|---|
| `DEMO_MODE` | si (=1/true/yes) | Activa todo el modo demo. Sin ella, la API se comporta como antes. |
| `DATABASE_URL` | si | Postgres de Railway. Se copia a `AJUSTES_DATABASE_URL` si esta no existe. |
| `AJUSTES_DATABASE_URL` | no | DSN que usa el módulo (nombre ya existente); tiene prioridad sobre `DATABASE_URL`. |
| `DEMO_PASSWORD` | recomendada | Contraseña compartida. Sin ella NO hay auth (se avisa en el log). |
| `DEMO_STATIC_DIR` | no (default `static_demo`) | Carpeta del build del front (`index.html` dentro). |
| `API_KEY` | para alertas de eventos 4D | Clave de OpenAI: el código ya usa este nombre (no `OPENAI_API_KEY`). Solo por entorno. |
| `PORT` | Railway la inyecta | Usada por `python -m src.api.main`. Con uvicorn directo: `--port $PORT`. |
| `DEMO_REFRESH_DATA` | no | `1` permite refrescar datos desde la red (por defecto la demo es solo lectura). |
| `DEMO_SKIP_MIGRATIONS` | no | `1` omite migraciones al arrancar. |
| `AJUSTES_EVENTOS_CACHE_DIR` | no | Cache de eventos (existente). |
No definir `AJUSTES_SERVICE_TOKEN` en la demo (el front no manda esa cabecera); la protección es `DEMO_PASSWORD`.

## D0 - snapshot
- `scripts/demo_snapshot.py` -> `demo_snapshot/` (en .gitignore, igual que `static_demo/`). Layout de repo:
  `data/raw/{Antioquia,Atlantico}/*`, `data/features/{ucp}/data_with_features_latest.csv`, `models/{ucp}/` completo,
  `models/{hourly,special_days}_disaggregator.pkl`, `models/registry/{champion_model.joblib,registry_metadata.json}`,
  `reports/ajustes/curvas_modelo_k1_{ucp}.csv` (solo si existen) y `MANIFEST.json` (ruta, bytes, sha256).
- Tamaño medido: 44.8 MB, 24 archivos (data 50M / models 101M / reports 37M en el repo completo) -> cabe en la imagen; no hace falta volumen.
- Las curvas `curvas_modelo_k1_*.csv` NO existen en esta máquina ni en otro worktree: el snapshot sale sin ellas (aviso en consola y manifiesto). La base del pronóstico en la demo es el modelo real (champion en CPU); si se quieren curvas precalculadas hay que generarlas con `scripts/backtest_modelo.py` en el PC que las tenga y re-correr el snapshot.
- Verificación: `scripts/demo_snapshot.py --verify [--dest dir]`.
- Datos reales: NO a git. Dockerfile (D3): `COPY demo_snapshot/ /app/` desde contexto de build local (construir la imagen en local con `railway up` desde el directorio, o subir al volumen). El `.dockerignore` no debe excluir `demo_snapshot/` pero git sí.

## D1 - backend
- `src/api/demo.py` (`install_demo`, invocado al final de `main.py` solo si `DEMO_MODE`), `src/adjustments/migrations.py`.
- Migraciones `src/adjustments/sql/001_init.sql` y `002_versionado.sql` (las únicas que existen; la ruta real NO es `sql/`) se aplican al arrancar, en orden, idempotentes, con advisory lock y 15 reintentos de conexión (30 s). Si fallan, se registra el error y el servicio arranca igual.
- Auth: HTTP Basic (cualquier usuario, clave = `DEMO_PASSWORD`, comparación en tiempo constante) o cabecera `X-Demo-Password`. El navegador pregunta una vez y reenvía credenciales en todas las peticiones same-origin; el front no necesita pantalla de login. `/health` exento. Respuesta 401 con `WWW-Authenticate`.
- Estáticos: SPA catch-all registrada al final: sirve archivos reales del build; para rutas sin extensión con `Accept: text/html` devuelve `index.html`; el resto 404. `/` pasa a ser la SPA (se retira el JSON informativo de `/` solo en modo demo). Protegido contra path traversal. Las rutas de la API (`/adjustments/*`, `/predict`, `/health`, `/docs`, ...) no cambian.
- Modo solo lectura: en demo se desactiva el refrescador de datos externo (`full_update_csv`), así no hay red ni escritura sobre `data/raw`. Caveat: el flujo de pronóstico del modelo (`run_predict_flow`) puede escribir archivos temporales bajo `data/features/{ucp}/` del contenedor (efímero; ya lo hacía antes).
- `main.py`: `__main__` usa `PORT` (default 8000). Sin otros cambios; algoritmos y defaults del módulo intactos.
- El front debe llamar a la API en el mismo origen con rutas `/adjustments/...` (no hay prefijo `/api` para ajustes).

## Arranque local
```
.venv/bin/python scripts/demo_snapshot.py --dest demo_snapshot      # opcional en local (ya hay data/ y models/)
createdb demo_local
DEMO_MODE=1 DEMO_PASSWORD=algo DATABASE_URL=postgresql://localhost/demo_local \
  DEMO_STATIC_DIR=/ruta/al/dist PORT=8000 API_KEY=<openai, opcional> .venv/bin/python -m src.api.main
curl localhost:8000/health ; curl -u x:algo localhost:8000/adjustments/aprendizaje?ucp=Antioquia
```
En contenedor (cwd con data/ y models/): `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`.

## Probado / no probado
- Probado: `tests/test_demo_mode.py` (13 tests: flag, auth Basic/cabecera/exentos, DSN, refresco opt-in, estáticos + SPA + traversal, sin-estáticos, main sin DEMO_MODE intacto, snapshot+hashes, migraciones idempotentes y arranque con Postgres local temporal; estos 2 se saltan sin `AJUSTES_DATABASE_URL`). Ejecución real con uvicorn contra Postgres local: arranque, tablas `ajustes.*` creadas, `/health` 200, 401 sin clave, SPA en `/`, `/docs` 200, `/adjustments/aprendizaje` 200.
- Suite completa e import de `src.api.main`: ver LOG.md.
- NO probado: corrida completa de ajustes con el modelo real dentro de la demo (D4), Docker, Railway, llamada real a OpenAI, front real (D2).
- Pendiente para D2/D3: el front debe usar mismo origen; Dockerfile debe copiar `demo_snapshot/` y el build del front a `DEMO_STATIC_DIR`; healthcheck `/health`.
