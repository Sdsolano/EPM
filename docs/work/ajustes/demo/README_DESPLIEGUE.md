# Despliegue de la demo (EPM + Sphere) en Railway

La demo es un solo servicio (FastAPI de EPM sirviendo el build del front) + el plugin Postgres de Railway.
Es un entorno INTERNO con datos reales: `static_demo/` y `demo_snapshot/` NO van a git (EPM es repo publico).

## 1. Armar el contexto de build (en tu Mac/PC, una vez por version)
Requisitos: node 24 (`/Users/sdsolano/.local/tools/node/bin`), `.venv` de EPM, Sphere en `../Sphere` (rama `test`).
```
cd EPM
SPHERE_DIR=../Sphere scripts/demo_build.sh
```
Hace: `npm ci && npm run build:demo` en Sphere -> copia `dist/` a `static_demo/`; `scripts/demo_snapshot.py` -> `demo_snapshot/`
(data/raw, features, models, ~45 MB) y verifica sus hashes. El `Dockerfile` los copia (`demo_snapshot/` a `/app/`, `static_demo/`).
Opcional: si tienes `reports/ajustes/curvas_modelo_k1_{ucp}.csv`, genera antes el snapshot (se incluyen solos). Sin ellas la base es el modelo real.

## 2. Probar local con Docker (opcional)
```
DEMO_PASSWORD=algo docker compose up --build      # http://localhost:8000
```
(app + Postgres 16; `API_KEY` opcional para las alertas de eventos con IA).

## 3. Railway (acciones del usuario)
1. `railway login` y crear proyecto: `railway init` (o desde la web).
2. Agregar plugin **PostgreSQL** al proyecto (Railway crea `DATABASE_URL`).
3. Crear el servicio de la app (vacio) y en **Variables**:
   | Variable | Valor |
   |---|---|
   | `DEMO_MODE` | `1` |
   | `DEMO_PASSWORD` | la clave compartida que daras al equipo |
   | `DATABASE_URL` | referencia al plugin: `${{Postgres.DATABASE_URL}}` |
   | `API_KEY` | clave de OpenAI (opcional; sin ella las alertas de eventos IA se omiten en silencio) |
   | `DEMO_REFRESH_DATA` / `DEMO_SKIP_MIGRATIONS` / `DEMO_ALLOW_NOAUTH` | opcionales; no definir en Railway |
   | `DEMO_STATIC_DIR` | `/app/static_demo` (ya es el ENV por defecto del Dockerfile) |
   `PORT` lo inyecta Railway. No definas `AJUSTES_SERVICE_TOKEN`.
4. Desde `EPM/` (con `static_demo/` y `demo_snapshot/` ya generados): `railway up --no-gitignore` (usa `Dockerfile` y `railway.json`;
   sube el contexto local, incluidas las carpetas ignoradas por git; el `.dockerignore` NO las excluye).
   Si `railway up` respeta `.gitignore` en tu version del CLI y omite esas carpetas, usa `railway up --no-gitignore` o
   construye la imagen local y publicala (`docker build` + registro).
5. Healthcheck `/health` (en `railway.json`). Al arrancar la app aplica las migraciones SQL de `ajustes.*` sola.
6. Generar dominio publico del servicio y entrar con cualquier nombre + la `DEMO_PASSWORD`.

## 4. Notas
- Acceso: pantalla propia del front; la clave se valida contra `/api/demo/ucps`. Sin `DEMO_PASSWORD` la app NO arranca (salvo `DEMO_ALLOW_NOAUTH=1`, solo para desarrollo local).
- Solo lectura sobre datos: la demo no refresca datos ni entrena. Para actualizar datos: re-generar snapshot y `railway up`.
- El calculo de una semana tarda ~ver REPORTE_D3D4.md; CPU del contenedor puede ser mas lento.
- Docker NO se probo en la maquina de desarrollo (no instalado): Dockerfile, compose y railway.json revisados a mano; la prueba E2E se hizo con uvicorn + Postgres local (ver REPORTE_D3D4.md).
