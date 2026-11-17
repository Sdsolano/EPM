# Estado del despliegue de la demo (2026-11-13/14) — leer primero

## Hecho
- Fase 4 cerrada en main (RESUMEN_NOCHE.md). Demo D0-D4 completa: backend (`src/api/demo.py`), front (Sphere `test` 8a6302c, `npm run build:demo`), Dockerfile/compose/railway.json, README_DESPLIEGUE.md, revision unica (1 agente) y correcciones (commit 2cd04c1: sin `DEMO_PASSWORD` la app NO arranca salvo `DEMO_ALLOW_NOAUTH=1`; `demo_build.sh` aborta si Sphere esta sucio).
- **DESPLEGADO en Railway** (cuenta del usuario, workspace personal): proyecto `epm-demo-ajustes`, entorno `production`, servicios `app` + `Postgres`, dominio publico generado (consultar con `railway status` / `railway domain --service app`). Verificado por HTTP: /health 200, /api/demo/ucps 401 sin clave y 200 con clave (Antioquia, Atlantico), /adjustments/aprendizaje 200, front 200. NO se verifico visualmente en navegador ni un calculo de ajustes en la nube (en local: 13 s Antioquia, 33 s Atlantico).
- Variables del servicio `app`: `DEMO_MODE=1`, `DEMO_PASSWORD` (generada; la copia local esta en `.demo_password.local`, ignorado por git; cambiar en Railway si se quiere), `DATABASE_URL=${{Postgres.DATABASE_URL}}`, `API_KEY` (OpenAI, tomada del `.env` local). Nunca van a git.

## Como redesplegar (importante)
`railway up --no-gitignore` desde la raiz de EPM FALLA con 413 (sube .venv, ~342 MB). Se despliega un contexto minimo:
```bash
bash scripts/demo_build.sh                      # genera static_demo/ y demo_snapshot/ (no versionados)
S=/tmp/demo_ctx && rm -rf $S && mkdir $S && cp -R src config demo_snapshot static_demo Dockerfile requirements.txt railway.json $S/
find $S -name __pycache__ -prune -exec rm -rf {} +
cd $S && railway link -p <project-id> -e production -s app && railway up --no-gitignore --service app --ci
```
(`railway status` desde el repo muestra el project id). Pendiente de mejora: automatizar esto en `scripts/demo_build.sh` o un `.railwayignore`.

## Pendientes / riesgos conocidos
- Docker nunca se probo localmente (no instalado); Railway construyo el Dockerfile sin errores.
- Dominio publico con datos reales protegido solo por la contrasena compartida; para cerrarlo, quitar el dominio en Railway.
- Rutas de la API principal con escritura (`/update`, etc.) siguen alcanzables con la clave; opcion: allowlist de rutas en modo demo (hallazgo medio aceptado, entorno interno).
- Repo versiona CSV de datos y modelos (`data/raw/*.csv`, `incident/`, `string/`, `models/**`) desde antes de la demo: confirmar sensibilidad si el repo es publico.
- No existen `curvas_modelo_k1_*.csv` en esta maquina: la base de la demo es el modelo en CPU.
- El E2E local escribe en `data/features/{ucp}/`: correr la demo en local sobre una copia, no sobre el checkout.
- Fase 4: segmentos R2 `no_aplicar` solo se informan (decision de producto); backend intermedio debe aceptar `usuario` en aprobar; modulo sin registrar en BD de Sphere (ver RESUMEN_NOCHE.md).
- Tests: 4 fallos de entorno aceptados (christmas por locale; 3 async por falta de pytest-asyncio).
