# Entorno de demo autocontenido (EPM + front Sphere) para Railway

## Contexto y decisiones del usuario
Mostrar el avance del módulo de ajustes (pasos, Mesa de ajustes, resultados de backtest, antes/después) en un entorno de test individual, con funcionalidades completas, autocontenido (archivos internos) y desplegable en Railway.
- Es INTERNO (no público): datos reales de las UCP permitidos; la API key de OpenAI puede usarse (alertas de eventos 4D activas). Aun así: nada de secretos en git, variables de entorno en Railway.
- Depende de que Fase 4 esté integrada en main (ver LOG.md "ESTADO ACTUAL v3"). Si aún no lo está, el agente termina primero esa integración.

## Diseño
Un servicio en Railway + plugin Postgres.
- Backend: FastAPI de EPM (`src/api/main.py`) con el router de ajustes y los 6 pasos. Corridas persistidas en Postgres (`DATABASE_URL`; esquema en `sql/001_init.sql`, `002_versionado.sql`; aplicarlas al arrancar si faltan).
- Front: build estático de Sphere (repo Sphere, rama `test`, `npx vite build`), servido por el mismo FastAPI (StaticFiles) o contenedor nginx aparte. Perfil "demo": sin login real del backend intermedio (auth mínima o una contraseña compartida), menú con la Mesa de ajustes y una pantalla "Avance". Fallback: `VITE_AJUSTES_MOCK`.
- Datos: snapshot de `data/raw/{ucp}` y los `models/` necesarios dentro de la imagen o en un volumen de Railway (medir tamaño primero: si > ~1–2 GB, volumen). Solo lectura: la demo no entrena ni escribe sobre ellos. Base de pronóstico: modelo real si es viable en CPU del contenedor; si no, curvas precalculadas (`reports/ajustes/curvas_modelo_k1_*.csv`, fuera de git: copiarlas al snapshot).
- Pantalla "Avance": tabla antes/después por fase (ASR, 2a, 2b, re-backtest, Fase 4) leída de `reports/ajustes/*.md|csv` y `docs/work/ajustes/RESUMEN_NOCHE.md`.
- Despliegue: Dockerfile multi-stage (build del front + imagen Python), `railway.json`/variables, healthcheck `/health`, puerto `$PORT`.

## Partes (agentes separados, sub-plan propio en docs/work/ajustes/demo/)
- D0: medir tamaño de data/ y models/, decidir imagen vs volumen; script `scripts/demo_snapshot.py` que arma el snapshot reproducible.
- D1: backend demo (arranque, migraciones SQL, StaticFiles, endpoint de avance, auth mínima), tests.
- D2: front demo en Sphere (perfil demo, pantalla Avance, apuntar a la API del mismo origen), `vite build`, tsc sin sumar errores.
- D3: Dockerfile, railway.json, variables (`DATABASE_URL`, `OPENAI_API_KEY` / nombre que use el código, contraseña demo), README de despliegue; prueba local con docker compose (app + postgres).
- D4: verificación E2E local (corrida completa: calcular → pasos → resumen → aprobar; 6 pasos; eventos con y sin key) y reporte.
No ejecutar el despliegue real en Railway sin confirmación del usuario (cuenta/credenciales son suyas); dejar todo listo y documentado.

## Criterio de terminado
`docker compose up` levanta app+postgres y la demo funciona de punta a punta; documentación en `docs/work/ajustes/demo/README_DESPLIEGUE.md` con los pasos exactos de Railway; commits en main (EPM) y `test` (Sphere).
