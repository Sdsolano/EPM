# Sub-plan D0 + D1 (snapshot de datos y backend demo)

Plan maestro: `PLAN.md` (esta carpeta). Reglas: `../REGLAS_AGENTES.md`.

## Hallazgos del código (base de las decisiones)
- Rutas de datos/modelos son relativas al cwd (`data/raw/{ucp}`, `data/features/{ucp}`, `models/{ucp}`, `models/registry`):
  el snapshot replica ese layout y se copia a `/app` en la imagen.
- Variables existentes: `AJUSTES_DATABASE_URL` (Postgres del módulo), `AJUSTES_SERVICE_TOKEN` (opcional),
  `API_KEY` (OpenAI; eventos 4D y análisis), `AJUSTES_EVENTOS_CACHE_DIR`, `PORT` no se usaba.
- Migraciones en `src/adjustments/sql/` (no `sql/`): 001_init.sql y 002_versionado.sql, idempotentes.
- `reports/ajustes/curvas_modelo_k1_*.csv` NO existen en esta máquina (ni worktree 2a): el script las incluye si aparecen.

## D0 - scripts/demo_snapshot.py
Arma `demo_snapshot/` (en .gitignore) con layout de repo: data/raw/{Antioquia,Atlantico}/{datos,clima_new}.csv, data/raw/clima*.csv,
data/features/{ucp}/data_with_features_latest.csv, models/ necesarios (por ucp + registry/champion + disaggregators),
reports/ajustes/curvas_modelo_k1_*.csv (si existen). Escribe `MANIFEST.json` (rutas, bytes, sha256) y verifica. NO se versiona
(repo público); el Dockerfile (D3) lo toma de un contexto de build local no versionado (`COPY demo_snapshot/ /app/`).

## D1 - backend demo (todo tras DEMO_MODE=1)
- `src/api/demo.py::install_demo(app)`, invocado al final de main.py solo si `DEMO_MODE` activo.
- Migraciones al arrancar (`src/adjustments/migrations.py`, advisory lock, reintentos, idempotente).
- `DATABASE_URL` -> `AJUSTES_DATABASE_URL` si esta última no está definida.
- Auth mínima: HTTP Basic (cualquier usuario) con `DEMO_PASSWORD`; sin variable, sin auth; `/health` exento.
- Estáticos: `DEMO_STATIC_DIR` + fallback SPA (solo GET sin extensión y Accept text/html); `/` pasa a servir la SPA.
- Sin refresco de datos externo (solo lectura) salvo `DEMO_REFRESH_DATA=1`.
- `$PORT` en el `__main__` de main.py (default 8000).
- Tests: `tests/test_demo_mode.py` (sin red/OpenAI; Postgres solo si hay uno local).
