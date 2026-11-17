# LOG demo (D0 + D1)

## Estado actual
- **2026-11-14: demo DESPLEGADA en Railway.** Ver `ESTADO_DESPLIEGUE.md` (lo primero que debe leer un agente que continue): como redesplegar, variables, pendientes.
- D0 y D1 TERMINADOS y verificados; commit en main (ver historial). Entrega: REPORTE_D0D1.md.
- Suite: 689 passed, 7 skipped, 4 failed; los 4 son los preexistentes permitidos (future_events, christmas, 2 de openai_analysis). `import src.api.main` OK.
- tests/test_demo_mode.py: 13 passed con AJUSTES_DATABASE_URL local (2 de Postgres se saltan sin ella).
- Siguiente: D2 (front), D3 (Dockerfile/railway.json), D4 (E2E). Ver PLAN.md.
- Verificar: `AJUSTES_DATABASE_URL=postgresql://localhost/postgres .venv/bin/python -m pytest tests/test_demo_mode.py -q -p no:cacheprovider`

## Historial
- Lectura de codigo; sub-plan (PLAN_D0D1.md).
- Implementado: src/adjustments/migrations.py, src/api/demo.py, hook en main.py (DEMO_MODE, PORT), scripts/demo_snapshot.py, .gitignore, tests.
- Smoke con uvicorn + Postgres local temporal: migraciones, 401/200, SPA, /docs OK (base borrada).
- Decisiones sin el usuario: auth = HTTP Basic (la mas simple, sin login en el front); OpenAI var = API_KEY (nombre existente); migraciones viven en src/adjustments/sql; demo sin refresco de red por defecto; curvas k1 no existen en esta maquina (snapshot sin ellas).

## D3 + D4 (2026-11-13)
- Hecho: reconciliacion contrato (401 sin WWW-Authenticate, /api/demo/ucps y /reales, publicos GET / y estaticos), no-op de full_update_csv en demo,
  Dockerfile/.dockerignore/railway.json/docker-compose/scripts/demo_build.sh/README_DESPLEGUE, E2E local OK (ver REPORTE_D3D4.md).
- Falta (usuario): probar Docker, `railway up`, eventos con key real, revision visual.
