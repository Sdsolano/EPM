# LOG demo (D0 + D1)

## Estado actual
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
