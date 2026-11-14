# LOG_B — Pista B

## Estado actual
- Hecho: lectura de contexto; CONTRATO_B.md v1 y PLAN_B.md escritos.
- En curso: lanzar subagentes B1 (API insights) y B2 (backtest de mejoras).
- Falta: implementación, integración, revisión ronda 1, REPORTE_B.md.
- Sin commitear: nada tras el commit del contrato.
- Verificar: `C:/Users/samue/OneDrive/Documentos/GitHub/EPM/.venv/Scripts/python.exe -m pytest tests -q --ignore=tests/test_hourly_disaggregation.py -p no:cacheprovider` (fallos aceptados: christmas + 3 async).

## Decisiones tomadas sin el usuario
- Prefijo `/adjustments/insights/*` con router propio (sin conflicto con `/{run_id}`); endpoints dependientes de semana usan `run_id` (el demo persiste con Postgres).
- Historial de pronósticos emitidos para precisión/banda = curvas k1 versionadas compactas en `src/adjustments/precision_data/` (el demo no tiene `reports/`).

## Bitácora
- 2026-11-14 contrato publicado.
