# LOG_B — Pista B

## Estado actual
- PISTA B TERMINADA. B1-B6 hechos; 3 rondas de revision (R3 sin criticos/altos); verificacion estandar: 783 passed, 12 skipped, 1 christmas; import API OK. Reporte: REPORTE_B.md. Contrato v1.5.
- Falta: nada de la pista; el orquestador integra (rebase/merge, demo_build, redeploy).

## Decisiones tomadas sin el usuario
- Prefijo `/adjustments/insights/*` con router propio (sin conflicto con `/{run_id}`); endpoints dependientes de semana usan `run_id` (el demo persiste con Postgres).
- Historial de pronósticos emitidos para precisión/banda = curvas k1 versionadas compactas en `src/adjustments/precision_data/` (el demo no tiene `reports/`).

## Bitácora
- 2026-11-14 contrato publicado.
- 2026-11-14 B1, B2, B3 terminados y commiteados (b00a20a + commit de B3). La Pista A commitea sus docs en esta misma rama (docs only).
- B2 no pudo escribir REPORTE.md (herramienta rechaza); el coordinador lo consolida en REPORTE_B.md.
