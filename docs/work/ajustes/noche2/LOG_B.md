# LOG_B — Pista B

## Estado actual
- Hecho: contrato v1.3; B1 API insights; B2 (ninguna candidata activada); B3 banda; B4 correcciones de ronda 1 (commit 27afbd3; suite 770 passed, 12 skipped, 1 christmas).
- En curso: ronda 2 COMPLETA (por el ALTO A1 de la ronda 1): 8 angulos separados + regresion, sonnet, contra `git diff main...HEAD -- src tests scripts`. Lanzados en 2 tandas de 4 + regresion.
- Falta: leer resultados; si sin critico/alto -> corregir medios/bajos y UNA verificacion estandar; REPORTE_B.md; commit final.

## Decisiones tomadas sin el usuario
- Prefijo `/adjustments/insights/*` con router propio (sin conflicto con `/{run_id}`); endpoints dependientes de semana usan `run_id` (el demo persiste con Postgres).
- Historial de pronósticos emitidos para precisión/banda = curvas k1 versionadas compactas en `src/adjustments/precision_data/` (el demo no tiene `reports/`).

## Bitácora
- 2026-11-14 contrato publicado.
- 2026-11-14 B1, B2, B3 terminados y commiteados (b00a20a + commit de B3). La Pista A commitea sus docs en esta misma rama (docs only).
- B2 no pudo escribir REPORTE.md (herramienta rechaza); el coordinador lo consolida en REPORTE_B.md.
