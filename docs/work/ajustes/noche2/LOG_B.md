# LOG_B — Pista B

## Estado actual
- Hecho: contrato v1 (+v1.1, v1.2 en Cambios); B1 API insights (5 endpoints, tests); B2 investigacion de mejoras (ninguna candidata cumple la regla; bandas empiricas escala 1.2 medidas; correccion.py opt-in); B3 banda calibrada integrada. Suite: 755 passed, 12 skipped, 1 failed (christmas).
- En curso: ronda 1 de revision (8 angulos + regresion, sonnet) contra `git diff main...HEAD` (solo archivos de la pista B; commits de docs de la Pista A tambien estan en la rama).
- Falta: corregir hallazgos, verificacion estandar, REPORTE_B.md, commit final.
- Verificar: pytest suite (ver arriba).

## Decisiones tomadas sin el usuario
- Prefijo `/adjustments/insights/*` con router propio (sin conflicto con `/{run_id}`); endpoints dependientes de semana usan `run_id` (el demo persiste con Postgres).
- Historial de pronósticos emitidos para precisión/banda = curvas k1 versionadas compactas en `src/adjustments/precision_data/` (el demo no tiene `reports/`).

## Bitácora
- 2026-11-14 contrato publicado.
- 2026-11-14 B1, B2, B3 terminados y commiteados (b00a20a + commit de B3). La Pista A commitea sus docs en esta misma rama (docs only).
- B2 no pudo escribir REPORTE.md (herramienta rechaza); el coordinador lo consolida en REPORTE_B.md.
