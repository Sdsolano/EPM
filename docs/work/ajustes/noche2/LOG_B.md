# LOG_B — Pista B

## Estado actual
- Hecho: contrato v1.2; B1 API insights; B2 investigacion (ninguna candidata activada); B3 banda calibrada. Commits b00a20a, f9352e2. Ronda 1 de revision COMPLETA (8 angulos + regresion + medicion de cobertura): regresion limpia (755 passed, 12 skipped, 1 christmas).
- Hallazgos ronda 1: ALTO A1 (recomendaciones de sesgo prometen mejora que el backtest rolling NO confirma: Antioquia 2.63->2.58, Atlantico 5.22->5.54 peor); MEDIOS: nivel recomendado contradice el sesgo cuando final!=base, cobertura declarada inflada (medida real: Antioquia 87.6 %, Atlantico 80.2 % prueba, escala 1.2; sin escala 81.3/73.2), cache/event-loop del historico, list_runs 500; BAJOS: fuga banda con run_id (corte = semana-1), n_semanas_historial ignora filtros, ELECCIONES_IA sin texto, version null, duplicacion de helpers, codigo redundante, A3 historial CSV estatico (documentar).
- En curso: agente de correccion B4. Al terminar: por el ALTO, repetir ronda COMPLETA (8 angulos + regresion) contra el diff actual.
- Falta: ronda 2, REPORTE_B.md, commit final.

## Decisiones tomadas sin el usuario
- Prefijo `/adjustments/insights/*` con router propio (sin conflicto con `/{run_id}`); endpoints dependientes de semana usan `run_id` (el demo persiste con Postgres).
- Historial de pronósticos emitidos para precisión/banda = curvas k1 versionadas compactas en `src/adjustments/precision_data/` (el demo no tiene `reports/`).

## Bitácora
- 2026-11-14 contrato publicado.
- 2026-11-14 B1, B2, B3 terminados y commiteados (b00a20a + commit de B3). La Pista A commitea sus docs en esta misma rama (docs only).
- B2 no pudo escribir REPORTE.md (herramienta rechaza); el coordinador lo consolida en REPORTE_B.md.
