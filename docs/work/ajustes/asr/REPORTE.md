# Reporte — fase ASR (comparación ASR vs mesa vs real)

Commit: `0e8de2e` (rama `feat/ajustes-asr-comparacion`). Detalle completo: `LOG.md` de esta carpeta y
`reports/ajustes/asr_comparacion_20261112.md` (incluye la guía de 1 página para la sesión del viernes).

## Resultados (WAPE horario %, mismos días para todas las series)
| base / UCP | base | mesa | ASR (escenario `anual`) |
|---|---|---|---|
| ingenuo / Antioquia (364 días) | 4.05 | 3.71 | 5.26 |
| ingenuo / Atlántico (231 días) | 6.36 | 6.00 | 8.24 |
| archivo / Antioquia (147 días) | 3.03 | 2.79 | 5.42 |
El ASR mejora la base solo en 26–33 % de los días; la mesa en 54–61 %.

## Diagnóstico
1. El ASR = h1 + (h2 − h3): el pronóstico base se cancela; no lo usa.
2. Su corrección diaria casi no correlaciona con la necesaria (−0.02 a 0.25): es ruido de días sueltos.
3. Festivos/Semana Santa en h1/h2/h3 lo disparan (39.6 % WAPE el 2026-01-12; 22–27 % post Semana Santa).
4. Bug de orden en el front: h1 se ordena por fecha pero `desvi2` se congela en orden de clic → sesgo −4 a −9 %.
5. `tipo_dia` fijo y el validador del cluster NO son la causa (≤ 0.04 pp).

## Recomendación
Reemplazar el ASR por la mesa (Fase 3). Paliativo: marcarlo como no recomendado o que el botón llame a
`/adjustments/preview`.

## Decisiones sin el usuario
Selección de históricos supuesta (5 escenarios; principal `anual`); festivos de la librería en el histórico;
`base: "modelo"` no permitido en el endpoint (422); corte = último dato real ≤ sábado previo.

## Pendientes
Datos al día para el caso real del viernes (el histórico local llega a 2026-07); validar con el analista su
selección real de históricos; base modelo (Fase 2a).
