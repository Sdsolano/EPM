# Agente 2 — sub-plan: `scripts/backtest_mesa_vs_jamer.py` (parte C del PLAN de festivos_r1)

Leer antes: `../PLAN.md` (sección C y "Criterio de aceptación"), `../LOG.md`, `../../REGLAS_AGENTES.md`.
No toca `src/adjustments/*` (dueño: Agente 1). Sin commit. Repo público: el .md versionable solo trae agregados.

## Entradas (CLI, nada hardcodeado; viven fuera del repo)
`--historico` (FECHA,P1..P24), `--festivos` (fecha,nombre), `--envios` (sesion,envio,version,fecha,P1..P24).
Opcionales: `--desde` (lunes mínimo de la parte 1; default: todas las semanas con VF), `--partes 1,2`,
`--seed` (default 7), `--desde-ampliada` (default 2022-01-01), `--n-control` (default 40),
`--salida` (md, default `reports/ajustes/`), `--salida-detalle` (CSV, default `reports/ajustes/jamer/`;
`/reports` ya está en `.gitignore`, y el .md se versiona con `git add -f` como los demás reportes).

## Configs (data-driven, extra params a nivel Y forma)
`jamer`, `jamer+festivo_por_tipo`, `jamer+r1_limpia`, `jamer+saltar_puente` (nivel y forma aceptan el flag hoy),
`jamer+ambas` (festivo_por_tipo + r1_limpia), sensibilidad `jamer+r1_limpia_u04` / `_u08`
(`r1_umbral_atipico` 0.04/0.08, solo informativa). Si un paso rechaza una clave (ValueError "params desconocidos")
la config se salta con aviso (funciona antes y después del cambio del Agente 1).

## Parte 1 — vs Jamer
Semana = lunes de la sesión; envíos con fecha ≤ domingo. VF = primer envío, final = último envío; corte = envío − 1;
Mesa con base `ingenuo`, histórico recortado ≤ corte (assert sin fuga). VF se evalúa en todos los días reales de
la semana; final en los días > fecha de envío. Series: `jamer_envio` + configs. Métricas: MAPE h, |err día|,
sesgo energía, p90 |err día|; por tipo de día (+ post_festivo) y por lead; pareado por semana (bootstrap IC95,
semilla) vs `jamer` y vs `jamer_envio`; partición temporal 60/40 (calibración / prueba).

## Parte 2 — ampliada contra real (sin Jamer), 2022-01..último dato
Semanas: (a) con algún festivo, (b) con Semana Santa o con algún día cuyo R1 (de `jamer`) saltaría `r1_limpia`
(clasificador propio, independiente del código del Agente 1: Semana Santa Ramos–Pascua, 24/dic–6/ene,
|E/mediana(4 mismos-dow normales previos) − 1| > 6 %), (c) control: muestra aleatoria con semilla de semanas
normales. Corte = lunes − 6 (martes; emisión miércoles). Segmentos: festivo por `tipo_festivo`
(`data.tipo_festivo` si existe; si no, fallback lunes/semana_santa/fijo), semanas de Semana Santa, días con R1
saltada, control.

## Criterio (PLAN "Criterio de aceptación"), por regla
1. Mejora MAPE h en su segmento (festivo_por_tipo → festivos; r1_limpia → días con R1 saltada; saltar_puente →
   días cuyo R1 cambia; ambas → festivos ∪ R1 saltada) en la parte 2 **y** en las semanas de prueba (parte 1, VF).
   Sin días del segmento en prueba → NO PASA (no evaluable).
2. Parte 1 VF todas las semanas, pareado config − jamer: media ≤ 0 y el IC95 no es enteramente > 0.
3. Parte 1 VF: ningún tipo de día (laboral/sabado/domingo/festivo) empeora > 0,1 pp.

## Archivos
- `scripts/backtest_mesa_vs_jamer.py` (funciones puras + main).
- `tests/test_backtest_mesa_vs_jamer.py` (sintético con `tests/ajustes_sinteticos.py`; sin red ni datos reales).
- `REPORTE.md` en esta carpeta al terminar; línea del Agente 2 en `../LOG.md`.

## Hecho / en curso / falta
- [x] sub-plan
- [x] script
- [x] tests (22, pasan)
- [x] corrida real end-to-end (229 s; ver REPORTE.md)
- [x] suite completa, REPORTE, LOG
