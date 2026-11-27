# REPORTE agente de implementación — post_festivo_factor

- Flag `post_festivo_factor` (default off) + `post_festivo_fuente` (homologo|recientes|combinado). Off = salida idéntica.
- Suite: 1094 passed / 16 skipped / 1 failed (preexistente permitido).
- Caso 3/nov/2026 (corte 27/oct): jamer −5,51 %; pf_homologo +0,77 % (F 0,9979); pf_recientes −3,45 % (F 0,9561);
  pf_combinado −3,36 % (F 0,9570). Jamer envío −1,04 %.
- Backtest: `--conjunto post_festivo` (default) con base `jamer` actual + 3 variantes; elección de variante solo con
  parte 2 2022–2024; `--conjunto festivos_r1` reproduce la fase anterior.
- Decisiones: festivo previo siempre d−1; recientes sin vecino normal se descartan; 2020 no se filtra; sin datos de
  calibración no se elige variante.
- Archivos: referencia_reciente.py, nivel.py, forma.py, pipeline.py, backtest_mesa_vs_jamer.py, CONTRATO_BACKEND.md,
  tests/test_adjustments_post_festivo.py (34), tests/test_backtest_mesa_vs_jamer.py (+6), ajustes menores de asserts.
