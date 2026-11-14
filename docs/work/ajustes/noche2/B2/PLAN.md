# B2 — Mejora de pronósticos con backtest sin fuga (noche 2)

Padre: `docs/work/ajustes/noche2/PLAN.md` y `PLAN_B.md` (parte B2). Reglas: `docs/work/ajustes/REGLAS_AGENTES.md`.

## Objetivo
Probar con rigor (sin fuga) candidatas que mejoren el pronóstico sobre la base `modelo_k1` (reentreno semanal, 52 semanas por UCP) y sobre
la salida final del pipeline (nivel + forma default). Solo se activa lo que mejora de forma consistente en calibración Y prueba en AMBAS UCP.

## Diseño experimental
- Datos: curvas k1 (solo lectura, repo EPM principal) + real/festivos (`data/raw`, solo lectura). Partición por UCP: primeras 60 % de las 52 semanas
  (31) = calibración (todo se elige ahí), últimas 21 = prueba (solo se reporta).
- Sin fuga: para la semana objetivo W (lunes), corte c = sábado anterior. Los errores usables son los de días con fecha <= c (pronóstico emitido
  antes de ese día y real ya observado). Las ventanas móviles solo ven esos errores. El pipeline se corre por semana con `construir_contexto` (verifica no fuga).
- Componentes por día (cache en scratch, regenerable con `scripts/noche2_b2_datos.py`): base k1, nivel, forma (= final default), final con análogos
  (nivel ancla análogos α .25 + forma default) y final con forma referencia; real; tipo de día.
- Métricas: WAPE horario, MAPE energía diaria y semanal, sesgo, error de pico (%), por UCP × partición × tipo de día.
- Diferencia < 0.05 pp = ruido (prueba = 21 semanas).

## Candidatas
a) Sesgo multiplicativo tipo-día×hora y dow×hora (ventana N semanas, shrinkage, mínimo casos) sobre base y sobre final.
b) Mezcla ponderada base/final/análogos/referencia con pesos por tipo de día en ventana móvil.
c) Festivo / puente (día adyacente a festivo) / clima observado (limitado: Antioquia solo tiene clima desde 2025-12-01).
d) Intervalos empíricos p10–p90 por tipo de día × hora (cobertura en prueba).
e) Selección automática del mejor método por tipo de día (elegida en calibración).
f) Otras: suavizado de factores entre horas, corrección de pico, factor de energía diaria.

## Entregables
`REPORTE.md` (tabla antes/después, veredictos, cobertura de bandas), `LOG.md`, scripts `scripts/noche2_b2_*.py`,
y si algo cumple la regla: `src/adjustments/correccion.py` + `tests/test_adjustments_correccion.py`.
