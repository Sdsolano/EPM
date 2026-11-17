# Fase 2a — Parte C: `alpha_por_tipo` en el paso nivel (reporte)

## Cambios (aditivos, sin reordenar/reformatear)
- `src/adjustments/steps/nivel.py`:
  - Docstring del módulo: +2 líneas describiendo `alpha_por_tipo`.
  - Import nuevo (bloque aparte) de `TIPO_*` y `tipo_dia` desde `src.adjustments.data`.
  - `TIPOS_ALPHA` y `_validar_alpha_por_tipo` nuevos; `validar_params` extrae la clave antes de `_params.completar`
    (DEFAULTS no cambia) y la reinserta validada solo si no es `None`.
  - `step`: 2 líneas nuevas tras `alpha = validar_params(params)["alpha"]` (línea existente intacta, para merge
    trivial con fase 2b) + override de `alpha_d` por tipo de día; diagnósticos aditivos.

## Semántica
- `alpha_por_tipo`: dict opcional, claves ⊆ {laboral, sabado, domingo, festivo} (las de `data.tipo_dia(d, ctx.festivos)`),
  valores finitos en [0, 1]. α del día = `alpha_por_tipo[tipo_dia(d)]` si la clave existe, si no `alpha`.
- Sin evidencia → α = 0 ese día (sin cambios).
- Ausente o `None` → params normalizados, curvas y diagnósticos idénticos a antes (no se agrega ninguna clave).
- Con el param: `params` normalizado incluye `alpha_por_tipo` (floats, claves en orden canónico); diagnósticos añaden
  `alpha_por_tipo` (top-level) y `tipo_dia` por día; `dias[f]["alpha"]` ya era el α efectivo del día.
- Validación: no-dict, claves desconocidas (case-sensitive), no numérico/bool/str/None, NaN/inf, fuera de [0,1] →
  `ValueError` (mismo tipo que `alpha` inválido; el router lo convierte en 422).

## Decisiones
- No se tocó DEFAULTS (evita que records/params persistidos cambien); `None` explícito se trata como ausente y se
  descarta del normalizado.
- `step` llama `validar_params` dos veces para no modificar la línea existente de `alpha` (costo despreciable).

## Tests
`tests/test_nivel_alpha_por_tipo.py` (21 casos, sintéticos): idéntico sin la clave / con `None`; todas las claves = alpha
→ idéntico; solo `festivo` cambia solo jueves/viernes santo 2025; sin evidencia sigue α=0; validación (12 entradas
inválidas, vía `validar_params` y `step`); normalización a float; ida y vuelta JSON de records del pipeline.

## Suite
`pytest tests -q --ignore=tests/test_hourly_disaggregation.py` → 263 passed, 4 skipped, 1 failed
(`test_christmas_profile_uses_prior_christmases`, preexistente/permitido). Línea base 242 + 21 nuevos.
