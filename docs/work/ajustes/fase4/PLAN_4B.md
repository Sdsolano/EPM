# Sub-plan 4B — paso `especiales` (días especiales)

## Objetivo
Nuevo paso del pipeline, entre `forma` y `control`, que detecta y (solo si se activa) corrige
**Semana Santa**, **24 y 31 de diciembre** y **día posterior a festivo**. Default: sin efecto.
La evidencia previa con α 0.5 + `ancla: analogos` **NO se sostiene** tras el re-backtest con
`modelo_k1` (`REPORTE_PASO0.md`): no hay evidencia para activar el candidato analógico, así que
el default queda `aplicar = false`, `alpha = 0.25`, `ancla = "actual"` (sin efecto por defecto).

## Contrato
- Archivo nuevo `src/adjustments/steps/especiales.py`.
- `STEP_ORDER = ["base", "referencia", "nivel", "forma", "especiales", "control"]` (6 pasos).
- `_MODULOS["especiales"] = especiales`; `_PASOS_QUE_MODIFICAN = ("base", "nivel", "forma", "especiales")`.
- `DEFAULTS = {"aplicar": False, "alpha": 0.25, "ancla": "actual"}`.
  - `aplicar=false` → calcula sugerencias en `diagnostics` y NO toca `curves` (salida idéntica a hoy).
  - `aplicar=true` → para cada día especial aplica `E(d) = (1−α)·E_previa(d) + α·E*(d)` y
    (con `ancla="analogos"`) mezcla la forma de los análogos escalada a `E(d)`.
  - `alpha <= 0` → no modifica ni energía ni forma; el día queda `aplicado=false`.
  - `ancla="actual"` (sin análogos) → solo energía por homólogo/referencia; si no hay evidencia,
    deja el día intacto con alerta info.
- Detección (usa `ctx.festivos`, `analogos.es_semana_santa` y `analogos.descriptor_dia`):
  - Semana Santa: días con `es_semana_santa(d)`.
  - 24/31 dic.
  - Post-festivo: día laboral/sábado cuyo día anterior es festivo, confirmado por la
    `relacion_festivo` real del descriptor (`despues` o `puente`).
  - Por día se registra `motivo` (`semana_santa` | `24_dic` | `31_dic` | `post_festivo`).
  - Elecciones: días de `ctx.extras["eventos"]` con texto de elecciones/votación/consulta popular
    se marcan `elecciones_ia` SOLO en diagnóstico/alerta; nunca tocan curvas. Sin extras, nada.
- Evidencia: reutilizar `analogos.seleccionar` + `steps/_analogos.py` (`seleccion_dia`,
  `forma_ponderada`, `evidencia_analogos`, `filtros_efectivos`, `anios_excluidos_ctx`).
  Filtros por motivo: Semana Santa → `semana_santa: true`; 24/31 dic → `meses:[12]`;
  post-festivo → la `relacion_festivo` del descriptor (`despues` o `puente`) y su distancia si
  aplica. Mantener `dia_semana:"auto"`. Respetar `filtros_analogos.limite` del analista.
  Sin análogos observados → sin cambio de ese día + info `ESPECIALES_SIN_EVIDENCIA`.
- Ediciones manuales previas (`fechas_editadas` de los records que llegan del pipeline): el paso
  salta esos días, conserva la curva editada y emite info `ESPECIALES_EDICION_PRESERVADA`.
- Diagnóstico (siempre): `motivos`, por día `E_previa`, `E_objetivo`, `E_final`, `alpha`,
  `ratio`, `ancla`, `analogos`/`fechas_fuente`, `aplicado` (bool), `formas_fuente`.
  Alerta info `ESPECIALES_SUGERIDOS` cuando hay días especiales y no se aplicó.
- No modifica días de brecha.

## Integración en el pipeline
- `_energia_esperada` (reconciliación de `forma`→`control`): excluir los días que `especiales`
  ajustó (`diagnostics.dias_ajustados` con `aplicado=true`), igual que hoy se excluyen los días
  editados de `forma`. Si no, `RECONCILIACION_ROTA` bloquearía corridas válidas.
- `_energia_nivel` (enclavamiento del control): para los días que `especiales` ajustó
  (`aplicado=true`) usar su `E_final` como baseline, de modo que el ajuste intencional no emita
  `TENDENCIA_FUERA_BANDA`.
- `final_curves` y `approval_blockers` funcionan por `_PASOS_QUE_MODIFICAN` sin cambios extra.
- **Migración de corridas viejas de 5 pasos** en `pipeline._records_por_paso`: si los nombres son
  exactamente los 5 antiguos, insertar un record `especiales` `omitido` (orden 4, `diagnostics={"omitido": True,
  "migrado": True}`) y renumerar `control` a orden 5 (`dataclasses.replace`). No persistir hasta el
  siguiente PATCH (entonces ya se guardan los 6).
- `run_pipeline`/`rerun_from` siguen indexando por nombre.

## Tests nuevos (`tests/test_adjustments_especiales.py`)
- Default `aplicar=false`: curvas idénticas a una corrida sin el paso (6 pasos vs 5) y
  `ESPECIALES_SUGERIDOS` con los días detectados.
- Semana Santa, 24/12, 31/12 y post-festivo detectados con `historico_sintetico` + festivos.
- `aplicar=true` con análogos inyectados (`params.analogos` o histórico sintético con patrón claro):
  cambia solo los días especiales, conserva el resto, registra `dias_ajustados`.
- `alpha`/`ancla` inválidos → `ValueError`; `validar_params` de claves desconocidas.
- Reconciliación: `forma` intacta + `especiales` aplicado NO genera `RECONCILIACION_ROTA`.
- Migración: records de 5 pasos se aceptan en `rerun_from` y devuelven 6 con `especiales.omitido`.
- `run_pipeline` con `overrides`/params por nombre `especiales`.

## Verificación
`pytest tests/test_adjustments_especiales.py tests/test_adjustments_pipeline.py -q`, luego la suite
completa (`--ignore=tests/test_hourly_disaggregation.py -p no:cacheprovider`) y
`python -c "import src.api.main"`. Actualizar `docs/work/ajustes/fase4/LOG.md`.
