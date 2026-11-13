# Sub-plan 4C — automático por segmento (R2)

## Objetivo
Exponer y aplicar (opcionalmente) la configuración por **UCP × tipo de día** derivada del backtest
`reports/ajustes/backtest_modelo_20261113.md` (sección 8, `modelo_k1`), sin cambiar defaults.
Por defecto **off**: ninguna curva cambia si no se pide.

## Configuración (derivada del backtest)
`src/adjustments/automatico.py` (nuevo):
- `CONFIG_R2 = {"Antioquia": {"laboral": {"forma": "automatico"}},
                "Atlantico": {"laboral": {"forma": "automatico"},
                              "domingo": {"forma": "automatico"},
                              "sabado":  {"nivel": "automatico"}}}`.
- `TIPOS = ("laboral","sabado","domingo","festivo")`, `PASOS = ("nivel","forma")`.
- `configuracion(ucp)` → dict anidado completo con `"sugerido"` por defecto para lo no listado
  (UCP desconocida → todo `sugerido`), incluyendo `fuente` = reporte y `criterio`.
- `estado(ucp, tipo_dia, paso)` → `"automatico" | "sugerido"`; `pasos_automaticos(ucp, tipo_dia)`.
- `modos_paso(ucp)` → `{paso: "automatico" | "sugerido"}`: un paso es `"automatico"` solo si
  **los 4 tipos de día** lo marcan automático; cualquier tipo sin regla cuenta como `"sugerido"`
  (una UCP cuyo backtest solo respalda un segmento queda en `"sugerido"`). Es el resumen que
  consume el front.
- `activar(ucp, tipo_dia, paso)` NO muta el global: el request lleva `aplicar_automatico: bool`
  (default false) en `RunRequest`; se guarda en `meta.extras["aplicar_automatico"]`.

## Integración
- **Decisión (sin usuario):** `aplicar_automatico` **no altera curvas**; el backend aplica siempre
  los pasos del pipeline. R2 `"automatico"` es una **clasificación de revisión para el front**, no
  una instrucción de ejecución. `modos_paso` es el contrato que consume el front; el detalle por
  día queda en `diagnostics["automatico"]`.
- `RunRequest.aplicar_automatico: bool = False` (router).
- En `pipeline._ejecutar`, tras calcular `nivel`/`forma`/`especiales`, si
  `ctx.extras.get("aplicar_automatico")` es verdadero, añadir a `result.diagnostics["automatico"]`
  un dict por fecha de la semana objetivo: `{fecha: {"tipo_dia": ..., "estado": ..., "paso": ...}}`
  usando `automatico.estado(...)`. No altera `curves` ni `params`.
- `meta.extras["aplicar_automatico"]` en `_compute` y `_context_from_meta` (para PATCH).
- **Siempre** (flag on u off): `meta.extras["modos_paso"] = automatico.modos_paso(ucp)` (barato, no
  requiere store ni datos) y `_context_from_meta` lo restaura (recalculándolo si la corrida vieja no
  lo trae).
- Endpoint de solo lectura `GET /adjustments/automatico?ucp=...` → configuración completa + estado.
  Requiere el mismo token que el resto; no requiere store.

## Tests (`tests/test_adjustments_automatico.py`)
- `configuracion` completa: los 4 tipos × 2 pasos, valores del backtest, `sugerido` en el resto.
- `estado` por UCP/tipo/paso y UCP desconocida.
- `run_pipeline` con `aplicar_automatico` off (default) → curvas y `diagnostics` idénticos a antes.
- Con `aplicar_automatico` on → `diagnostics["automatico"]` por día (con `paso`), curvas idénticas.
- `modos_paso` presente en `meta.extras` con y sin el flag, y conservado en PATCH.
- Endpoint 200 con y sin token; 422 UCP inválida (pattern del router).

## Verificación
Suite completa + `import src.api.main`. Documentar en `docs/work/ajustes/fase4/LOG.md`.
