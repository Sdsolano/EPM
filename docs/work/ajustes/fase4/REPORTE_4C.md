# Reporte 4C — automático por segmento (R2)

Fecha: 2026-11-13 · Rama: `feat/ajustes-fase4` · Alcance original: **solo núcleo**
(sin tocar `router.py` ni `pipeline.py`; la integración la hizo un agente posterior). La ronda 3
(corrección del ALTO de diseño) actualizó además `router.py`/`pipeline.py` y sus tests.

## Archivos

| Archivo | Tipo | Contenido |
|---|---|---|
| `src/adjustments/automatico.py` | nuevo | Configuración R2: `estado` (2 estados), `recomendacion`/`segmentos` (3 estados), `pasos_automaticos`, `modos_paso`, `configuracion`. Sin imports del paquete ni efectos secundarios. |
| `tests/test_adjustments_automatico.py` | nuevo | 93 tests (configuración, `estado`, `recomendacion`, `segmentos`, `modos_paso`, claves inválidas, UCP desconocida, import aislado). |
| `docs/work/ajustes/fase4/REPORTE_4C.md` | nuevo | Este reporte. |

En la ronda 3 se tocaron además `src/adjustments/router.py`, `src/adjustments/pipeline.py`,
`src/adjustments/eventos.py`, `src/adjustments/aprendizaje.py`,
`src/adjustments/steps/especiales.py` y los tests de integración.

## API pública

```python
CONFIG_R2 = {
    "Antioquia": {"laboral": {"forma": "automatico"}},
    "Atlantico": {"laboral": {"forma": "automatico"},
                  "domingo": {"forma": "automatico"},
                  "sabado":  {"nivel": "automatico"}},
}
CONFIG_NO_APLICAR = {
    "Antioquia": {"domingo": ("nivel",)},
    "Atlantico": {"laboral": ("nivel",)},
}
TIPOS = ("laboral", "sabado", "domingo", "festivo")
PASOS = ("nivel", "forma")
ESTADOS = ("automatico", "sugerido")                      # contrato del pipeline
RECOMENDACIONES = ("automatico", "sugerido", "no_aplicar")  # clasificación R2
AUTOMATICO, SUGERIDO, NO_APLICAR
FUENTE, SECCION, CRITERIO, UCP_CON_AUTOMATICO
```

- `estado(ucp, tipo_dia, paso) -> "automatico" | "sugerido"` (2 estados, sin
  cambios). Valida `tipo_dia` contra `TIPOS` y `paso` contra `PASOS` **antes** de
  mirar la UCP (`ValueError` con mensaje `tipo_dia inválido ...` / `paso inválido
  ...`). UCP desconocida, vacía, `None` o no-str → `"sugerido"`.
- `recomendacion(ucp, tipo_dia, paso) -> "no_aplicar" | "automatico" | "sugerido"`
  (3 estados): `"no_aplicar"` para los pares del §8 (Antioquia/domingo/nivel,
  Atlantico/laboral/nivel), `"automatico"` para los 4 segmentos respaldados y
  `"sugerido"` para el resto. Misma validación/UCP que `estado`.
- `pasos_automaticos(ucp, tipo_dia) -> list[str]` en el orden de `PASOS`
  (`[]` si no hay). Valida `tipo_dia` (mismo `ValueError`).
- `segmentos(ucp) -> dict` → `{tipo: {paso: recomendacion}}` con los 4 tipos × 2
  pasos y los 3 estados. Es el contrato que el router publica siempre en
  `meta.extras["segmentos_paso"]`.
- `modos_paso(ucp) -> dict[str, str]` → `{paso: "automatico" | "sugerido"}`. Un paso
  es `"automatico"` solo si **los 4 tipos de día** lo marcan automático; cualquier
  tipo sin regla en `CONFIG_R2` cuenta como `"sugerido"` (p. ej. Atlántico/nivel y
  Antioquia/forma quedan `sugerido`). `no_aplicar` **no** cuenta como automático.
  UCP desconocida → todo `"sugerido"`. Es el resumen que el router publica siempre
  en `meta.extras["modos_paso"]`; el front lo usa para decidir qué revisiones puede
  automatizar. No altera curvas ni params.
- `configuracion(ucp) -> dict`:

```python
{
  "ucp": ucp,
  "fuente": "reports/ajustes/backtest_modelo_20261113.md",
  "seccion": "8. R2 (modelo_k1, config default)",
  "criterio": "automático si el paso ayuda en calibración y en prueba, ...",
  "tipos": {  # contrato del pipeline (2 estados)
    "laboral": {"nivel": "sugerido", "forma": "automatico"},
    "sabado":  {"nivel": "sugerido", "forma": "sugerido"},
    "domingo": {"nivel": "sugerido", "forma": "sugerido"},
    "festivo": {"nivel": "sugerido", "forma": "sugerido"},
  },
  "segmentos": {  # 4 × 2 con los 3 estados (igual que segmentos(ucp))
    "laboral": {"nivel": "sugerido", "forma": "automatico"},
    "domingo": {"nivel": "no_aplicar", "forma": "sugerido"},
    ...
  },
  "recomendaciones": [  # solo los segmentos que no son `sugerido`
    {"tipo_dia": "domingo", "paso": "nivel", "recomendacion": "no_aplicar"},
    ...
  ],
}
```

  Siempre trae los 4 tipos × 2 pasos; UCP desconocida → todo `"sugerido"` y
  `recomendaciones` vacío. Devuelve estructuras nuevas en cada llamada (mutarlas no
  altera `CONFIG_R2`/`CONFIG_NO_APLICAR`). Nota: la clave `tipos` conserva los 2
  estados del pipeline (`domingo/nivel` de Antioquia queda `sugerido` allí y
  `no_aplicar` solo en `segmentos`/`recomendaciones`).

## Tests

Comando (desde `EPM-wt-4`):

```
C:\Users\samue\OneDrive\Documentos\GitHub\EPM\.venv\Scripts\python.exe -m pytest tests/test_adjustments_automatico.py -q -p no:cacheprovider
```

Resultado: **93 passed** (incluye `recomendacion` y `segmentos`). `python -c "import
src.adjustments.automatico"` → `ok`. Cobertura de los tests:

- Constantes exactas (`TIPOS`, `PASOS`, `ESTADOS`, `RECOMENDACIONES`, `CONFIG_R2`,
  `CONFIG_NO_APLICAR`) y metadatos.
- `configuracion` completa para Antioquia/Atlantico/desconocida/`None`/minúsculas,
  con los valores del backtest, `sugerido` en el resto, `segmentos` (3 estados) y
  `recomendaciones`. 
- `estado` para las 20 combinaciones representativas UCP × tipo × paso.
- `recomendacion` para los 4 automáticos, los 2 `no_aplicar` y el resto; UCP
  desconocida; claves inválidas.
- `segmentos` 4 × 2 completo, UCP desconocida y estructuras nuevas.
- Claves inválidas de tipo y paso (incluye `None`, enteros y mayúsculas) y orden
  de validación (claves antes que UCP).
- `pasos_automaticos` por segmento y tipo inválido.
- Import aislado: `importlib.reload` no altera los `DEFAULTS` de los 6 pasos;
  el módulo no expone `router` ni retiene módulos `src.*`; un subproceso
  comprueba que importarlo no carga `src.adjustments.router` ni
  `src.adjustments.pipeline`.

## Decisiones

- **Alcance R2**: `recomendacion` expone los 3 estados del §8: `"automatico"` en
  los 4 segmentos respaldados, `"no_aplicar"` en Antioquia/domingo/nivel y
  Atlantico/laboral/nivel, y `"sugerido"` en el resto. `estado` se mantiene en 2
  estados porque es el contrato del pipeline. `no_aplicar` es una recomendación
  de revisión: el backend no omite el paso (eso sería una decisión de producto /
  front).
- **UCP exacta**: comparación literal `"Antioquia"` / `"Atlantico"` (sin
  normalizar mayúsculas ni quitar tildes/espacios). Refleja los nombres reales
  de `data/raw/{ucp}` y evita que una variante ortográfica active reglas del
  backtest por accidente.
- **UCP no-str** → `"sugerido"` (no lanza): la validación estricta es solo para
  `tipo_dia`/`paso`, como pide el contrato.
- **Sin imports del paquete**: el módulo solo usa `typing`, así que importarlo no
  toca curvas, `DEFAULTS`, datos ni router. Es seguro para el endpoint de solo
  lectura y para los tests.
- `configuracion` incluye `fuente`, `seccion` y `criterio` además del mapa
  `tipos`; el endpoint puede devolverlo tal cual y el pipeline consumir
  `estado(...)`.

## Riesgos / notas para integración

- El agente de integración debe leer la configuración en
  `configuracion(ucp)["tipos"]`; el nivel superior contiene metadatos.
- `tipo_dia` esperado es el de `data.tipo_dia` (sin tilde, `festivo` con
  prioridad). Valores como `"miércoles"` o `"sábado"` lanzan `ValueError`.
- **Decisión de integración (sin usuario):** `aplicar_automatico` **no activa ni
  desactiva pasos**; el backend aplica siempre los pasos del pipeline. R2
  `"automatico"` es una clasificación de revisión para el front. El flag (default
  `False`) solo añade el detalle por día en `diagnostics["automatico"]` (que ahora
  incluye `estado` y `recomendacion`), `modos_paso` viaja siempre en
  `meta.extras["modos_paso"]` y `segmentos` en `meta.extras["segmentos_paso"]`.
  Este módulo no modifica curvas ni `params`.
- Si en el futuro se recalibra el backtest, actualizar `CONFIG_R2`, `FUENTE` y
  `SECCION` juntos; los tests fijan los valores actuales.

## Limitaciones aceptadas (ronda 2)

- `modos_paso` exige los 4 tipos: con los datos actuales ninguna UCP queda `automatico` en
  `nivel`/`forma` (solo hay respaldo parcial por segmento). Es intencional: el front recibe
  `sugerido` y el detalle por tipo sigue en `configuracion()`.
- R2 es solo una clasificación de revisión: no activa/desactiva pasos ni altera curvas.

## Limitación de diseño y contrato expuesto (ronda 3)

- **Limitación**: el backend aplica **siempre** los pasos del pipeline; R2 no omite ni activa
  `nivel`/`forma`. La automatización por segmento y la omisión de los segmentos `no_aplicar` son
  una decisión de front/producto, no un comportamiento del backend. `modos_paso` sigue siendo
  conservador (solo `automatico` si los 4 tipos lo son), así que hoy devuelve `sugerido` para
  `nivel`/`forma` en ambas UCP.
- **Contrato expuesto**:
  - `GET /adjustments/automatico` → `configuracion(ucp)` con `tipos` (2 estados),
    `segmentos` (4 × 2, 3 estados) y `recomendaciones` (solo los segmentos explícitos).
  - `meta.extras["segmentos_paso"]` = `segmentos(ucp)` en `preview`/`run`/`GET`/`PATCH`
    (recalculado si la corrida es vieja), además de `meta.extras["modos_paso"]`.
  - `diagnostics["automatico"]` de `nivel`/`forma` (solo con `aplicar_automatico=true`): cada día
    trae `{"tipo_dia", "estado", "paso", "recomendacion"}`.
- **Pendiente del usuario (decisión de producto)**: definir si el front debe omitir los segmentos
  `no_aplicar` (y cómo se refleja en la revisión/aprobación). El backend no lo hace por sí solo.
