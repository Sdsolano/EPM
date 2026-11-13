# Reporte 4C — automático por segmento (R2)

Fecha: 2026-11-13 · Rama: `feat/ajustes-fase4` · Alcance: **solo núcleo** (sin
tocar `router.py` ni `pipeline.py`; la integración la hace un agente posterior).

## Archivos

| Archivo | Tipo | Contenido |
|---|---|---|
| `src/adjustments/automatico.py` | nuevo | Configuración R2, `estado`, `pasos_automaticos`, `modos_paso`, `configuracion`. Sin imports del paquete ni efectos secundarios. |
| `tests/test_adjustments_automatico.py` | nuevo | 60 tests (configuración, estado, `modos_paso`, claves inválidas, UCP desconocida, import aislado). |
| `docs/work/ajustes/fase4/REPORTE_4C.md` | nuevo | Este reporte. |

No se modificó ningún archivo existente.

## API pública

```python
CONFIG_R2 = {
    "Antioquia": {"laboral": {"forma": "automatico"}},
    "Atlantico": {"laboral": {"forma": "automatico"},
                  "domingo": {"forma": "automatico"},
                  "sabado":  {"nivel": "automatico"}},
}
TIPOS = ("laboral", "sabado", "domingo", "festivo")
PASOS = ("nivel", "forma")
ESTADOS = ("automatico", "sugerido")
AUTOMATICO, SUGERIDO
FUENTE, SECCION, CRITERIO, UCP_CON_AUTOMATICO
```

- `estado(ucp, tipo_dia, paso) -> "automatico" | "sugerido"`.
  Valida `tipo_dia` contra `TIPOS` y `paso` contra `PASOS` **antes** de mirar la
  UCP (`ValueError` con mensaje `tipo_dia inválido ...` / `paso inválido ...`).
  UCP desconocida, vacía, `None` o no-str → `"sugerido"`.
- `pasos_automaticos(ucp, tipo_dia) -> list[str]` en el orden de `PASOS`
  (`[]` si no hay). Valida `tipo_dia` (mismo `ValueError`).
- `modos_paso(ucp) -> dict[str, str]` → `{paso: "automatico" | "sugerido"}`. Un paso
  es `"automatico"` solo si **todos** los tipos de día con regla en `CONFIG_R2` lo
  marcan automático; mixto o sin reglas → `"sugerido"`. UCP desconocida → todo
  `"sugerido"`. Es el resumen que el router publica siempre en
  `meta.extras["modos_paso"]`; el front lo usa para decidir qué revisiones puede
  automatizar. No altera curvas ni params.
- `configuracion(ucp) -> dict`:

```python
{
  "ucp": ucp,
  "fuente": "reports/ajustes/backtest_modelo_20261113.md",
  "seccion": "8. R2 (modelo_k1, config default)",
  "criterio": "automático si el paso ayuda en calibración y en prueba, ...",
  "tipos": {
    "laboral": {"nivel": "sugerido", "forma": "automatico"},
    "sabado":  {"nivel": "sugerido", "forma": "sugerido"},
    "domingo": {"nivel": "sugerido", "forma": "sugerido"},
    "festivo": {"nivel": "sugerido", "forma": "sugerido"},
  },
}
```

  Siempre trae los 4 tipos × 2 pasos; UCP desconocida → todo `"sugerido"`.
  Devuelve estructuras nuevas en cada llamada (mutarlas no altera `CONFIG_R2`).

## Tests

Comando (desde `EPM-wt-4`):

```
C:\Users\samue\OneDrive\Documentos\GitHub\EPM\.venv\Scripts\python.exe -m pytest tests/test_adjustments_automatico.py -q -p no:cacheprovider
```

Resultado: **54 passed**. `python -c "import src.adjustments.automatico"` → `ok`.
Cobertura de los tests:

- Constantes exactas (`TIPOS`, `PASOS`, `ESTADOS`, `CONFIG_R2`) y metadatos.
- `configuracion` completa para Antioquia/Atlantico/desconocida/`None`/minúsculas,
  con los valores del backtest y `sugerido` en el resto.
- `estado` para las 20 combinaciones representativas UCP × tipo × paso.
- Claves inválidas de tipo y paso (incluye `None`, enteros y mayúsculas) y orden
  de validación (claves antes que UCP).
- `pasos_automaticos` por segmento y tipo inválido.
- Import aislado: `importlib.reload` no altera los `DEFAULTS` de los 6 pasos;
  el módulo no expone `router` ni retiene módulos `src.*`; un subproceso
  comprueba que importarlo no carga `src.adjustments.router` ni
  `src.adjustments.pipeline`.

## Decisiones

- **Alcance R2**: solo lo que el backtest marcó `automático` en `modelo_k1` con
  la config default se expone como `"automatico"`. Los `no aplicar` del reporte
  (Antioquia/domingo/nivel y Atlantico/laboral/nivel) quedan como `"sugerido"`,
  porque el contrato R2 solo distingue dos estados; "no aplicar" se sigue
  expresando con el paso `omitido` en el pipeline.
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
  `False`) solo añade el detalle por día en `diagnostics["automatico"]`, y
  `modos_paso` viaja siempre en `meta.extras["modos_paso"]`. Este módulo no
  modifica curvas ni `params`.
- Si en el futuro se recalibra el backtest, actualizar `CONFIG_R2`, `FUENTE` y
  `SECCION` juntos; los tests fijan los valores actuales.
