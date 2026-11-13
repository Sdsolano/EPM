# Sub-plan 4D — alertas de eventos y clima (F06)

## Objetivo
Alertas informativas en `control` sobre eventos/elecciones y clima esperado de la semana objetivo,
obtenidos de la consulta OpenAI **ya existente** (`get_future_events_from_openai`, `src/api/main.py:765`,
key `API_KEY` en `.env`). El usuario no entrega ningún dato. Nunca modifican curvas.

## Módulo nuevo `src/adjustments/eventos.py`
- `consultar_openai(ucp, inicio, fin)` (sync, cliente inyectable): replica el prompt y la llamada
  `client.responses.create(model="gpt-5-mini", tools=[{"type":"web_search"}])`; parsea JSON y
  devuelve `{fecha ISO: "evento"}`. Sin `API_KEY`/sin `openai` → `{}`.
- `async def obtener_eventos(ucp, inicio, fin, *, consultar=None, cache_dir=None, ttl_dias=7)`:
  - cache en disco `*.cache*` por `(ucp, semana)` (JSON con `ucp`, `semana`, `generado`, `eventos`);
    dir default `Path(os.getenv("AJUSTES_EVENTOS_CACHE_DIR", ROOT/".cache"/"eventos_ajustes"))`
    (`.cache/` ya está en .gitignore). Lee si está fresco; si no, consulta y escribe.
  - cualquier excepción/fallo → devuelve la cache vieja si existe o `{}`, en silencio (log warning).
  - sanea: claves `YYYY-MM-DD`, valores str ≤ 120 chars, solo fechas del rango pedido ±7 días.
- `eventos_de_semana(ucp, lunes, ...)` → rango lunes..domingo.
- No importa `fastapi`; usa `asyncio.to_thread` para la llamada bloqueante.

## `main.py`
- `get_future_events_from_openai(ucp, fecha_inicio, fecha_fin)` delega en este módulo (misma firma y
  tipo de retorno) para no duplicar prompt/cliente. El endpoint existente sigue igual.

## Router (`src/adjustments/router.py`)
- En `_compute` (preview/run), tras `ctx.extras[...]`, llamar una vez
  `ctx.extras["eventos"] = await eventos.eventos_de_semana(...)` con try/except total → `{}`.
  No se ejecuta dentro del threadpool del pipeline.
- Guardar `eventos` en `meta.extras` (reproducible en PATCH); en `_context_from_meta` reusar
  `extras["eventos"]` y solo consultar si falta. `_run_out` no expone `eventos` completo en `meta`
  (filtrarlo igual que `festivos`); los detalles van en las alertas de `control`.
- No bloquear el request si OpenAI tarda demasiado: timeout/Except → `{}` (degradación silenciosa).

## `steps/control.py`
- Leer solo `ctx.extras` (`eventos`, y `ctx.festivos`); nunca llamar a la red.
- Alertas nuevas:
  - `FESTIVO_EN_SEMANA` (info) con los festivos de la semana objetivo (determinista).
  - `EVENTOS_SEMANA` (info, solo si hay eventos) con mensaje "Eventos sugeridos por IA, no verificados"
    y `detalle={"eventos": {...}, "fuente": "openai", "verificado": False}`.
  - `CLIMA_OBSERVADO` (info/warning) si `analogos.cargar_clima(ucp)` cubre días de la semana
    (replay/semanas pasadas): temp media y lluvia; warning si `lluvia_mm` > p90 del propio histórico
    o `|temp_media − media histórica|` > 2σ. Si no hay clima, no emite nada (silencio).
- Las alertas NO alteran `curves` ni bloqueos; test que las curvas son idénticas con/sin eventos.

## Tests (`tests/test_adjustments_eventos.py`)
- Cliente falso (callable sync) que devuelve dict; `obtener_eventos` lo llama una vez y reusa cache
  (contador), cache dir temporal (`tmp_path`).
- Excepción del cliente → `{}` y no lanza; sin key → `{}`; eventos malformados se descartan.
- `eventos_de_semana` pide lunes..domingo.
- `control.step` con `ctx.extras["eventos"]` emite `EVENTOS_SEMANA` y curvas idénticas; sin eventos,
  sin alerta. Festivos → `FESTIVO_EN_SEMANA`. Clima inyectado (monkeypatch a `cargar_clima`) → alerta.
- Router: preview/run con provider de prueba y `monkeypatch` de `eventos.eventos_de_semana` → el
  record `control` trae la alerta; fallo del módulo → corrida normal (degradación).
- Nada de tests llama a OpenAI (ni a `API_KEY` real).

## Verificación
Suite completa + `import src.api.main`. Documentar en `docs/work/ajustes/fase4/LOG.md` y dejar
`AJUSTES_EVENTOS_CACHE_DIR` documentado en `.env.example`.
