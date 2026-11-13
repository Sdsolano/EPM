# Reporte 4D — alertas de eventos y clima (F06)

## Qué se hizo
- Módulo nuevo `src/adjustments/eventos.py`: consulta OpenAI con web search,
  cache en disco exacta por `(UCP, inicio, fin)`, saneo al rango pedido y
  degradación silenciosa.
- `src/api/main.py`: `get_future_events_from_openai` delega en el módulo nuevo
  (misma firma y retorno); import perezoso y protegido.
- `src/adjustments/steps/control.py`: alertas de solo lectura `FESTIVO_EN_SEMANA`,
  `EVENTOS_SEMANA` y `CLIMA_OBSERVADO`; no tocan curvas ni bloqueos. Los eventos
  se filtran a lunes..domingo de la semana objetivo.
- `src/adjustments/analogos.py`: cache de módulo de `cargar_clima` por
  `(ruta, mtime, tamaño)` (misma firma/retorno).
- `src/adjustments/router.py`: timeout de eventos 5 s; las corridas nuevas
  persisten `meta.extras.eventos` y los PATCH viejos ya no consultan OpenAI.
- `tests/test_adjustments_eventos.py`, `tests/test_adjustments_fase4_router.py`:
  cobertura de cache, saneo, fallos de guardado, TTL de vacíos, filtrado de la
  semana, cache de clima y no-consulta en PATCH viejos.
- `.env.example`: `AJUSTES_EVENTOS_CACHE_DIR` documentado (comentado).

## API del módulo (`src/adjustments/eventos.py`)
- `consultar_openai(ucp, inicio, fin, *, client=None) -> dict`: sync; replica el
  prompt y `client.responses.create(model="gpt-5-mini", tools=[{"type":"web_search"}])`,
  limpia markdown y parsea JSON. Sin `API_KEY`/`openai` (y sin `client`) lanza
  `EventosNoDisponibles`, igual que ante cualquier fallo del cliente: un fallo no se
  confunde con "sin eventos". Un `{}` parseado sí es un éxito.
- `await obtener_eventos(ucp, inicio, fin, *, consultar=None, cache_dir=None,
  ttl_dias=7, ttl_vacios_dias=1)`: cache JSON
  `{ucp, inicio, fin, generado, eventos}` en `.cache/eventos_ajustes` (o
  `AJUSTES_EVENTOS_CACHE_DIR`; si la variable está vacía se usa el default), con
  archivo `{ucp}_{inicio}_{fin}.json` y clave exacta del rango. TTL 7 días para
  resultados no vacíos y 1 día para `{}` (permite reintentar). Un fallo de la
  consulta devuelve la cache vieja o `{}` **sin sobrescribir** esa cache; un fallo al
  guardar la cache **no** descarta los eventos frescos. La llamada síncrona usa
  `asyncio.to_thread`.
- `await eventos_de_semana(ucp, lunes, ...)`: rango lunes..domingo.

Saneo: claves `YYYY-MM-DD` válidas, valores `str` no vacíos truncados a 120
caracteres y fechas dentro de `[inicio, fin]` exactos. No importa `fastapi`.

## Alertas en `control` (solo lectura de `ctx.extras` / `ctx.festivos`)
- `FESTIVO_EN_SEMANA` (info): festivos de `ctx.festivos` que caen en la semana.
- `EVENTOS_SEMANA` (info): eventos de `ctx.extras["eventos"]` que caen en
  lunes..domingo de `ctx.semana_inicio`; los de ±7 días alrededor se descartan
  del listado y del detalle. Mensaje "Eventos sugeridos por IA, no verificados";
  detalle `{eventos, fuente: "openai", verificado: false}`.
- `CLIMA_OBSERVADO` (info/warning): si `analogos.cargar_clima(ucp)` cubre días
  de la semana. `info` con temp media y lluvia total; `warning` si la lluvia
  diaria máxima de la semana supera el p90 histórico (excluyendo la propia
  semana) o si la temp media semanal se aleja > 2σ de la histórica. Sin clima
  (semanas futuras) no emite nada. Nunca altera `curves` ni bloqueos. El clima
  se lee del disco pero cacheado por archivo; no se persiste en la corrida.

## Verificación
- `pytest tests/test_adjustments_eventos.py tests/test_adjustments_steps.py
  tests/test_adjustments_fase4_router.py -q -p no:cacheprovider`
  → **132 passed**.
- Suite completa (`pytest tests -q -p no:cacheprovider`, sin
  `tests/test_hourly_disaggregation.py` que ya falla al colectar en `main` por
  imports relativos preexistentes)
  → **607 passed, 5 skipped, 1 failed**: solo el test preexistente de
  `test_holiday_homology.py::...christmas...` (`"Navidad"` vs
  `"Christmas Day"`).
- `python -m py_compile` de los archivos tocados → OK.

## Correcciones de la ronda 1
1. Cache exacta `(ucp, inicio, fin)` y salida saneada solo a `[inicio, fin]`
   (el margen ya no se devuelve). Tests: rangos distintos no comparten cache y
   no salen fechas fuera del rango.
2. Guardado de cache en try/except propio: si falla el disco se devuelven los
   eventos frescos. Test con `cache_dir` que es un archivo.
3. `AJUSTES_EVENTOS_CACHE_DIR=""` se trata como no definido. Test.
4. Los `{}` se cachean con `ttl_vacios_dias=1` y se reintentan al vencer. Test y
   documentación.
5. `EVENTOS_TIMEOUT = 5` (constante) con degradación silenciosa a `{}` y log.
   Test del valor.
6. `_context_from_meta` usa `{}` para corridas sin `meta.extras.eventos` (no
   consulta OpenAI en PATCH); las nuevas sí persisten el dict. Test.
7. `EVENTOS_SEMANA` filtra a lunes..domingo de la semana objetivo en `fechas` y
   `detalle`. Tests con eventos dentro/fuera.
8. `analogos.cargar_clima` cachea por `(ruta, mtime, tamaño)` con lock; dos
   llamadas devuelven el mismo objeto sin releer. Tests de cache y de control.
9. Documentado que el clima observado sigue leyéndose del disco (cacheado) y no
   se persiste: limitación aceptada.

## Pendientes aceptados
- El clima observado no se persiste en `meta.extras`; se relee (cacheado) al
  recalcular. Aceptado: solo alimenta una alerta informativa.
- La cache de clima en memoria es un LRU de hasta 4 entradas por
  `(ruta, mtime_ns, tamaño)`; una UCP que deje de consultarse puede requerir una
  relectura, pero nunca se vacía la cache de las demás.
- El cliente OpenAI se crea con `timeout=5, max_retries=0`: un fallo/API_KEY ausente
  devuelve la cache vieja (o `{}`) y no sobrescribe la buena.
- `tests/test_hourly_disaggregation.py` no colecta en el entorno actual
  (imports relativos `prediction.hourly`), igual que en `main`; no es de 4D.
- El test de Navidad de `test_holiday_homology.py` sigue fallando por el nombre
  del festivo en la librería; preexistente.

