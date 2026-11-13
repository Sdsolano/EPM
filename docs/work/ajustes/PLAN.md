# Servicio de ajustes por pasos — Fase 1 (back + backtest)

## Context
El analista no usa el output del modelo tal cual: cada semana repite a mano el flujo descrito en
"Procedimiento para llevar a cabo el pronóstico…" (referencia semanal → nivel diario → forma con
Curva ref → festivos → revisar tendencia semanal). Hoy el único ajuste "inteligente" (ASR) vive en
el front (`Sphere/src/views/pages/pronosticos/Pronosticos.tsx:5146`, llama a
`/validate-hourly-adjustments`), no deja rastro y no se puede backtestear.

Objetivo de la Fase 1: un **módulo de ajustes encadenados** en la API de predicción de EPM que tome el
pronóstico raw y aplique pasos ordenados, guardando el resultado de cada paso en Postgres, más un
**backtest** que mida el WAPE de cada paso. Así se sabe qué paso aporta antes de construir UI.

Decisiones ya confirmadas con el usuario:
- Módulo nuevo dentro de la API de EPM (`src/adjustments/`, router propio), no servicio aparte.
- Persistencia: **Postgres propio de EPM** (la API hoy no tiene base; se agrega conexión).
- Sphere llegará vía el **backend intermedio** (`admin/dashboard/...` → EPM). Eso es Fase 3, fuera de este plan.

Fuera de alcance (fases siguientes): pantalla "Mesa de ajustes" en Sphere, rutas en el backend
intermedio, ranking automático de semanas de referencia (F01), alertas clima/eventos (F06),
aplicación automática por segmento, retirar el ASR viejo del front.

## Diseño

### Contrato de un paso
Cada paso es una función pura (sin IO):
`step(ctx: RunContext, prev: Curves, params: dict) -> StepResult`
- `Curves = dict[fecha_iso, list[float]]` (24 periodos P1..P24, MW como en `datos.csv`).
- `StepResult(curves, diagnostics: dict, alerts: list[Alert])`.
- `RunContext`: ucp, semana objetivo (lunes–domingo), fecha de corte, histórico observado ≤ corte
  (DataFrame de `data/raw/{ucp}/datos.csv`), festivos, semana de referencia.
- El pipeline (`pipeline.py`) recorre los pasos en orden; un paso puede estar `aplicado`, `omitido`
  (pasa `prev` sin cambios) o `editado` (curvas manuales del analista). Al cambiar un paso se
  recalculan los posteriores.

### Pasos de la Fase 1
| # | Paso | Qué hace |
|---|---|---|
| 0 | `base` | Pronóstico raw para corte+1 … domingo objetivo. Registra fecha de corte (último dato real ≤ fecha de emisión), días de brecha (estimados) y versión del modelo. |
| 1 | `referencia` | Fase 1: no ranquea. Default = semana equivalente del año anterior alineada por día de semana (−364 d); si la semana objetivo tiene festivo, se busca el año donde el festivo homólogo cae en el mismo día de semana (reusar `find_homologous_holiday_date`), hasta `max_anios_atras` (default 4). Años anómalos excluidos (`anios_excluidos`, default [2020]): nunca se eligen (el −364 retrocede a −728…) y nivel tampoco los usa. El analista puede fijar otra semana por parámetro. No modifica curvas. |
| 2 | `nivel` | Para cada día d: ratio en el año de referencia entre el día equivalente d′ y su ancla (último día real comparable con el **mismo desfase de corte**); energía objetivo E*(d) = E_ancla_actual × ratio. Final: E(d) = (1−α)·E_base(d) + α·E*(d), α default 0.25 (calibrado en backtest 2026-11-12; antes 0.5). Si falta evidencia, α baja a 0 y se emite alerta. Escala la curva previa al nuevo total (no cambia forma). |
| 3 | `forma` | Reemplaza la forma conservando la energía del paso 2: curva normalizada × E(d). Fuente de forma por día: festivo → `average_holiday_hourly_profile` (holiday_homology); resto → configurable `reciente` (último día real del mismo tipo), `referencia` (día equivalente del año ref) o `promedio_n` (default desde backtest 2026-11-12, n=4). Mezcla β entre forma base y forma ref (default 1.0). Diagnóstico: pico (MW y hora) antes/después, check suma = E(d). |
| 5 | `control` | No modifica. Dos controles de tendencia DISTINTOS (fix ronda 2): **(1) Enclavamiento** `TENDENCIA_FUERA_BANDA` (warning, control de EDICIÓN): desvio_pp = E_semana_final / E_semana_nivel − 1, con E_nivel = energía que calculó el paso nivel (pre-edición; si nivel se omitió, la del paso base); fuera de ±`banda_pp` (1.0) → warning con los días que más contribuyen; sin ediciones = 0. No ve alpha=0, nivel omitido ni base editada. **(2) Coherencia con la referencia** `TENDENCIA_REFERENCIA_FUERA_BANDA` (info, control de PROCESO): diferencia_tendencia_pp = tendencia resultante − tendencia de referencia (Σ E*(d) vs semana previa, E* del nivel, independiente de α y de la base); |dif| > `banda_referencia_pp` (default 2.0 pp; null la desactiva) → info. Se dispara también con alpha=0, nivel omitido o base editada. Bloqueos (solo semana objetivo): periodos faltantes/no finitos/None/negativos (`validar_curva`), curva que no reconcilia con E(d) (días no editados). Brecha (corte+1..lunes−1): es estimación, no se aprueba ni bloquea; día faltante/inválido → warning `BRECHA_INVALIDA`. |

(Paso 4 "días especiales" queda cubierto en Fase 1 por la regla festivo→homólogo del paso 3; como paso propio en Fase 4.)

### Reuso
- Pronóstico base: `run_predict_flow` (`src/api/main.py:1573`). Para evitar import circular, el
  pipeline recibe un `base_provider` inyectado; el router lo cablea a `run_predict_flow`.
- Histórico diario: mismo origen que `_get_historical_day_profile` (`src/api/main.py:2257`,
  `data/raw/{ucp}/datos.csv`); se extrae a `src/adjustments/data.py` una carga única por corrida.
- Festivos/homólogos: `src/prediction/holiday_homology.py` (`find_homologous_holiday_date`,
  `average_holiday_hourly_profile`, `colombia_holiday_map`).
- Patrón Postgres: `fastapi_factores/app/db.py` (psycopg2 + `DATABASE_URL`).

### Archivos nuevos
```
src/adjustments/
  __init__.py
  types.py         # RunContext, Curves, StepResult, Alert, StepState
  data.py          # carga histórico ≤ corte, energía diaria, tipo de día, festivos
  steps/base.py, referencia.py, nivel.py, forma.py, control.py
  pipeline.py      # orden de pasos, recalcular desde el paso n
  store.py         # interfaz Store + PostgresStore + InMemoryStore (tests)
  sql/001_init.sql # esquema `ajustes`
  router.py        # APIRouter /adjustments
scripts/apply_adjustments_schema.py   # aplica sql/ (no hay alembic en el repo)
scripts/backtest_adjustments.py
tests/test_adjustments_steps.py, test_adjustments_pipeline.py, test_adjustments_router.py
docs/work/ajustes/PLAN.md (maestro) y LOG.md (progreso resumible)
```
Modificados: `src/api/main.py` (solo `app.include_router(...)` + wiring de `base_provider`),
`requirements.txt` (`psycopg2-binary`), `.env.example` (`AJUSTES_DATABASE_URL`).

### Persistencia (esquema `ajustes`; sql/001_init.sql + sql/002_versionado.sql)
- `corrida`: id uuid, ucp, semana_inicio, fecha_corte, fecha_emision, modelo_version, params jsonb (= params
  del request original; la API lo expone como `meta.params_iniciales`, los vigentes van por paso),
  extras jsonb (fuente_base modelo|externa, fuente_festivos api|libreria, ultimo_dato_real, festivos usados,
  base_ref opcional),
  estado (`borrador`/`aprobada`/`reemplazada`), version int (1, +1 en cada PATCH/approve), creado_por,
  created_at, approved_at, approved_by, reemplazada_por, reemplazada_at.
  Índice único parcial (ucp, semana_inicio) WHERE estado='aprobada' → una sola aprobada vigente.
- `corrida_paso`: corrida_id, paso, orden (informativo), estado, params, curvas, curvas_calculadas,
  diagnostico, alertas, updated_at. PK (corrida_id, paso).
- `corrida_version`: (corrida_id, version) PK, records jsonb (snapshot completo), creado_por, motivo, created_at.
- `corrida_evento`: auditoría (corrida_id, usuario, accion, paso (nombre), motivo, detalle, version, ts).
- Concurrencia optimista: PATCH/approve exigen `version_esperada`, verificada con la fila bloqueada
  (`FOR UPDATE`); approve re-evalúa bloqueos dentro del lock y serializa por (ucp, semana) con advisory lock.
- Conexión por `AJUSTES_DATABASE_URL` (.env raíz); sin ella los endpoints que persisten responden 503.
- Corrida `aprobada`/`reemplazada` congelada (PATCH/approve → 409).

### Endpoints (`/adjustments`)
- Auth servicio-a-servicio: si `AJUSTES_SERVICE_TOKEN` está definido, TODOS los endpoints exigen la cabecera
  `X-Ajustes-Token` igual (`hmac.compare_digest`) → 401 si falta/no coincide (se evalúa antes que el body). Sin la
  variable se permite y se loguea un warning una sola vez (solo desarrollo). `usuario` sigue viniendo en el body
  (lo pone el backend intermedio).
- UCP: `^[A-Za-z0-9_-]{1,64}$` (422) y debe existir `data/raw/{ucp}/datos.csv` dentro de `data.RAW_DATA_DIR`
  (404), validado antes de refresco, red, lock o lectura. `semana_inicio`/`fecha_emision` fuera de 1990-01-01..2100-12-31 → 422 (en el modelo Pydantic).
  `ValueError`/`OverflowError` del pipeline → 422.
- `POST /adjustments/run` (201) / `POST /adjustments/preview` (200, sin persistir)
  {ucp, semana_inicio, params?, usuario?, base_curvas?, modelo_version?, fecha_emision?, base_ref? (≤200)}.
  `base_ref` → `meta.extras.base_ref` (enlace al pronóstico emitido en Sphere). `meta.params_iniciales` = `params`.
  El evento `crear` se inserta en la misma transacción que la corrida.
  Sin `base_curvas`: verifica modelo entrenado (hook `set_model_checker` = `check_model_exists` +
  `check_hourly_disaggregation_trained`; si falta → 409 "use /retrain", sin entrenar en el request); refresca datos (hook `full_update_csv`), corte = último dato real, modelo con
  end_date = corte (fuera del event loop); semana ya observada → 422; `fecha_emision`/`modelo_version` → 422;
  error del modelo o del refresco → 502. Con `base_curvas` {fecha:[24]} (pronóstico ya emitido, Fase 3): no
  llama modelo ni refresco; debe cubrir corte+1..domingo (422 si no).
- `GET /adjustments/aprobada?ucp=&semana_inicio=` → vigente (404 si no hay).
- `GET /adjustments/{id}` → meta (con version), pasos (curves, curvas_calculadas, diagnostics, alerts), bloqueos, eventos.
- `GET /adjustments/{id}/versiones` y `/versiones/{v}` → historial y snapshot de una versión.
- `PATCH /adjustments/{id}/steps/{paso}` (paso = base|referencia|nivel|forma|control)
  {estado, version_esperada, params?, curvas?, motivo?, usuario?} → recalcula siguientes; 409 versión vieja.
  Re-aplicar base: 409 sin modelo, refresca, 422 si hay datos reales posteriores al corte o la base fue externa;
  actualiza meta.modelo_version. Editar base varias veces: `curvas_calculadas` conserva la salida del modelo.
- `POST /adjustments/{id}/approve` {usuario, version_esperada, motivo?, excepciones_justificadas?} → 422 con
  bloqueos; 409 versión vieja/no borrador; reemplaza la aprobada anterior de la misma semana (evento `reemplazar`
  con la versión de la reemplazada). `excepciones_justificadas`: bajo el lock se calculan los warnings vigentes
  de esa versión; códigos no vigentes → 422 {codigos_invalidos, warnings_vigentes}; no hace falta justificar
  todos. El evento `aprobar` registra excepciones_justificadas, warnings_vigentes y warnings_sin_justificar.

### Backtest (`scripts/backtest_adjustments.py`)
- Para cada UCP (Antioquia, Atlántico) y las últimas N semanas (default 52): corte = sábado de la
  semana anterior a la objetivo (reproduce la brecha real), corre el pipeline con params default
  vía `preview`/pipeline directo, y compara cada paso contra el real.
- Métricas por paso: WAPE horario, error de energía diaria y semanal, sesgo, error de pico (MW y hora);
  cortes por UCP y tipo de día (laboral/sábado/domingo/festivo).
- Barrido opcional de α (0, 0.25, 0.5, 0.75, 1) y fuente de forma.
- Salida: `reports/ajustes/backtest_<fecha>.csv` + resumen `.md`.
- Limitaciones a declarar en el reporte: el clima es observado (no hay pronóstico archivado) y el
  modelo base puede haber visto esas semanas al entrenar → el error base es optimista; lo que importa
  es la **mejora relativa entre pasos**.

## Ejecución (según metodología global)
1. Crear `docs/work/ajustes/PLAN.md` (copia de este plan) y `LOG.md` apenas se apruebe.
2. Subagentes en paralelo (prompts autocontenidos con rutas a PLAN.md):
   - **A – Núcleo**: `types.py` primero (contrato fijo de este plan), `data.py`, los 5 pasos,
     `pipeline.py`, tests unitarios con datos sintéticos.
   - **B – Persistencia y API**: `store.py` (Postgres + InMemory), `sql/001_init.sql`,
     `apply_adjustments_schema.py`, `router.py`, wiring en `main.py`, tests del router con InMemoryStore
     y `TestClient`.
   - **C – Backtest** (después de A): script + primera corrida real + reporte.
3. Ronda de revisión multi-ángulo (line-by-line, removed-behavior, cross-file, reuse, simplification,
   efficiency, altitude, conventions) como agentes separados + agente de regresión (suite completa
   `pytest tests/`, arranque de la API, endpoints existentes intactos). Repetir solo si hay
   hallazgos críticos/altos.
4. Commit en rama nueva (no en main) tras revisión limpia.

## Verificación
- `pytest tests/` completo (incluye los existentes: holiday_homology, hourly_disaggregation, pipeline).
- Tests clave: suma diaria del paso 3 = E(d) del paso 2 (tolerancia 1e-6); paso omitido = identidad;
  PATCH recalcula pasos siguientes; corrida aprobada no se modifica; sin `AJUSTES_DATABASE_URL` la API
  arranca y `/predict` sigue respondiendo.
- Manual: levantar la API, `POST /adjustments/preview` para Antioquia semana 2026-11-02 (lunes
  festivo) y revisar que el paso 3 use forma de festivo homólogo y el control emita/no emita alerta.
- Con Postgres local: aplicar esquema, `run` → `PATCH` → `approve`, verificar filas.
- Backtest: correr 52 semanas y entregar la tabla WAPE por paso al usuario.

---

## Contratos entre módulos (fijos — A y B trabajan en paralelo contra esto)

Python 3.11, venv: `.venv/bin/python`. Tests: `.venv/bin/python -m pytest tests -q`.

### `src/adjustments/types.py` (lo escribe A, B lo importa)
```python
Curves = Dict[str, List[float]]          # {"YYYY-MM-DD": [P1..P24]}

class StepState(str, Enum): APLICADO="aplicado"; OMITIDO="omitido"; EDITADO="editado"

@dataclass
class Alert:
    nivel: str            # "info" | "warning" | "bloqueo"
    codigo: str           # ej. "TENDENCIA_FUERA_BANDA", "SIN_EVIDENCIA_NIVEL", "PERIODOS_FALTANTES"
    mensaje: str
    fechas: List[str] = field(default_factory=list)
    detalle: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StepResult:
    curves: Curves
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    alerts: List[Alert] = field(default_factory=list)

@dataclass
class RunContext:
    ucp: str
    semana_inicio: date          # lunes de la semana objetivo
    fecha_emision: date
    fecha_corte: date            # último dato real <= fecha_emision - 1
    historico: pd.DataFrame      # columnas: fecha (datetime64), P1..P24, TOTAL; solo fechas <= fecha_corte
    festivos: List[str]          # "YYYY-MM-DD"
    semana_ref_inicio: Optional[date] = None   # la fija el paso 1
    extras: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StepRecord:            # lo que se persiste por paso
    orden: int
    paso: str                # "base" | "referencia" | "nivel" | "forma" | "control"
    estado: StepState
    params: Dict[str, Any]
    result: StepResult
```
Todos los dataclasses deben tener `to_dict()` JSON-serializable y `from_dict()` (para Store).

**Actualización fix ronda 1 (núcleo):**
- `StepResult.curvas_calculadas: Curves = {}`: en `editado`, curva calculada (pre-edición) de cada día editado.
- Curvas JSON-safe: NaN/±inf → `None` en `curves_to_dict`, y `curves_from_dict` deja `None`.
- `validar_curva(valores) -> Optional[str]` (código PERIODOS_FALTANTES / VALORES_NO_FINITOS / VALORES_NEGATIVOS o None).
- `validar_curvas_editadas(curvas, fechas_permitidas) -> None` (ValueError, para el 422 de ediciones).
- Cada step: `DEFAULTS` + `validar_params(params) -> dict` (ValueError; sin TypeError). Rechaza claves desconocidas. `base.DEFAULTS = {}`; `modelo_version` va en `ctx.extras["modelo_version"]` → `diagnostics.modelo_version` del paso base.
- Pipeline: `orden_de(paso)`. `rerun_from(..., orden: int | str, ...)` exige records con exactamente los nombres de STEP_ORDER y orden coherente. `overrides` acepta claves int o nombre. Todo se identifica por `paso`.

### `src/adjustments/pipeline.py` (A)
```python
STEP_ORDER = ["base", "referencia", "nivel", "forma", "control"]   # orden 0..4
BaseProvider = Callable[[str, date, int], Curves]   # (ucp, end_date=fecha_corte, n_days) -> curvas raw

def build_context(ucp: str, semana_inicio: date, fecha_emision: Optional[date] = None,
                  historico: Optional[pd.DataFrame] = None, festivos: Optional[List[str]] = None) -> RunContext
    # si historico/festivos son None los carga desde data/raw/{ucp}/datos.csv (src/adjustments/data.py)

def run_pipeline(ctx: RunContext, base_provider: BaseProvider,
                 params: Optional[Dict[str, Dict[str, Any]]] = None,     # params por nombre de paso
                 overrides: Optional[Dict[int, StepRecord]] = None       # pasos omitidos/editados
                 ) -> List[StepRecord]

def rerun_from(ctx: RunContext, records: List[StepRecord], orden: int,
               base_provider: BaseProvider, nuevo: StepRecord) -> List[StepRecord]
    # reemplaza el paso `orden` (omitido/editado/params nuevos) y recalcula los posteriores

def approval_blockers(records: List[StepRecord]) -> List[Alert]   # alertas nivel "bloqueo"
    # alcance: SOLO la semana objetivo; los días de brecha son estimación y no se aprueban
    # ni bloquean (si son inválidos, control emite warning BRECHA_INVALIDA)
def final_curves(records: List[StepRecord]) -> Curves             # curvas del último paso que modifica
```
Paso `omitido` => curves = curves del paso anterior. Paso `editado` => curves vienen del analista
(validar 24 periodos, no negativos). El paso `referencia` escribe `ctx.semana_ref_inicio` (y lo deja
en diagnostics) — al recalcular, se toma de diagnostics del record.
Las curvas del paso `base` cubren desde fecha_corte+1 hasta domingo objetivo (incluye brecha);
los pasos siguientes solo ajustan los 7 días de la semana objetivo y conservan la brecha tal cual.

### `src/adjustments/store.py` (B)
```python
class Store(Protocol):
    def create_run(self, meta: Dict[str, Any], records: List[StepRecord]) -> str   # devuelve id
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]   # {"meta":..., "records":[StepRecord]}
    def replace_records(self, run_id: str, records: List[StepRecord], evento: Dict[str, Any]) -> None
    def approve(self, run_id: str, usuario: str, motivo: Optional[str]) -> None
    def add_event(self, run_id: str, evento: Dict[str, Any]) -> None
class InMemoryStore(Store); class PostgresStore(Store)   # psycopg2, AJUSTES_DATABASE_URL
def get_store() -> Optional[Store]   # None si no hay AJUSTES_DATABASE_URL (router -> 503)
```
meta: ucp, semana_inicio, fecha_corte, fecha_emision, modelo_version, params_iniciales, estado, creado_por.
(Contrato actualizado en ronda 2: `create_run(meta, records, evento=None)`, `replace_records(..., version_esperada,
meta_updates)`, `approve(..., version_esperada, check, detalle)`; ver docstring de store.py.)

### `src/adjustments/router.py` (B)
`router = APIRouter(prefix="/adjustments", tags=["ajustes"])` + `set_base_provider(fn)`.
`main.py` solo agrega: `from src.adjustments.router import router as adjustments_router, set_base_provider`,
`app.include_router(adjustments_router)` y un `base_provider` que llama a `run_predict_flow(PredictRequest(ucp, end_date, n_days))`
y convierte `predictions` (fecha + P1..P24) a `Curves`. `run_predict_flow` es async: el router debe
soportar provider async (await) o el pipeline recibir curvas base ya calculadas — B decide, pero
`run_pipeline` acepta un provider síncrono; para el endpoint, B calcula la base con await y pasa
`lambda *_: base_curves`.

**Actualización fix ronda 2 (núcleo):**
- Params acotados, todo error = ValueError (nunca OverflowError/TypeError): enteros validados sin pasar por float;
  `forma.n` 1..60, `referencia.max_anios_atras` 1..20, `control.top_dias` 1..7, `anios_excluidos` años 1990..2100
  (≤200 elementos, solo list/tuple/set), `semana_ref_inicio` 1990-01-01..2100-12-31 (validar_params) y no posterior a
  la semana objetivo (referencia.step → ValueError). `build_context` exige semana_inicio en 1990-01-01..2100-12-31.
- `control.DEFAULTS["banda_referencia_pp"] = 2.0` (antes None). Records guardados con `null` explícito siguen sin la
  alerta al recalcular (respetan su param).
- `approval_blockers` cubre solo la semana objetivo; `BRECHA_INVALIDA` (warning) para la brecha.
