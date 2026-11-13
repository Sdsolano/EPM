# Log — Servicio de ajustes por pasos (Fase 1)

Rama: `feat/ajustes-pipeline` (desde main 7880239). Plan maestro: `docs/work/ajustes/PLAN.md`.

## Decisiones confirmadas con el usuario
- Módulo dentro de la API EPM (`src/adjustments/`), router propio.
- Persistencia en Postgres propio de EPM (`AJUSTES_DATABASE_URL`).
- Sphere accederá vía backend intermedio (Fase 3, fuera de alcance).
- Fase 1 = back (pasos 0,1,2,3,5) + persistencia + endpoints + backtest.

## Línea base de tests (antes de tocar código)
`.venv/bin/python -m pytest tests -q --ignore=tests/test_hourly_disaggregation.py` → 16 passed, 4 failed
(preexistentes: test_future_events::test_get_future_events, test_holiday_homology::...christmas_profile_uses_prior_christmases,
test_openai_analysis x2). `tests/test_hourly_disaggregation.py` falla en colección (ImportError relativo) — preexistente.
Cambios no commiteados ajenos en el árbol (fastapi_factores/test_*, reports/performance/README.md, .claude/settings.local.json): NO tocar ni commitear.

## Progreso real
- [x] Rama + PLAN.md + contratos
- [x] Agente A (núcleo) — terminado 2026-11-12 (ver sección "Agente A" abajo)
- [x] Agente B (persistencia/API) — terminado 2026-11-12 (ver sección "Agente B" abajo)
- [x] Agente C (backtest) — terminado 2026-11-12 (ver sección "Agente C" abajo)
- [x] Fix H1/H2 + defaults (α 0.25, `promedio_n`) + backtest v2 — terminado 2026-11-12 (ver sección "Fix H1/H2 + defaults")
- [x] Ronda 1 multi-ángulo (ver "Ronda 1 de revisión"); fix ronda 1 API/store terminado (ver "Fix ronda 1 — API/store")
- [ ] Ronda 2 multi-ángulo + regresión
- [ ] Commit

## Agente B — persistencia y API (terminado)
Archivos nuevos: `src/adjustments/sql/001_init.sql`, `scripts/apply_adjustments_schema.py`,
`src/adjustments/store.py`, `src/adjustments/router.py`, `tests/test_adjustments_router.py`.
Modificados: `src/api/main.py` (1 import + bloque "AJUSTES POR PASOS" antes de INICIALIZACIÓN:
`_adjustments_base_provider` async → `run_predict_flow`, `set_base_provider`, `app.include_router`),
`requirements.txt` (`psycopg2-binary>=2.9`, `httpx>=0.25` para TestClient; httpx se instaló en .venv),
`.env.example` (`AJUSTES_DATABASE_URL` comentado).

Decisiones:
- uuid generado en Python (uuid4); DEFAULT gen_random_uuid() (PG13+) solo de respaldo. `corrida` lleva además
  `approved_by`; `corrida_evento` lleva `id bigserial` y `detalle jsonb`.
- PostgresStore: una conexión por operación; replace_records/approve bloquean la fila (`SELECT ... FOR UPDATE`),
  verifican estado y hacen DELETE+INSERT+evento en una transacción. Excepciones: `RunNotFoundError`,
  `RunApprovedError` (subclases de `StoreError`). id no-uuid → get_run None / RunNotFoundError.
- `get_run` devuelve además `"eventos"` (extensión compatible del contrato).
- base_provider del router: sync o async; devuelve `Curves` o `(Curves, {"modelo_version": ...})`.
  main.py usa `metadata["modelo_usado"]` como modelo_version. n_days = domingo objetivo − fecha_corte (1..90, si no 422).
- `set_context_loader(fn(ucp, semana_inicio, fecha_emision) -> RunContext)` para tests.
- PATCH reconstruye ctx desde meta y fuerza la `fecha_corte` guardada (recorta histórico si llegaron datos nuevos).
  Solo re-llama al modelo si se PATCHea el paso 0 con estado=aplicado; si no reusa las curvas base guardadas.
  Edición: el router envía solo los días editados + `diagnostics.fechas_editadas` (acumula ediciones previas del
  mismo paso); el rango permitido lo valida el pipeline (ValueError → 422).
- approve: bloqueos → 422 `{mensaje, bloqueos}` + evento `aprobacion_rechazada`. `excepciones_justificadas` NO
  saltan bloqueos; solo se registran (evento `justificar_excepciones`).
- Eventos: crear, aplicar_paso / omitir_paso / editar_paso (paso=orden, detalle con estado anterior/params/fechas),
  justificar_excepciones, aprobacion_rechazada, aprobar.
- Códigos: preview 200, run 201, 503 sin store (o sin base_provider), 404 corrida/UCP sin histórico, 409 aprobada,
  422 validación/bloqueos/orden inválido/omitir base.

Verificación: `tests/test_adjustments_router.py` 21 passed + 1 skipped (Postgres) sin DB; con Postgres local
temporal (`epm_ajustes_tmp_test`, ya eliminada) 22 passed y flujo run→PATCH→approve→409 por router OK.
`import src.api.main` OK, rutas /adjustments en OpenAPI, /health 200, GET /adjustments/{id} → 503 sin DB.
Suite completa: 37 passed, 1 skipped, mismos 4 fallos preexistentes.

Dudas abiertas (para revisión):
- Semántica de A en paso `editado`: los días NO editados toman las curvas del paso previo (se pierde el ajuste
  calculado del propio paso para esos días). ¿Debería conservar el resultado calculado del paso?
- `excepciones_justificadas`: hoy no permite aprobar con bloqueos. Confirmar con usuario.
- Sin autenticación: `usuario` viene en el body (lo pondrá el backend intermedio en Fase 3).
- Import de router en main.py sin try/except: un error de import en src/adjustments tumba la API completa.

## Agente A — núcleo (terminado)
Archivos nuevos: `src/adjustments/__init__.py`, `types.py`, `data.py`, `pipeline.py`,
`src/adjustments/steps/{__init__,base,referencia,nivel,forma,control}.py`,
`tests/ajustes_sinteticos.py` (generador sintético + `ProveedorPlano`, reutilizable),
`tests/test_adjustments_steps.py` (25), `tests/test_adjustments_pipeline.py` (26, incluye humo con CSV real).
No se tocó main.py/store.py/router.py.

Contrato: firmas exactas del PLAN. Extensiones compatibles: `RunContext.to_dict(include_historico=False)` /
`from_dict(d, historico=None)` (el histórico no se serializa por defecto); helpers `to_jsonable`, `parse_date`
en types.py; `base.step(..., *, base_provider)` y `control.step(..., *, energia_esperada)` reciben esos
argumentos como keyword (no van en params, que se persisten). `record.params` se guarda con defaults resueltos.

Decisiones de algoritmo:
- data: una lectura de CSV; filtra UCP y VARIABLE=Demanda_Real; TOTAL = suma P1..P24; fila válida = 24 periodos
  finitos y >= 0. Festivos = `colombia_holiday_map` (offline). NO se reusa `CalendarClassifier` de
  `calculate_base_curves`: consulta la API externa de festivos (red) → no reproducible ni testeable.
- build_context: `fecha_emision` default = min(hoy, lunes objetivo); corte = último día válido
  <= min(emisión, lunes objetivo) − 1 (nunca dentro de la semana objetivo); histórico recortado a <= corte.
- base: llama provider(ucp, corte, n_dias = domingo − corte); días faltantes → warning BASE_INCOMPLETA
  (el bloqueo lo pone control). diagnostics: fecha_corte, dias_brecha, rango, n_dias, modelo_version, energía.
- referencia: param `semana_ref_inicio` (se lleva al lunes, alerta info); sin festivos → −364; con festivos →
  k=1..6 años, año más reciente que maximiza homólogos en el mismo día de semana Y con 7 días de datos;
  semana = la que contiene el homólogo del primer festivo coincidente; si ninguno → −364 + info
  REFERENCIA_SIN_HOMOLOGO. Semana ref sin datos → warning REFERENCIA_SIN_DATOS. El paso no muta ctx: el
  pipeline copia `diagnostics.semana_ref_inicio` a `ctx` (también al reconstruir en rerun_from).
  Referencia omitida → sin semana ref (nivel sin evidencia, control sin banda).
- nivel: ancla a = d − 7·ceil((d−corte)/7); a' = d' − lag. Anclas nunca festivas (se retrocede de a 7 días
  ambas, hasta 6 semanas extra — junio 2025/2026 tiene lunes festivos seguidos). d no festivo con d' festivo →
  d' ± 7/14 manteniendo el lag. d festivo → d' = festivo de la ref en la misma posición o el homólogo del año
  ref; sin homólogo con datos → sin evidencia (α=0, warning SIN_EVIDENCIA_NIVEL). alpha fuera de [0,1] → ValueError.
- forma: suma diaria = energía del paso previo (verificado <= 1e-6). "reciente" = último día real del MISMO día
  de semana no festivo (más estricto que "mismo tipo"); "referencia" cae a reciente si d' no tiene datos o es
  festivo (info FORMA_FUENTE_ALTERNA); festivo → forma de `average_holiday_hourly_profile`
  (scale_with_recent_sundays=False, solo forma); sin homólogos → domingo reciente.
- control: tendencia esperada = E(sem ref)/E(sem ref −7) − 1; resultante = E(sem ajustada)/E(sem previa) − 1
  (sem previa = real si <= corte, si no la brecha estimada). Desvío por día vs E_previa(d−7)·E_ref(d')/E_ref(d'−7);
  top 3 en la dirección del desvío. Bloqueos: PERIODOS_FALTANTES, VALORES_NO_FINITOS, VALORES_NEGATIVOS,
  RECONCILIACION_ROTA (vs energía del paso nivel; se omite si la forma fue editada a mano).
- pipeline: base no se puede omitir; editar solo base/nivel/forma (referencia se fuerza con param; control
  no produce curvas) → ValueError. Edición: solo fechas de la semana objetivo (base: todo su rango).
  **Editado = el paso se calcula como aplicado y luego los días editados lo reemplazan** (los no editados
  conservan el ajuste calculado; responde la duda abierta de B). `approval_blockers` = bloqueos de los records
  + validación estructural de `final_curves` en la semana (cubre control omitido). `final_curves` incluye brecha.

Verificación: `pytest tests/test_adjustments_steps.py tests/test_adjustments_pipeline.py -q` → 51 passed.
Humo real: Antioquia termina en 2026-07-04, así que NO hay datos para la semana 2026-11-02; se usa la última
semana disponible 2026-06-29 (lunes festivo San Pedro): ref 2025-06-30 por homólogo, nivel festivo ratio 0.81,
forma festivo homólogo, sin bloqueos.

PENDIENTE de coordinación (fuera del alcance de A): `tests/test_adjustments_router.py::test_patch_editar_curvas`
(línea ~211) asume la semántica anterior de editado (días no editados = paso previo) y ahora falla. Hay que cambiar
esa aserción, p. ej. a que el día no editado sea igual al de una corrida sin edición, o a que
`steps[2]["curves"][otro] != steps[1]["curves"][otro]`. El resto de la suite: mismos 4 fallos preexistentes.


## Orquestador — integración A+B (2026-11-12)
- Decisión: semántica de paso `editado` = se calcula el paso normal y los días editados lo sobrescriben (versión de A). Se corrigió la aserción en tests/test_adjustments_router.py::test_patch_editar_curvas.
- Suite: 88 passed, 1 skipped (Postgres), 4 failed preexistentes. test_hourly_disaggregation sigue fallando en colección (preexistente).
- Dudas abiertas a reportar al usuario: referencia omitida = sin referencia (no default); excepciones_justificadas no desbloquean; sin auth (usuario en body, lo pondrá backend intermedio); import del router no protegido en main.py (evaluar en revisión).
- Siguiente: Agente C (backtest) en curso.


## Agente C — backtest (terminado 2026-11-12)
Archivos nuevos: `scripts/backtest_adjustments.py`, `tests/test_backtest_adjustments.py` (6 tests, sintético, sin red),
salidas `reports/ajustes/backtest_20261112.csv` (39 375 filas) y `.md`. OJO: `/reports` y `/docs` están en .gitignore →
para commitear el reporte hace falta `git add -f`. No se tocó `src/adjustments/` ni `main.py`; data/raw, models y
data/features verificados sin cambios (diff contra copia previa).

Método: por UCP, últimas 52 semanas completas (Antioquia 2025-06-30..2026-06-22, Atlántico 2025-05-05..2026-04-27);
corte = sábado anterior (fecha_emision = domingo), histórico recortado al corte y verificado (test de no-fuga: alterar
datos posteriores al corte no cambia ninguna curva). `run_pipeline` por config (α ∈ {0,.25,.5,.75,1} × fuente ∈
{reciente, referencia, promedio_n}). Corre en ~17 s.

Base: `run_predict_flow` NO viable offline: llama `full_update_csv` (red + reescribe data/raw/{ucp}/datos.csv), consulta
festivos por red, y con pandas 3.0.6 del venv falla en `src/pipeline/feature_engineering.py:493`
(`fillna(method='ffill')` → TypeError; bug de entorno preexistente: requirements dice `pandas>=2.0.0` sin tope);
además los modelos de Antioquia se entrenaron 2026-07-08 (fuga). Se usó `--base ingenuo` (mismo día de la semana real
anterior, ancla igual a la del paso nivel) y `--base archivo` (reports/performance/antioquia_ene_jun_2026/
resultados_horarios_ene_jun_2026.csv: una emisión con corte 2025-12-31 + ajustes post-hoc; 21 semanas Antioquia).
`--base modelo` queda implementado (parchea full_update_csv) pero falla en este venv.

Resultados (default α=0.5, reciente; WAPE horario base → nivel → forma | MAPE energía semanal base → final):
- ingenuo Antioquia: 4.05 → 3.74 → 3.67 % | 2.69 → 2.26 %
- ingenuo Atlántico: 6.27 → 6.33 → 6.27 % | 3.58 → 3.29 %  (16 semanas sin referencia por hueco de datos 2024)
- archivo Antioquia: 3.03 → 3.01 → 2.99 % | 2.11 → 1.97 %
Mejor config: α=0.25 + `promedio_n` (archivo Antioquia 2.78 %, Atlántico 5.92 %); ingenuo Antioquia α=0.5 + promedio_n
(3.63 %). Por tipo de día: festivos α 0.75–1, domingos α=0. `referencia` siempre peor fuente.
Recomendación: defaults α=0.25 y fuente `promedio_n`; a futuro α por tipo de día.

Alerta TENDENCIA_FUERA_BANDA: precision 0.75–0.93 vs tasa base 0.62–0.81 (lift 0.02–0.13), recall 0.41–0.69 → poco útil.

Hallazgos (documentados, NO corregidos — para la ronda de revisión / decisión del usuario):
- H1 (diseño control/referencia, severidad media-alta para la utilidad de la alerta): semana objetivo sin festivo con
  ref −364 que sí tiene festivo (o semanas previas con distinta estructura de festivos) → tendencia esperada errónea.
  Ej. Antioquia 2026-06-22, ref 2025-06-23 (lunes festivo): esperada −4.07 %, real +4.95 %, resultante +0.24 % →
  alerta falsa (desvío 4.31 pp). En semanas con desajuste el error de t_esp es 5.0–6.5 pp vs 1.8–2.2 pp sin desajuste.
  Sugerencia: tendencia esperada a partir de E*(d) del paso nivel, o excluir días festivos.
- H2 (referencia, severidad media): con festivo de fecha fija busca hasta 6 años atrás y puede elegir 2020 (pandemia).
  Antioquia/Atlántico 2026-04-27 → ref 2020-04-27; nivel empeora el error semanal (ingenuo Antioquia +3.14 % → +5.19 %,
  archivo −0.73 % → +3.26 %, Atlántico WAPE 2.80 → 5.08 %). Sugerencia: excluir 2020 y limitar a 3–4 años.
- H3 (datos, no bug): hueco Atlántico 2024-09-01..2024-11-29 y dic-2024 → REFERENCIA_SIN_DATOS, α=0 (degrada bien).
- 0 bloqueos y reconciliación forma = nivel exacta en todas las corridas.

Tests: `pytest tests -q --ignore=tests/test_hourly_disaggregation.py` → 94 passed, 1 skipped, 4 failed (los mismos
preexistentes; +6 nuevos del backtest).


## Fix H1/H2 + defaults (terminado 2026-11-12)
Archivos tocados (todos dentro del alcance del módulo; nada en fastapi_factores/, reports/performance/, .claude/, main.py, router.py):
- `src/adjustments/steps/control.py` — **H1**: tendencia esperada = Σ E_esp(d) / E(semana previa real+estimada) − 1,
  E_esp(d) = E*(d) (`E_objetivo` del paso nivel, use o no α<1). Día sin E* → energía base de ese día, listado en
  `diagnostics.dias_sin_evidencia` y `dias[f].fuente_esperada = "energia_base"`. Sin E* en ningún día (o sin ref) →
  info SIN_TENDENCIA_ESPERADA, no evalúa banda. Desvío por día = E_ajustada(d) − E_esp(d). Nuevo kw
  `nivel_dias` (diagnostics["dias"] del nivel); si es None (nivel omitido / llamada directa) recalcula con
  `nivel.evidencias(ctx)` y toma E_base = energía de la curva recibida. Nuevo diag `energia_semana_esperada`.
  `energia_esperada` sigue siendo SOLO para reconciliación (sin cambios).
- `src/adjustments/steps/nivel.py` — default α 0.25; función pública `evidencias(ctx)` (E* por día, la usa
  step y control); `anios_excluidos(ctx)` lee `ctx.extras["anios_excluidos"]` (default (2020,)); d′, desplazamientos
  ±7/±14, homólogo festivo y a′ se descartan si caen en año excluido. diag `anios_excluidos`.
- `src/adjustments/steps/referencia.py` — **H2**: DEFAULTS `max_anios_atras: 4` (reemplaza `max_anios: 6`),
  `anios_excluidos: [2020]`. Semana excluida = algún día en año excluido. −364 retrocede −728… hasta max (si todas
  excluidas: −364 + warning REFERENCIA_ANIO_EXCLUIDO). Búsqueda por festivo salta años excluidos. Semana forzada
  por param en año excluido: se respeta + warning REFERENCIA_ANIO_EXCLUIDO (nivel no la usará salvo
  `anios_excluidos: []`). diag `anios_excluidos`, `max_anios_atras`. Records viejos con `max_anios` quedan con la
  clave sobrante (inofensiva); al recalcular usan el default nuevo.
- `src/adjustments/steps/forma.py` — default `fuente: "promedio_n"` (n=4).
- `src/adjustments/pipeline.py` — `_aplicar_referencia(ctx, diag)` copia semana ref y `anios_excluidos` a ctx
  (run_pipeline, _ejecutar y rerun_from); `_nivel_dias(previos)` pasa la evidencia del nivel al control (None si
  nivel omitido). Contrato del router intacto (control recibe kw nuevos solo desde el pipeline).
- `scripts/backtest_adjustments.py` — `CONFIG_DEFAULT` se lee de los DEFAULTS de nivel/forma (0.25, promedio_n).
- `docs/work/ajustes/PLAN.md` — tabla de pasos con los defaults/reglas nuevas.
- Tests: `tests/test_adjustments_steps.py` +8 (H2: festivo fijo nunca 2020 y respeta max, max limita, −364 salta
  2020, festivo móvil salta 2020, forzada en 2020 alerta + nivel sin evidencia, anclas ref no caen en 2020; H1:
  día sin E* usa base y se marca, sin E* en ningún día no evalúa banda) usando `historico_sintetico(2015..2026-06)`;
  `tests/test_adjustments_pipeline.py` +2 (semana 2025-05-12 sin festivo con ref 2024-05-13 festiva → sin alerta
  falsa y esperado = Σ E_objetivo del nivel, y la fórmula vieja habría desviado > 1 pp; nivel omitido → control
  recalcula E*) y aserción de defaults actualizada; `tests/test_backtest_adjustments.py` CONFIGS incluye
  `bt.CONFIG_DEFAULT` (sin él `evaluar_alerta` queda vacío). Ninguna aserción de reconciliación se tocó.

Backtest v2: `.venv/bin/python scripts/backtest_adjustments.py --base ingenuo archivo --fecha 20261112_v2` (~17 s)
→ `reports/ajustes/backtest_20261112_v2.{csv,md}` (39 375 filas); sección 6 del .md escrita a mano con la comparación.
Resumen (WAPE final v1 (0.5,reciente) → v2 mismos params → v2 default (0.25,promedio_n)):
ingenuo Antioquia 3.67 → 3.61 → 3.71; ingenuo Atlántico 6.27 → 6.22 → 5.90; archivo Antioquia 2.99 → 2.93 → 2.79.
|t_esp − t_real| en semanas con festivos desalineados: ingenuo Antioquia 4.31 → 2.48 pp, archivo 6.09 → 2.79 pp
(≈ semanas alineadas); 2026-06-22 y 2026-04-20 ya no alertan en ingenuo Antioquia. 2026-04-27: ref 2020 → 2025-04-28,
error semanal final ingenuo Antioquia +5.19 → +2.25 %, archivo +3.26 → −0.66 %, Atlántico +1.37 → −0.86 %.
Ninguna ref en 2020; 0 bloqueos.
Pendiente / no resuelto:
- Alerta TENDENCIA_FUERA_BANDA sigue sin poder predictivo (lift ≈ 0 con los defaults nuevos; con α=0.25 alerta más:
  34/52 Antioquia). Es estructural: el desvío = (1−α)·Σ(E_base − E*)/E_prev mide desacuerdo base vs referencia.
  Propuestas en la sección 6 del reporte v2 (banda por volatilidad, alerta informativa).
- α=0.25 empeora festivos (WAPE festivo ingenuo Antioquia 9.28 → 12.57 %); se recomienda α por tipo de día (fase siguiente).
- En ingenuo/Antioquia α=0.5 sigue siendo mejor que 0.25 (3.57 vs 3.71 %).
- Exclusión estricta de a′: semanas objetivo de inicio de enero 2022 (ref enero 2021) quedan sin evidencia porque a′
  cae en dic-2020 (cubierto por test; irrelevante para fechas actuales).
Tests: `pytest tests -q --ignore=tests/test_hourly_disaggregation.py` → 104 passed, 1 skipped, 4 failed (los mismos
preexistentes: future_events, christmas profile, 2 openai_analysis). +10 tests nuevos.
Sin ronda de revisión multi-ángulo todavía (la hace el orquestador); no se commiteó.

## Ronda 1 de revisión multi-ángulo (2026-11-12) — resultado: HAY ALTOS → corregir + ronda 2 completa
Regresión: sin regresiones (endpoints existentes intactos, 4 fallos preexistentes idénticos, fastapi_factores 14/14).

### Altos (obligan ronda 2)
- R1-A [line-by-line, efficiency, cross-file] `fecha_corte` se calcula del CSV ANTES de que `run_predict_flow` lo refresque → /run da 422 "brecha 141 días" para la semana próxima, o usa corte viejo.
- R1-B [removed-behavior] preview/run de semana pasada llama `run_predict_flow(end_date=pasado)` → puede entrenar/guardar modelo truncado y sobrescribe `data_with_features_latest.csv` truncado (afecta /predict, /health).
- R1-C [altitude] paso `editado` pierde la curva calculada de los días editados → imposible comparar automático vs analista (F09).
- R1-D [altitude + line-by-line] sin versionado: PATCH sobrescribe todo; sin control de concurrencia (última escritura gana); approve evalúa bloqueos fuera del lock; N corridas aprobadas para la misma semana.

### Medios a corregir en esta pasada
- Pasos identificados por posición (choca con paso 4 futuro) → identificar por nombre en API/store.
- Control circular tras fix H1 (desvío = (1−α)·Σ(E_base−E*)) → rediseño: enclavamiento mide desvío de la semana final vs energía del paso nivel (detecta ediciones que rompen la tendencia); tendencia de referencia como diagnóstico.
- NaN/inf en curvas rompe PostgresStore; params null/inf → 500 o corrida irrecuperable.
- `HTTP_422_UNPROCESSABLE_CONTENT` requiere Starlette ≥0.48 → usar 422 literal.
- Festivos: ajustes usan librería `holidays`, modelo usa FestivosAPIClient por UCP (difieren: 2025-12-31 Antioquia, 2026-07-13 Atlántico).
- `run_predict_flow` bloquea el event loop desde los endpoints nuevos.
- `modelo_version` nunca llega a diagnostics del paso base; re-aplicar base descarta la versión nueva.
- docs/ y reports/ ignorados por git → versionar solo .md con `git add -f` (CSV de 10 MB no).
### Bajos (se corrigen si son baratos)
validadores de curvas triplicados, ANIOS_EXCLUIDOS duplicado, regla de ancla triplicada, `_n_days` duplica `rango_base`, router sin logging, `.env.example` sin indicar archivo, test_postgres_store no limpia, fixture no restaura provider, limpiezas de simplification.
### Aceptados (no se corrigen ahora)
- Backtest in-sample y con base no-modelo: se declara en el reporte; defaults son provisionales hasta backtest con modelo real (requiere arreglar pandas/fillna del pipeline, fuera de alcance).
- PATCH reconstruye contexto desde el CSV actual recortado a la fecha_corte guardada (no snapshot del histórico): con versionado queda trazable; snapshot completo = fase siguiente.
- Pool de conexiones Postgres, payload de curvas duplicadas por paso: aceptable para 1–3 analistas.

## Fix ronda 1 — API/store (terminado 2026-11-12)
Archivos: `src/adjustments/router.py` (reescrito), `src/adjustments/store.py` (reescrito), `src/adjustments/sql/002_versionado.sql`
(nuevo), `src/api/main.py` (bloque ajustes), `tests/test_adjustments_router.py` (reescrito), `.env.example`.
No se tocaron archivos del núcleo (consume `curvas_calculadas`, `validar_curvas_editadas` de types, `pl.orden_de`,
`ctx.extras["modelo_version"]`). `scripts/apply_adjustments_schema.py` sin cambios (ya aplica todos los .sql en orden).
- R1-A/R1-B: sin `base_curvas` → (lock por UCP) refresco (`set_data_refresher`, main = `full_update_csv`) → histórico →
  último dato real → `build_context(fecha_emision = último+1)` ⇒ corte = último real → modelo con end_date = corte.
  Semana con datos reales → 422; `fecha_emision`/`modelo_version` sin `base_curvas` → 422. Con `base_curvas` no hay
  refresco ni modelo; deben cubrir corte+1..domingo con 24 periodos (NaN permitidos → bloqueo del pipeline).
  Provider de main corre `asyncio.run(run_predict_flow(...))` en threadpool. Errores del modelo/refresco → 502
  `{mensaje, causa}` (primera línea, ≤300 chars). Re-aplicar base: refresca; 422 si llegaron datos posteriores al corte
  o si la base fue externa; actualiza `meta.modelo_version` y lo registra (detalle.modelo_version[_anterior]).
- R1-D: `corrida.version` + `corrida_version` (snapshot por versión, incluida la inicial y el approve);
  `version_esperada` obligatoria en PATCH/approve, verificada bajo `FOR UPDATE` (VersionConflictError → 409 con
  version_actual); approve re-evalúa `approval_blockers` sobre los records leídos dentro del lock; advisory lock por
  (ucp, semana) + índice único parcial; la aprobada anterior pasa a `reemplazada` (+ `reemplazada_por`, evento
  `reemplazar`). `GET /adjustments/aprobada`, `GET /{id}/versiones[/{v}]`. `excepciones_justificadas` va en el
  detalle del evento `aprobar` (ya no es evento aparte).
- Pasos por nombre: `PATCH /{id}/steps/{paso}`; PK `corrida_paso (corrida_id, paso)`; `corrida_evento.paso` → text
  (migra 0..4 a nombres); columna `curvas_calculadas`; respuesta con `version` y `curvas_calculadas` por paso.
- Festivos: loader default = `FestivosAPIClient.get_festivos` por UCP (threadpool); si falla/vacío → librería,
  `meta.extras.fuente_festivos = "libreria"` + alerta info `FESTIVOS_FUENTE_ALTERNA` en base (se re-anota tras recalcular).
  Los festivos usados se guardan en `extras.festivos` y se reutilizan en PATCH (no se exponen en la respuesta).
- Bajos: literales 422/…; logger (crear/editar/aprobar/rechazo/503/502); `.env.example` indica .env raíz;
  setters devuelven el anterior y el fixture los restaura; `_n_days` usa `rango_base`; `parse_date` en vez de `_to_date`;
  store serializa JSON estricto (NaN/inf → null). Re-read post-escritura se conserva (devuelve eventos/meta del store).
- 002 probado sobre una base con datos de 001 (2 aprobadas duplicadas → la vieja queda `reemplazada`; eventos 2→'nivel';
  backfill de versión 1) y re-aplicado (idempotente). Bases temporales borradas (no queda ninguna `epm_ajustes_*`).
Tests: router 44 passed + 3 skipped sin DSN; con `AJUSTES_DATABASE_URL=postgresql:///postgres` 47 passed (base temporal
creada y borrada por el fixture; incluye aprobaciones concurrentes reales y PATCH concurrente). Suite completa:
183 passed, 3 skipped, 4 failed (los mismos preexistentes). `import src.api.main` sin DSN OK, /health 200, rutas en OpenAPI.
Aceptado/pendiente: con el modelo hay doble refresco (`full_update_csv` del router + el interno de `run_predict_flow`),
costo de red aceptado; si entre ambos llega un día nuevo, la base empieza un día después → BASE_INCOMPLETA/bloqueo (no
corrompe). El lock por UCP solo serializa /adjustments, no /predict concurrente (preexistente).

## Fix ronda 1 — núcleo (terminado 2026-11-12)
Alcance: types.py, data.py, pipeline.py, steps/*, backtest y sus tests. No se tocaron router/store/sql/main/.env.example ni test_adjustments_router.py.

Contrato final (lo consume el agente de router/store):
1. **R1-C**: `StepResult.curvas_calculadas: Curves` (default `{}`), incluido en `to_dict`/`from_dict` (from_dict tolera que falte). En un paso `editado` guarda la curva CALCULADA (pre-edición) de cada día editado. `curves` trae la editada. Store: hay que persistir `result["curvas_calculadas"]` (nueva columna/jsonb).
2. **No finitos**: `curves_to_dict` convierte NaN/±inf/no numérico a `None`, y `curves_from_dict` deja `None`. En types.py:
   - `validar_curva(valores) -> Optional[str]` devuelve el código `PERIODOS_FALTANTES` | `VALORES_NO_FINITOS` | `VALORES_NEGATIVOS` o None, y nunca lanza (los textos están en `MOTIVOS_CURVA`).
   - `validar_curvas_editadas(curvas, fechas_permitidas) -> None` lanza ValueError si no hay curvas, si una fecha está fuera de rango o si una curva es inválida.
   - Las usan pipeline (`_aplicar_edicion`), `control.validar_curvas` y `approval_blockers`.
   - nivel y forma dejan intactos los días con curva previa inválida; control los bloquea. No hay excepción con NaN/None.
3. **Params**: cada step expone `DEFAULTS` + `validar_params(params) -> dict`, apoyadas en el helper nuevo `steps/_params.py`. Completan defaults, castean (int→float, 4.0→4, fecha→ISO) y lanzan ValueError ante:
   - claves desconocidas (un typo no se ignora en silencio);
   - None, no finitos o bool;
   - rangos: alpha y beta en [0,1], n ≥ 1, max_anios_atras ≥ 1, banda_pp > 0, top_dias ≥ 1, banda_referencia_pp None o > 0;
   - anios_excluidos que no sea lista de int; fuente fuera de FUENTES; semana_ref_inicio que no sea str/date ISO.
   Claves obsoletas que se descartan sin error: base `modelo_version` y referencia `max_anios`. El pipeline llama `validar_params` siempre en `_ejecutar` (run y rerun). `run_pipeline` rechaza también nombres de paso desconocidos en `params`.
4. **Por nombre**:
   - `orden_de(paso) -> int` (ValueError si el paso no existe).
   - `rerun_from(ctx, records, orden: int|str, ...)` acepta orden o nombre y exige `nuevo.paso == STEP_ORDER[orden]`. Exige además que `records` traiga exactamente los 5 nombres, sin repetidos ni desconocidos, y `r.orden == orden_de(r.paso)`; si no, ValueError.
   - `run_pipeline(overrides=...)` acepta claves int o nombre, y el `paso` del override debe coincidir con la clave.
   - Todo se indexa por `paso` internamente. `final_curves` va por nombre.
5. **modelo_version**: fuera de `base.DEFAULTS` (base no tiene params). `base.step` la toma de `ctx.extras["modelo_version"]`. En base editado, `diagnostics.energia` refleja las curvas editadas y la calculada queda en `curvas_calculadas`.

Lógica:
- **Control rediseñado (enclavamiento)**:
  - `desvio_pp = (E_semana_final / E_semana_nivel − 1)·100`.
  - E_nivel = energía que CALCULÓ el paso nivel (para días editados, `curvas_calculadas`). Con nivel omitido se usa la energía del paso base (`diagnostics.origen_energia_nivel` = "nivel" | "base").
  - Fuera de `banda_pp` (1.0, sin recalibrar: sin ediciones el backtest no informa) → warning TENDENCIA_FUERA_BANDA con los top días por contribución.
  - Sin ediciones → desvío 0, sin alerta.
  - Diagnóstico informativo (fórmula post-H1): tendencia_esperada_pct, tendencia_resultante_pct, `diferencia_tendencia_pp`, `dias_sin_evidencia`, `dias[f].energia_esperada`. Solo alerta (TENDENCIA_REFERENCIA_FUERA_BANDA) si se pasa el param `banda_referencia_pp` (default None).
  - Infos nuevas: SIN_ENERGIA_NIVEL (llamada directa sin energía) y SIN_CONTROL_TENDENCIA (faltan días). Se eliminaron SIN_TENDENCIA_RESULTANTE y SIN_TENDENCIA_ESPERADA como alertas: el motivo queda en `diagnostics.tendencia_motivo`.
  - Reconciliación: con forma editada, ahora se reconcilian los días NO editados (antes se saltaba toda la forma).
- **Limpiezas**:
  - `data.ANIOS_EXCLUIDOS_DEFAULT` como constante única.
  - `data.ancla_mismo_dow(d, corte)` usado por nivel, forma y backtest (se borró `bt.ancla`).
  - Se eliminaron: la rama base-editada-sin-proveedor, `_PASOS_EDITABLES` (queda `_PASOS_QUE_MODIFICAN`), `e_obj … else 0.0`, la duplicación de alertas default/excluido en referencia (`_default_con_alerta`, `_alerta_anio_excluido`) y la normalización doble en forma.
  - `data.energia_curvas` ya no lanza con None.
  - base convierte valores no convertibles a NaN.

Backtest v3: `.venv/bin/python scripts/backtest_adjustments.py --base ingenuo archivo --fecha 20261112_v3` → `reports/ajustes/backtest_20261112_v3.{csv,md}`.
- Las curvas son idénticas a v2 (diferencia máxima 0.0 en las 39 375 filas), así que el WAPE por paso es igual.
- TENDENCIA_FUERA_BANDA pasa de 34/31/14 alertas a 0/0/0, con |desvío| máx 2e-14.
- La alerta de v2 se reproduce exacta como diagnóstico `diferencia_tendencia_pp` (tabla "hipotética", lift 0).
- 0 bloqueos.
- El .md trae la sección 3 reescrita, las Limitaciones ampliadas (in-sample, base no-modelo, corte en sábado, defaults provisionales, alerta sin evaluar) y la sección 6 de comparación.

Tests:
- Los 3 archivos del núcleo pasan: 123 passed.
- Nuevos: curvas_calculadas (editado, base editado, rerun, ida y vuelta, ausencia), NaN/inf en base → bloqueo + `json.dumps(allow_nan=False)` + rerun desde records con None, `validar_curvas_editadas`/`validar_curva`, params inválidos (unitarios y en run/rerun), records desalineados (5 casos), `orden_de`, rerun por nombre, override con paso que no coincide, control sin ediciones = 0, edición +10 % en nivel y en forma → alerta, `modelo_version` desde ctx, pasos que toleran curva previa inválida, backtest sin ediciones = 0 alertas.
- Suite completa (`--ignore=tests/test_hourly_disaggregation.py`): 183 passed, 3 skipped, solo los 4 fallos preexistentes. En una corrida intermedia fallaron 2 tests NaN del router mientras el otro agente los editaba; al repetir pasaron.

## Orquestador — cierre fix ronda 1 (2026-11-12)
- Ambos agentes de corrección terminaron (se reanudaron tras corte por límite de uso). Suite: 183 passed, 3 skipped, 4 failed preexistentes.
- Siguiente: ronda 2 COMPLETA (8 ángulos + regresión, agentes separados, modelo Sonnet por pedido del usuario) contra el diff completo actual.

## Ronda 2 de revisión (Sonnet, 8 ángulos + regresión) — 2026-11-12
Regresión: sin regresiones (16→183 passed, mismos 4 fallos, rutas/esquemas previos intactos, import sin IO).
### Tratado como ALTO (regla "ante la duda, alto") → corregir + ronda 3 completa
- R2-A [line-by-line, removed-behavior] `ucp` sin validar llega a `full_update_csv` → `ucp="../x"` crea dirs/CSV fuera de data/raw y llama a red con ese valor, en endpoint sin auth (preview ni siquiera exige store). Mismo patrón preexistente en /predict, superficie nueva aquí.
### Medios a corregir
- Re-editar `base` pierde la curva del modelo en `curvas_calculadas` de días ya editados (3 ángulos, CONFIRMED).
- `meta.params` queda desactualizado tras PATCH.
- UCP sin modelo → el request entrena en línea con el lock tomado (minutos) → responder 409.
- `OverflowError` en params/fechas extremas → 500.
- `excepciones_justificadas` no se valida contra alertas vigentes ni registra los warnings vistos al aprobar.
- Sin auth: agregar secreto servicio-a-servicio opcional (`AJUSTES_SERVICE_TOKEN`).
- Control: la coherencia con la tendencia de referencia quedó apagada; emitirla como alerta `info` por defecto (control de proceso, distinto del enclavamiento).
- Reporte v3: filas de festivo con base ingenua no son interpretables; decirlo.
### Bajos a corregir si baratos
evento `reemplazar` version distinta InMemory/Postgres; evento `crear` atómico con create_run; `base_ref` opcional en RunRequest→extras; validación UCP antes de crear lock.
### Aceptados
params por día/tipo de día (fase siguiente), des-editar un solo día, preview sobre corrida existente, trabajo duplicado entre corridas en cola de la misma UCP, duplicación corrida_paso vs corrida_version y paridad InMemory/Postgres, brecha no validada en aprobación (la aprobación cubre la semana objetivo; se documenta), RAW_DATA_DIR local en data.py (evita efectos de importar settings), logs sin emojis.
- Corrección del usuario (2026-11-12): R2-A se reclasifica como MEDIO (ningún agente lo reportó alto; patrón preexistente en /predict). Ronda 2 sin críticos/altos → NO hay ronda 3: tras los fixes, una verificación estándar (suite completa + Postgres + import API) y commit.

## Fix ronda 2 — núcleo (terminado 2026-11-12)
Alcance: types/data/pipeline/steps, backtest y sus tests, reporte v3. No se tocaron router/store/sql/main ni
test_adjustments_router.py (los edita otro agente en paralelo).
- `steps/_params.py`: `numero` captura OverflowError (float(10**400)) → ValueError; `entero(minimo, maximo)` obligatorio,
  ints sin pasar por float; `lista_enteros(minimo, maximo, max_items=200)` solo list/tuple/set; mensajes con repr acotado
  (`_corto`); constantes `FECHA_MIN=1990-01-01`, `FECHA_MAX=2100-12-31`.
- Cotas: `forma.MAX_N=60`, `referencia.MAX_ANIOS_ATRAS=20`, `control.top_dias ≤ 7`, `anios_excluidos` en 1990..2100.
  `referencia.semana_ref_inicio` en [FECHA_MIN, FECHA_MAX] (validar_params) y no posterior a la semana objetivo
  (`referencia.step` → ValueError; el router ya convierte ValueError del pipeline en 422). "9999-12-27" antes
  desbordaba en `dias_semana` (+6 d) → ahora ValueError. `pipeline.build_context` exige semana_inicio en
  [FECHA_MIN, FECHA_MAX] (protege −364·20 y festivos). Verificado con tests que ref=1990-01-01, max_anios_atras=20, n=60,
  top_dias=7 corren referencia→nivel→forma→control sin excepción.
  NOTA para el agente de router: si el router hace aritmética con `semana_inicio` ANTES de `pl.build_context`
  (p. ej. rango del loader de festivos), una semana 0001/9999 podría desbordar ahí; conviene validar el rango en RunRequest.
- `control.py`: docstring reescrito con los DOS controles (enclavamiento = edición, warning; coherencia con referencia =
  proceso, info). `TENDENCIA_REFERENCIA_FUERA_BANDA` ahora `info`, default `banda_referencia_pp=2.0` (null desactiva);
  detalle con tendencias y dias_sin_evidencia. Nueva `_brecha`: `diagnostics.dias_brecha` y warning `BRECHA_INVALIDA`
  (días de brecha faltantes/inválidos, `detalle.motivos`), no bloquea.
- `pipeline.approval_blockers`: docstring — solo semana objetivo; brecha no se reporta ni bloquea.
- `scripts/backtest_adjustments.py`: columnas `alerta_tendencia_referencia_info`, `banda_referencia_pp`;
  `evaluar_alerta(..., columna_banda)`; sección 3 "Alertas de tendencia" con las dos alertas + la hipotética ±1 pp;
  `nota_festivos(df)` bajo "Mejor configuración…" (números del propio detalle); LIMITACIONES con festivos ingenuo no
  interpretables / archivo no concluyentes / α por tipo de día pendiente; `secciones_manuales(md)` preserva "## 6." en
  adelante al regenerar.
- Backtest v3 re-corrido (mismos flags) → curvas idénticas (dif. máx 0.0, 39 375 filas), 0 bloqueos, 0 BRECHA_INVALIDA,
  enclavamiento 0. Alerta info con defaults: ingenuo Antioquia 20/52 (precision 0.65 vs tasa base 0.56), ingenuo
  Atlántico 21/52 (0.48 vs 0.56), archivo 7/21 (0.43 vs 0.38). Sección 7 escrita a mano en el .md; bullet de la sección 6
  sobre "apagada por defecto" marcado como superado. Backup del v3 previo en el scratchpad de la sesión (no versionado).
- Tests: steps +31 (params extremos 10**400/fechas 0001 y 9999/cotas, límites válidos, ref posterior, valores límite que
  no desbordan, build_context fuera de rango); pipeline +7 (info con alpha=0 / nivel omitido / base editada sin
  enclavamiento, null desactiva, BRECHA_INVALIDA no bloquea + JSON-safe, brecha válida sin alerta, ref posterior en
  run_pipeline); backtest +3. Default de control actualizado en el test de defaults.
- Verificación: núcleo `162 passed`; suite completa `--ignore=tests/test_hourly_disaggregation.py` → 238 passed,
  4 skipped, 4 failed (los mismos preexistentes).
- Aceptado: records persistidos con `banda_referencia_pp: null` explícito (corridas creadas antes de este fix) no emiten la
  alerta info al recalcular — respetan el param guardado.

## Fix ronda 2 — API/store (terminado 2026-11-12)
Archivos: `src/adjustments/router.py`, `src/adjustments/store.py`, `src/api/main.py` (bloque ajustes + import),
`tests/test_adjustments_router.py`, `.env.example`. Sin SQL nuevo (no hizo falta 003). No se tocaron
steps/_params.py, steps/control.py, pipeline.py, types.py ni el reporte (otro agente).
- R2-A (ALTO) UCP: `RunRequest.ucp` y query de `/aprobada` con `pattern=^[A-Za-z0-9_-]{1,64}$` (422 de Pydantic);
  `_validar_ucp` (primera línea de `_compute` y antes de re-aplicar base) exige `data/raw/{ucp}/datos.csv` resuelto
  dentro de `data.RAW_DATA_DIR` (leído en cada llamada, misma constante que data.py) → 404. `_ucp_lock` valida antes
  de crear la entrada del dict. Tests: `../x`, `a/b`, `..`, vacío, 65 chars, con espacio → 422; `Foo` → 404; symlink
  que escapa → 404; con refresher/provider que fallan si se invocan, sin cambios en el árbol de data/ y sin locks nuevos.
- Re-editar base: provider = `{**base.curves, **base.curvas_calculadas}` → `curvas_calculadas[X]` sigue siendo la del
  modelo tras editar X y luego Y (InMemory y Postgres; verificado que el test falla sin el fix).
- `meta.params` → `meta.params_iniciales` (request original). En BD la columna sigue siendo `corrida.params`
  (`SELECT params AS params_iniciales`): lo más simple, sin migración.
- Modelo sin entrenar: `set_model_checker(fn)`; main.py `_adjustments_model_checker` = `check_model_exists(ucp)[0] and
  check_hourly_disaggregation_trained(ucp)` (run_predict_flow entrena ambos si faltan). Sin modelo → 409 "La UCP no
  tiene modelo entrenado; use /retrain" ANTES del lock/refresco (run y re-aplicar base); con base_curvas no se verifica.
  Si el checker lanza → 502.
- `OverflowError`: `ERRORES_ENTRADA = (ValueError, OverflowError)` en todos los puntos que convierten a 422 (histórico,
  build_context, fechas/curvas, run_pipeline, validar_curvas_editadas, rerun_from). Además `semana_inicio`/`fecha_emision`
  fuera de 1990-01-01..2100-12-31 (`_params.FECHA_MIN/FECHA_MAX`, misma cota que el núcleo) → 422 en el modelo Pydantic, antes de cualquier cálculo de fechas (rango de festivos, contexto); pedido del agente del núcleo, test con 0001-01-01, 1989-12-25, 9999-12-27, 2101-01-03. Test con alpha = 10**400 en preview y PATCH.
- `excepciones_justificadas`: `_approval_check` (corre bajo lock en el store) devuelve bloqueos; si no hay, calcula los
  warnings vigentes de esa versión, rechaza códigos no vigentes (`JustificacionInvalidaError` → 422 con
  codigos_invalidos/warnings_vigentes, sin aprobar) y completa el `detalle` del evento `aprobar` con warnings_vigentes y
  warnings_sin_justificar (dedup de justificadas). No exige justificar todos. Store: `detalle if detalle is not None`.
- Auth: dependencia de router `verificar_token`; `AJUSTES_SERVICE_TOKEN` leído por request; cabecera `X-Ajustes-Token`,
  `hmac.compare_digest` → 401 (antes que la validación del body). Sin variable: permitido + warning una vez.
  `.env.example` documentado (placeholder). Tests de ambos modos.
- Bajos: evento `reemplazar` con la versión de la corrida reemplazada en ambos stores (Postgres `RETURNING id, version`);
  `create_run(meta, records, evento=None)` inserta `crear` en la misma transacción (InMemory valida antes de mutar);
  `base_ref` (≤200) → `meta.extras.base_ref` y detalle del evento `crear`.
Tests: router 64 passed + 4 skipped sin DSN; con `AJUSTES_DATABASE_URL=postgresql:///postgres` 68 passed (bases temporales
borradas; 0 `epm_ajustes_*`). Suite (`--ignore=tests/test_hourly_disaggregation.py`): 242 passed, 4 skipped, 4 failed
(los mismos preexistentes). `import src.api.main` sin variables de entorno OK: /health 200, rutas /adjustments en
OpenAPI, preview `../x` → 422, `Foo` → 404, GET sin DSN → 503; checker real: Antioquia/Atlantico True.

## Verificación estándar final y commit (2026-11-12)
- Suite: 242 passed, 4 skipped, 4 failed preexistentes. Router con Postgres: 68 passed. Import de la API sin env: /health 200, 8 rutas /adjustments. Sin bases temporales.
- Ronda 2 sin críticos/altos → sin ronda 3 (regla del framework). Commit en `feat/ajustes-pipeline`.
- Pendientes Fase 2/3: backtest con el modelo real (requiere arreglar fillna/pandas 3 en feature_engineering y aislar full_update_csv), α por tipo de día, pantalla Mesa de ajustes en Sphere + rutas en backend intermedio (enviar `X-Ajustes-Token`, `base_curvas`, `base_ref`), aplicar `sql/001`+`002` en el Postgres de despliegue y definir `AJUSTES_DATABASE_URL`/`AJUSTES_SERVICE_TOKEN`.
