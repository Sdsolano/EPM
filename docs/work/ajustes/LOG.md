# Log â€” Servicio de ajustes por pasos (Fase 1)

Rama: `feat/ajustes-pipeline` (desde main 7880239). Plan maestro: `docs/work/ajustes/PLAN.md`.

## Decisiones confirmadas con el usuario
- MÃ³dulo dentro de la API EPM (`src/adjustments/`), router propio.
- Persistencia en Postgres propio de EPM (`AJUSTES_DATABASE_URL`).
- Sphere accederÃ¡ vÃ­a backend intermedio (Fase 3, fuera de alcance).
- Fase 1 = back (pasos 0,1,2,3,5) + persistencia + endpoints + backtest.

## LÃ­nea base de tests (antes de tocar cÃ³digo)
`.venv/bin/python -m pytest tests -q --ignore=tests/test_hourly_disaggregation.py` â†’ 16 passed, 4 failed
(preexistentes: test_future_events::test_get_future_events, test_holiday_homology::...christmas_profile_uses_prior_christmases,
test_openai_analysis x2). `tests/test_hourly_disaggregation.py` falla en colecciÃ³n (ImportError relativo) â€” preexistente.
Cambios no commiteados ajenos en el Ã¡rbol (fastapi_factores/test_*, reports/performance/README.md, .claude/settings.local.json): NO tocar ni commitear.

## Progreso real
- [x] Rama + PLAN.md + contratos
- [x] Agente A (nÃºcleo) â€” terminado 2026-11-12 (ver secciÃ³n "Agente A" abajo)
- [x] Agente B (persistencia/API) â€” terminado 2026-11-12 (ver secciÃ³n "Agente B" abajo)
- [x] Agente C (backtest) â€” terminado 2026-11-12 (ver secciÃ³n "Agente C" abajo)
- [x] Fix H1/H2 + defaults (Î± 0.25, `promedio_n`) + backtest v2 â€” terminado 2026-11-12 (ver secciÃ³n "Fix H1/H2 + defaults")
- [x] Ronda 1 multi-Ã¡ngulo (ver "Ronda 1 de revisiÃ³n"); fix ronda 1 API/store terminado (ver "Fix ronda 1 â€” API/store")
- [ ] Ronda 2 multi-Ã¡ngulo + regresiÃ³n
- [ ] Commit

## Agente B â€” persistencia y API (terminado)
Archivos nuevos: `src/adjustments/sql/001_init.sql`, `scripts/apply_adjustments_schema.py`,
`src/adjustments/store.py`, `src/adjustments/router.py`, `tests/test_adjustments_router.py`.
Modificados: `src/api/main.py` (1 import + bloque "AJUSTES POR PASOS" antes de INICIALIZACIÃ“N:
`_adjustments_base_provider` async â†’ `run_predict_flow`, `set_base_provider`, `app.include_router`),
`requirements.txt` (`psycopg2-binary>=2.9`, `httpx>=0.25` para TestClient; httpx se instalÃ³ en .venv),
`.env.example` (`AJUSTES_DATABASE_URL` comentado).

Decisiones:
- uuid generado en Python (uuid4); DEFAULT gen_random_uuid() (PG13+) solo de respaldo. `corrida` lleva ademÃ¡s
  `approved_by`; `corrida_evento` lleva `id bigserial` y `detalle jsonb`.
- PostgresStore: una conexiÃ³n por operaciÃ³n; replace_records/approve bloquean la fila (`SELECT ... FOR UPDATE`),
  verifican estado y hacen DELETE+INSERT+evento en una transacciÃ³n. Excepciones: `RunNotFoundError`,
  `RunApprovedError` (subclases de `StoreError`). id no-uuid â†’ get_run None / RunNotFoundError.
- `get_run` devuelve ademÃ¡s `"eventos"` (extensiÃ³n compatible del contrato).
- base_provider del router: sync o async; devuelve `Curves` o `(Curves, {"modelo_version": ...})`.
  main.py usa `metadata["modelo_usado"]` como modelo_version. n_days = domingo objetivo âˆ’ fecha_corte (1..90, si no 422).
- `set_context_loader(fn(ucp, semana_inicio, fecha_emision) -> RunContext)` para tests.
- PATCH reconstruye ctx desde meta y fuerza la `fecha_corte` guardada (recorta histÃ³rico si llegaron datos nuevos).
  Solo re-llama al modelo si se PATCHea el paso 0 con estado=aplicado; si no reusa las curvas base guardadas.
  EdiciÃ³n: el router envÃ­a solo los dÃ­as editados + `diagnostics.fechas_editadas` (acumula ediciones previas del
  mismo paso); el rango permitido lo valida el pipeline (ValueError â†’ 422).
- approve: bloqueos â†’ 422 `{mensaje, bloqueos}` + evento `aprobacion_rechazada`. `excepciones_justificadas` NO
  saltan bloqueos; solo se registran (evento `justificar_excepciones`).
- Eventos: crear, aplicar_paso / omitir_paso / editar_paso (paso=orden, detalle con estado anterior/params/fechas),
  justificar_excepciones, aprobacion_rechazada, aprobar.
- CÃ³digos: preview 200, run 201, 503 sin store (o sin base_provider), 404 corrida/UCP sin histÃ³rico, 409 aprobada,
  422 validaciÃ³n/bloqueos/orden invÃ¡lido/omitir base.

VerificaciÃ³n: `tests/test_adjustments_router.py` 21 passed + 1 skipped (Postgres) sin DB; con Postgres local
temporal (`epm_ajustes_tmp_test`, ya eliminada) 22 passed y flujo runâ†’PATCHâ†’approveâ†’409 por router OK.
`import src.api.main` OK, rutas /adjustments en OpenAPI, /health 200, GET /adjustments/{id} â†’ 503 sin DB.
Suite completa: 37 passed, 1 skipped, mismos 4 fallos preexistentes.

Dudas abiertas (para revisiÃ³n):
- SemÃ¡ntica de A en paso `editado`: los dÃ­as NO editados toman las curvas del paso previo (se pierde el ajuste
  calculado del propio paso para esos dÃ­as). Â¿DeberÃ­a conservar el resultado calculado del paso?
- `excepciones_justificadas`: hoy no permite aprobar con bloqueos. Confirmar con usuario.
- Sin autenticaciÃ³n: `usuario` viene en el body (lo pondrÃ¡ el backend intermedio en Fase 3).
- Import de router en main.py sin try/except: un error de import en src/adjustments tumba la API completa.

## Agente A â€” nÃºcleo (terminado)
Archivos nuevos: `src/adjustments/__init__.py`, `types.py`, `data.py`, `pipeline.py`,
`src/adjustments/steps/{__init__,base,referencia,nivel,forma,control}.py`,
`tests/ajustes_sinteticos.py` (generador sintÃ©tico + `ProveedorPlano`, reutilizable),
`tests/test_adjustments_steps.py` (25), `tests/test_adjustments_pipeline.py` (26, incluye humo con CSV real).
No se tocÃ³ main.py/store.py/router.py.

Contrato: firmas exactas del PLAN. Extensiones compatibles: `RunContext.to_dict(include_historico=False)` /
`from_dict(d, historico=None)` (el histÃ³rico no se serializa por defecto); helpers `to_jsonable`, `parse_date`
en types.py; `base.step(..., *, base_provider)` y `control.step(..., *, energia_esperada)` reciben esos
argumentos como keyword (no van en params, que se persisten). `record.params` se guarda con defaults resueltos.

Decisiones de algoritmo:
- data: una lectura de CSV; filtra UCP y VARIABLE=Demanda_Real; TOTAL = suma P1..P24; fila vÃ¡lida = 24 periodos
  finitos y >= 0. Festivos = `colombia_holiday_map` (offline). NO se reusa `CalendarClassifier` de
  `calculate_base_curves`: consulta la API externa de festivos (red) â†’ no reproducible ni testeable.
- build_context: `fecha_emision` default = min(hoy, lunes objetivo); corte = Ãºltimo dÃ­a vÃ¡lido
  <= min(emisiÃ³n, lunes objetivo) âˆ’ 1 (nunca dentro de la semana objetivo); histÃ³rico recortado a <= corte.
- base: llama provider(ucp, corte, n_dias = domingo âˆ’ corte); dÃ­as faltantes â†’ warning BASE_INCOMPLETA
  (el bloqueo lo pone control). diagnostics: fecha_corte, dias_brecha, rango, n_dias, modelo_version, energÃ­a.
- referencia: param `semana_ref_inicio` (se lleva al lunes, alerta info); sin festivos â†’ âˆ’364; con festivos â†’
  k=1..6 aÃ±os, aÃ±o mÃ¡s reciente que maximiza homÃ³logos en el mismo dÃ­a de semana Y con 7 dÃ­as de datos;
  semana = la que contiene el homÃ³logo del primer festivo coincidente; si ninguno â†’ âˆ’364 + info
  REFERENCIA_SIN_HOMOLOGO. Semana ref sin datos â†’ warning REFERENCIA_SIN_DATOS. El paso no muta ctx: el
  pipeline copia `diagnostics.semana_ref_inicio` a `ctx` (tambiÃ©n al reconstruir en rerun_from).
  Referencia omitida â†’ sin semana ref (nivel sin evidencia, control sin banda).
- nivel: ancla a = d âˆ’ 7Â·ceil((dâˆ’corte)/7); a' = d' âˆ’ lag. Anclas nunca festivas (se retrocede de a 7 dÃ­as
  ambas, hasta 6 semanas extra â€” junio 2025/2026 tiene lunes festivos seguidos). d no festivo con d' festivo â†’
  d' Â± 7/14 manteniendo el lag. d festivo â†’ d' = festivo de la ref en la misma posiciÃ³n o el homÃ³logo del aÃ±o
  ref; sin homÃ³logo con datos â†’ sin evidencia (Î±=0, warning SIN_EVIDENCIA_NIVEL). alpha fuera de [0,1] â†’ ValueError.
- forma: suma diaria = energÃ­a del paso previo (verificado <= 1e-6). "reciente" = Ãºltimo dÃ­a real del MISMO dÃ­a
  de semana no festivo (mÃ¡s estricto que "mismo tipo"); "referencia" cae a reciente si d' no tiene datos o es
  festivo (info FORMA_FUENTE_ALTERNA); festivo â†’ forma de `average_holiday_hourly_profile`
  (scale_with_recent_sundays=False, solo forma); sin homÃ³logos â†’ domingo reciente.
- control: tendencia esperada = E(sem ref)/E(sem ref âˆ’7) âˆ’ 1; resultante = E(sem ajustada)/E(sem previa) âˆ’ 1
  (sem previa = real si <= corte, si no la brecha estimada). DesvÃ­o por dÃ­a vs E_previa(dâˆ’7)Â·E_ref(d')/E_ref(d'âˆ’7);
  top 3 en la direcciÃ³n del desvÃ­o. Bloqueos: PERIODOS_FALTANTES, VALORES_NO_FINITOS, VALORES_NEGATIVOS,
  RECONCILIACION_ROTA (vs energÃ­a del paso nivel; se omite si la forma fue editada a mano).
- pipeline: base no se puede omitir; editar solo base/nivel/forma (referencia se fuerza con param; control
  no produce curvas) â†’ ValueError. EdiciÃ³n: solo fechas de la semana objetivo (base: todo su rango).
  **Editado = el paso se calcula como aplicado y luego los dÃ­as editados lo reemplazan** (los no editados
  conservan el ajuste calculado; responde la duda abierta de B). `approval_blockers` = bloqueos de los records
  + validaciÃ³n estructural de `final_curves` en la semana (cubre control omitido). `final_curves` incluye brecha.

VerificaciÃ³n: `pytest tests/test_adjustments_steps.py tests/test_adjustments_pipeline.py -q` â†’ 51 passed.
Humo real: Antioquia termina en 2026-07-04, asÃ­ que NO hay datos para la semana 2026-11-02; se usa la Ãºltima
semana disponible 2026-06-29 (lunes festivo San Pedro): ref 2025-06-30 por homÃ³logo, nivel festivo ratio 0.81,
forma festivo homÃ³logo, sin bloqueos.

PENDIENTE de coordinaciÃ³n (fuera del alcance de A): `tests/test_adjustments_router.py::test_patch_editar_curvas`
(lÃ­nea ~211) asume la semÃ¡ntica anterior de editado (dÃ­as no editados = paso previo) y ahora falla. Hay que cambiar
esa aserciÃ³n, p. ej. a que el dÃ­a no editado sea igual al de una corrida sin ediciÃ³n, o a que
`steps[2]["curves"][otro] != steps[1]["curves"][otro]`. El resto de la suite: mismos 4 fallos preexistentes.


## Orquestador â€” integraciÃ³n A+B (2026-11-12)
- DecisiÃ³n: semÃ¡ntica de paso `editado` = se calcula el paso normal y los dÃ­as editados lo sobrescriben (versiÃ³n de A). Se corrigiÃ³ la aserciÃ³n en tests/test_adjustments_router.py::test_patch_editar_curvas.
- Suite: 88 passed, 1 skipped (Postgres), 4 failed preexistentes. test_hourly_disaggregation sigue fallando en colecciÃ³n (preexistente).
- Dudas abiertas a reportar al usuario: referencia omitida = sin referencia (no default); excepciones_justificadas no desbloquean; sin auth (usuario en body, lo pondrÃ¡ backend intermedio); import del router no protegido en main.py (evaluar en revisiÃ³n).
- Siguiente: Agente C (backtest) en curso.


## Agente C â€” backtest (terminado 2026-11-12)
Archivos nuevos: `scripts/backtest_adjustments.py`, `tests/test_backtest_adjustments.py` (6 tests, sintÃ©tico, sin red),
salidas `reports/ajustes/backtest_20261112.csv` (39 375 filas) y `.md`. OJO: `/reports` y `/docs` estÃ¡n en .gitignore â†’
para commitear el reporte hace falta `git add -f`. No se tocÃ³ `src/adjustments/` ni `main.py`; data/raw, models y
data/features verificados sin cambios (diff contra copia previa).

MÃ©todo: por UCP, Ãºltimas 52 semanas completas (Antioquia 2025-06-30..2026-06-22, AtlÃ¡ntico 2025-05-05..2026-04-27);
corte = sÃ¡bado anterior (fecha_emision = domingo), histÃ³rico recortado al corte y verificado (test de no-fuga: alterar
datos posteriores al corte no cambia ninguna curva). `run_pipeline` por config (Î± âˆˆ {0,.25,.5,.75,1} Ã— fuente âˆˆ
{reciente, referencia, promedio_n}). Corre en ~17 s.

Base: `run_predict_flow` NO viable offline: llama `full_update_csv` (red + reescribe data/raw/{ucp}/datos.csv), consulta
festivos por red, y con pandas 3.0.6 del venv falla en `src/pipeline/feature_engineering.py:493`
(`fillna(method='ffill')` â†’ TypeError; bug de entorno preexistente: requirements dice `pandas>=2.0.0` sin tope);
ademÃ¡s los modelos de Antioquia se entrenaron 2026-07-08 (fuga). Se usÃ³ `--base ingenuo` (mismo dÃ­a de la semana real
anterior, ancla igual a la del paso nivel) y `--base archivo` (reports/performance/antioquia_ene_jun_2026/
resultados_horarios_ene_jun_2026.csv: una emisiÃ³n con corte 2025-12-31 + ajustes post-hoc; 21 semanas Antioquia).
`--base modelo` queda implementado (parchea full_update_csv) pero falla en este venv.

Resultados (default Î±=0.5, reciente; WAPE horario base â†’ nivel â†’ forma | MAPE energÃ­a semanal base â†’ final):
- ingenuo Antioquia: 4.05 â†’ 3.74 â†’ 3.67 % | 2.69 â†’ 2.26 %
- ingenuo AtlÃ¡ntico: 6.27 â†’ 6.33 â†’ 6.27 % | 3.58 â†’ 3.29 %  (16 semanas sin referencia por hueco de datos 2024)
- archivo Antioquia: 3.03 â†’ 3.01 â†’ 2.99 % | 2.11 â†’ 1.97 %
Mejor config: Î±=0.25 + `promedio_n` (archivo Antioquia 2.78 %, AtlÃ¡ntico 5.92 %); ingenuo Antioquia Î±=0.5 + promedio_n
(3.63 %). Por tipo de dÃ­a: festivos Î± 0.75â€“1, domingos Î±=0. `referencia` siempre peor fuente.
RecomendaciÃ³n: defaults Î±=0.25 y fuente `promedio_n`; a futuro Î± por tipo de dÃ­a.

Alerta TENDENCIA_FUERA_BANDA: precision 0.75â€“0.93 vs tasa base 0.62â€“0.81 (lift 0.02â€“0.13), recall 0.41â€“0.69 â†’ poco Ãºtil.

Hallazgos (documentados, NO corregidos â€” para la ronda de revisiÃ³n / decisiÃ³n del usuario):
- H1 (diseÃ±o control/referencia, severidad media-alta para la utilidad de la alerta): semana objetivo sin festivo con
  ref âˆ’364 que sÃ­ tiene festivo (o semanas previas con distinta estructura de festivos) â†’ tendencia esperada errÃ³nea.
  Ej. Antioquia 2026-06-22, ref 2025-06-23 (lunes festivo): esperada âˆ’4.07 %, real +4.95 %, resultante +0.24 % â†’
  alerta falsa (desvÃ­o 4.31 pp). En semanas con desajuste el error de t_esp es 5.0â€“6.5 pp vs 1.8â€“2.2 pp sin desajuste.
  Sugerencia: tendencia esperada a partir de E*(d) del paso nivel, o excluir dÃ­as festivos.
- H2 (referencia, severidad media): con festivo de fecha fija busca hasta 6 aÃ±os atrÃ¡s y puede elegir 2020 (pandemia).
  Antioquia/AtlÃ¡ntico 2026-04-27 â†’ ref 2020-04-27; nivel empeora el error semanal (ingenuo Antioquia +3.14 % â†’ +5.19 %,
  archivo âˆ’0.73 % â†’ +3.26 %, AtlÃ¡ntico WAPE 2.80 â†’ 5.08 %). Sugerencia: excluir 2020 y limitar a 3â€“4 aÃ±os.
- H3 (datos, no bug): hueco AtlÃ¡ntico 2024-09-01..2024-11-29 y dic-2024 â†’ REFERENCIA_SIN_DATOS, Î±=0 (degrada bien).
- 0 bloqueos y reconciliaciÃ³n forma = nivel exacta en todas las corridas.

Tests: `pytest tests -q --ignore=tests/test_hourly_disaggregation.py` â†’ 94 passed, 1 skipped, 4 failed (los mismos
preexistentes; +6 nuevos del backtest).


## Fix H1/H2 + defaults (terminado 2026-11-12)
Archivos tocados (todos dentro del alcance del mÃ³dulo; nada en fastapi_factores/, reports/performance/, .claude/, main.py, router.py):
- `src/adjustments/steps/control.py` â€” **H1**: tendencia esperada = Î£ E_esp(d) / E(semana previa real+estimada) âˆ’ 1,
  E_esp(d) = E*(d) (`E_objetivo` del paso nivel, use o no Î±<1). DÃ­a sin E* â†’ energÃ­a base de ese dÃ­a, listado en
  `diagnostics.dias_sin_evidencia` y `dias[f].fuente_esperada = "energia_base"`. Sin E* en ningÃºn dÃ­a (o sin ref) â†’
  info SIN_TENDENCIA_ESPERADA, no evalÃºa banda. DesvÃ­o por dÃ­a = E_ajustada(d) âˆ’ E_esp(d). Nuevo kw
  `nivel_dias` (diagnostics["dias"] del nivel); si es None (nivel omitido / llamada directa) recalcula con
  `nivel.evidencias(ctx)` y toma E_base = energÃ­a de la curva recibida. Nuevo diag `energia_semana_esperada`.
  `energia_esperada` sigue siendo SOLO para reconciliaciÃ³n (sin cambios).
- `src/adjustments/steps/nivel.py` â€” default Î± 0.25; funciÃ³n pÃºblica `evidencias(ctx)` (E* por dÃ­a, la usa
  step y control); `anios_excluidos(ctx)` lee `ctx.extras["anios_excluidos"]` (default (2020,)); dâ€², desplazamientos
  Â±7/Â±14, homÃ³logo festivo y aâ€² se descartan si caen en aÃ±o excluido. diag `anios_excluidos`.
- `src/adjustments/steps/referencia.py` â€” **H2**: DEFAULTS `max_anios_atras: 4` (reemplaza `max_anios: 6`),
  `anios_excluidos: [2020]`. Semana excluida = algÃºn dÃ­a en aÃ±o excluido. âˆ’364 retrocede âˆ’728â€¦ hasta max (si todas
  excluidas: âˆ’364 + warning REFERENCIA_ANIO_EXCLUIDO). BÃºsqueda por festivo salta aÃ±os excluidos. Semana forzada
  por param en aÃ±o excluido: se respeta + warning REFERENCIA_ANIO_EXCLUIDO (nivel no la usarÃ¡ salvo
  `anios_excluidos: []`). diag `anios_excluidos`, `max_anios_atras`. Records viejos con `max_anios` quedan con la
  clave sobrante (inofensiva); al recalcular usan el default nuevo.
- `src/adjustments/steps/forma.py` â€” default `fuente: "promedio_n"` (n=4).
- `src/adjustments/pipeline.py` â€” `_aplicar_referencia(ctx, diag)` copia semana ref y `anios_excluidos` a ctx
  (run_pipeline, _ejecutar y rerun_from); `_nivel_dias(previos)` pasa la evidencia del nivel al control (None si
  nivel omitido). Contrato del router intacto (control recibe kw nuevos solo desde el pipeline).
- `scripts/backtest_adjustments.py` â€” `CONFIG_DEFAULT` se lee de los DEFAULTS de nivel/forma (0.25, promedio_n).
- `docs/work/ajustes/PLAN.md` â€” tabla de pasos con los defaults/reglas nuevas.
- Tests: `tests/test_adjustments_steps.py` +8 (H2: festivo fijo nunca 2020 y respeta max, max limita, âˆ’364 salta
  2020, festivo mÃ³vil salta 2020, forzada en 2020 alerta + nivel sin evidencia, anclas ref no caen en 2020; H1:
  dÃ­a sin E* usa base y se marca, sin E* en ningÃºn dÃ­a no evalÃºa banda) usando `historico_sintetico(2015..2026-06)`;
  `tests/test_adjustments_pipeline.py` +2 (semana 2025-05-12 sin festivo con ref 2024-05-13 festiva â†’ sin alerta
  falsa y esperado = Î£ E_objetivo del nivel, y la fÃ³rmula vieja habrÃ­a desviado > 1 pp; nivel omitido â†’ control
  recalcula E*) y aserciÃ³n de defaults actualizada; `tests/test_backtest_adjustments.py` CONFIGS incluye
  `bt.CONFIG_DEFAULT` (sin Ã©l `evaluar_alerta` queda vacÃ­o). Ninguna aserciÃ³n de reconciliaciÃ³n se tocÃ³.

Backtest v2: `.venv/bin/python scripts/backtest_adjustments.py --base ingenuo archivo --fecha 20261112_v2` (~17 s)
â†’ `reports/ajustes/backtest_20261112_v2.{csv,md}` (39 375 filas); secciÃ³n 6 del .md escrita a mano con la comparaciÃ³n.
Resumen (WAPE final v1 (0.5,reciente) â†’ v2 mismos params â†’ v2 default (0.25,promedio_n)):
ingenuo Antioquia 3.67 â†’ 3.61 â†’ 3.71; ingenuo AtlÃ¡ntico 6.27 â†’ 6.22 â†’ 5.90; archivo Antioquia 2.99 â†’ 2.93 â†’ 2.79.
|t_esp âˆ’ t_real| en semanas con festivos desalineados: ingenuo Antioquia 4.31 â†’ 2.48 pp, archivo 6.09 â†’ 2.79 pp
(â‰ˆ semanas alineadas); 2026-06-22 y 2026-04-20 ya no alertan en ingenuo Antioquia. 2026-04-27: ref 2020 â†’ 2025-04-28,
error semanal final ingenuo Antioquia +5.19 â†’ +2.25 %, archivo +3.26 â†’ âˆ’0.66 %, AtlÃ¡ntico +1.37 â†’ âˆ’0.86 %.
Ninguna ref en 2020; 0 bloqueos.
Pendiente / no resuelto:
- Alerta TENDENCIA_FUERA_BANDA sigue sin poder predictivo (lift â‰ˆ 0 con los defaults nuevos; con Î±=0.25 alerta mÃ¡s:
  34/52 Antioquia). Es estructural: el desvÃ­o = (1âˆ’Î±)Â·Î£(E_base âˆ’ E*)/E_prev mide desacuerdo base vs referencia.
  Propuestas en la secciÃ³n 6 del reporte v2 (banda por volatilidad, alerta informativa).
- Î±=0.25 empeora festivos (WAPE festivo ingenuo Antioquia 9.28 â†’ 12.57 %); se recomienda Î± por tipo de dÃ­a (fase siguiente).
- En ingenuo/Antioquia Î±=0.5 sigue siendo mejor que 0.25 (3.57 vs 3.71 %).
- ExclusiÃ³n estricta de aâ€²: semanas objetivo de inicio de enero 2022 (ref enero 2021) quedan sin evidencia porque aâ€²
  cae en dic-2020 (cubierto por test; irrelevante para fechas actuales).
Tests: `pytest tests -q --ignore=tests/test_hourly_disaggregation.py` â†’ 104 passed, 1 skipped, 4 failed (los mismos
preexistentes: future_events, christmas profile, 2 openai_analysis). +10 tests nuevos.
Sin ronda de revisiÃ³n multi-Ã¡ngulo todavÃ­a (la hace el orquestador); no se commiteÃ³.

## Ronda 1 de revisiÃ³n multi-Ã¡ngulo (2026-11-12) â€” resultado: HAY ALTOS â†’ corregir + ronda 2 completa
RegresiÃ³n: sin regresiones (endpoints existentes intactos, 4 fallos preexistentes idÃ©nticos, fastapi_factores 14/14).

### Altos (obligan ronda 2)
- R1-A [line-by-line, efficiency, cross-file] `fecha_corte` se calcula del CSV ANTES de que `run_predict_flow` lo refresque â†’ /run da 422 "brecha 141 dÃ­as" para la semana prÃ³xima, o usa corte viejo.
- R1-B [removed-behavior] preview/run de semana pasada llama `run_predict_flow(end_date=pasado)` â†’ puede entrenar/guardar modelo truncado y sobrescribe `data_with_features_latest.csv` truncado (afecta /predict, /health).
- R1-C [altitude] paso `editado` pierde la curva calculada de los dÃ­as editados â†’ imposible comparar automÃ¡tico vs analista (F09).
- R1-D [altitude + line-by-line] sin versionado: PATCH sobrescribe todo; sin control de concurrencia (Ãºltima escritura gana); approve evalÃºa bloqueos fuera del lock; N corridas aprobadas para la misma semana.

### Medios a corregir en esta pasada
- Pasos identificados por posiciÃ³n (choca con paso 4 futuro) â†’ identificar por nombre en API/store.
- Control circular tras fix H1 (desvÃ­o = (1âˆ’Î±)Â·Î£(E_baseâˆ’E*)) â†’ rediseÃ±o: enclavamiento mide desvÃ­o de la semana final vs energÃ­a del paso nivel (detecta ediciones que rompen la tendencia); tendencia de referencia como diagnÃ³stico.
- NaN/inf en curvas rompe PostgresStore; params null/inf â†’ 500 o corrida irrecuperable.
- `HTTP_422_UNPROCESSABLE_CONTENT` requiere Starlette â‰¥0.48 â†’ usar 422 literal.
- Festivos: ajustes usan librerÃ­a `holidays`, modelo usa FestivosAPIClient por UCP (difieren: 2025-12-31 Antioquia, 2026-07-13 AtlÃ¡ntico).
- `run_predict_flow` bloquea el event loop desde los endpoints nuevos.
- `modelo_version` nunca llega a diagnostics del paso base; re-aplicar base descarta la versiÃ³n nueva.
- docs/ y reports/ ignorados por git â†’ versionar solo .md con `git add -f` (CSV de 10 MB no).
### Bajos (se corrigen si son baratos)
validadores de curvas triplicados, ANIOS_EXCLUIDOS duplicado, regla de ancla triplicada, `_n_days` duplica `rango_base`, router sin logging, `.env.example` sin indicar archivo, test_postgres_store no limpia, fixture no restaura provider, limpiezas de simplification.
### Aceptados (no se corrigen ahora)
- Backtest in-sample y con base no-modelo: se declara en el reporte; defaults son provisionales hasta backtest con modelo real (requiere arreglar pandas/fillna del pipeline, fuera de alcance).
- PATCH reconstruye contexto desde el CSV actual recortado a la fecha_corte guardada (no snapshot del histÃ³rico): con versionado queda trazable; snapshot completo = fase siguiente.
- Pool de conexiones Postgres, payload de curvas duplicadas por paso: aceptable para 1â€“3 analistas.

## Fix ronda 1 â€” API/store (terminado 2026-11-12)
Archivos: `src/adjustments/router.py` (reescrito), `src/adjustments/store.py` (reescrito), `src/adjustments/sql/002_versionado.sql`
(nuevo), `src/api/main.py` (bloque ajustes), `tests/test_adjustments_router.py` (reescrito), `.env.example`.
No se tocaron archivos del nÃºcleo (consume `curvas_calculadas`, `validar_curvas_editadas` de types, `pl.orden_de`,
`ctx.extras["modelo_version"]`). `scripts/apply_adjustments_schema.py` sin cambios (ya aplica todos los .sql en orden).
- R1-A/R1-B: sin `base_curvas` â†’ (lock por UCP) refresco (`set_data_refresher`, main = `full_update_csv`) â†’ histÃ³rico â†’
  Ãºltimo dato real â†’ `build_context(fecha_emision = Ãºltimo+1)` â‡’ corte = Ãºltimo real â†’ modelo con end_date = corte.
  Semana con datos reales â†’ 422; `fecha_emision`/`modelo_version` sin `base_curvas` â†’ 422. Con `base_curvas` no hay
  refresco ni modelo; deben cubrir corte+1..domingo con 24 periodos (NaN permitidos â†’ bloqueo del pipeline).
  Provider de main corre `asyncio.run(run_predict_flow(...))` en threadpool. Errores del modelo/refresco â†’ 502
  `{mensaje, causa}` (primera lÃ­nea, â‰¤300 chars). Re-aplicar base: refresca; 422 si llegaron datos posteriores al corte
  o si la base fue externa; actualiza `meta.modelo_version` y lo registra (detalle.modelo_version[_anterior]).
- R1-D: `corrida.version` + `corrida_version` (snapshot por versiÃ³n, incluida la inicial y el approve);
  `version_esperada` obligatoria en PATCH/approve, verificada bajo `FOR UPDATE` (VersionConflictError â†’ 409 con
  version_actual); approve re-evalÃºa `approval_blockers` sobre los records leÃ­dos dentro del lock; advisory lock por
  (ucp, semana) + Ã­ndice Ãºnico parcial; la aprobada anterior pasa a `reemplazada` (+ `reemplazada_por`, evento
  `reemplazar`). `GET /adjustments/aprobada`, `GET /{id}/versiones[/{v}]`. `excepciones_justificadas` va en el
  detalle del evento `aprobar` (ya no es evento aparte).
- Pasos por nombre: `PATCH /{id}/steps/{paso}`; PK `corrida_paso (corrida_id, paso)`; `corrida_evento.paso` â†’ text
  (migra 0..4 a nombres); columna `curvas_calculadas`; respuesta con `version` y `curvas_calculadas` por paso.
- Festivos: loader default = `FestivosAPIClient.get_festivos` por UCP (threadpool); si falla/vacÃ­o â†’ librerÃ­a,
  `meta.extras.fuente_festivos = "libreria"` + alerta info `FESTIVOS_FUENTE_ALTERNA` en base (se re-anota tras recalcular).
  Los festivos usados se guardan en `extras.festivos` y se reutilizan en PATCH (no se exponen en la respuesta).
- Bajos: literales 422/â€¦; logger (crear/editar/aprobar/rechazo/503/502); `.env.example` indica .env raÃ­z;
  setters devuelven el anterior y el fixture los restaura; `_n_days` usa `rango_base`; `parse_date` en vez de `_to_date`;
  store serializa JSON estricto (NaN/inf â†’ null). Re-read post-escritura se conserva (devuelve eventos/meta del store).
- 002 probado sobre una base con datos de 001 (2 aprobadas duplicadas â†’ la vieja queda `reemplazada`; eventos 2â†’'nivel';
  backfill de versiÃ³n 1) y re-aplicado (idempotente). Bases temporales borradas (no queda ninguna `epm_ajustes_*`).
Tests: router 44 passed + 3 skipped sin DSN; con `AJUSTES_DATABASE_URL=postgresql:///postgres` 47 passed (base temporal
creada y borrada por el fixture; incluye aprobaciones concurrentes reales y PATCH concurrente). Suite completa:
183 passed, 3 skipped, 4 failed (los mismos preexistentes). `import src.api.main` sin DSN OK, /health 200, rutas en OpenAPI.
Aceptado/pendiente: con el modelo hay doble refresco (`full_update_csv` del router + el interno de `run_predict_flow`),
costo de red aceptado; si entre ambos llega un dÃ­a nuevo, la base empieza un dÃ­a despuÃ©s â†’ BASE_INCOMPLETA/bloqueo (no
corrompe). El lock por UCP solo serializa /adjustments, no /predict concurrente (preexistente).

## Fix ronda 1 â€” nÃºcleo (terminado 2026-11-12)
Alcance: types.py, data.py, pipeline.py, steps/*, backtest y sus tests. No se tocaron router/store/sql/main/.env.example ni test_adjustments_router.py.

Contrato final (lo consume el agente de router/store):
1. **R1-C**: `StepResult.curvas_calculadas: Curves` (default `{}`), incluido en `to_dict`/`from_dict` (from_dict tolera que falte). En un paso `editado` guarda la curva CALCULADA (pre-ediciÃ³n) de cada dÃ­a editado. `curves` trae la editada. Store: hay que persistir `result["curvas_calculadas"]` (nueva columna/jsonb).
2. **No finitos**: `curves_to_dict` convierte NaN/Â±inf/no numÃ©rico a `None`, y `curves_from_dict` deja `None`. En types.py:
   - `validar_curva(valores) -> Optional[str]` devuelve el cÃ³digo `PERIODOS_FALTANTES` | `VALORES_NO_FINITOS` | `VALORES_NEGATIVOS` o None, y nunca lanza (los textos estÃ¡n en `MOTIVOS_CURVA`).
   - `validar_curvas_editadas(curvas, fechas_permitidas) -> None` lanza ValueError si no hay curvas, si una fecha estÃ¡ fuera de rango o si una curva es invÃ¡lida.
   - Las usan pipeline (`_aplicar_edicion`), `control.validar_curvas` y `approval_blockers`.
   - nivel y forma dejan intactos los dÃ­as con curva previa invÃ¡lida; control los bloquea. No hay excepciÃ³n con NaN/None.
3. **Params**: cada step expone `DEFAULTS` + `validar_params(params) -> dict`, apoyadas en el helper nuevo `steps/_params.py`. Completan defaults, castean (intâ†’float, 4.0â†’4, fechaâ†’ISO) y lanzan ValueError ante:
   - claves desconocidas (un typo no se ignora en silencio);
   - None, no finitos o bool;
   - rangos: alpha y beta en [0,1], n â‰¥ 1, max_anios_atras â‰¥ 1, banda_pp > 0, top_dias â‰¥ 1, banda_referencia_pp None o > 0;
   - anios_excluidos que no sea lista de int; fuente fuera de FUENTES; semana_ref_inicio que no sea str/date ISO.
   Claves obsoletas que se descartan sin error: base `modelo_version` y referencia `max_anios`. El pipeline llama `validar_params` siempre en `_ejecutar` (run y rerun). `run_pipeline` rechaza tambiÃ©n nombres de paso desconocidos en `params`.
4. **Por nombre**:
   - `orden_de(paso) -> int` (ValueError si el paso no existe).
   - `rerun_from(ctx, records, orden: int|str, ...)` acepta orden o nombre y exige `nuevo.paso == STEP_ORDER[orden]`. Exige ademÃ¡s que `records` traiga exactamente los 5 nombres, sin repetidos ni desconocidos, y `r.orden == orden_de(r.paso)`; si no, ValueError.
   - `run_pipeline(overrides=...)` acepta claves int o nombre, y el `paso` del override debe coincidir con la clave.
   - Todo se indexa por `paso` internamente. `final_curves` va por nombre.
5. **modelo_version**: fuera de `base.DEFAULTS` (base no tiene params). `base.step` la toma de `ctx.extras["modelo_version"]`. En base editado, `diagnostics.energia` refleja las curvas editadas y la calculada queda en `curvas_calculadas`.

LÃ³gica:
- **Control rediseÃ±ado (enclavamiento)**:
  - `desvio_pp = (E_semana_final / E_semana_nivel âˆ’ 1)Â·100`.
  - E_nivel = energÃ­a que CALCULÃ“ el paso nivel (para dÃ­as editados, `curvas_calculadas`). Con nivel omitido se usa la energÃ­a del paso base (`diagnostics.origen_energia_nivel` = "nivel" | "base").
  - Fuera de `banda_pp` (1.0, sin recalibrar: sin ediciones el backtest no informa) â†’ warning TENDENCIA_FUERA_BANDA con los top dÃ­as por contribuciÃ³n.
  - Sin ediciones â†’ desvÃ­o 0, sin alerta.
  - DiagnÃ³stico informativo (fÃ³rmula post-H1): tendencia_esperada_pct, tendencia_resultante_pct, `diferencia_tendencia_pp`, `dias_sin_evidencia`, `dias[f].energia_esperada`. Solo alerta (TENDENCIA_REFERENCIA_FUERA_BANDA) si se pasa el param `banda_referencia_pp` (default None).
  - Infos nuevas: SIN_ENERGIA_NIVEL (llamada directa sin energÃ­a) y SIN_CONTROL_TENDENCIA (faltan dÃ­as). Se eliminaron SIN_TENDENCIA_RESULTANTE y SIN_TENDENCIA_ESPERADA como alertas: el motivo queda en `diagnostics.tendencia_motivo`.
  - ReconciliaciÃ³n: con forma editada, ahora se reconcilian los dÃ­as NO editados (antes se saltaba toda la forma).
- **Limpiezas**:
  - `data.ANIOS_EXCLUIDOS_DEFAULT` como constante Ãºnica.
  - `data.ancla_mismo_dow(d, corte)` usado por nivel, forma y backtest (se borrÃ³ `bt.ancla`).
  - Se eliminaron: la rama base-editada-sin-proveedor, `_PASOS_EDITABLES` (queda `_PASOS_QUE_MODIFICAN`), `e_obj â€¦ else 0.0`, la duplicaciÃ³n de alertas default/excluido en referencia (`_default_con_alerta`, `_alerta_anio_excluido`) y la normalizaciÃ³n doble en forma.
  - `data.energia_curvas` ya no lanza con None.
  - base convierte valores no convertibles a NaN.

Backtest v3: `.venv/bin/python scripts/backtest_adjustments.py --base ingenuo archivo --fecha 20261112_v3` â†’ `reports/ajustes/backtest_20261112_v3.{csv,md}`.
- Las curvas son idÃ©nticas a v2 (diferencia mÃ¡xima 0.0 en las 39 375 filas), asÃ­ que el WAPE por paso es igual.
- TENDENCIA_FUERA_BANDA pasa de 34/31/14 alertas a 0/0/0, con |desvÃ­o| mÃ¡x 2e-14.
- La alerta de v2 se reproduce exacta como diagnÃ³stico `diferencia_tendencia_pp` (tabla "hipotÃ©tica", lift 0).
- 0 bloqueos.
- El .md trae la secciÃ³n 3 reescrita, las Limitaciones ampliadas (in-sample, base no-modelo, corte en sÃ¡bado, defaults provisionales, alerta sin evaluar) y la secciÃ³n 6 de comparaciÃ³n.

Tests:
- Los 3 archivos del nÃºcleo pasan: 123 passed.
- Nuevos: curvas_calculadas (editado, base editado, rerun, ida y vuelta, ausencia), NaN/inf en base â†’ bloqueo + `json.dumps(allow_nan=False)` + rerun desde records con None, `validar_curvas_editadas`/`validar_curva`, params invÃ¡lidos (unitarios y en run/rerun), records desalineados (5 casos), `orden_de`, rerun por nombre, override con paso que no coincide, control sin ediciones = 0, ediciÃ³n +10 % en nivel y en forma â†’ alerta, `modelo_version` desde ctx, pasos que toleran curva previa invÃ¡lida, backtest sin ediciones = 0 alertas.
- Suite completa (`--ignore=tests/test_hourly_disaggregation.py`): 183 passed, 3 skipped, solo los 4 fallos preexistentes. En una corrida intermedia fallaron 2 tests NaN del router mientras el otro agente los editaba; al repetir pasaron.

## Orquestador â€” cierre fix ronda 1 (2026-11-12)
- Ambos agentes de correcciÃ³n terminaron (se reanudaron tras corte por lÃ­mite de uso). Suite: 183 passed, 3 skipped, 4 failed preexistentes.
- Siguiente: ronda 2 COMPLETA (8 Ã¡ngulos + regresiÃ³n, agentes separados, modelo Sonnet por pedido del usuario) contra el diff completo actual.

## Ronda 2 de revisiÃ³n (Sonnet, 8 Ã¡ngulos + regresiÃ³n) â€” 2026-11-12
RegresiÃ³n: sin regresiones (16â†’183 passed, mismos 4 fallos, rutas/esquemas previos intactos, import sin IO).
### Tratado como ALTO (regla "ante la duda, alto") â†’ corregir + ronda 3 completa
- R2-A [line-by-line, removed-behavior] `ucp` sin validar llega a `full_update_csv` â†’ `ucp="../x"` crea dirs/CSV fuera de data/raw y llama a red con ese valor, en endpoint sin auth (preview ni siquiera exige store). Mismo patrÃ³n preexistente en /predict, superficie nueva aquÃ­.
### Medios a corregir
- Re-editar `base` pierde la curva del modelo en `curvas_calculadas` de dÃ­as ya editados (3 Ã¡ngulos, CONFIRMED).
- `meta.params` queda desactualizado tras PATCH.
- UCP sin modelo â†’ el request entrena en lÃ­nea con el lock tomado (minutos) â†’ responder 409.
- `OverflowError` en params/fechas extremas â†’ 500.
- `excepciones_justificadas` no se valida contra alertas vigentes ni registra los warnings vistos al aprobar.
- Sin auth: agregar secreto servicio-a-servicio opcional (`AJUSTES_SERVICE_TOKEN`).
- Control: la coherencia con la tendencia de referencia quedÃ³ apagada; emitirla como alerta `info` por defecto (control de proceso, distinto del enclavamiento).
- Reporte v3: filas de festivo con base ingenua no son interpretables; decirlo.
### Bajos a corregir si baratos
evento `reemplazar` version distinta InMemory/Postgres; evento `crear` atÃ³mico con create_run; `base_ref` opcional en RunRequestâ†’extras; validaciÃ³n UCP antes de crear lock.
### Aceptados
params por dÃ­a/tipo de dÃ­a (fase siguiente), des-editar un solo dÃ­a, preview sobre corrida existente, trabajo duplicado entre corridas en cola de la misma UCP, duplicaciÃ³n corrida_paso vs corrida_version y paridad InMemory/Postgres, brecha no validada en aprobaciÃ³n (la aprobaciÃ³n cubre la semana objetivo; se documenta), RAW_DATA_DIR local en data.py (evita efectos de importar settings), logs sin emojis.
- CorrecciÃ³n del usuario (2026-11-12): R2-A se reclasifica como MEDIO (ningÃºn agente lo reportÃ³ alto; patrÃ³n preexistente en /predict). Ronda 2 sin crÃ­ticos/altos â†’ NO hay ronda 3: tras los fixes, una verificaciÃ³n estÃ¡ndar (suite completa + Postgres + import API) y commit.

## Fix ronda 2 â€” nÃºcleo (terminado 2026-11-12)
Alcance: types/data/pipeline/steps, backtest y sus tests, reporte v3. No se tocaron router/store/sql/main ni
test_adjustments_router.py (los edita otro agente en paralelo).
- `steps/_params.py`: `numero` captura OverflowError (float(10**400)) â†’ ValueError; `entero(minimo, maximo)` obligatorio,
  ints sin pasar por float; `lista_enteros(minimo, maximo, max_items=200)` solo list/tuple/set; mensajes con repr acotado
  (`_corto`); constantes `FECHA_MIN=1990-01-01`, `FECHA_MAX=2100-12-31`.
- Cotas: `forma.MAX_N=60`, `referencia.MAX_ANIOS_ATRAS=20`, `control.top_dias â‰¤ 7`, `anios_excluidos` en 1990..2100.
  `referencia.semana_ref_inicio` en [FECHA_MIN, FECHA_MAX] (validar_params) y no posterior a la semana objetivo
  (`referencia.step` â†’ ValueError; el router ya convierte ValueError del pipeline en 422). "9999-12-27" antes
  desbordaba en `dias_semana` (+6 d) â†’ ahora ValueError. `pipeline.build_context` exige semana_inicio en
  [FECHA_MIN, FECHA_MAX] (protege âˆ’364Â·20 y festivos). Verificado con tests que ref=1990-01-01, max_anios_atras=20, n=60,
  top_dias=7 corren referenciaâ†’nivelâ†’formaâ†’control sin excepciÃ³n.
  NOTA para el agente de router: si el router hace aritmÃ©tica con `semana_inicio` ANTES de `pl.build_context`
  (p. ej. rango del loader de festivos), una semana 0001/9999 podrÃ­a desbordar ahÃ­; conviene validar el rango en RunRequest.
- `control.py`: docstring reescrito con los DOS controles (enclavamiento = ediciÃ³n, warning; coherencia con referencia =
  proceso, info). `TENDENCIA_REFERENCIA_FUERA_BANDA` ahora `info`, default `banda_referencia_pp=2.0` (null desactiva);
  detalle con tendencias y dias_sin_evidencia. Nueva `_brecha`: `diagnostics.dias_brecha` y warning `BRECHA_INVALIDA`
  (dÃ­as de brecha faltantes/invÃ¡lidos, `detalle.motivos`), no bloquea.
- `pipeline.approval_blockers`: docstring â€” solo semana objetivo; brecha no se reporta ni bloquea.
- `scripts/backtest_adjustments.py`: columnas `alerta_tendencia_referencia_info`, `banda_referencia_pp`;
  `evaluar_alerta(..., columna_banda)`; secciÃ³n 3 "Alertas de tendencia" con las dos alertas + la hipotÃ©tica Â±1 pp;
  `nota_festivos(df)` bajo "Mejor configuraciÃ³nâ€¦" (nÃºmeros del propio detalle); LIMITACIONES con festivos ingenuo no
  interpretables / archivo no concluyentes / Î± por tipo de dÃ­a pendiente; `secciones_manuales(md)` preserva "## 6." en
  adelante al regenerar.
- Backtest v3 re-corrido (mismos flags) â†’ curvas idÃ©nticas (dif. mÃ¡x 0.0, 39 375 filas), 0 bloqueos, 0 BRECHA_INVALIDA,
  enclavamiento 0. Alerta info con defaults: ingenuo Antioquia 20/52 (precision 0.65 vs tasa base 0.56), ingenuo
  AtlÃ¡ntico 21/52 (0.48 vs 0.56), archivo 7/21 (0.43 vs 0.38). SecciÃ³n 7 escrita a mano en el .md; bullet de la secciÃ³n 6
  sobre "apagada por defecto" marcado como superado. Backup del v3 previo en el scratchpad de la sesiÃ³n (no versionado).
- Tests: steps +31 (params extremos 10**400/fechas 0001 y 9999/cotas, lÃ­mites vÃ¡lidos, ref posterior, valores lÃ­mite que
  no desbordan, build_context fuera de rango); pipeline +7 (info con alpha=0 / nivel omitido / base editada sin
  enclavamiento, null desactiva, BRECHA_INVALIDA no bloquea + JSON-safe, brecha vÃ¡lida sin alerta, ref posterior en
  run_pipeline); backtest +3. Default de control actualizado en el test de defaults.
- VerificaciÃ³n: nÃºcleo `162 passed`; suite completa `--ignore=tests/test_hourly_disaggregation.py` â†’ 238 passed,
  4 skipped, 4 failed (los mismos preexistentes).
- Aceptado: records persistidos con `banda_referencia_pp: null` explÃ­cito (corridas creadas antes de este fix) no emiten la
  alerta info al recalcular â€” respetan el param guardado.

## Fix ronda 2 â€” API/store (terminado 2026-11-12)
Archivos: `src/adjustments/router.py`, `src/adjustments/store.py`, `src/api/main.py` (bloque ajustes + import),
`tests/test_adjustments_router.py`, `.env.example`. Sin SQL nuevo (no hizo falta 003). No se tocaron
steps/_params.py, steps/control.py, pipeline.py, types.py ni el reporte (otro agente).
- R2-A (ALTO) UCP: `RunRequest.ucp` y query de `/aprobada` con `pattern=^[A-Za-z0-9_-]{1,64}$` (422 de Pydantic);
  `_validar_ucp` (primera lÃ­nea de `_compute` y antes de re-aplicar base) exige `data/raw/{ucp}/datos.csv` resuelto
  dentro de `data.RAW_DATA_DIR` (leÃ­do en cada llamada, misma constante que data.py) â†’ 404. `_ucp_lock` valida antes
  de crear la entrada del dict. Tests: `../x`, `a/b`, `..`, vacÃ­o, 65 chars, con espacio â†’ 422; `Foo` â†’ 404; symlink
  que escapa â†’ 404; con refresher/provider que fallan si se invocan, sin cambios en el Ã¡rbol de data/ y sin locks nuevos.
- Re-editar base: provider = `{**base.curves, **base.curvas_calculadas}` â†’ `curvas_calculadas[X]` sigue siendo la del
  modelo tras editar X y luego Y (InMemory y Postgres; verificado que el test falla sin el fix).
- `meta.params` â†’ `meta.params_iniciales` (request original). En BD la columna sigue siendo `corrida.params`
  (`SELECT params AS params_iniciales`): lo mÃ¡s simple, sin migraciÃ³n.
- Modelo sin entrenar: `set_model_checker(fn)`; main.py `_adjustments_model_checker` = `check_model_exists(ucp)[0] and
  check_hourly_disaggregation_trained(ucp)` (run_predict_flow entrena ambos si faltan). Sin modelo â†’ 409 "La UCP no
  tiene modelo entrenado; use /retrain" ANTES del lock/refresco (run y re-aplicar base); con base_curvas no se verifica.
  Si el checker lanza â†’ 502.
- `OverflowError`: `ERRORES_ENTRADA = (ValueError, OverflowError)` en todos los puntos que convierten a 422 (histÃ³rico,
  build_context, fechas/curvas, run_pipeline, validar_curvas_editadas, rerun_from). AdemÃ¡s `semana_inicio`/`fecha_emision`
  fuera de 1990-01-01..2100-12-31 (`_params.FECHA_MIN/FECHA_MAX`, misma cota que el nÃºcleo) â†’ 422 en el modelo Pydantic, antes de cualquier cÃ¡lculo de fechas (rango de festivos, contexto); pedido del agente del nÃºcleo, test con 0001-01-01, 1989-12-25, 9999-12-27, 2101-01-03. Test con alpha = 10**400 en preview y PATCH.
- `excepciones_justificadas`: `_approval_check` (corre bajo lock en el store) devuelve bloqueos; si no hay, calcula los
  warnings vigentes de esa versiÃ³n, rechaza cÃ³digos no vigentes (`JustificacionInvalidaError` â†’ 422 con
  codigos_invalidos/warnings_vigentes, sin aprobar) y completa el `detalle` del evento `aprobar` con warnings_vigentes y
  warnings_sin_justificar (dedup de justificadas). No exige justificar todos. Store: `detalle if detalle is not None`.
- Auth: dependencia de router `verificar_token`; `AJUSTES_SERVICE_TOKEN` leÃ­do por request; cabecera `X-Ajustes-Token`,
  `hmac.compare_digest` â†’ 401 (antes que la validaciÃ³n del body). Sin variable: permitido + warning una vez.
  `.env.example` documentado (placeholder). Tests de ambos modos.
- Bajos: evento `reemplazar` con la versiÃ³n de la corrida reemplazada en ambos stores (Postgres `RETURNING id, version`);
  `create_run(meta, records, evento=None)` inserta `crear` en la misma transacciÃ³n (InMemory valida antes de mutar);
  `base_ref` (â‰¤200) â†’ `meta.extras.base_ref` y detalle del evento `crear`.
Tests: router 64 passed + 4 skipped sin DSN; con `AJUSTES_DATABASE_URL=postgresql:///postgres` 68 passed (bases temporales
borradas; 0 `epm_ajustes_*`). Suite (`--ignore=tests/test_hourly_disaggregation.py`): 242 passed, 4 skipped, 4 failed
(los mismos preexistentes). `import src.api.main` sin variables de entorno OK: /health 200, rutas /adjustments en
OpenAPI, preview `../x` â†’ 422, `Foo` â†’ 404, GET sin DSN â†’ 503; checker real: Antioquia/Atlantico True.

## VerificaciÃ³n estÃ¡ndar final y commit (2026-11-12)
- Suite: 242 passed, 4 skipped, 4 failed preexistentes. Router con Postgres: 68 passed. Import de la API sin env: /health 200, 8 rutas /adjustments. Sin bases temporales.
- Ronda 2 sin crÃ­ticos/altos â†’ sin ronda 3 (regla del framework). Commit en `feat/ajustes-pipeline`.
- Pendientes Fase 2/3: backtest con el modelo real (requiere arreglar fillna/pandas 3 en feature_engineering y aislar full_update_csv), Î± por tipo de dÃ­a, pantalla Mesa de ajustes en Sphere + rutas en backend intermedio (enviar `X-Ajustes-Token`, `base_curvas`, `base_ref`), aplicar `sql/001`+`002` en el Postgres de despliegue y definir `AJUSTES_DATABASE_URL`/`AJUSTES_SERVICE_TOKEN`.

## Cierre de sesiÃ³n (2026-11-12)
- Por pedido del usuario, la rama `feat/ajustes-pipeline` se pusheÃ³ como fast-forward a `origin/main`: `8501faa` (feature) y `c7b9d32` (guÃ­a `docs/work/ajustes/README.md` para continuar en otro PC). Sphere no se tocÃ³ en esta sesiÃ³n (Ã¡rbol limpio, nada pusheado a `test`).
- Cambios ajenos del usuario que siguen sin commitear en el Ã¡rbol, sin tocar: `fastapi_factores/test_*`, `reports/performance/README.md`, `.claude/settings.local.json`, carpeta `env.migracion-pc-epm-usb/`.
- Framework global (`~/.claude/CLAUDE.md`) actualizado con dos reglas: la severidad la deciden los agentes revisores (el orquestador no sube medios a altos para forzar otra ronda); con Opus, los revisores corren con Sonnet.
- Para retomar: leer `docs/work/ajustes/README.md` (estado, arranque, datos a copiar, pendientes por fase).

## RetroalimentaciÃ³n del analista y backlog (2026-11-12, post-cierre)
- Usuario reafirma el objetivo final: reducir mucho el error de las predicciones semanales; la mesa da las herramientas y el paso a paso, y el sistema debe hacer gran parte del trabajo del analista.
- Analista: el ajuste inteligente por escenario de referencia (ASR, Sphere `handleASR` â†’ `/validate-hourly-adjustments`) no le sirve como quisiera; revisiÃ³n el viernes 2026-11-13.
- Nuevo requerimiento: buscador de dÃ­as/escenarios anÃ¡logos muy especÃ­ficos (p. ej. martes de mayo despuÃ©s de festivo) usable como fuente de forma/nivel/referencia.
- Todo documentado en `docs/work/ajustes/REQUERIMIENTOS.md` (R1 ASR, R2 automatizar, R3 anÃ¡logos, backlog priorizado). README apunta a ese archivo.

## SesiÃ³n autÃ³noma 2026-11-12/13 (orquestador)
Plan maestro: `docs/work/ajustes/PLAN_NOCHE.md`. Logs por fase en `docs/work/ajustes/<fase>/LOG.md` (en la rama/worktree de cada fase).
- [x] Framework global: agregadas las reglas "quiÃ©n decide la severidad" y "modelo de revisores = sonnet" a `~/.claude/CLAUDE.md` (no estaban en este PC).
- [x] Entorno: este PC no tenÃ­a `.venv` â†’ creado con `py -3.11 -m venv .venv` + `pip install -r requirements.txt pytest`. Python: `.venv/Scripts/python.exe`.
- [x] Worktrees: `../EPM-wt-asr` (feat/ajustes-asr-comparacion), `../EPM-wt-2a` (feat/ajustes-fase2a), `../EPM-wt-2b` (feat/ajustes-fase2b) desde main 74bc100, con copias de data/raw/{Antioquia,Atlantico}, models, data/features; `../Sphere-wt-mesa` (feat/mesa-ajustes) desde origin/test 57eb4fb.
- Hallazgo temprano ASR (Sphere `handleASR`): manda `tipo_dia: "laboral"` FIJO para cualquier dÃ­a; ajuste = pred + (h2âˆ’h3) âˆ’ (predâˆ’h1) = h1 + (h2âˆ’h3) hora a hora (redondeado a entero), luego el validador Â±15 % del cluster.
- DecisiÃ³n tomada sin el usuario: fases en paralelo en worktrees separados (ASR, 2a, 2b, 3) para aprovechar la noche; cada una en su rama y con su propia ronda de revisiÃ³n; se integran a main en orden de llegada con rebase y suite.
### Progreso (hecho / en curso / falta)
- En curso: ola 1 (ASR, 2a, 2b, 3).
- LÃ­nea base suite en este PC: 242 passed, 4 skipped, 1 failed (christmas). Fase 3 lanzada.
- Ola 1 lanzada (4 coordinadores en background, cada uno con su worktree, docs de fase, revisiÃ³n propia con revisores sonnet y commit LOCAL; el orquestador verifica, hace push y fast-forward):
  - Fase 3 â†’ `../Sphere-wt-mesa` (feat/mesa-ajustes), docs en EPM `docs/work/ajustes/fase3/`.
  - Fase 2a â†’ `../EPM-wt-2a`, docs `docs/work/ajustes/fase2a/`.
  - Fase 2b â†’ `../EPM-wt-2b`, docs `docs/work/ajustes/fase2b/`.
  - ASR â†’ `../EPM-wt-asr`, docs `docs/work/ajustes/asr/`.
  Si la sesiÃ³n se corta: revisar `git -C ../EPM-wt-<x> log main..HEAD` y el LOG.md de cada fase; si un coordinador quedÃ³ esperando a sus hijos, reanudarlo.
- 2026-11-13 04:41 Los 4 coordinadores se cortaron por lÃ­mite de uso (5 h); reanudados con SendMessage tras el reinicio.
- 2026-11-13 07:45 Segundo corte por lÃ­mite; reanudados ASR, 2b, hijo backtest de 2a (a8f9f127) y coordinador de Fase 3 (foundation terminado). Coordinador 2a: reanudar cuando su hijo termine.

### ESTADO ACTUAL (snapshot 2026-11-13, orquestador) â€” leer esto primero para retomar
Regla del usuario: documentar estado antes de cada fase y despuÃ©s de cada paso (agregada a REGLAS_AGENTES.md).
Ninguna fase commiteada aÃºn; todo el trabajo estÃ¡ sin commitear en cada worktree.
| Fase | Worktree / rama | Coordinador (agentId) | Estado | Docs de fase |
|---|---|---|---|---|
| ASR | `../EPM-wt-asr` / feat/ajustes-asr-comparacion | ac820e2f693544d69 | cÃ³digo + tests hechos (asr.py, comparar_asr.py, router_asr.py, script); falta REPORTE, revisiÃ³n, commit | `../EPM-wt-asr/docs/work/ajustes/asr/` |
| 2a | `../EPM-wt-2a` / feat/ajustes-fase2a | a8e631a8352b58974 (espera a su hijo a8f9f127063692673 = backtest+recalibraciÃ³n) | pandas3 + aislamiento + alpha_por_tipo hechos (A_REPORTE); backtest modelo en curso (dry run OK); falta corrida completa, recalibraciÃ³n, revisiÃ³n, commit | `../EPM-wt-2a/docs/work/ajustes/fase2a/` |
| 2b | `../EPM-wt-2b` / feat/ajustes-fase2b | a63d0d1fa7ad341bc | analogos.py, endpoint, integraciÃ³n en pasos y backtest en curso (agente B); falta mediciÃ³n, revisiÃ³n, commit | `../EPM-wt-2b/docs/work/ajustes/fase2b/` |
| 3 | `../Sphere-wt-mesa` / feat/mesa-ajustes | a3e65c24035ba882f | Foundation hecho y verificado (tipos, servicio+mock, shell, router/menÃº); falta P1â€“P4, integraciÃ³n, revisiÃ³n, commit | `docs/work/ajustes/fase3/` (en este repo) |
CÃ³mo retomar si el orquestador se corta: para cada fase leer su LOG de fase; si el coordinador estÃ¡ detenido, `SendMessage` a su agentId ("continÃºa desde tu LOG"); si un coordinador espera a un hijo que ya terminÃ³, reanudar al coordinador. Cuando una fase tenga commit local: verificar suite en su worktree, `git rebase main`, push de la rama y `git push origin <rama>:main` (Sphere: `<rama>:test`).
Pendiente despuÃ©s de la ola 1: Fase 4 (dÃ­as especiales, alertas clima/eventos, aprendizaje F09, auto por segmento), rerun de 2b y ASR con base modelo, RESUMEN_NOCHE.md.
- 2026-11-13 ASR TERMINADA y en main: `0e8de2e` + `c94c5d1` (push de la rama y fast-forward a main; suite 306 passed, 4 skipped, 1 preexistente). WAPE baseâ†’mesaâ†’ASR: ingenuo Antioquia 4.05â†’3.71â†’5.26; AtlÃ¡ntico 6.36â†’6.00â†’8.24; archivo 3.03â†’2.79â†’5.42. DiagnÃ³stico: ASR = h1+(h2âˆ’h3) ignora la base; ruido de dÃ­as sueltos; festivos lo disparan; bug de orden desvi2 en el front. RecomendaciÃ³n: reemplazar por la mesa. Reporte: `reports/ajustes/asr_comparacion_20261112.md`.
  Nota: las ramas 2a/2b deberÃ¡n rebasear sobre main (c94c5d1) antes de subir.
- 2026-11-13 Fase 2a: hijo B terminÃ³. Backtest con modelo real sin fuga (reentreno semanal k=1, 21 semanas holdout): WAPE baseâ†’final Antioquia 2.44â†’2.33, AtlÃ¡ntico 5.86â†’5.46; energÃ­a semanal MAPE 1.52â†’1.37 / 4.57â†’4.41. Holdout confirma defaults (Î± 0.25, promedio_n n=4, banda 2.0); alpha_por_tipo no mejora â†’ no se cambia ningÃºn default. R2: forma automÃ¡tica en laborales (ambas UCP) y domingos AtlÃ¡ntico. Coordinador 2a reanudado para revisiÃ³n + commit.

### ESTADO ACTUAL v2 (2026-11-13, orquestador al ~80 % de uso) â€” REEMPLAZA a la tabla anterior; leer primero
main local y origin/main = `c94c5d1` (ASR integrada). LOG.md maestro, PLAN_NOCHE.md y REGLAS_AGENTES.md estÃ¡n SIN commitear en el checkout principal (docs/ estÃ¡ en .gitignore â†’ `git add -f`).
| Fase | Worktree / rama | Coordinador | Estado | Siguiente paso |
|---|---|---|---|---|
| ASR | `../EPM-wt-asr` | ac820e2f693544d69 | **TERMINADA, en main** (0e8de2e, c94c5d1) | â€” |
| 2a | `../EPM-wt-2a` / feat/ajustes-fase2a | a8e631a8352b58974 | A+B+C hechos, sin commit; ronda 1 de revisiÃ³n (9 revisores sonnet) en curso | coordinador consolida hallazgos, corrige, verifica, commit local |
| 2b | `../EPM-wt-2b` / feat/ajustes-fase2b | a63d0d1fa7ad341bc | A+B+C hechos (backtest_analogos_20261113.md, defaults sin cambio), sin commit; ronda 1 en curso | Ã­dem |
| 3 | `../Sphere-wt-mesa` / feat/mesa-ajustes | a3e65c24035ba882f | Foundation hecho; P1â€“P4 (componentes) en curso; sin commit | integrar, build/lint, revisiÃ³n, commit local |
Al recibir el commit local de cada fase (lo hace el orquestador o quien lo reemplace):
1. `cd ../EPM-wt-<f> && git rebase main` (2a y 2b tocan `nivel.py`/`forma.py`: el segundo en llegar tendrÃ¡ conflictos â†’ resolver conservando ambos cambios, que son aditivos).
2. Suite en el worktree: `../EPM/.venv/Scripts/python.exe -m pytest tests -q --ignore=tests/test_hourly_disaggregation.py -p no:cacheprovider` (solo falla christmas) + `python -c "import src.api.main"`.
3. `git push origin <rama>` y `git push origin <rama>:main`; luego en `../EPM`: `git merge --ff-only origin/main`.
4. Sphere: en `../Sphere-wt-mesa`, `git fetch && git rebase origin/test`, `npx vite build`, `git push origin feat/mesa-ajustes` y `git push origin feat/mesa-ajustes:test`.
5. Anotar commits aquÃ­.
DespuÃ©s: re-correr backtest de anÃ¡logos y comparador ASR con base modelo (scripts de 2a: `scripts/backtest_modelo.py`, `scripts/recalibrar_ajustes.py`); Fase 4 (dÃ­as especiales â€” festivos y post-Semana Santa son el segmento con mÃ¡s error: WAPE festivo ~12.6 % â€”, alertas clima/eventos, aprendizaje F09); `docs/work/ajustes/RESUMEN_NOCHE.md` con tabla antes/despuÃ©s (datos en esta secciÃ³n y en los REPORTE.md de cada fase) y push a main.
Resultados ya medidos para el RESUMEN: ASR (ver entrada arriba); 2a modelo real holdout 21 sem: WAPE baseâ†’final Antioquia 2.44â†’2.33, AtlÃ¡ntico 5.86â†’5.46; energÃ­a semanal 1.52â†’1.37 / 4.57â†’4.41.
- 2026-11-13 El usuario DETUVO los coordinadores de 2b (a63d0d1fa7ad341bc) y Fase 3 (a3e65c24035ba882f). No se relanzan sin instrucciÃ³n del usuario. Su trabajo queda sin commitear en `../EPM-wt-2b` y `../Sphere-wt-mesa`; para retomar: SendMessage a esos IDs ("continÃºa desde tu LOG de fase"). 2a (a8e631a8352b58974) sigue con su ronda de revisiÃ³n.
- 2026-11-13 El usuario tambiÃ©n DETUVO el coordinador de 2a (a8e631a8352b58974) al lanzar su ronda 1 de revisiÃ³n. Las 3 fases quedan sin commitear en sus worktrees; ninguna se relanza sin instrucciÃ³n del usuario.
- 2026-11-13 Usuario pidiÃ³ reanudar. Los coordinadores detenidos no se pueden reanudar â†’ lanzados coordinadores NUEVOS que retoman desde los LOG de fase: 2a = a80fc7bf7fb60bfeb, 2b = ae2dd8e4a4163dd29, Fase 3 = a3c115c92dee443b1 (los IDs viejos ya no sirven).
- 2026-11-13 SesiÃ³n nueva retoma "ESTADO ACTUAL v2". Los coordinadores del punto anterior ya no estaban activos (ninguna fase con commit; 2a, 2b y Fase 3 siguen sin commitear en sus worktrees). Lanzados coordinadores NUEVOS en background: 2a = a1fe2455967ca2f06, 2b = aa9a9b1a80034e740, Fase 3 = a5d2fc058f221f772 (los IDs anteriores ya no sirven). Cada uno: revisiÃ³n ronda 1 â†’ correcciÃ³n â†’ verificaciÃ³n â†’ commit local. El orquestador hace luego rebase/push/merge segÃºn los pasos 1â€“5 de ESTADO ACTUAL v2.
- 2026-11-13 Fase 2a TERMINADA y en main: commit `06c6432` (rebaseado sobre a2a3239; suite 344 passed, 5 skipped, 1 preexistente christmas; import API OK). Sin crÃ­ticos/altos en revisiÃ³n; medios/bajos aceptados en el LOG de fase. 2b (aa9a9b1a80034e740) y Fase 3 (a5d2fc058f221f772) siguen en curso; 2b deberÃ¡ rebasear sobre main (06c6432) y resolver conflictos aditivos en nivel.py.
- 2026-11-13 Fase 2b TERMINADA y en main: `git log -1` tras rebase (conflicto aditivo en nivel.py resuelto conservando alpha_por_tipo + anÃ¡logos; suite 481 passed, 5 skipped, 1 preexistente christmas). RevisiÃ³n ronda 1 sin crÃ­ticos/altos; agente de regresiÃ³n no pudo lanzarse (lÃ­mite concurrencia) â†’ cubierta con suite completa. Pendiente: Fase 3 (a5d2fc058f221f772), re-backtest anÃ¡logos/ASR con base modelo, Fase 4, RESUMEN_NOCHE.md.
- 2026-11-13 Fase 3 TERMINADA y en Sphere `test`: commit `22ef6d1` (feat/mesa-ajustes; vite build OK, tsc -b 178 = base, 0 errores en archivos nuevos). RevisiÃ³n: 3 rondas (R1 2 altos, R2 1 alto por regresiÃ³n propia, R3 limpia). Aceptados y pendientes en `docs/work/ajustes/fase3/LOG.md` ("FASE 3 TERMINADA"). Pendientes de integraciÃ³n: el backend intermedio debe aÃ±adir `usuario` en aprobar; registrar el mÃ³dulo en BD para quitar la inyecciÃ³n de menÃº. Quedan: re-backtest anÃ¡logos/ASR con base modelo, Fase 4, RESUMEN_NOCHE.md.
- 2026-11-13 Lanzado agente de re-backtest (anÃ¡logos + ASR con base modelo) = a8157874d0b22e115 (commit local en main de reportes, sin push). Fase 4 pendiente de plan: alcance sin definir en PLAN.md, se pide plan al usuario antes de implementar.
- 2026-11-13 Plan de Fase 4 + re-backtest + resumen documentado en docs/work/ajustes/fase4/PLAN.md; prompt para otro agente en fase4/PROMPT_EJECUTAR.md. El usuario pidiÃ³ detenerse por crÃ©ditos. 4D usa OpenAI (sin datos del usuario). Agente de re-backtest a8157874d0b22e115 quedÃ³ bloqueado por modo plan sin cambios (su plan: reusar curvas k1 de 2a); no hay commits nuevos de ese paso.

### ESTADO ACTUAL v3 (2026-11-13 12:40) â€” traspaso a otro PC; leer primero
- `main` (EPM): ASR, 2a, 2b y re-backtest con base modelo_k1 (`6ad685d`) + estos docs. Sphere `test`: Fase 3 (`22ef6d1`).
- **Fase 4 NO integrada**: rama `feat/ajustes-fase4` (2 commits sobre main previo: `f04ee27` feat, `c9c9a2a` fix ronda 2), subida al remoto como rama aparte. Docs en esa rama: `docs/work/ajustes/fase4/` (LOG.md, PLAN_4B..4E, REPORTE_4B..4E, REPORTE_INTEGRACION). Plan general: `fase4/PLAN.md` (en main); prompt: `fase4/PROMPT_EJECUTAR.md`.
- QuÃ© falta: (1) en la rama fase4 leer `fase4/LOG.md` y confirmar si la ronda 2 cerrÃ³ limpia (segÃºn la regla: solo repetir ronda completa si hubo crÃ­ticos/altos; si no, una verificaciÃ³n estÃ¡ndar); (2) `git rebase main`; (3) suite `python -m pytest tests -q --ignore=tests/test_hourly_disaggregation.py -p no:cacheprovider` (solo falla christmas) + `python -c "import src.api.main"`; (4) `git push origin feat/ajustes-fase4:main` y ff en el checkout principal; (5) `docs/work/ajustes/RESUMEN_NOCHE.md` (tabla antes/despuÃ©s, decisiones sin usuario, pendientes) y push a main.
- Pendientes fuera de EPM: backend intermedio debe aÃ±adir `usuario` en aprobar; registrar mÃ³dulo en BD para quitar la inyecciÃ³n de menÃº de Sphere; front Sphere para el paso `especiales` (6 pasos) y alertas de eventos.
- Fuera de git (copiar a mano al otro PC): `.env` (ver `.env.migracion-pc-epm-usb.zip`), `data/`, `models/`, y `reports/ajustes/curvas_modelo_k1_*.csv` (solo para re-correr backtests). `docs/` estÃ¡ en .gitignore â†’ usar `git add -f`.
- La sesiÃ³n anterior se detuvo sin avisar (Ãºltimo commit 11:58).
- 2026-11-13 AÃ±adido plan de demo Railway: docs/work/ajustes/demo/PLAN.md y demo/PROMPT_EJECUTAR.md (ejecutar DESPUÃ‰S de integrar Fase 4). Decisiones del usuario: interno (no pÃºblico), datos reales y API key OpenAI permitidos.

## Cierre ronda 3 â€” R2, eventos y documentaciÃ³n (2026-11-13, EPM-wt-4)
La ronda 3 dejÃ³ un ALTO de diseÃ±o en 4C (R2 no distinguÃ­a "no aplicar") y varios bajos. Cierre:
- `automatico.py`: `estado` sigue en 2 estados (`automatico`/`sugerido`) para el pipeline;
  `recomendacion` expone 3 (`no_aplicar`/`automatico`/`sugerido`); `segmentos(ucp)` da el mapa
  4 tipos Ã— 2 pasos; `configuracion` aÃ±ade `segmentos` y `recomendaciones`. `modos_paso` sigue
  conservador (solo `automatico` si los 4 tipos lo son).
- Router: `meta.extras["segmentos_paso"]` viaja **siempre** (se restaura/recalcula en
  `_context_from_meta` y `_run_out`) ademÃ¡s de `modos_paso`; `/automatico` ya devuelve lo nuevo.
- Pipeline: el diagnÃ³stico R2 por dÃ­a aÃ±ade `recomendacion` junto a `estado` (no cambia curvas).
- **R2 es clasificaciÃ³n para el front**: `aplicar_automatico` no cambia curvas ni activa/omite
  pasos. `no_aplicar` estÃ¡ disponible en `segmentos`/`recomendaciones` y en el diagnÃ³stico diario.
- **Pendiente (decisiÃ³n de producto/usuario)**: que el front omita los segmentos `no_aplicar`;
  el backend los sigue aplicando. `modos_paso` conserva los 2 estados.
- Bajos: temporal de cache de eventos Ãºnico (`pid`+`uuid`); `aprendizaje._energia` tolera ndarray;
  `ESPECIALES_SIN_BACKTEST` solo se emite si `especiales` ajustÃ³ dÃ­as de verdad.
- Docs: `REPORTE_INTEGRACION` (el PATCH viejo no consulta OpenAI), `PLAN_4D`/`REPORTE_4D` (saneo
  exacto `[inicio, fin]`, sin Â±7), `REPORTE_4C` (limitaciÃ³n de diseÃ±o + contrato expuesto).

### ESTADO ACTUAL v4 (2026-11-13, cierre) — Fase 4 integrada en main
- `main` = `ce4b6fb` (incluye `6ad685d` re-backtest y los 3 commits de Fase 4). `origin/main`
  actualizado por fast-forward. La rama remota vieja `feat/ajustes-fase4` quedó en `c9c9a2a`
  (historia previa al rebase); la historia final está en `main` y en `feat/ajustes-fase4-final`.
  No se hizo force-push.
- Fase 4 (4A verificado; 4B `especiales` 6 pasos con `aplicar=false`; 4C R2 como clasificación para
  el front con `recomendacion`/`segmentos_paso`; 4D eventos/clima con cache y degradación; 4E
  aprendizaje F09 de solo lectura) integrada. Detalle y pendientes en `RESUMEN_NOCHE.md`.
- Verificación: worktree/rama final 678 passed, 5 skipped, 1 failed (christmas preexistente);
  `main` 486 passed, 5 skipped, 4 failed preexistentes (christmas + 3 async por entorno); import API
  OK; `/automatico` 200 y `/aprendizaje` 503 sin store. Postgres local no autenticó: tests de
  Postgres pendientes en un entorno con DSN.
- Revisión: 3 rondas multi-ángulo + regresión. R1/R2 corrigieron altos/medios; R3 solo dejó un ALTO
  de diseño en 4C, cerrado exponiendo `no_aplicar` en `recomendacion`/`segmentos` y documentando que
  el backend siempre aplica (la omisión por segmento es decisión de producto/front).
- Pendientes: front Sphere 6 pasos + endpoints/alertas; backend intermedio (`usuario`, módulo en BD,
  token/base_curvas/base_ref); decisión de producto sobre `no_aplicar`; aplicar SQL y variables en
  despliegue; demo Railway. Datos fuera de git: `.env`, `data/`, `models/`, `data/features/`,
  `reports/ajustes/curvas_modelo_k1_*.csv`.
- Para retomar: leer `RESUMEN_NOCHE.md` y `README.md`, luego `fase4/LOG.md`. Comandos de verificación
  en `RESUMEN_NOCHE.md` §5.
- 2026-11-13 Fase 4 CERRADA e integrada en main (8f86f03, febb595, ce4b6fb). Suite en main: 678 passed, 5 skipped, 4 failed preexistentes (christmas + 3 async sin pytest-asyncio); import API OK. Resumen final en `docs/work/ajustes/RESUMEN_NOCHE.md`.

- 2026-11-14 Demo (D0-D4) terminada y DESPLEGADA en Railway (proyecto epm-demo-ajustes). Estado, redespliegue y pendientes: `docs/work/ajustes/demo/ESTADO_DESPLIEGUE.md`. Commits: EPM e24428b, 322ec58, 2cd04c1; Sphere test 8a6302c.
- 2026-11-14 SESIÓN NOCTURNA 2: plan en docs/work/ajustes/noche2/PLAN.md (UX wow + curvas, mejora de pronósticos medida, redeploy). Agente de despliegue previo ya desplegó demo en Railway (ver demo/ESTADO_DESPLIEGUE.md).
