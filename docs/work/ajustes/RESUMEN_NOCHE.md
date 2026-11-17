# RESUMEN NOCHE — ajustes EPM (sesión 2026-11-12/13)

Documento de traspaso. Si solo lees un archivo, lee este y `docs/work/ajustes/LOG.md`.
Estado: **ASR, Fase 2a, Fase 2b, re-backtest con base `modelo_k1` y Fase 4 están en `main`.**
Último commit de `main` al cerrar: `ce4b6fb`.

## 1. Tabla antes/después

| Fase | Qué se midió | Antes | Después | Decisión |
|---|---|---|---|---|
| ASR | WAPE horario base → mesa → ASR (52 sem) | ingenuo Antioquia 4.05; Atlántico 6.36; archivo 3.03 | mesa 3.71 / 6.00 / 2.79; ASR 5.26 / 8.24 / 5.42 | El ASR actual empeora; reemplazarlo por la mesa en Sphere |
| 2a | WAPE base → final, modelo real sin fuga, holdout 21 sem | Antioquia 2.44; Atlántico 5.86 | 2.33; 5.46 | Defaults confirmados: α 0.25, `promedio_n` n=4, banda 2.0 pp |
| 2a | MAPE energía semanal base → final | 1.52 / 4.57 % | 1.37 / 4.41 % | `alpha_por_tipo` no mejora; no se cambia |
| 2b | Buscador de análogos, 104 sem, base ingenua | default forma 3.66/3.71 y 5.46/5.90 | ninguna variante cumple la regla (mejora ≥2 % en calibración y holdout, ambas UCP) | Se expone el buscador y `fuente="analogos"`; defaults sin cambio |
| Paso 0 | Re-backtest de análogos y ASR con `modelo_k1` sin fuga (52 sem) | reportes con base ingenua | WAPE forma default Antioquia 2.57/2.60 (calib/holdout), Atlántico 4.96/5.15; ASR `asr_anual` 5.26 / 8.24 | El candidato α 0.5 + ancla análogos en festivos **no se sostiene**; única señal `nivel_analogos_a025` en `rel:puente` (n=13, no concluyente) |
| 4A | R1-C: `curvas_calculadas` de días editados | ya implementado en `ded1df6` | — | Verificado con la suite; no requirió cambios |
| 4B | Paso `especiales` (Semana Santa, 24/31 dic, post-festivo, elecciones IA) | no existía | paso 5 (orden 4), entre `forma` y `control`; **default `aplicar=false`** (curvas idénticas) | Sin evidencia para activarlo; se deja como sugerencia/diagnóstico |
| 4C | Automático por segmento R2 | no existía | módulo `automatico.py`, endpoint `/automatico`, `meta.extras.segmentos_paso`, `aplicar_automatico` | Clasificación para el front; el backend siempre aplica. `no_aplicar` queda como recomendación |
| 4D | Alertas F06 (festivos, eventos IA, clima observado) | no existía | `eventos.py` con cache por `(ucp, inicio, fin)`, TTL 7 d, vacíos 1 d, timeout 5 s; alertas info/warning en `control` | Nunca modifica curvas; sin key o fallo degrada en silencio |
| 4E | Aprendizaje F09 (calculado vs editado) | no existía | `aprendizaje.py` + `Store.list_runs` + `GET /aprendizaje` (solo lectura) | Agrupa por UCP × paso × tipo de día; `min_casos=3`; porcentajes ×100 |

### Números de los reportes
- `reports/ajustes/asr_comparacion_20261112.md`: ASR = `h1 + (h2 − h3)`, no usa la base; festivos lo disparan.
- `reports/ajustes/backtest_modelo_20261113.md`: modelo real sin fuga (reentreno semanal k=1); R2 automático: forma en laborales (ambas UCP) y domingo Atlántico; nivel solo sábado Atlántico (borde).
- `reports/ajustes/backtest_analogos_20261113_modelo.md`: re-backtest con `modelo_k1`.
- `reports/ajustes/asr_comparacion_20261113_modelo.md`: ASR con base real; sigue peor que mesa/base.

## 2. Qué se implementó en Fase 4

- **4A (prerrequisito F09):** ya estaba: `StepResult.curvas_calculadas`, `_aplicar_edicion`, columna en
  `store/postgres` y tests. Verificado, sin cambios.
- **4B `especiales`:** `src/adjustments/steps/especiales.py`; `STEP_ORDER` de 6; migración transparente de
  corridas viejas de 5 pasos en lectura/PATCH (`pipeline.migrar_records`). Detecta Semana Santa, 24/12,
  31/12, post-festivo (`despues`/`puente`) y elecciones leídas de eventos IA. Con `aplicar=true` ajusta
  energía/formas de esos días, respeta ediciones manuales previas y no dispara
  `RECONCILIACION_ROTA`/`TENDENCIA_FUERA_BANDA` por su propio ajuste. Default: α 0.25, ancla `actual`,
  `aplicar=false`; si se activa emite `ESPECIALES_SIN_BACKTEST`.
- **4C R2:** `src/adjustments/automatico.py`; `estado` (2 estados) para el pipeline, `recomendacion`
  (3 estados, incluye `no_aplicar`) y `segmentos` por UCP × tipo × paso. `GET /adjustments/automatico`.
  `aplicar_automatico` solo añade diagnóstico por día; **no cambia curvas**.
- **4D F06:** `src/adjustments/eventos.py` (cliente OpenAI inyectable, cache en `.cache/eventos_ajustes`,
  saneo al rango exacto, no sobrescribe cache buena en fallo); `main.get_future_events_from_openai`
  delega; `control` emite `FESTIVO_EN_SEMANA`, `EVENTOS_SEMANA` (marcados “sugeridos por IA, no
  verificados”) y `CLIMA_OBSERVADO` (clima observado cacheado; no se persiste). Sin key/fallo → `{}`.
- **4E F09:** `src/adjustments/aprendizaje.py`, `Store.list_runs` (InMemory/Postgres, 2 queries en
  Postgres), `GET /adjustments/aprendizaje` en threadpool. Solo lectura; no reentrena.

## 3. Decisiones tomadas sin el usuario

1. **Fase 4 en el orden 4B→4C→4D→4E**, con 4A verificado (ya estaba en main).
2. **4B no se activa por defecto**: el re-backtest `modelo_k1` refutó α 0.5 + ancla análogos; se usa
   α 0.25/ancla `actual` y `aplicar=false`. Las elecciones se cubren como alerta IA, no como curva.
3. **R2 es una clasificación de revisión, no una ejecución**: el backend aplica siempre los pasos;
   `segmentos_paso`/`recomendacion` permiten al front automatizar u omitir. Queda pendiente que el
   producto/front decida si omite los segmentos `no_aplicar`.
4. **Eventos**: consulta OpenAI una vez por request (fuera del pipeline), cache por rango exacto,
   TTL 7 d, vacíos 1 d, timeout 5 s; corridas viejas sin eventos no consultan en PATCH.
5. **Clima**: se cachea el CSV por `(ruta, mtime, tamaño)`; la alerta es informativa y no se persiste
   (limitación aceptada: no es reproducible si el CSV cambia).
6. **F09**: `min_casos=3` por defecto; agrega todos los estados (borrador/aprobada/reemplazada) — se
   documenta como limitación; porcentajes ×100.
7. **Postgres local no autenticó** (`password authentication failed for user "samue"`): los tests de
   Postgres se saltaron; el SQL de `list_runs` queda pendiente de ejecutar en un entorno con DSN.
8. **Push**: `origin/main` avanzó con commits de docs de otra sesión; no se hizo force-push. `main`
   quedó en `ce4b6fb`. La rama remota `feat/ajustes-fase4` quedó obsoleta en `c9c9a2a`; la historia
   final está en `main` y en la rama nueva `feat/ajustes-fase4-final`.

## 4. Pendientes (orden recomendado)

1. **Front Sphere (Fase 3)**: el contrato pasa a 6 pasos (`especiales` entre `forma` y `control`) y
   aparecen `GET /adjustments/automatico`, `GET /adjustments/aprendizaje`,
   `meta.extras.modos_paso`/`segmentos_paso` y `aplicar_automatico`. Actualizar
   `docs/work/ajustes/fase3/CONTRATO_BACKEND.md` y el whitelist del backend intermedio. No se
   implementó el front en esta sesión.
2. **Backend intermedio**: añadir `usuario` en `aprobar`; registrar el módulo en BD para quitar la
   inyección de menú; propagar `X-Ajustes-Token`, `base_curvas`, `base_ref`, `aplicar_automatico`.
3. **Producto**: decidir si el front omite segmentos `no_aplicar` (Antioquia domingo/nivel,
   Atlántico laboral/nivel) y si `especiales` se activa con evidencia adicional. Hoy el backend no
   omite nada por R2.
4. **Postgres**: aplicar `src/adjustments/sql/001_init.sql` + `002_versionado.sql` en el Postgres de
   despliegue, definir `AJUSTES_DATABASE_URL` y `AJUSTES_SERVICE_TOKEN`, y correr los tests de
   Postgres/`list_runs`.
5. **Datos fuera de git para el otro PC**: `.env` (ver `.env.migracion-pc-epm-usb.zip`), `data/`,
   `models/`, `data/features/`, y `reports/ajustes/curvas_modelo_k1_*.csv` (solo para re-correr
   backtests). `docs/` y `reports/` están en `.gitignore`: usar `git add -f` para los `.md`.
6. **Demo Railway**: `docs/work/ajustes/demo/PLAN.md` y `PROMPT_EJECUTAR.md` (de otra sesión) se
   ejecutan DESPUÉS de integrar Fase 4 (ya integrada).

## 5. Qué revisar al despertar (5 minutos)

```powershell
# en C:\Users\samue\OneDrive\Documentos\GitHub\EPM
git log --oneline -6
git status --short

.venv\Scripts\python.exe -m pytest tests -q --ignore=tests/test_hourly_disaggregation.py -p no:cacheprovider
# main: 486 passed, 5 skipped, 4 failed preexistentes (christmas + 3 async sin pytest-asyncio en main)
# worktree/rama final: 678 passed, 5 skipped, 1 failed (solo christmas)

.venv\Scripts\python.exe -c "import src.api.main; print('ok')"
```

- `/health` 200; `/adjustments/automatico?ucp=Antioquia` 200; `/adjustments/aprendizaje` 503 sin
  `AJUSTES_DATABASE_URL` (esperado).
- `git status` solo debe mostrar cambios ajenos preexistentes: `.claude/settings.local.json` y
  `fastapi_factores/nul`.
- Reportes para revisar: `reports/ajustes/backtest_modelo_20261113.md`,
  `backtest_analogos_20261113_modelo.md`, `asr_comparacion_20261113_modelo.md`.
- Docs de la fase: `docs/work/ajustes/fase4/LOG.md` + `REPORTE_4B..4E`/`REPORTE_INTEGRACION.md`.

## 6. Archivos tocados en Fase 4

Código: `src/adjustments/{pipeline,router,store,analogos,data,types,automatico,eventos,aprendizaje}.py`,
`src/adjustments/steps/{control,especiales}.py`, `src/api/main.py`, `.env.example`.
Tests: `tests/test_adjustments_{especiales,automatico,eventos,aprendizaje,fase4_router}.py` (nuevos) y
ajustes en `test_adjustments_{pipeline,analogos_pasos,router}.py`.
Paso 0: `scripts/{backtest_adjustments,backtest_analogos,comparar_asr}.py`,
`tests/test_backtest_analogos_modelo_k1.py`, reportes `*_modelo.md`.
Docs: `docs/work/ajustes/fase4/` (planes, reportes, LOG), este resumen y `LOG.md` maestro.

## Verificación final en main (2026-11-13, tras cerrar Fase 4)
- `pytest tests -q --ignore=tests/test_hourly_disaggregation.py -p no:cacheprovider`: 678 passed, 5 skipped, 4 failed.
- Los 4 fallos son preexistentes y de entorno: `test_christmas_profile_uses_prior_christmases` (locale) y 3 tests async (`test_future_events`, 2 de `test_openai_analysis`) por falta de pytest-asyncio. Ninguno viene de la Fase 4.
- `python -c "import src.api.main"`: OK.
