# Servicio de ajustes por pasos — cómo continuar

Punto de entrada para retomar este trabajo en otro PC o con otro agente. Leer en este orden:
1. `RESUMEN_NOCHE.md`: estado final, tabla antes/después, decisiones sin el usuario y pendientes.
2. `REQUERIMIENTOS.md`: objetivo, retroalimentación del analista (ASR, análogos) y backlog.
3. `PLAN.md`: diseño, pasos, esquema y endpoints. Si algo difiere, manda el código y `LOG.md`.
4. `LOG.md`: decisiones, hallazgos de revisión y lo aceptado sin corregir.

## Estado (2026-11-13, `main` = `ce4b6fb`)
- **Fase 1/ASR/2a/2b/Fase 4 terminadas e integradas en `main`.** Fase 3 (UI Mesa) está en Sphere
  `test` = `22ef6d1`. Re-backtest con base `modelo_k1`: `6ad685d`.
- Fase 4 añade el paso `especiales` (6 pasos), el automático por segmento R2 (clasificación para el
  front, off por defecto), alertas de eventos/clima F06 y aprendizaje F09 de solo lectura.
- **No desplegado todavía**: falta aplicar el esquema en el Postgres del servidor y definir las
  variables de entorno. En este PC los tests de Postgres se saltaron por falta de DSN.

## Qué hace
`base` (modelo o `base_curvas`) → `referencia` → `nivel` (α=0.25) → `forma` (`promedio_n` n=4) →
`especiales` (Semana Santa, 24/31-dic, post-festivo; `aplicar=false` por defecto) → `control`
(alertas/bloqueos) → aprobación versionada (una aprobada vigente por UCP y semana).

## Arranque en un PC nuevo
1. `git clone` y `git checkout main` (o `git pull`). Crear venv con Python 3.11 y
   `pip install -r requirements.txt` + `pytest`.
2. **Datos que NO están en git** (ignorados por `.gitignore`); copiarlos aparte:
   - `data/raw/Antioquia/` (`datos.csv`, `clima_new.csv`); `data/raw/Atlantico/` y `data/features/`.
   - `models/` (sin ellos `/adjustments/run` responde 409 "use /retrain").
   - `reports/performance/antioquia_ene_jun_2026/resultados_horarios_ene_jun_2026.csv` (base archivo).
   - `reports/ajustes/curvas_modelo_k1_{Antioquia,Atlantico}.csv` (solo para re-correr backtests).
   - `.env` raíz (copiar de `.env.example` y completar, incluye `API_KEY` de OpenAI).
3. Variables en el `.env` **raíz**:
   - `AJUSTES_DATABASE_URL=postgresql://<usuario>:<clave>@<host>:5432/<db>`; sin ella `preview`
     funciona y los endpoints que guardan responden 503.
   - `AJUSTES_SERVICE_TOKEN=<secreto>` (opcional): todas las llamadas a `/adjustments/*` deben mandar
     `X-Ajustes-Token`.
   - `API_KEY` de OpenAI (opcional): eventos/alertas F06; sin ella degrada a `{}`.
   - `AJUSTES_EVENTOS_CACHE_DIR` (opcional; default `.cache/eventos_ajustes`).
4. Esquema (idempotente): `python scripts/apply_adjustments_schema.py` (aplica `sql/001` y `002`).

## Verificar
- Suite: `python -m pytest tests -q --ignore=tests/test_hourly_disaggregation.py -p no:cacheprovider`.
  En `main` son preexistentes: `test_future_events`, `test_holiday_homology::...christmas...` y los 2
  de `test_openai_analysis` (falta pytest-asyncio/red en este entorno). En el worktree de Fase 4 solo
  falla christmas.
- `python -c "import src.api.main"`.
- Postgres: `AJUSTES_DATABASE_URL=postgresql:///postgres python -m pytest tests/test_adjustments_router.py -q`
  (crea y borra una base temporal).
- Backtest: `python scripts/backtest_adjustments.py --base ingenuo archivo --fecha <sufijo>`.
  `.md` versionados; los CSV grandes se regeneran.

## Endpoints nuevos de Fase 4
- `GET /adjustments/automatico?ucp=...`: configuración R2 por tipo de día y `recomendacion`
  (`automatico`/`sugerido`/`no_aplicar`). No cambia curvas.
- `GET /adjustments/aprendizaje?ucp=&desde=&hasta=&limite=&min_casos=`: intervenciones editadas vs
  calculadas (solo lectura; requiere store).
- En preview/run: `meta.extras.modos_paso`, `meta.extras.segmentos_paso`, `aplicar_automatico`
  (diagnóstico, no automatiza en backend) y alertas `FESTIVO_EN_SEMANA`, `EVENTOS_SEMANA`,
  `CLIMA_OBSERVADO`.

## Pendiente (ver `RESUMEN_NOCHE.md` §4)
1. Front Sphere: contrato de 6 pasos + endpoints/alertas nuevos; actualizar
   `fase3/CONTRATO_BACKEND.md` y el whitelist del backend intermedio.
2. Backend intermedio: `usuario` en aprobar, registro del módulo en BD, token/base_curvas/base_ref.
3. Producto: decidir si el front omite segmentos `no_aplicar` y si se activa `especiales` con más
   evidencia.
4. Despliegue: esquema SQL, `AJUSTES_DATABASE_URL`/`AJUSTES_SERVICE_TOKEN`, tests Postgres.
5. Demo Railway: `docs/work/ajustes/demo/PLAN.md` (después de Fase 4, ya integrada).

## Cosas a saber
- Una corrida con el modelo **refresca datos** primero (`full_update_csv`: red y reescritura de
  `data/raw/{ucp}`) y usa como corte el último dato real. Semana ya observada → 422.
- Festivos: API por UCP; si falla, librería `holidays` + alerta `FESTIVOS_FUENTE_ALTERNA`.
- Dos controles: `TENDENCIA_FUERA_BANDA` (warning, ediciones vs nivel) y
  `TENDENCIA_REFERENCIA_FUERA_BANDA` (info, vs semana de referencia). Ninguno bloquea; bloquean
  periodos faltantes, NaN, negativos o reconciliación rota.
- `especiales` respeta ediciones manuales previas y su ajuste no dispara falsos positivos de
  reconciliación/enclavamiento.
- Metodología: plan → implementación por agentes → revisión multi-ángulo + regresión → commit.
