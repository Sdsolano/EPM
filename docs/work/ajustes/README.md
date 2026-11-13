# Servicio de ajustes por pasos — cómo continuar

Punto de entrada para retomar este trabajo en otro PC o con otro agente. Leer en este orden:
1. Este archivo (estado y arranque) y `REQUERIMIENTOS.md`, con el objetivo final, la
   retroalimentación del analista (ASR, buscador de análogos) y el backlog priorizado.
2. `PLAN.md`: diseño, pasos, esquema y endpoints. La sección "Contratos entre módulos" es la
   versión inicial; si algo difiere, manda el código y las secciones "Fix ronda 1/2" de `LOG.md`.
3. `LOG.md`: decisiones, hallazgos de revisión, lo que se aceptó sin corregir y por qué.

## Estado (2026-11-12)
- **Fase 1 terminada** y en `main` (commit `8501faa`): módulo `src/adjustments/`, API `/adjustments`,
  persistencia versionada en Postgres y backtest. Pasó 2 rondas de revisión multi-ángulo y la
  verificación final: 242 tests pasan; los 4 que fallan ya fallaban antes de la feature.
- **No desplegado todavía:** falta aplicar el esquema en el Postgres del servidor y definir las
  variables de entorno (ver abajo).

## Qué hace (resumen)
El pronóstico raw pasa por pasos en orden; cada paso se puede aplicar, omitir o editar, y editar uno
recalcula los siguientes:

`base` (modelo o `base_curvas` externas) → `referencia` (semana equivalente) → `nivel` (energía diaria,
α=0.25) → `forma` (perfil `promedio_n`, festivos con perfil homólogo) → `control` (alertas y bloqueos)
→ aprobación (una sola aprobada vigente por UCP y semana, con historial de versiones).

## Arranque en un PC nuevo
1. `git clone` y `git checkout main`. Crear venv con Python 3.11 y `pip install -r requirements.txt`.
2. **Datos que NO están en git** (ignorados por `.gitignore`); hay que copiarlos aparte. La carpeta
   `env.migracion-pc-epm-usb/` del PC original existe para esto:
   - `data/raw/Antioquia/` (`datos.csv`, `clima_new.csv`). Atlántico sí está versionado.
   - `models/` (modelos entrenados). Sin ellos `/adjustments/run` responde 409 "use /retrain".
   - `reports/performance/antioquia_ene_jun_2026/resultados_horarios_ene_jun_2026.csv`, solo para
     `--base archivo` del backtest.
   - `.env` raíz (copiar de `.env.example` y completar).
3. Variables en el `.env` **raíz** (el de la API de predicción, no el de `fastapi_factores/`):
   - `AJUSTES_DATABASE_URL=postgresql://<usuario>:<clave>@<host>:5432/<db>`. Sin ella, `preview`
     funciona y los endpoints que guardan responden 503.
   - `AJUSTES_SERVICE_TOKEN=<secreto>`, opcional. Si está definido, todas las llamadas a
     `/adjustments/*` deben mandar la cabecera `X-Ajustes-Token`.
4. Esquema (idempotente, se puede correr varias veces):
   `python scripts/apply_adjustments_schema.py` (o con `--dsn ...`). Aplica `sql/001_init.sql` y
   `sql/002_versionado.sql`.

## Verificar
- Suite: `.venv/bin/python -m pytest tests -q --ignore=tests/test_hourly_disaggregation.py`.
  Fallos que ya existían antes de esta feature (no son de ajustes): `test_future_events`,
  `test_holiday_homology::...christmas...` y los 2 de `test_openai_analysis`.
  `test_hourly_disaggregation.py` falla al importar (ImportError relativo), también desde antes.
- Tests contra Postgres real: `AJUSTES_DATABASE_URL=postgresql:///postgres .venv/bin/python -m pytest tests/test_adjustments_router.py -q`
  (crea y borra su propia base temporal).
- Los tests que usan el CSV real de Antioquia se saltan si ese archivo no está.
- Backtest (unos 17 s; no usa red):
  `.venv/bin/python scripts/backtest_adjustments.py --base ingenuo archivo --fecha <sufijo>`.
  Escribe en `reports/ajustes/`. Los `.md` están versionados; los `.csv` (~10 MB) no, se regeneran.

## Pendiente (siguientes fases)
El orden y el detalle actualizados viven en `REQUERIMIENTOS.md`, que manda sobre esta lista. Lo más
próximo: la revisión del ASR con el analista el viernes 2026-11-13 y el buscador de días análogos
(R3), junto con el punto 1.
1. **Backtest con el modelo real.** `--base modelo` existe, pero hoy falla por dos razones:
   - `src/pipeline/feature_engineering.py:493` usa `fillna(method='ffill')`, que pandas 3 ya no acepta.
   - `run_predict_flow` llama a `full_update_csv`, que va a la red y reescribe `datos.csv`.
   Con el modelo real habría que recalibrar α (posiblemente por tipo de día) y la banda del control.
   Los valores por defecto actuales son provisionales: se eligieron sobre las mismas semanas evaluadas.
2. **Despliegue:** aplicar el esquema y definir las variables de entorno en el servidor.
3. **Fase 3 – UI:** pantalla "Mesa de ajustes" en Sphere (rama `test`) y rutas
   `admin/dashboard/ajustes/*` en el backend intermedio (no está en estos repos). Ese backend debe
   mandar `X-Ajustes-Token`, el `usuario` real y, si ya tiene el pronóstico emitido, `base_curvas` y
   `base_ref`.
4. **Fase 4:** paso propio de días especiales (Semana Santa, 24/31-dic, elecciones), ranking de
   semanas de referencia (F01) y alertas de clima y eventos (F06).

## Cosas a saber
- Una corrida con el modelo **refresca datos** primero (`full_update_csv`: red y reescritura de
  `data/raw/{ucp}`) y usa como corte el último dato real. Una semana ya observada da 422.
- Los festivos salen de la API por UCP (`FestivosAPIClient`). Si falla, se usa la librería
  `holidays` y aparece la alerta `FESTIVOS_FUENTE_ALTERNA`. El backtest siempre usa la librería.
- Dos controles distintos:
  - `TENDENCIA_FUERA_BANDA` (warning): las ediciones movieron la energía semanal respecto al paso nivel.
  - `TENDENCIA_REFERENCIA_FUERA_BANDA` (info): el resultado se aleja de la tendencia de la semana de
    referencia.
  Ninguno bloquea. Bloquean solo los problemas estructurales: periodos faltantes, NaN, negativos o
  una forma que no cuadra con la energía del nivel.
- Metodología usada (framework global del usuario): plan → implementación por agentes → ronda de
  revisión multi-ángulo (revisores con Sonnet) + agente de regresión → commit.
