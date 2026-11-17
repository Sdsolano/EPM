# LOG fase ASR

## Estado actual (2026-11-13 ~07:50)
- **Hecho:**
  - implementación: asr.py, comparar_asr.py, router_asr.py, script y tests;
  - corrida histórica y reporte;
  - ronda 1 de revisión (con un ALTO) y fixes;
  - ronda 2 completa: sin críticos ni altos; fixes de medios y bajos aplicados;
  - verificación estándar: 306 passed, 4 skipped, 1 failed preexistente (christmas). `import src.api.main`
    OK. Hash de data/raw, models y data/features sin cambios.
- **Hecho también:** commit local 0e8de2e en `feat/ajustes-asr-comparacion` (sin push).
- **Falta:** nada de esta fase después del commit. No hay push (lo hace el coordinador).
- **Archivos sin commitear** (hasta el commit):
  - código: src/adjustments/{asr,comparar_asr,router_asr}.py, src/api/main.py (2 líneas),
    scripts/comparar_asr.py, tests/test_asr.py y tests/test_comparar_asr.py;
  - docs: docs/work/ajustes/asr/{PLAN,LOG}.md;
  - reporte: reports/ajustes/asr_comparacion_20261112.md (con `git add -f`; el CSV NO se versiona).
- **Subagentes vivos:** ninguno.
- **Siguiente paso exacto:** `git add` por ruta, con `-f` para los .md, y commit con mensaje en español
  terminado en `Co-Authored-By`. Después, avisar al coordinador.

## Progreso (histórico)
- [x] Lectura de docs, del front (Sphere test 57eb4fb) y del endpoint. Hallazgos en PLAN.md.
- [x] PLAN.md
- [x] asr.py y tests: equivalencia con el endpoint vía TestClient, con motor falso y con models/Antioquia.
- [x] comparar_asr.py, scripts/comparar_asr.py, router_asr.py (+2 líneas en main.py) y tests.
- [x] Corrida histórica (`--historico --base ingenuo archivo --fecha 20261112`, ~25 s) y reporte
      reports/ajustes/asr_comparacion_20261112.md. Las secciones 6–9 son manuales: diagnóstico, ejemplos,
      recomendación y guía del viernes.
- [x] Ronda 1 de revisión: 9 agentes sonnet, 7 relanzados tras corte por límite de uso.
- [x] Ronda 2 completa (por el ALTO de la ronda 1): 9 agentes sonnet, sin críticos ni altos.
- [x] Verificación estándar.
- [x] Commit local 0e8de2e.

## Decisiones tomadas sin el usuario
- Implementé yo el núcleo (asr.py), porque la fidelidad al front depende de detalles leídos en este
  hilo. La revisión sí fue con agentes separados.
- Escenario principal de selección de históricos: `anual` (ver PLAN.md). Se evalúan 5 escenarios, todos
  supuestos; los veredictos numéricos se validan con el caso real del analista.
- En el backtest, `is_special_day` usa los festivos de la librería `holidays` (sin red) en lugar de la API.
- Los modelos de desagregación (clusters) son los actuales, entrenados en 2026-07 con 3 meses. En semanas
  históricas el cluster no es el que existía entonces, pero es el que usa hoy el ASR.
- Endpoint comparar-asr con base `modelo` → 422, porque refresca datos y llama al modelo; para el viernes
  se usa base_curvas. En el script, `--base modelo` queda enchufado a `proveedor_modelo` del backtest
  (Fase 2a).
- Corte por defecto del endpoint: último dato real, a lo sumo el sábado previo (como el backtest).
- Festivos para `is_special_day` en el comparador: los mismos del contexto (API por UCP o librería).
- Tablas del reporte sobre días comunes a todas las series comparadas.
- REPORTE.md: el harness bloquea que este agente escriba archivos de reporte. El resumen tipo REPORTE
  está en la sección final de este LOG.

## Ronda 1 de revisión (cerrada)
- **line-by-line:** medio (SERIES_PRINCIPALES fija → reporte vacío con --escenarios reducido) más
  bajos. Corregidos.
- **removed-behavior:** sin hallazgos.
- **regresión:** sin regresión.
- **cross-file:** medio (caché de motores sin invalidación ni lock) más bajos (InvalidOperation → 500;
  fallo de carga reintentado). Corregidos: lock + firma por mtime de los .pkl; valores no finitos →
  ValueError; motor cargado una vez con el error cacheado.
- **reuse:** medio (paridad de es_dia_especial sin test). Test agregado. `proveedor_ingenuo` duplicado
  aceptado: src no importa scripts y hay test de paridad.
- **simplification:** bajos.
  - Aplicados: `**d` y los filtros redundantes de reales.
  - Aceptados:
    - `motor`, `festivos` y `congelar_desvi2` en asr_dia: son las variantes de diagnóstico pedidas y se
      usan en tests;
    - `base: "modelo"` con 422 explícito, porque el mensaje es útil.
- **efficiency:** medio (caché y lock), corregido. `curvas_reales` se calcula una sola vez en el
  endpoint. Lo demás se acepta (costo < 7 %).
- **altitude:** un **ALTO**: la guía del viernes no era ejecutable con el histórico local, que llega a
  jul-2026. Además, medios: alcance de la evidencia (selección supuesta, circularidad con base ingenua) y
  evidencia en contra de la mesa omitida. Corregidos en el reporte (Paso 0, "Alcance de la evidencia",
  evidencia en contra, cifras de domingos) y con `advertencias` en el comparador.
- **conventions:** medio (falta REPORTE.md; ver decisión). Bajos corregidos: HTTP_422 del router, toggle
  del logger eliminado, parámetro sin uso en test y rutas `python` genéricas.

## Ronda 2 (completa, sin críticos ni altos) → fixes inline + una verificación estándar
- **line-by-line y cross-file:** bajos (división por cero si el real es todo cero; perfil promedio de
  festivos None → TypeError). Corregidos: métricas en None y ValueError → día omitido.
  `Math.round` en 0.49999999999999994 aceptado (teórico).
- **reuse:** medio (validación de la selección y de las curvas CSV divergente entre router y script;
  `seleccion` fuera de semana descartada en silencio). Corregidos con `comparar_asr.validar_curvas`
  compartida, ValueError por fechas objetivo fuera de semana y validación de los CSV en el script.
  Duplicación de la regla de corte aceptada (no se puede importar el backtest desde src).
- **altitude:** medios (el Paso 0 decía que el endpoint refresca, y no lo hace; `full_update_csv` usa la
  API de clima y `.env`; el reporte está bajo /reports, ignorado). Corregidos en la guía. El reporte se
  versiona con `git add -f`, como los backtest_*.md previos. Bajo: "consistente" → "mejora pequeña,
  mediana cercana a cero".
- **simplification:** bajos. Quitados `% 24` redundante y el comentario viejo; el guard `h_tipos` se
  acepta (defensivo para CSV antiguos). `motor_estado` aceptado.
- **efficiency:** bajos aceptados: unos 50–100 ms síncronos en el endpoint de diagnóstico y un
  `curvas_reales` extra por semana.
- **conventions:** bajos aceptados: uso de helpers privados de router.py (mismo paquete, no se puede
  modificar router.py) e importlib para el backtest (scripts/ no es paquete).
- **removed-behavior y regresión:** sin hallazgos. Suite: 303 passed (después 306 con los tests nuevos),
  1 preexistente.

## Resumen tipo REPORTE
- **Archivos:**
  - `src/adjustments/asr.py`: reproducción fiel del ASR + senda/validador del endpoint + escenarios.
  - `src/adjustments/comparar_asr.py`: comparador de una semana con métricas y advertencias.
  - `src/adjustments/router_asr.py`: `POST /adjustments/comparar-asr`.
  - `scripts/comparar_asr.py`: modos semana e histórico.
  - Tests: 64 nuevos.
  - Reporte: `reports/ajustes/asr_comparacion_20261112.md`.
- **WAPE horario, base / mesa / ASR `anual`, días comunes:**

  | base | UCP | base | mesa | ASR `anual` |
  |---|---|---|---|---|
  | ingenuo | Antioquia | 4.05 | 3.71 | 5.26 |
  | ingenuo | Atlántico | 6.36 | 6.00 | 8.24 |
  | archivo | Antioquia | 3.03 | 2.79 | 5.42 |

- **Diagnóstico:**
  1. El ASR es h1 + (h2 − h3): descarta la base.
  2. La corrección que aplica no se relaciona con la necesaria (correlación −0.02 a 0.25) y es tan grande
     como el error.
  3. Los festivos o la Semana Santa en h1, h2 o h3 lo disparan (39.6 % el 2026-01-12).
  4. Hay un bug de orden: desvi2 se congela en orden de clic, mientras h1 sale por fecha.
  5. `tipo_dia` y el validador son irrelevantes para el error.
- **Recomendación** (a confirmar con el caso real): reemplazar el ASR por la mesa (Fase 3). Paliativo:
  marcar el ASR como no recomendado, corregir el orden de desvi2 o llamar a `/adjustments/preview`.
- **Pendientes y riesgos:**
  - Datos al día para el caso del viernes (Paso 0 de la guía).
  - Validar la selección real del analista.
  - Base `modelo` (Fase 2a).
  - Puntos débiles de la mesa: festivos y días después de Semana Santa (fases 2b/4).
