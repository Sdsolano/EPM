# Fase 2b — Buscador de días análogos (R3): sub-plan

Coordinador: agente de la Fase 2b (sesión autónoma 2026-11-12/13). Worktree
`C:/Users/samue/OneDrive/Documentos/GitHub/EPM-wt-2b`, rama `feat/ajustes-fase2b` (desde `main` 74bc100).
Reglas comunes: `C:/Users/samue/OneDrive/Documentos/GitHub/EPM/docs/work/ajustes/REGLAS_AGENTES.md`.
Contrato FIJO de la API: "Contrato de análogos" en
`C:/Users/samue/OneDrive/Documentos/GitHub/EPM/docs/work/ajustes/PLAN_NOCHE.md` (copiado abajo en §2; solo se
admiten campos opcionales nuevos, documentados aquí).

Contexto de negocio: `docs/work/ajustes/REQUERIMIENTOS.md` (R3), `README.md`, `PLAN.md`, `LOG.md` del worktree.

Convivencia con la Fase 2a (rama `feat/ajustes-fase2a`, en paralelo): esa fase toca `nivel.py` (α por tipo de
día), `scripts/backtest_adjustments.py` y arreglos de pandas 3. Aquí los cambios en `nivel.py`, `forma.py`,
`referencia.py`, `router.py` son **aditivos y acotados** (no reordenar, no reformatear, no renombrar). El
backtest de análogos va en un script NUEVO (`scripts/backtest_analogos.py`) que importa funciones de
`scripts/backtest_adjustments.py` sin modificarlo.

## 1. Partes y dueños

| Parte | Archivos | Dueño | Estado |
|---|---|---|---|
| A. Módulo `analogos.py` + endpoint + tests | `src/adjustments/analogos.py`, `src/adjustments/router.py` (aditivo), `tests/test_adjustments_analogos.py` | subagente A | ver LOG |
| B. Integración en pasos + tests | `steps/forma.py`, `steps/nivel.py`, `steps/referencia.py` (aditivo), `tests/test_adjustments_analogos_pasos.py` | subagente B (tras A) | ver LOG |
| C. Backtest + reporte | `scripts/backtest_analogos.py`, `reports/ajustes/backtest_analogos_<AAAAMMDD>.md` | subagente C (tras B) | ver LOG |
| D. Revisión (8 ángulos + regresión, sonnet), fixes, commit local | — | coordinador | ver LOG |

## 2. Contrato (copia del PLAN_NOCHE, fijo)

`POST /adjustments/analogos` (dependencia `verificar_token` del router, igual que el resto).
Request:
```json
{"ucp": "Antioquia", "fecha": "2026-05-19", "fecha_corte": "2026-05-16",
 "filtros": {"dia_semana": "auto", "meses": [5], "temporada_dias": null, "relacion_festivo": "auto",
             "distancia_festivo": 1, "tipo_festivo": null, "semana_santa": null, "temporada_escolar": null,
             "clima": null, "anio_min": null, "anio_max": null, "anios_excluidos": [2020], "limite": 20}}
```
Respuesta 200:
```json
{"ucp": "...", "fecha": "...", "fecha_corte": "...",
 "descriptor_objetivo": {"dia_semana": 1, "mes": 5, "festivo": false, "relacion_festivo": "despues",
                         "distancia_festivo": 1, "semana_santa": false, "temporada_escolar": "clases"},
 "filtros_aplicados": {...},
 "candidatos": [{"fecha": "2025-05-20", "score": 0.93, "razones": ["martes", "mayo", "día después de festivo (2025-05-19)"],
                 "observado": true, "energia": 31200.5, "pico": {"mw": 1500.1, "hora": 20},
                 "curva": [24], "curva_normalizada": [24], "clima": {"temp_media": 23.1, "lluvia_mm": 3.0} | null}],
 "resumen": {"n_candidatos": 12, "curva_promedio_normalizada": [24], "energia_promedio": 31000.0}}
```

### 2.1 Semántica fijada por esta fase (decisiones; ver LOG "decisiones sin el usuario")
- `filtros` ausente o `null` en una clave = NO filtra por ese criterio, salvo `anios_excluidos` (ausente →
  `[2020]`; `[]` explícito = no excluye) y `limite` (ausente → 20; rango 1..366).
- `fecha_corte` ausente → último dato real (último día con 24 periodos válidos). Si llega una posterior al
  último dato real se usa el último dato real (queda en `filtros_aplicados.fecha_corte_efectiva`). El día
  objetivo nunca es candidato de sí mismo. Solo días `<= corte`.
- `dia_semana`: `"auto"` = el del objetivo; int 0..6; lista de ints; null.
- `meses`: lista 1..12. `temporada_dias`: int 1..183, ventana ± N días del día del año del objetivo
  (circular sobre el año). Si vienen ambos se exige los dos.
- `relacion_festivo`: `"auto"` (= la del objetivo) | `"festivo"` | `"puente"` | `"antes"` | `"despues"` |
  `"ninguna"` | null. `distancia_festivo`: int 1..7 | `"auto"` (= la del objetivo) | null; solo filtra cuando la
  relación (resuelta) es `antes`/`despues`.
- `tipo_festivo`: clave del festivo relacionado (slug, ver §3.2) o nombre; también `"auto"`. `semana_santa`:
  bool | null. `temporada_escolar`: `"clases"` | `"vacaciones"` | null. `anio_min`/`anio_max`: int.
- `clima`: `{"temp_tolerancia": float>0, "lluvia": bool|null, "temp_media": float|null}` (campo opcional
  nuevo `temp_media`: temperatura objetivo pronosticada para días futuros; si no llega se usa la observada del
  objetivo; si no hay ninguna, el filtro de temperatura no se aplica y se anota en
  `filtros_aplicados.notas`). Candidatos sin clima observado se descartan cuando el filtro de clima está activo.
- Extensiones opcionales de la respuesta (aditivas): `candidatos[].descriptor`, `candidatos[].componentes`
  (aporte de cada criterio al score), `filtros_aplicados.notas`, `filtros_aplicados.fecha_corte_efectiva`,
  `resumen.n_observados`, `resumen.ponderado_por_score: true`, `descriptor_objetivo` con claves extra
  (`anio`, `festivo_relacionado`, `tipo_festivo`, `nombre_festivo`, `clima`).
- Orden de candidatos: score desc, luego fecha desc (más reciente primero). Los no observados (sin 24 periodos
  válidos) aparecen con `observado: false`, `energia/pico/curva/curva_normalizada: null`; no entran al
  `resumen`. `resumen.curva_promedio_normalizada` y `energia_promedio` = promedio ponderado por score de los
  observados (null si no hay).

## 3. Diseño de `src/adjustments/analogos.py` (funciones puras, sin red)

### 3.1 API pública
```python
FILTROS_DEFAULT: dict            # todas las claves del contrato con su default (None, [2020], 20)
def validar_filtros(filtros: Optional[dict]) -> dict      # ValueError (→ 422) ante claves/tipos/rangos inválidos
def descriptor_dia(d: date, festivos: set[str], clima: Optional[dict] = None) -> dict
def clima_diario(df_clima: pd.DataFrame) -> dict[str, dict]   # {"YYYY-MM-DD": {"temp_media", "lluvia_mm", "humedad_media"}}
def cargar_clima(ucp: str, data_dir: Optional[Path] = None) -> dict  # data/raw/{ucp}/clima_new.csv si existe, si no {}
def puntuar(obj: dict, cand: dict) -> tuple[float, dict, list[str]]  # (score 0..1, componentes, razones)
def buscar_analogos(historico, festivos, fecha, fecha_corte=None, filtros=None, clima=None) -> dict
    # dict con descriptor_objetivo, filtros_aplicados, candidatos, resumen, fecha_corte (sin "ucp": lo pone el router)
def seleccionar(historico, festivos, fecha, fecha_corte, filtros=None) -> list[tuple[str, float, np.ndarray]]
    # helper para pasos: [(fecha, score, curva_24)] solo observados, ya ordenados (usa buscar internamente o el índice)
def puntuar_fechas(historico, festivos, fecha, fechas: list[str]) -> list[tuple[str, float, Optional[np.ndarray]]]
    # para selección explícita del analista: score de cada fecha elegida (curva None si no observada)
```
Rendimiento: un índice de descriptores del histórico se construye una vez por (rango de fechas, festivos) y se
cachea a nivel de módulo (cache pequeño, p. ej. `functools.lru_cache` con clave hashable o dict con tope);
una búsqueda sobre ~10 años < 100 ms tras construir el índice. Nunca escribe archivos.

### 3.2 Descriptor de un día
- `dia_semana` (0=lunes), `mes`, `anio`, `dia_del_anio`.
- `festivo`: fecha en el set de festivos (de la UCP/ctx). `nombre_festivo`: `holiday_name_for_date` (librería
  `holidays`, español, p. ej. "Día de San José (observado)").
- `tipo_festivo`: slug del festivo RELACIONADO (el propio si es festivo; el más cercano si la relación es
  antes/después/puente): minúsculas, sin tildes, sin "(observado)", espacios → `_` (p. ej. `dia_de_san_jose`).
  Jueves y Viernes Santo → `semana_santa` (alias), festivo sin nombre en la librería → `festivo_ucp`.
- `relacion_festivo` (prioridad en este orden):
  1. `festivo`: el día es festivo.
  2. `puente`: el día no es festivo y pertenece a un bloque contiguo de días no laborables (sábado, domingo,
     festivo) que contiene al menos un festivo (p. ej. sábado y domingo antes de un lunes festivo), o es un día
     laboral entre dos días no laborables con al menos un festivo adyacente ("sándwich").
  3. `antes`/`despues`: festivo más cercano a ≤ `MAX_DIST_FESTIVO` = 3 días (calendario). Empate → `despues`.
     `distancia_festivo` = días a ese festivo (1..3).
  4. `ninguna` (con `distancia_festivo` = null).
  `festivo_relacionado` = ISO del festivo usado (o null).
- `semana_santa`: Domingo de Ramos .. Domingo de Resurrección (Pascua por `dateutil.easter`).
- `temporada_escolar` (aproximado, calendario A de Colombia, documentado en el módulo): `vacaciones` si
  1-ene..25-ene, semana santa (lunes..domingo), 15-jun..10-jul, semana del lunes festivo de octubre (receso
  escolar) o diciembre completo; si no `clases`.
- `clima` (opcional): `{"temp_media": °C media de p_t, "lluvia_mm": suma de p_i, "humedad_media": media p_h}`
  del día si está en `clima_new.csv` (columnas `fecha, periodo, p_t, p_h, p_v, p_i`); días con < 20 periodos
  o valores no finitos → sin clima.

### 3.3 Score (0..1, explicable)
Pesos (constantes del módulo, documentadas): día de semana 0.25 (mismo día 1.0; ambos laborales lun–vie 0.4);
festivo/relación 0.25 (misma relación y misma distancia 1.0; misma relación distinta distancia 0.6; en
festivos mismo `tipo_festivo` suma, distinto 0.6); estacionalidad 0.20 (`max(0, 1 − dist_días_del_año/60)`);
recencia 0.10 (`exp(−años/4)`); Semana Santa 0.05 (coincide); temporada escolar 0.05 (coincide); clima 0.10
solo si objetivo y candidato tienen temperatura (`max(0, 1 − |Δt|/3)`), si no se omite y se renormaliza.
`score = Σ w·s / Σ w` (pesos presentes), redondeado a 4 decimales. `componentes` = {criterio: aporte}.
`razones`: solo criterios que coinciden, en español legible: `"martes"`, `"mayo"` (mismo mes),
`"día después de festivo (2025-05-19)"`, `"2 días antes de festivo (…)"`, `"puente festivo (…)"`,
`"festivo: Día del Trabajo"`, `"Semana Santa"`, `"temporada de clases"`/`"vacaciones escolares"`,
`"temperatura similar (23.1 °C)"`, `"con lluvia"`.

### 3.4 Validación de UCP / endpoint
Router (aditivo, al final de la sección de modelos y endpoints, sin tocar lo existente):
- `AnalogosRequest` (pydantic): `ucp` con `UCP_PATTERN`, `fecha: date` y `fecha_corte: Optional[date]` en
  [FECHA_MIN, FECHA_MAX], `filtros: Optional[Dict[str, Any]]`.
- Flujo: `_validar_ucp` → `_load_history` → `_load_festivos(ucp, hist, fecha)` → clima con un loader inyectable
  `set_clima_loader(fn)` (default `analogos.cargar_clima`; fallo de lectura → sin clima + nota) →
  `analogos.validar_filtros` (ValueError → 422) → `buscar_analogos` en threadpool (ValueError → 422).
- No refresca datos, no llama al modelo, no usa store (funciona sin base). Mismo token.
- Declarar la ruta ANTES de `GET /{run_id}` no es necesario (es POST), pero no debe chocar con `/{run_id}`.

## 4. Integración en pasos (Parte B)
Params nuevos opcionales; defaults sin cambiar (salvo evidencia de la Parte C).
- `forma`: `fuente` acepta además `"analogos"`. Params nuevos `analogos: {"YYYY-MM-DD": ["fecha", ...]} | None`
  (default None) y `filtros_analogos: dict | None` (default None = filtros automáticos `FILTROS_AUTO_FORMA`).
  Con `fuente="analogos"`: día con entrada en el mapa → esas fechas (deben ser ≤ corte; las no observadas se
  omiten con alerta info), pesos = score de `puntuar_fechas`; día sin entrada → búsqueda automática
  (`seleccionar`, limite = `n`). Forma = promedio ponderado por score de formas normalizadas (sin suavizar).
  Festivos sin entrada en el mapa → se conserva el perfil homólogo actual. Sin análogos → fallback
  `promedio_n` con alerta `FORMA_FUENTE_ALTERNA` (motivo `sin_analogos`). Diagnóstico por día: `fechas_fuente`,
  `pesos`, `seleccion` (`"analista"`/`"automatica"`).
- `nivel`: param `ancla: "actual" | "analogos"` (default `"actual"` = comportamiento actual) + `analogos` y
  `filtros_analogos` como en forma. Con `"analogos"`: para cada análogo c, `lag` = el del ancla actual del día
  objetivo (d − a), ancla del análogo c_a = c − lag (no festivo, observada; si no, se retrocede de a 7 días
  hasta 6 semanas); ratio = Σ score·E(c)/E(c_a) / Σ score; `E* = E(a)·ratio`. Sin ratios válidos → evidencia de
  la semana de referencia (comportamiento actual) con alerta info `NIVEL_ANCLA_ALTERNA`. Diagnóstico por día con
  `regla: "analogos"` y las fechas usadas.
- `referencia`: diagnóstico `semanas_sugeridas`: para los días de la semana objetivo, búsqueda automática de
  análogos (limite pequeño) → lunes de cada análogo; ranking de semanas por suma de scores de días cubiertos;
  top 5 `[{"semana_inicio", "score", "dias": {fecha_objetivo: fecha_analoga}}]`. No cambia la semana elegida.
- Los mapas `analogos` validan: claves dentro de la semana objetivo, valores lista (≤ 60) de fechas ISO ≤
  `fecha_corte` (ValueError → 422). Recalcular con PATCH funciona igual que otros params (los params se
  guardan validados en el record).

## 5. Backtest (Parte C)
`scripts/backtest_analogos.py`, importa de `scripts/backtest_adjustments.py` (`construir_contexto`,
`semanas_objetivo`, `metricas_dia`, `proveedor_ingenuo`, `cargar_archivo`, `proveedor_desde_curvas`,
`proveedor_modelo`, `con_cache`, `_agg_dias`). Bases `ingenuo`, `archivo` (y `modelo` disponible por flag,
sin correrla: depende de la Fase 2a). Últimas 104 semanas por UCP: calibración = 52 más antiguas, holdout =
52 recientes. Configs: forma `promedio_n` (default), `referencia`, `analogos` automático (variantes de k y
ventana de temporada); nivel `ancla actual` vs `analogos` (α 0.25 y 0.5). Segmentos: tipo de día, relación
con festivo (antes/después/puente/ninguna), Semana Santa, diciembre, festivo. Reporte
`reports/ajustes/backtest_analogos_<AAAAMMDD>.md` (CSV fuera de git). Regla para cambiar un default: la
variante gana en calibración y en holdout, en ambas UCP con base ingenuo, por ≥ 2 % relativo de WAPE del paso
afectado (y no empeora el pico); si se aplica, es por segmento y retrocompatible.

## 6. Criterio de terminado
Código + tests verdes (suite sin fallos nuevos), endpoint con el contrato, backtest con reporte, revisión
limpia (ronda 1 completa: 8 ángulos + regresión con sonnet), REPORTE.md, commit local en `feat/ajustes-fase2b`.
