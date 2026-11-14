# CONTRATO_B — endpoints de insights para el front (Pista B, noche 2)

Versión 1 (2026-11-14). El front (Pista A) construye contra este contrato; cualquier cambio se anota al final en "Cambios".
Todo es **solo lectura**, JSON, mismo estilo que `src/adjustments/router.py`: prefijo `/adjustments/insights`, mismas cabeceras
(`X-Ajustes-Token` si `AJUSTES_SERVICE_TOKEN`; en demo, Basic/`X-Demo-Password`), validación de `ucp` igual (422 formato / 404 UCP
desconocida), `run_id` inexistente → 404, sin base configurada → 503. Errores: `{"detail": "..."}`.
Porcentajes en **puntos porcentuales ×100** (3.2 = 3.2 %). Energía en **MWh/día** (suma de P1..P24), potencia en **MW**. Horas: índice
0..23 = periodo P1..P24 (hora 0 = 00:00–01:00). Fechas ISO `YYYY-MM-DD`. Valores no calculables = `null`.

Todos los endpoints con `run_id` usan la corrida guardada (la que devuelve `POST /adjustments/run`); su **`final_curves` es el
pronóstico "ajustado"**, el paso `base` es el "base" del modelo y `referencia` es la curva real de la semana de referencia
(`steps[referencia].diagnostics.semana_ref_inicio`). Si la corrida está aprobada, `estado:"aprobada"` y las recomendaciones pasan a
informativas (`aplicable:false`).

## 1. `GET /adjustments/insights/resumen?run_id=`
Pantalla de entrada "Resumen ejecutivo": todo lo necesario para entender la semana en <10 s.

```json
{
  "run_id": "b1f2...", "ucp": "Antioquia", "nombre_ucp": "Antioquia",
  "semana_inicio": "2026-06-29", "semana_fin": "2026-07-05",
  "estado": "borrador", "version": 2,
  "titulo": "Semana del 29 jun al 5 jul: demanda prevista 28.4 GWh, un 1.8 % más que la semana pasada",
  "confianza": {"nivel": "alta", "texto": "Los pronósticos de las últimas 8 semanas se equivocaron en promedio 2.4 %.",
                "wape_reciente_pct": 2.4, "semanas_evaluadas": 8},
  "kpis": {
    "energia_semana_mwh": 28412.3, "energia_semana_anterior_mwh": 27910.0, "variacion_vs_anterior_pct": 1.8,
    "pico_mw": 1312.4, "pico_fecha": "2026-07-02", "pico_hora": 19,
    "error_reciente_pct": 2.4, "error_reciente_tendencia": "mejorando",
    "alertas": {"total": 3, "warning": 1, "info": 2}, "recomendaciones": {"total": 2, "alta": 1}
  },
  "dias": [
    {"fecha": "2026-06-29", "dia": "lunes", "tipo_dia": "laboral", "energia_mwh": 4101.2, "pico_mw": 218.5, "pico_hora": 19,
     "ajuste_pct": 0.8, "estado": "ok", "n_recomendaciones": 0}
  ],
  "insights": [
    {"id": "festivo-2026-06-29", "nivel": "warning", "titulo": "Lunes festivo (San Pedro y San Pablo)",
     "texto": "El lunes es festivo: se espera menos demanda de lo normal. Revisa que el pronóstico baje.", "fechas": ["2026-06-29"]}
  ],
  "siguiente_paso": {"accion": "revisar_recomendaciones", "texto": "Revisa las 2 recomendaciones y aprueba la semana."}
}
```
Notas: `estado` del día ∈ `ok|revisar|critico` (según alertas/recomendaciones del día). `error_reciente_tendencia` ∈
`mejorando|estable|empeorando|sin_datos`. `confianza.nivel` ∈ `alta|media|baja|sin_datos` (alta: WAPE ≤ 3 %; media ≤ 6 %; baja > 6 %,
umbrales documentados en `insights.py`). `insights` ordenados por severidad; `texto` sin jerga (sin α, ratio, ancla).
`siguiente_paso.accion` ∈ `revisar_recomendaciones|revisar_alertas|aprobar|ya_aprobada`.

## 2. `GET /adjustments/insights/precision?ucp=&semanas=12&hasta=`
Precisión reciente pronóstico vs real. `semanas` 1..52 (default 12); `hasta` = lunes máximo (default: última semana con datos
completos). Fuente: pronósticos emitidos históricos (reentreno semanal del modelo, corte = sábado previo; ver `fuente`) ∪ corridas
aprobadas de la base si existen (estas prevalecen; `fuente` por semana = `corrida_aprobada`).

```json
{
  "ucp": "Antioquia", "desde": "2026-04-13", "hasta": "2026-06-22",
  "resumen": {"semanas": 12, "wape_pct": 2.4, "mape_energia_semana_pct": 1.4, "sesgo_energia_pct": -0.5,
              "mejor_semana": "2026-05-04", "peor_semana": "2026-06-08", "tendencia": "mejorando"},
  "semanas": [
    {"semana_inicio": "2026-06-15", "fuente": "modelo_historico", "wape_pct": 2.1, "energia_pronosticada_mwh": 27800.0,
     "energia_real_mwh": 28100.0, "error_energia_pct": -1.1, "pico_error_pct": 1.7, "dias_con_datos": 7,
     "dias": [
       {"fecha": "2026-06-15", "dia": "lunes", "tipo_dia": "laboral", "wape_pct": 1.9, "error_energia_pct": -0.8,
        "energia_pronosticada_mwh": 4000.0, "energia_real_mwh": 4032.0, "pico_pronosticado_mw": 215.0,
        "pico_real_mw": 218.0, "hora_pico_pronosticada": 19, "hora_pico_real": 19}
     ]}
  ],
  "por_tipo_dia": [{"tipo_dia": "laboral", "wape_pct": 2.2, "sesgo_energia_pct": -0.5, "n_dias": 60}],
  "por_hora": {"sesgo_pct": [/* 24 */], "wape_pct": [/* 24 */]}
}
```
`semanas` en orden cronológico; solo semanas con real completo se evalúan (las demás no aparecen). `error_energia_pct` = (pronóstico −
real)/real ×100 (negativo = se quedó corto).

## 3. `GET /adjustments/insights/recomendaciones?run_id=`
Recomendaciones por día con explicación en lenguaje de negocio y un ajuste sugerido aplicable con 1 clic.

```json
{
  "run_id": "b1f2...", "version": 2, "estado": "borrador",
  "recomendaciones": [
    {
      "id": "sesgo-2026-07-04", "fecha": "2026-07-04", "dia": "sábado", "tipo_dia": "sabado",
      "prioridad": "alta", "categoria": "nivel",
      "titulo": "El sábado suele quedar corto",
      "explicacion": "En los últimos 8 sábados el pronóstico se quedó 1.9 % por debajo de lo real. Sube el sábado en esa proporción.",
      "evidencia": {"n": 8, "sesgo_pct": -1.9, "fuente": "historial"},
      "impacto_esperado": "Reduce el error esperado de ~2.6 % a ~2.1 %" ,
      "aplicable": true,
      "ajuste": {
        "resumen": "Subir el sábado 1.9 % (+95 MWh)", "delta_energia_pct": 1.9, "delta_energia_mwh": 95.0,
        "curva_sugerida": [/* 24 MW, día completo ya ajustado */],
        "patch": {"paso": "forma", "body": {"estado": "editado", "curvas": {"2026-07-04": [/* 24 */]},
                  "motivo": "Recomendación del sistema: sesgo reciente del sábado", "usuario": null}}
      }
    }
  ],
  "ya_resueltas": ["..."]
}
```
`categoria` ∈ `nivel|forma|festivo|clima|alerta`; `prioridad` ∈ `alta|media|baja`. `ajuste` es `null` si la recomendación es solo
informativa (p. ej. una alerta que requiere criterio humano). **Aplicar con 1 clic** = `PATCH /adjustments/{run_id}/steps/{patch.paso}`
con `patch.body` + `version_esperada` (el front pone su `version` actual y su `usuario`). Las curvas editadas conservan el flujo
estándar de ediciones (`curvas_calculadas`). Solo se emiten recomendaciones respaldadas por evidencia medida (sesgo/forma históricos
con n ≥ 6 y efecto ≥ 0.5 %, o alertas reales del pipeline); nada inventado. `ya_resueltas` = ids de recomendaciones ya aplicadas
(día ya editado en el paso correspondiente).

## 4. `GET /adjustments/insights/banda?ucp=&run_id=`  (o `&fecha_inicio=&fecha_fin=` + `tipo_dia`)
Banda de incertidumbre **empírica por hora**: cuantiles de los errores relativos pasados (real/pronóstico − 1) por `tipo_dia` × hora
a partir del historial de pronósticos emitidos. Con `run_id`, la banda se aplica a las curvas finales de cada día de la semana.

```json
{
  "ucp": "Antioquia", "nivel_confianza": 0.8, "metodo": "cuantiles_empiricos_error_relativo",
  "n_semanas_historial": 52,
  "por_tipo_dia": {"laboral": {"n_dias": 243, "p10_pct": [/* 24 */], "p50_pct": [/* 24 */], "p90_pct": [/* 24 */]},
                   "sabado": {"...": "..."}, "domingo": {"...": "..."}, "festivo": {"...": "..."}},
  "dias": {"2026-06-29": {"tipo_dia": "festivo", "inferior": [/* 24 MW */], "superior": [/* 24 MW */]}}
}
```
`dias` solo con `run_id`. `p*_pct` en % (p10 = −4.1 → el real fue ≥ 4.1 % menor que el pronóstico en el 10 % de los casos).
Festivo con menos de 10 casos usa la banda de domingo y lo indica en `por_tipo_dia.festivo.nota`.

## 5. `GET /adjustments/insights/series?run_id=`
Series continuas de la semana (168 puntos = 7 días × 24 h, lunes 00:00 → domingo 23:00) para dibujar **curvas** (no barras).

```json
{
  "run_id": "b1f2...", "ucp": "Antioquia", "semana_inicio": "2026-06-29", "unidad": "MW", "puntos": 168,
  "fechas": ["2026-06-29", "...", "2026-07-05"],
  "etiquetas_eje": [{"i": 0, "texto": "lun 29"}, {"i": 24, "texto": "mar 30"}],
  "series": {
    "base": [/* 168 */], "ajustado": [/* 168 */],
    "referencia": [/* 168 o null si no hay semana de referencia */], "referencia_semana_inicio": "2025-06-30",
    "real": [/* 168 con null donde aún no hay dato (semana futura = todo null) */],
    "banda_inferior": [/* 168 */], "banda_superior": [/* 168 */]
  },
  "energia_diaria": {"fechas": ["..."], "base": [/*7*/], "ajustado": [/*7*/], "referencia": [/*7*/], "real": [/*7*/]},
  "dias": [{"fecha": "2026-06-29", "tipo_dia": "festivo", "editado": false, "etiqueta": "Festivo"}]
}
```
`banda_*` = misma banda que el endpoint 4 (omitidas = `null` si no hay historial suficiente). `real` sale del histórico cargado
(en demo, solo lectura). `dias[].editado` = el día tiene edición manual en algún paso.

## Compatibilidad
- No se modifican endpoints existentes. Modo demo: nada requiere red ni modelo; `GET` de lectura de histórico y archivos de
  `src/adjustments/precision_data/` (versionados, se copian con `src/` en el contexto de despliegue).
- Si el historial de pronósticos de una UCP no existe: `precision.semanas=[]`, `banda` con `n_dias=0` y arrays `null`,
  `confianza.nivel="sin_datos"`; nunca 500.

## Cambios
- v1 (2026-11-14): versión inicial.
- v1.1 (2026-11-14, B1; implementado en `src/adjustments/router_insights.py`, compatible hacia atrás):
    - `recomendaciones[].ajuste.patch.paso` es `forma` si ese paso no está omitido; si lo está, `especiales`; si ambos están omitidos
      `ajuste = null` y `aplicable = false` (editar un paso omitido lo activaría para los demás días). El front aplica `patch.body`
      tal cual (+ `version_esperada`, `usuario`). Tras el PATCH, `final_curves[fecha]` == `curva_sugerida`.
    - `recomendaciones[]` de alertas del pipeline (festivo, clima, eventos, avisos) vienen con `ajuste: null`, `aplicable: false`,
      `evidencia.fuente = "alerta_del_sistema"` y `impacto_esperado: null`. `categoria` de sesgo: `nivel` o `forma`.
    - `resumen.kpis.alertas` agrega la clave `critico` (bloqueos que impiden aprobar; cuentan también en `total`); `insights[].nivel`
      ∈ `critico|warning|info`. `confianza.texto` con sin_datos: "Todavía no hay pronósticos anteriores comparables...".
    - `resumen.kpis.energia_semana_anterior_mwh` usa real y, donde falta (brecha), el pronóstico de la corrida; si falta un día, `null`.
    - La confianza/error reciente usa las 8 semanas anteriores a la semana de la corrida (sin fuga).
    - `banda`: sin muestra suficiente (< 5 días) `p10_pct/p50_pct/p90_pct = null`; `fecha_inicio`/`fecha_fin`/`tipo_dia` son filtros
      opcionales de la muestra/salida (con `ucp` o `run_id`; `run_id` implica su `ucp`; 422 si falta ambos o no coinciden).
    - `precision`: `hasta` debe ser lunes (422); solo semanas con los 7 días de pronóstico y de real; `ucp` en la respuesta.
    - Evidencia de sesgo: n ≥ 6 (últimas 8 ocurrencias del día de semana no festivo, 20 festivos), |sesgo| ≥ 0.5 %, mismo signo en
      ≥ 65 % de los días, mejora del error medida sin el día evaluado, y que la última muestra no tenga > 20 semanas de antigüedad.
    - Festivos: en endpoints por `ucp` calendario oficial (sin red); por `run_id` los festivos guardados en la corrida.
