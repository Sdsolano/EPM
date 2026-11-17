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
      "prioridad": "baja", "categoria": "nivel", "respaldo": "tendencia_reciente", "validado_fuera_de_muestra": false,
      "titulo": "Tendencia reciente: el sábado quedó corto",
      "explicacion": "Tendencia reciente observada; las pruebas del sistema no confirmaron que corregirla mejore el pronóstico. Úsala como referencia, tú decides. Observado: en los últimos 8 sábados el pronóstico se quedó 1.9 % por debajo de lo real. Sugerencia: subir el sábado 1.9 % (+95 MWh) sobre lo que muestra hoy. En las últimas semanas (ventana con la que se eligió la regla) habría reducido el error en 8 de 8 días.",
      "evidencia": {"n": 8, "sesgo_pct": -1.9, "fuente": "historial",
                    "ventana_seleccion": {"dias_evaluados": 8, "dias_mejora": 8, "pct_dias_mejora": 100, "wape_antes_pct": 2.1, "wape_despues_pct": 1.2}},
      "impacto_esperado": "Basado en lo ocurrido en los últimos 8 sábados; no garantiza mejora.",
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
con n ≥ 6 y efecto ≥ 0.5 %, validadas fuera de muestra, o alertas reales del pipeline); nada inventado. **Las de sesgo son
sugerencias del analista sobre una tendencia reciente, no correcciones automáticas ni una predicción**: la prueba de B2/B4 midió que
las correcciones de sesgo no mejoran de forma consistente fuera de muestra (Antioquia WAPE 2.63→2.58 empeorando 44 % de los días;
Atlántico 5.22→5.54), por eso la UI no debe prometer mejora. `ya_resueltas` = ids de recomendaciones ya aplicadas
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
- v1.2 (2026-11-14, B3; compatible hacia atrás, solo agrega campos y ensancha la banda): la banda (endpoint 4 `banda` y
  `series.banda_inferior/superior`, ambos sobre las mismas tablas de `insights.tablas_banda`) usa la **banda calibrada**: los cuantiles
  p10 y p90 del error relativo se multiplican por `ESCALA_BANDA = 1.2` (misma constante que B2, `correccion.DEFAULTS_BANDA["escala"]`;
  p50 no se escala y p10/p90 nunca cruzan la mediana). Motivo: sin escala la banda de 80 % subcubría (cobertura 71-76 % en backtest sin
  fuga). `por_tipo_dia.*.p10_pct/p90_pct` ya vienen escalados. El JSON de banda agrega:
    - `escala_calibracion`: `1.2`.
    - `cobertura_medida`: `{"nominal": 0.8, "antioquia_prueba": 0.913, "atlantico_prueba": 0.819, "nota": "..."}`. Valores **medidos en
      backtest sin fuga de 21 semanas** (no una garantía): Antioquia calibración 79.1 % / prueba 91.3 %, Atlántico calibración 83.5 % /
      prueba 81.9 %; ancho medio ≈ 9-10 % (Antioquia) y 18-21 % (Atlántico). El front puede mostrarlo como "cubre ~8 de cada 10 días
      (medido en backtest)".
  `resumen` no usa la banda (la `confianza` sale del WAPE reciente), no cambia.
- v1.3 (2026-11-14, B4; corrección tras la ronda 1 de revisión; compatible hacia atrás salvo textos/prioridades):
    - **Recomendaciones de sesgo honestas (A1).** `impacto_esperado` ya no promete una cifra de mejora: texto "Basado en lo ocurrido en
      los últimos N <días>; no garantiza mejora." `prioridad` máxima `media` (`baja` si |sesgo| < 1 %). `titulo` empieza por
      "Tendencia reciente: ..." y la `explicacion` aclara que es una tendencia observada, no una predicción, e indica el delta real
      (% y MWh) y el resultado de la prueba fuera de muestra. `evidencia` agrega `validacion` `{dias_evaluados, dias_mejora, pct_dias_mejora}`.
      `categoria` sigue siendo `nivel|forma`. B2 midió que las correcciones de sesgo no mejoran de forma consistente: son sugerencias
      del analista, no automáticas.
    - **Validación rolling fuera de muestra en el gating.** Además de n ≥ 6, |sesgo| ≥ 0.5 %, consistencia ≥ 65 % y mejora leave-one-out,
      el segmento debe pasar `insights.validar_rolling`: para cada día del segmento en las últimas 8 semanas (52 para festivos) se estima
      el sesgo solo con días hasta el sábado previo a su semana, se aplica la misma regla (nivel o forma) y se compara el error; pasa si
      actuó en ≥ 4 días, redujo el error en ≥ 60 % y mejoró el WAPE agregado. Si no pasa **no se emite la recomendación** (ni cuenta en
      `ya_resueltas`). Con los datos reales (24 semanas): Antioquia 1 de 26 casos con evidencia simple pasa (4 %); Atlántico 8 de 50 (16 %,
      todos jueves/miércoles con 4-6 días de prueba): se suprime ~90 % de lo que antes se recomendaba.
    - **Nivel sobre lo que muestra `final` (M1).** `curva_sugerida = final × factor_nivel`; el delta (`delta_energia_pct/mwh`) y el
      `resumen` son el cambio real sobre `final`. Si el pipeline o el analista ya movieron el día (final vs base) en la dirección del sesgo
      al menos la mitad de lo necesario, no se sugiere; tampoco si el signo del delta contradice el sesgo.
    - **Cobertura de banda medida con esta implementación (M2).** `cobertura_medida` pasa a: Antioquia prueba 0.876, Atlántico 0.802
      (sin escala 0.813 / 0.732; ancho medio relativo 9.9 % / 20.1 %; mínimos por segmento: festivo Antioquia 71 % con 7 días, sábado Antioquia
      82 %). Nuevas claves `antioquia_sin_escala`, `atlantico_sin_escala`, `ancho_medio_relativo_pct`. `ESCALA_BANDA = 1.2` se define en
      `insights.py` sin importar `correccion`; `correccion.py` es opt-in y no está cableado al pipeline (su `tablas_banda` es el
      estimador experimental de B2: log-error, pool de horas, ventana móvil; distinto del de insights).
    - **Sin fuga con `run_id` (B-1).** `banda` (y las bandas de `series`) con `run_id` cortan la muestra en
      `min(semana_inicio − 1 día, fecha_corte de la corrida)` por defecto; un `fecha_fin` explícito se respeta. La respuesta de `banda`
      agrega `muestra: {fecha_inicio, fecha_fin}` (la muestra realmente usada). La evidencia de sesgo usa el mismo corte.
    - `banda.n_semanas_historial` respeta `fecha_inicio`/`fecha_fin`/`tipo_dia` (B-2). `version` de las respuestas nunca es `null`:
      si la corrida no la trae, 1 (B-5). Alerta `ELECCIONES_IA` con texto de negocio "Posibles elecciones o votación" (B-4).
    - **Historial estático (A3).** `src/adjustments/precision_data/*_k1.csv` es una foto: Antioquia termina el 2026-06-28, Atlántico el
      2026-05-03. La recomendación por sesgo caduca con `MAX_ANTIGUEDAD_DIAS` (140 días antes de la semana objetivo): para semanas
      posteriores no se emite y `precision`/`banda` no ven datos nuevos. Para refrescarlo: regenerar las curvas k1 y correr
      `scripts/generar_historial_pronosticos.py` (o, pendiente, persistir los pronósticos de las corridas aprobadas).
    - Rendimiento (M3): el histórico real se cachea por (ruta, mtime) con el cargador por defecto (lru de 8) y se normaliza en
      threadpool; `list_runs` de aprobadas se acota con `desde`/`limite` según el endpoint.
- v1.4 (2026-11-14, B5; corrección honestidad tras la ronda 2; compatible hacia atrás salvo textos/prioridad/`validacion`):
    - **Las recomendaciones de sesgo/nivel/forma ya no se presentan como respaldadas.** Un backtest rolling sin fuga (script
      `scripts/noche2_b5_backtest.py`) mostró que no mejoran fuera de muestra (ver B5/LOG.md). Cambios en cada recomendación:
        - `prioridad` = `"baja"` siempre en las de sesgo. Nuevo campo `respaldo` ∈ `"tendencia_reciente"` (sesgo/nivel/forma) |
          `"alerta_del_sistema"` (las que vienen de alertas del pipeline; solo estas son accionables en el resumen).
        - Nuevo campo `validado_fuera_de_muestra`: `false` en las de sesgo (honesto: nunca está validada fuera de muestra), `null` en las
          de alertas.
        - `explicacion` empieza por: "Tendencia reciente observada; las pruebas del sistema no confirmaron que corregirla mejore el
          pronóstico. Úsala como referencia, tú decides." Se eliminó la frase "En una prueba con semanas anteriores…"; la cifra pasa a
          "En las últimas semanas (ventana con la que se eligió la regla) habría reducido el error en X de Y días."
        - `evidencia.validacion` se renombra `evidencia.ventana_seleccion` `{dias_evaluados, dias_mejora, pct_dias_mejora,
          wape_antes_pct, wape_despues_pct}`: es una simulación sobre las MISMAS ~8 semanas que originan la regla (selección sobre una
          ventana corta), NO una validación independiente; el front no debe mostrarla como evidencia de mejora futura.
        - `ajuste` (curva_sugerida y patch) se sigue entregando para previsualizar/aplicar a mano. `aplicable` = hay `ajuste` y la
          corrida es borrador (la corrida no borrador, p. ej. aprobada o reemplazada, nunca es aplicable).
    - **Filtro de emisión más estricto** (constantes en `insights.py`): `ROLLING_MIN_DIAS = 8`, `ROLLING_MIN_MEJORA = 0.75` de los días
      y mejora del WAPE del segmento ≥ `ROLLING_MIN_MEJORA_WAPE_PP = 0.3` pp. Con datos reales se emite muy poco (B5/LOG.md).
    - **Resumen:** las recomendaciones `tendencia_reciente` NO cuentan como `alta`, NO marcan el día como `revisar` y NO empujan
      `siguiente_paso` a `revisar_recomendaciones` (solo las alertas del sistema accionables lo hacen). `kpis.recomendaciones` agrega
      `del_sistema` y `tendencia_reciente` (`total` = suma de ambas). `n_recomendaciones` por día cuenta todas.
    - Alertas `info` de clima (`clima-semana`) ya no son recomendaciones accionables (siguen en `insights`). Corrida con `estado !=
      borrador` (aprobada, reemplazada) → `siguiente_paso.accion = "ya_aprobada"`.
    - `banda` con `run_id`: `fecha_fin` explícito se recorta a `min(fecha_fin, min(semana − 1, fecha_corte))` (nunca hay fuga).
    - Textos de negocio: `FESTIVOS_FUENTE_ALTERNA` (info, visible), bloqueos por curva inválida y `RECONCILIACION_ROTA` incluyen los días;
      `SIN_FORMA` (warning) ya no se oculta. `COBERTURA_MEDIDA.nota` indica la fecha de la medición (2026-11-14).
- v1.5 (2026-11-14, B6; correcciones finales tras la ronda 3; compatible hacia atrás):
    - `resumen.confianza` / `kpis.error_reciente_*` solo cuentan semanas evaluadas dentro de las 12 anteriores a la corrida
      (`VENTANA_CONFIANZA_SEMANAS`); sin ninguna → `nivel = "sin_datos"`. Si la última semana evaluada tiene > 10 semanas de antigüedad
      el texto dice "de las N semanas hasta el <fecha>" en vez de "de las últimas N semanas".
    - Clima: ningún texto imprime `lluvia_mm` (es una suma de códigos de intensidad, no milímetros); se dice "lluvia por encima de lo
      habitual" / "dentro de lo habitual" comparando `lluvia_max_diaria_mm` con `p90_lluvia_mm`.
    - `banda.cobertura_medida.detalle_tecnico` es el nombre nuevo de la nota técnica; `nota` se mantiene con el mismo contenido (compat).
    - Días con real/pronóstico de energía <= 0 o no finitos se descartan de banda, sesgo y recomendaciones (sin 500).
    - Textos: `patch.body.motivo` = "...las pruebas del sistema no confirmaron que mejore"; la frase de ventana dice "en las últimas
      semanas" (festivos: "en el último año"), "que es el período con el que se eligió la regla"; "problema/problemas" en singular/plural;
      sin `desvio_pp` el texto de `TENDENCIA_FUERA_BANDA` no deja doble espacio.
    - Interno: `insights.PERIODOS/TIPOS_DIA` y `correccion.TIPOS` eliminados (usar `data.*`); `Curvas` → `CurvasNp`; `series()` sin parámetro
      `tablas`; scripts `noche2_b2_*` sin rutas absolutas (`EPM_RAW_DIR`, `EPM_K1_DIR`, `--raw-dir`, `--k1-dir`).
