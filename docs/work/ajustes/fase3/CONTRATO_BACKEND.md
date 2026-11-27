# Contrato del backend intermedio — Mesa de ajustes (`admin/dashboard/ajustes/*`)

Sphere (front) NO llama a la API EPM directo. El backend intermedio de Sphere (no está en estos repos) expone
las rutas de abajo y las traduce a la API EPM `/adjustments/*` (código: `EPM/src/adjustments/router.py`;
análogos: contrato fijo en `docs/work/ajustes/PLAN_NOCHE.md`). Copia de este documento en Sphere:
`src/providers/ajustes/CONTRATO_BACKEND.md`. Tipos TypeScript exactos: Sphere `src/types/ajustes.d.ts`.

## Reglas generales (obligatorias para el backend)
1. **Base**: `${VITE_API_URL}${VITE_API_VERSION}admin/dashboard/ajustes/...` (igual que el resto de Sphere).
   Autenticación de usuario: `Authorization: Bearer <token>` de Sphere (lo agrega `AxiosInstance`).
2. **Token de servicio**: en cada llamada a EPM el backend añade la cabecera `X-Ajustes-Token` con el valor
   de su variable de entorno `AJUSTES_SERVICE_TOKEN` (EPM responde 401 si falta o no coincide cuando EPM la
   tiene definida). El token nunca llega al navegador.
3. **Usuario**: el front NO manda `usuario`. En `run`, `PATCH` de pasos y `aprobar`, el backend pone
   `usuario` = identificador del usuario de la sesión (p. ej. `user.usuario`), y lo **sobrescribe** si llega
   en el cuerpo.
4. **Envoltorio de respuesta** (convención Sphere):
   - Éxito: HTTP 200 (201 en `run`) con `{ "success": true, "data": <cuerpo EPM sin cambios> }`.
   - Error: el MISMO status HTTP que devolvió EPM (401, 404, 409, 422, 502, 503) con
     `{ "success": false, "message": "<texto legible>", "detail": <detail de EPM tal cual> }`.
     `message`: si `detail` es string, ese string; si es objeto con `mensaje`, ese `mensaje`.
     Si EPM no responde: 502 con `message: "La API de ajustes no respondió"`.
   - El front distingue 409 (versión vieja / corrida no editable → recarga) y 422 (bloqueos o datos
     inválidos → muestra `detail`).
5. **Timeouts**: `preview`, `run` y `PATCH /pasos/base` pueden tardar (refresco de datos + modelo): timeout
   >= 180 s. El resto, 30 s.
6. **Sin transformación** de curvas: `{ "YYYY-MM-DD": [P1..P24] }`, `null` = valor no finito.

## Rutas

| Método | Ruta Sphere (`admin/dashboard/ajustes/…`) | EPM | Notas |
|---|---|---|---|
| GET | `ucps` | — (config del backend) | UCP habilitadas en la mesa |
| GET | `pronostico-emitido?ucp=&semana_inicio=` | — (datos Sphere) | 404 si no hay emitido |
| GET | `reales?ucp=&desde=&hasta=` | — (histórico Sphere) | curvas reales observadas |
| POST | `preview` | `POST /adjustments/preview` | no persiste |
| POST | `run` | `POST /adjustments/run` | 201; pone `usuario` |
| GET | `aprobada?ucp=&semana_inicio=` | `GET /adjustments/aprobada` | 404 si no hay |
| GET | `corridas/{id}` | `GET /adjustments/{id}` | |
| GET | `corridas/{id}/versiones` | `GET /adjustments/{id}/versiones` | |
| GET | `corridas/{id}/versiones/{version}` | `GET /adjustments/{id}/versiones/{version}` | |
| PATCH | `corridas/{id}/pasos/{paso}` | `PATCH /adjustments/{id}/steps/{paso}` | pone `usuario` |
| POST | `corridas/{id}/aprobar` | `POST /adjustments/{id}/approve` | pone `usuario` |
| POST | `analogos` | `POST /adjustments/analogos` | contrato PLAN_NOCHE |
| GET | `acumulado-anual?ucp=&anio=&banda_pct=&cupo=&hasta=` | `GET /adjustments/acumulado-anual` | solo lectura; 503 sin persistencia |

`{id}` es el UUID de la corrida (validar formato antes de reenviar). `{paso}` ∈
`base | referencia | nivel | forma | especiales | control` (6 pasos; validar antes de reenviar; si no, 422 local).

### GET `ucps`
Response `data`: `[{ "ucp": "Antioquia", "nombre": "Antioquia" }, { "ucp": "Atlantico", "nombre": "Atlántico" }]`.
`ucp` es el nombre EPM (`data/raw/{ucp}`; patrón `^[A-Za-z0-9_-]{1,64}$`); `nombre` es para mostrar.
El backend mapea sus nombres de mercado a los de EPM.

### GET `pronostico-emitido?ucp=Antioquia&semana_inicio=2026-11-16`
Pronóstico ya emitido en Sphere para usarlo como base (`base_curvas`).
Response `data`:
```json
{ "base_ref": "sphere:pronostico:12345", "fecha_emision": "2026-11-13", "modelo_version": "v2026.10",
  "base_curvas": { "2026-11-13": [24 números], "...": [], "2026-11-22": [24 números] } }
```
`base_curvas` debe cubrir desde el día siguiente al último dato real anterior a `fecha_emision` hasta el
domingo de la semana (EPM: `fecha_corte+1..domingo`, 422 si faltan días). 404 si no hay emitido.
El front reenvía `base_curvas`, `base_ref`, `fecha_emision` y `modelo_version` en `preview`/`run`.

### GET `reales?ucp=Antioquia&desde=2026-11-09&hasta=2026-11-22`
Curvas reales observadas (MW por periodo). Response `data`: `{ "2026-11-09": [24], ... }` solo con los días
que existan (puede ser `{}`). Opcional para el front: si falla, la gráfica no muestra la serie real.

### POST `preview` y POST `run`
Request (= `RunRequest` de EPM sin `usuario`):
```json
{ "ucp": "Antioquia", "semana_inicio": "2026-11-16",
  "params": { "nivel": { "alpha": 0.25 } },
  "perfil": null,
  "base_curvas": null, "base_ref": null, "fecha_emision": null, "modelo_version": null }
```
- `perfil` (opcional): `"jamer"` | `"jamer_modelo"`; cualquier otro valor → 422. Ver "Perfiles" abajo.
- `semana_inicio` lunes; `params` por nombre de paso (opcional). Con `base_curvas` no se refresca ni se
  llama al modelo. `fecha_emision`/`modelo_version` solo junto con `base_curvas` (si no, 422).
Response `data` (= `RunOut`):
```json
{ "id": "uuid | null (preview)", "persistido": true, "version": 1,
  "meta": { "ucp": "...", "semana_inicio": "...", "fecha_corte": "...", "fecha_emision": "...",
            "modelo_version": "...", "params_iniciales": {}, "estado": "borrador|aprobada|reemplazada|preview",
            "creado_por": "...", "extras": { "fuente_base": "modelo|externa", "fuente_festivos": "api|libreria",
            "ultimo_dato_real": "...", "base_ref": "...", "perfil": "jamer" } },
  "steps": [ { "orden": 0, "paso": "base", "estado": "aplicado|omitido|editado", "params": {},
               "curves": {"YYYY-MM-DD": [24]}, "curvas_calculadas": {}, "diagnostics": {},
               "alerts": [ { "nivel": "info|warning|bloqueo", "codigo": "...", "mensaje": "...",
                             "fechas": [], "detalle": {} } ] } ],
  "final_curves": {"YYYY-MM-DD": [24]},
  "bloqueos": [ <Alert> ],
  "eventos": [ { "usuario": "...", "accion": "crear|aplicar_paso|omitir_paso|editar_paso|aprobar|aprobacion_rechazada",
                 "paso": "...", "motivo": "...", "detalle": {}, "version": 2, "ts": "ISO-8601" } ] }
```
(`meta.extras.perfil` solo existe si la corrida se creó con `perfil`. `meta.params_iniciales` guarda los params ya
expandidos con el perfil. El `meta` persistido trae además `id`, `estado`, `version`, `created_at`, `approved_at`, `approved_by`,
`reemplazada_por`, `reemplazada_at`. En preview `id` es null, `version` null y `meta.estado = "preview"`.)
Errores EPM: 404 UCP desconocida o sin histórico; 409 sin modelo entrenado; 422 entrada inválida, semana ya
observada, `base_curvas` incompleta; 502 refresco/modelo falló; 503 sin persistencia (`run`).

### GET `aprobada?ucp=&semana_inicio=`
`data` = `RunOut` de la aprobada vigente. 404 si no hay.

### GET `corridas/{id}`
`data` = `RunOut` (con `eventos`). 404 si no existe.

### GET `corridas/{id}/versiones`
`data` = `[ { "version": 1, "creado_por": "...", "motivo": "...", "created_at": "..." } ]`.

### GET `corridas/{id}/versiones/{version}`
`data` = `{ "version", "creado_por", "motivo", "created_at", "steps": [StepOut], "final_curves": {} }`.

### PATCH `corridas/{id}/pasos/{paso}`
Request (= `StepPatchRequest` sin `usuario`):
```json
{ "estado": "aplicado|omitido|editado", "version_esperada": 3,
  "params": { "alpha": 0.4 }, "curvas": { "2026-11-18": [24] }, "motivo": "texto opcional" }
```
- `params` reemplaza los params del paso (omitido = conserva los vigentes). `curvas` solo con
  `estado=editado` (obligatorias en ese caso); días editados antes se conservan.
- `base` no se puede omitir; `base` + `aplicado` re-ejecuta el modelo (lento) salvo base externa (422).
Response `data` = `RunOut` con la nueva `version`.
Errores: 409 `{"mensaje","version_esperada","version_actual"}` si la versión cambió o la corrida no es
borrador (el front recarga `corridas/{id}`); 422 params/curvas inválidos (texto de EPM en `detail`).

Params por paso (EPM, `src/adjustments/steps/*.py`; claves desconocidas → 422). Las claves "Jamer" solo se
persisten en `steps[i].params` si difieren de su default (el front no debe asumir que vienen siempre):
- `referencia`: `semana_ref_inicio` (YYYY-MM-DD o null), `max_anios_atras` (1..), `anios_excluidos` (lista años).
- `nivel`: `alpha` (0..1, default 0.25). Análogos: `ancla: "analogos"`, `analogos: {fecha: [fechas]}`.
  Modo réplica: `modo` (`alpha` default | `reciente`), `peso_modelo` (0..1, 0), `saltar_puente` (bool, false),
  `g_fuente` (`interanual` default | `interanual_corto` | `interanual_estacional` | `fijo`), `g_fijo` (0.005),
  `g_ventana_dias` (28), `g_k` (1.0), `g_min` (-0.01), `g_max` (0.02), `g_min_pares` (14).
- `forma`: `fuente` (`promedio_n` default | `reciente` | `referencia` | `analogos` | `reciente_clase`), `n` (1..60, 4),
  `beta` (0..1, 1). Análogos: `analogos: {fecha: [fechas]}`. Réplica: `delta` (`proporcional` default | `plano`),
  `tramos` (`off` default | `modelo_anual`; exige `delta="plano"`, experimental, no recomendado), `umbral_tramo_mw` (20),
  `min_periodos_tramo` (3), `suavizado_borde` (0), `saltar_puente` (false).
- Selección de R1 (nivel `modo="reciente"` y forma `fuente="reciente_clase"`). **nivel y forma deben recibir los
  mismos valores** de estos tres flags (el perfil `jamer` expande `festivo_por_tipo=true` en ambos pasos —activado tras el backtest de festivos_r1—; `r1_limpia` y `r1_umbral_atipico` no se expanden: si el cliente los envía, en nivel y en forma; si difieren, la energía y la forma de un
  festivo salen de referencias distintas).
  - `festivo_por_tipo` (bool, false). Tipo de festivo (`data.tipo_festivo`): `lunes` (cae lunes, incluido uno
    trasladado a lunes dentro de Semana Santa, p. ej. San José 2024-03-25), `semana_santa` (solo Jueves y Viernes
    Santo) o `fijo` (el resto). Para `fijo`/`semana_santa`: **R1 = día NORMAL** reciente del mismo día de semana (la
    búsqueda de un día normal, con la limpieza si `r1_limpia`) y nivel usa E_obj = ΣR1·(1+g)·F con
    F = E(H)/E(N_H). H = homólogo (`find_homologous_holiday_date`) buscado **hasta 3 años atrás**, el más reciente
    <= corte con curva válida y E > 0, del **mismo tipo de día** (laboral/sábado/domingo) y, si el objetivo no es
    lunes, **que no sea lunes** (festivo de puente). N_H = día normal (fuera de Semana Santa y 24/dic–6/ene) del mismo
    día de semana que H a −7, −14, +7 o +14 días (en ese orden). forma usa la forma normalizada de H (fuente
    `festivo_homologo_tipo`). Festivos `lunes`: selección de siempre.
  - `r1_limpia` (bool, false): un día no festivo salta candidatos a R1 atípicos (`atipico`) y, **solo si el día
    objetivo está fuera de ese periodo**, los de Semana Santa (`semana_santa`, Domingo de Ramos–Pascua) o
    24/dic–6/ene (`temporada`). Estos motivos se suman a los de siempre en `saltados[].motivo` (`festivo`,
    `post_festivo`, `puente`, `sin datos`, `homologo`).
  - `r1_umbral_atipico` (número en (0, 0.5], 0.06): umbral de atípico de `r1_limpia`; **fijo, no calibrar** (elegido
    por magnitud física antes del backtest).
  - Diagnóstico por día con `festivo_por_tipo` (nivel y forma): `tipo_festivo`, `homologo`, `normal_homologo`,
    `factor_festivo` (null si no se aplicó), `homologo_anios_probados:[{anio, homologo, descarte}]` (`descarte`:
    `sin_festivo` | `sin_datos` | `otro_tipo_dia` | `lunes_puente` | null = el usado).
- `control`: `banda_pp` (>0, 1.0), `top_dias` (1..7, 3), `banda_referencia_pp` (2.0). Alertas Jamer (todas informativas,
  apagadas con `null`): `banda_salto_mw` (null), `periodos_valle` ([3,4,5]), `banda_crecimiento_pp` (null),
  `crecimiento_ventana_dias` (28), `crecimiento_min_pares` (14), `cupo_dias_anuales` (null), `banda_error_pct` (5.0).

## Perfiles (`RunRequest.perfil`)
Atajo que se expande en `src/adjustments/perfiles.py` y se mezcla con `params` (los params explícitos del usuario
ganan clave a clave). El resultado expandido queda en `meta.params_iniciales` y el nombre en `meta.extras.perfil`
(la clave solo aparece con perfil). Sin `perfil` el comportamiento es idéntico al anterior.

| Perfil | nivel | forma | control |
|---|---|---|---|
| `jamer` (réplica pura) | `{modo:"reciente", g_fuente:"interanual", peso_modelo:0}` | `{fuente:"reciente_clase", n:1, delta:"plano", tramos:"off"}` | `{banda_salto_mw:40}` |
| `jamer_modelo` | igual con `peso_modelo:0.3` (provisional) | igual | igual |

`banda_salto_mw=40` y `peso_modelo=0.3` son provisionales (los fija la calibración). Crecimiento y acumulado quedan apagados.

## Diagnósticos nuevos (aditivos; el front viejo los ignora)
`nivel.diagnostics` (con `modo="reciente"`):
- `modo`, `alpha` (= 1 − peso_modelo), `peso_modelo`, `semana_ref_inicio`, `desfase_dias`, `anios_excluidos`.
- `dias[YYYY-MM-DD]`: `modo`, `E_base` (Σ curva del modelo), `E_obj` (= `E_objetivo`, ΣR1·(1+g); con el factor festivo
  aplicado ΣR1·(1+g)·F), `E` (= `E_final`, energía del día
  resultante), `clase` (`normal|festivo|post_festivo|puente`), `saltados:[{fecha,motivo}]`, `peso_modelo`, `r1:{fecha,energia}|null`,
  `r1_extra:[{fecha,energia}]` (2.º histórico), `lag_dias`, `g_raw`, `g_usado`, `acotado`, `g_fuente`, `regla`
  (`reciente|reciente_factor_festivo|sin_evidencia`; `reciente_factor_festivo` = festivo por tipo con F aplicado), y por compatibilidad `ratio`, `alpha`, `ancla` (fecha R1), `ancla_ref` (R3), `dia_ref` (R2).
- `crecimiento`: `{Y, ventana:[desde,hasta], n_pares, fallback, fuente_g}`.
- `referencias[YYYY-MM-DD]`: `{r1:{fecha,curva[24]}|null, r1_extra:[{fecha,curva[24]}], r2:{fecha,curva[24]}|null, r3:{fecha,curva[24]}|null}`
  (R1 = último día real de la misma clase; R2 = día − 364 d o festivo homólogo; R3 = R1 − 364 d).

`forma.diagnostics`: `fuente`, `n`, `beta`, `dias`, y con réplica `delta`, `tramos` (modo). `dias[f]` añade: `energia_base`,
`delta_plano_mw`, `periodos_con_gancho`, `tramos_retenidos:[{desde,hasta,desv_mw,desv_mw_10}]`, `clase`, `saltados`, `tramos_motivo`.

`control.diagnostics`:
- `saltos`: `{banda_mw, periodos_valle, pares:[{de,a,delta_final,delta_ref,dif_mw}], motivo}` (siempre presente; `pares` vacío sin banda o sin referencia).
- `crecimiento`: indicador `(P/R−1)` por día y semana vs `Y` reciente (se publica siempre, con o sin banda; ver `fase_e4a`).
- `acumulado_anual`: `null` si `cupo_dias_anuales` es null; `{estado:"sin_datos", motivo}` si hay cupo pero ninguna aprobada del año
  (o sin store); si no, el cuerpo de `GET acumulado-anual` sin `alerta` (la alerta va en `alerts`). Las aprobadas del año las lee EPM del
  store (curvas finales de las corridas aprobadas vigentes, solo los 7 días de su semana) únicamente cuando hay cupo.

## Alertas nuevas (nivel `info` salvo indicación; no bloquean)
`SALTO_ENTRE_DIAS` (control, con `banda_salto_mw`), `CRECIMIENTO_FUERA_DE_TENDENCIA` (control, con `banda_crecimiento_pp`),
`ACUMULADO_ANUAL_FUERA_RANGO` (control, con `cupo_dias_anuales`), `G_ACOTADO`, `G_SIN_EVIDENCIA`, `R1_SIN_MISMA_CLASE` (nivel; forma solo si nivel no la reportó para ese día),
`FESTIVO_SIN_HOMOLOGO` (con `festivo_por_tipo`; se usa la referencia de siempre. `detalle.motivo` es un código:
`sin_homologo` (ningún año con homólogo con datos), `sin_homologo_mismo_tipo_dia` (los homólogos con datos no son del
mismo tipo de día), `homologo_lunes_puente` (los homólogos con datos del mismo tipo de día son lunes de puente y el
objetivo no es lunes), `sin_normal_homologo` (sin N_H), `sin_normal_reciente` (sin R1 normal); el texto legible va
solo en `mensaje`. `detalle` trae también `tipo_festivo`, `homologo`, `normal_homologo`, `anios_probados`), `R1_SIN_LIMPIA` (con `r1_limpia`: ningún candidato limpio, se usa la R1 de siempre;
`detalle.descartados`) — ambas como `R1_SIN_MISMA_CLASE`: las emite nivel y forma solo los días que nivel no reportó,
`FORMA_SIN_ENERGIA_BASE`, `FORMA_SIN_GANCHO` (forma). En nivel `modo=reciente` también `SIN_EVIDENCIA_NIVEL` y `CURVA_BASE_NULA` (warning).
Las alertas por día se unen por código: `fechas` = días afectados y `detalle.por_fecha[fecha]` trae el detalle y el `mensaje` de cada día (si los mensajes difieren, el `mensaje` unido es genérico).

### GET `acumulado-anual?ucp=&anio=&banda_pct=&cupo=&hasta=`
Solo lectura; EPM `GET /adjustments/acumulado-anual`. Store con esquema sin migrar o base inaccesible: 503 (no 500). `ucp` y `anio` obligatorios; `banda_pct` (≥0) y `cupo` (0..366) opcionales y
**apagados por defecto**; `hasta` (YYYY-MM-DD) tope inclusive. Compara la energía diaria de las aprobadas vigentes del año contra el real.
Errores: 503 sin persistencia, 404 UCP desconocida o sin histórico, 422 parámetros inválidos.
Response `data`:
```json
{ "ucp": "Antioquia", "estado": "apagado|sin_cupo|dentro|cerca|excedido", "anio": 2026, "banda_pct": 5.0,
  "cupo_dias": 25, "dias_fuera": 3, "dias_evaluados": 300, "dias_no_evaluables": 0,
  "fechas_fuera": [ { "fecha": "2026-03-04", "error_pct": 5.8 } ], "restantes": 22,
  "alerta": null, "dias_aprobados": 301 }
```
`alerta` es null o un `Alert` (`ACUMULADO_ANUAL_FUERA_RANGO`). Sin `banda_pct`: `estado="apagado"` y conteos en 0.
Con banda y sin `cupo`: `sin_cupo` (cuenta, no alerta). "25 días" NO es un valor oficial (pendiente de confirmar con Jamer).

### POST/GET `aprendizaje/error` (P1 "Lo que aprendió"; informativo, NO modifica curvas)
EPM (solo lectura, sin store, hereda `X-Service-Token`; registrada antes de `/{run_id}`):
- `POST /adjustments/aprendizaje/error` body `{ucp, semanas?=12 (4..26), hasta?, min_casos?=8, festivos?:[ISO], emisiones:[{semana_inicio(lunes), version?, curvas:{ISO:[24]}}], reales:{ISO:[24]}}`.
  422 con detalle claro si la entrada es inválida (fecha, curva != 24, NaN/negativos, semana_inicio no lunes, semanas fuera de 4..26).
- `GET /adjustments/aprendizaje/error?ucp=&semanas=` (demo): lee `retro/{ucp}.json` del snapshot (misma forma sin ucp/semanas). UCP válida sin archivo: 200 `sin_datos` (no 404); 422 ucp/semanas inválidas o archivo corrupto.
Respuesta 200: `{"estado":"ok",...}` o `{"estado":"sin_datos","ucp","motivo","avisos":[{"codigo":"RETRO_SIN_HISTORIAL",...}]}`. Forma completa en `replica_jamer/fase_retro_p1/CONTRATO.md`.
Campos: `sesgo_global = {pct, wape_pct, n}` (el campo es `pct`, NO `sesgo_pct`; `ventanas[]`, `por_tipo[]` y `por_bloque[]` sí usan `sesgo_pct`). `ventanas` con `semanas` > las pedidas vienen con `sesgo_pct/wape_pct` null y `n` 0. `hasta` muy posterior a los datos (> 7*semanas días) devuelve `sin_datos`. `hoy` (aviso de rezago) = fecha de America/Bogota. `ema.h112` usa todo el historial recibido: si el proxy recorta a `semanas` (12), queda limitada a esa ventana.
Signo: `(reportado - real)/real*100`. El backend Sphere lo expone como `GET ajustes/aprendizaje-error?ucp=&semanas=` (ver ese contrato).

### POST `corridas/{id}/aprobar`
Request: `{ "version_esperada": 3, "motivo": "texto", "excepciones_justificadas": ["TENDENCIA_FUERA_BANDA"] }`
(el backend añade `usuario`, obligatorio en EPM). Response `data` = `RunOut` aprobada.
Errores: 422 `{"mensaje","bloqueos":[Alert]}` si hay bloqueos; 422
`{"mensaje","codigos_invalidos","warnings_vigentes"}` si una justificación no es un warning vigente; 409
versión cambió o ya no es borrador.

### POST `analogos`
Request y response exactamente como el "Contrato de análogos" de `PLAN_NOCHE.md`
(`ucp`, `fecha`, `fecha_corte?`, `filtros{dia_semana, meses, temporada_dias, relacion_festivo,
distancia_festivo, tipo_festivo, semana_santa, temporada_escolar, clima, anio_min, anio_max,
anios_excluidos, limite}` → `descriptor_objetivo`, `filtros_aplicados`, `candidatos[]`, `resumen`).

### Clima / lluvia (cambio 2026-11-24, fase_clima C1)
`p_i` es un CODIGO de condicion del tiempo (no mm). Regla en `src/adjustments/clima.py`: periodo con lluvia si
`200<=p_i<600` (legado AccuWeather `1<=p_i<200`: lluvia si p_i en {12-18, 39-42}); `p_i` 0/NaN = dato faltante; dia
"con lluvia" si >=3 periodos con lluvia y >=12 periodos conocidos.
- Alerta `CLIMA_OBSERVADO` (`detalle`): SE ELIMINAN `lluvia_mm`, `lluvia_max_diaria_mm`, `p90_lluvia_mm`. NUEVAS:
  `lluvia_periodos` (suma semanal de periodos con lluvia, dias fiables), `lluvia_frac_semana`,
  `lluvia_frac_max_diaria` (0..1), `p90_lluvia_frac` (p90 historico de la fraccion diaria, solo dias con calidad
  suficiente). Mensaje: "... N periodos con lluvia (x %)". `detalle.dias[f]` trae por dia `temp_media`, `humedad_media`,
  `lluvia_periodos`, `periodos_conocidos`, `frac_lluvia`, `con_lluvia` (bool|null), `calidad_lluvia`
  (`suficiente|insuficiente|sin_dato`).
- `POST analogos`: `candidatos[].clima` pasa de `{temp_media, lluvia_mm}` a `{temp_media, frac_lluvia, con_lluvia}`
  (`frac_lluvia` 0..1|null, `con_lluvia` bool|null; null = lluvia desconocida). El filtro `filtros.clima.lluvia`
  (bool) conserva su semantica: true = dias con `con_lluvia` true, false = `con_lluvia` false; dias desconocidos se
  excluyen. La razon "con lluvia" solo aparece si objetivo y candidato tienen `con_lluvia` true.

## Pendientes conocidos del lado EPM / backend
- EPM no tiene "listar corridas por (ucp, semana)": el front recuerda el id (URL `?corrida=` y
  localStorage). Recomendado: `GET /adjustments?ucp=&semana_inicio=` → borradores y aprobada.
- Modo automático por segmento (R2): el front ya lee `meta.extras.modos_paso` (`{paso: "sugerido"|"automatico"}`)
  si llega; hoy no llega y todos se muestran "sugerido".
