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

`{id}` es el UUID de la corrida (validar formato antes de reenviar). `{paso}` ∈
`base | referencia | nivel | forma | control` (validar antes de reenviar; si no, 422 local).

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
  "base_curvas": null, "base_ref": null, "fecha_emision": null, "modelo_version": null }
```
- `semana_inicio` lunes; `params` por nombre de paso (opcional). Con `base_curvas` no se refresca ni se
  llama al modelo. `fecha_emision`/`modelo_version` solo junto con `base_curvas` (si no, 422).
Response `data` (= `RunOut`):
```json
{ "id": "uuid | null (preview)", "persistido": true, "version": 1,
  "meta": { "ucp": "...", "semana_inicio": "...", "fecha_corte": "...", "fecha_emision": "...",
            "modelo_version": "...", "params_iniciales": {}, "estado": "borrador|aprobada|reemplazada|preview",
            "creado_por": "...", "extras": { "fuente_base": "modelo|externa", "fuente_festivos": "api|libreria",
            "ultimo_dato_real": "...", "base_ref": "..." } },
  "steps": [ { "orden": 0, "paso": "base", "estado": "aplicado|omitido|editado", "params": {},
               "curves": {"YYYY-MM-DD": [24]}, "curvas_calculadas": {}, "diagnostics": {},
               "alerts": [ { "nivel": "info|warning|bloqueo", "codigo": "...", "mensaje": "...",
                             "fechas": [], "detalle": {} } ] } ],
  "final_curves": {"YYYY-MM-DD": [24]},
  "bloqueos": [ <Alert> ],
  "eventos": [ { "usuario": "...", "accion": "crear|aplicar_paso|omitir_paso|editar_paso|aprobar|aprobacion_rechazada",
                 "paso": "...", "motivo": "...", "detalle": {}, "version": 2, "ts": "ISO-8601" } ] }
```
(El `meta` persistido trae además `id`, `estado`, `version`, `created_at`, `approved_at`, `approved_by`,
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

Params por paso (EPM, `src/adjustments/steps/*.py`; claves desconocidas → 422):
- `referencia`: `semana_ref_inicio` (YYYY-MM-DD o null), `max_anios_atras` (1..), `anios_excluidos` (lista años).
- `nivel`: `alpha` (0..1, default 0.25). Con análogos (Fase 2b): `ancla: "analogos"`, `analogos: {fecha: [fechas]}`.
- `forma`: `fuente` (`promedio_n` default | `reciente` | `referencia` | `analogos` en 2b), `n` (1..60, default 4),
  `beta` (0..1, default 1). Con análogos: `analogos: {fecha: [fechas]}`.
- `control`: `banda_pp` (>0, default 1.0), `top_dias` (1..7, default 3), `banda_referencia_pp` (default 2.0).

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

## Pendientes conocidos del lado EPM / backend
- EPM no tiene "listar corridas por (ucp, semana)": el front recuerda el id (URL `?corrida=` y
  localStorage). Recomendado: `GET /adjustments?ucp=&semana_inicio=` → borradores y aprobada.
- Modo automático por segmento (R2): el front ya lee `meta.extras.modos_paso` (`{paso: "sugerido"|"automatico"}`)
  si llega; hoy no llega y todos se muestran "sugerido".
