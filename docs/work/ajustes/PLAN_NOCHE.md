# Plan maestro — sesión autónoma 2026-11-12/13 (backlog de la mesa de ajustes)

Documento fuente de verdad del orquestador para esta sesión. Cada fase tiene su carpeta
`docs/work/ajustes/<fase>/` con `PLAN.md` (sub-plan del agente) y `LOG.md` (progreso real).
El progreso global se registra en `docs/work/ajustes/LOG.md` (sección "Sesión autónoma 2026-11-12/13").

## Objetivo
Reducir de verdad el error de las predicciones semanales y quitarle trabajo al analista.
Métricas: WAPE horario, error de energía diaria y semanal (MAPE y sesgo), error de pico (MW y hora),
por mercado (UCP) y tipo de día (laboral/sábado/domingo/festivo). **Ningún cambio de defaults o de
algoritmo sin evidencia de backtest con holdout temporal.**

## Restricciones (obligatorias para todos los agentes)
- Python: `C:/Users/samue/OneDrive/Documentos/GitHub/EPM/.venv/Scripts/python.exe` (Windows; los docs
  viejos dicen `.venv/bin/python`). Se puede usar desde un worktree (no hay instalación editable).
- Suite: `python -m pytest tests -q --ignore=tests/test_hourly_disaggregation.py`. Fallos permitidos
  (preexistentes): `test_future_events::test_get_future_events`,
  `test_holiday_homology::...christmas_profile_uses_prior_christmases`, 2 de `test_openai_analysis`.
- No modificar ni commitear `data/raw`, `models/`, `data/features`. Backtests aislados (copias en
  temporales o en el scratchpad); sin red. Nada de CSV grandes, `.env` ni secretos en git.
- Repo EPM público: no subir `docs/work/ajustes/fuentes/` ni datos sensibles.
- No tocar cambios ajenos sin commitear: `fastapi_factores/test_*`, `fastapi_factores/nul`,
  `reports/performance/README.md`, `.claude/settings.local.json`, `env.migracion-pc-epm-usb/`.
- `git add` por ruta explícita; docs y reportes `.md` con `git add -f`.
- Sin force-push, amend, borrar ramas remotas, despliegues ni bases de producción. Postgres solo local
  con bases temporales que se borran.

## Fases y orden
| Fase | Carpeta | Rama (EPM desde main / Sphere desde test) | Depende de |
|---|---|---|---|
| ASR: comparación ASR vs mesa vs real | `asr/` | `feat/ajustes-asr-comparacion` | — |
| 2a: backtest con modelo real + recalibración | `fase2a/` | `feat/ajustes-fase2a` | — |
| 2b: buscador de análogos (R3) | `fase2b/` | `feat/ajustes-fase2b` | contrato abajo; recalibración final usa base modelo de 2a |
| 3: UI Mesa de ajustes | `fase3/` (en EPM) | Sphere `feat/mesa-ajustes` | contrato API (existente + análogos) |
| 4: días especiales, alertas, aprendizaje, auto por segmento | `fase4/` | `feat/ajustes-fase4` | 2a (base modelo) |

Ola 1 en paralelo (worktrees separados): ASR, 2a, 2b (módulo y endpoint), 3 (front).
Cada fase: implementación con subagentes → ronda 1 de revisión (8 ángulos separados + regresión, todos con
`model: "sonnet"`) → si un revisor reporta crítico/alto: corregir y ronda completa de nuevo; si no:
corregir medios/bajos y una verificación estándar → push de la rama y fast-forward a main (`git push origin
<rama>:main`; rebase previo si main avanzó, suite de nuevo).

Worktrees (fuera del repo, carpeta hermana): `../EPM-wt-asr`, `../EPM-wt-2a`, `../EPM-wt-2b`,
`../Sphere-wt-mesa`. A cada worktree de EPM se le COPIAN (no enlazan) `data/raw/Antioquia`,
`data/raw/Atlantico`, `models/` y `data/features` para que nada pueda escribir en los originales.

## Contrato de análogos (fijo; lo usan 2b y 3)
`POST /adjustments/analogos` (mismo token `X-Ajustes-Token` que el resto de `/adjustments`)
```json
{
  "ucp": "Antioquia",
  "fecha": "2026-05-19",                 // día objetivo (puede ser futuro)
  "fecha_corte": "2026-05-16",           // opcional; solo se buscan días <= corte (default: último dato real)
  "filtros": {                           // todos opcionales; null/ausente = no filtra
    "dia_semana": "auto",                // "auto" (= el del objetivo) | 0..6 (0=lunes) | lista | null
    "meses": [5],                        // lista 1..12 | null; "temporada_dias": ±N días del año del objetivo
    "temporada_dias": null,
    "relacion_festivo": "auto",          // "auto" | "antes" | "despues" | "puente" | "festivo" | "ninguna" | null
    "distancia_festivo": 1,              // n días al festivo más cercano (con relacion antes/despues)
    "tipo_festivo": null,                // nombre o clave del festivo (p.ej. "semana_santa") | null
    "semana_santa": null,                // true/false/null
    "temporada_escolar": null,           // "clases" | "vacaciones" | null (calendario aproximado documentado)
    "clima": null,                       // {"temp_tolerancia": 1.5, "lluvia": true} | null — usa clima observado
    "anio_min": null, "anio_max": null,
    "anios_excluidos": [2020],
    "limite": 20
  }
}
```
Respuesta 200:
```json
{
  "ucp": "...", "fecha": "...", "fecha_corte": "...",
  "descriptor_objetivo": {"dia_semana": 1, "mes": 5, "festivo": false, "relacion_festivo": "despues",
                           "distancia_festivo": 1, "semana_santa": false, "temporada_escolar": "clases"},
  "filtros_aplicados": {...},
  "candidatos": [
    {"fecha": "2025-05-20", "score": 0.93, "razones": ["martes", "mayo", "día después de festivo (2025-05-19)"],
     "observado": true, "energia": 31200.5, "pico": {"mw": 1500.1, "hora": 20},
     "curva": [24 floats MW], "curva_normalizada": [24 floats, suma 1],
     "clima": {"temp_media": 23.1, "lluvia_mm": 3.0} | null}
  ],
  "resumen": {"n_candidatos": 12, "curva_promedio_normalizada": [24], "energia_promedio": 31000.0}
}
```
Uso en pasos (params nuevos, opcionales, sin cambiar defaults salvo evidencia de backtest):
- `forma`: `fuente: "analogos"` + `analogos: {"YYYY-MM-DD": ["fecha1", ...]}` (fechas elegidas por día
  objetivo; si falta un día → búsqueda automática con filtros por defecto). Promedio ponderado por score de
  formas normalizadas, sin suavizar picos.
- `nivel`: `ancla: "analogos"` (opcional) con el mismo mapa → ratio de energía análogo/su ancla.
- `referencia`: diagnóstico `semanas_sugeridas` (semanas que contienen los escenarios de la semana objetivo).
- La selección queda en `params` del paso (trazabilidad, F09).

## Criterio de terminado global
Todo lo anterior mergeado (fast-forward) a `main` (EPM) y `test` (Sphere) con revisión limpia, más
`docs/work/ajustes/RESUMEN_NOCHE.md` con resultados antes/después, pendientes, decisiones sin el usuario y
qué revisar al despertar.
