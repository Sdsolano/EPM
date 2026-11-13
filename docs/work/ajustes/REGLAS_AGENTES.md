# Reglas para agentes (sesión autónoma 2026-11-12/13)

Léelas completas antes de empezar. El usuario duerme: NO preguntes nada (no uses AskUserQuestion ni
modo plan). Ante una duda de negocio, toma la opción más razonable y anótala en tu LOG como
"decisión tomada sin el usuario" con su justificación.

## Rutas
- Repo principal EPM (solo lectura para ti salvo que tu prompt diga otra cosa):
  `C:/Users/samue/OneDrive/Documentos/GitHub/EPM`. Plan maestro: `docs/work/ajustes/PLAN_NOCHE.md` de ese repo.
- Tu worktree y rama te los da tu prompt. Trabaja SOLO dentro de tu worktree.
- Python (Windows): `C:/Users/samue/OneDrive/Documentos/GitHub/EPM/.venv/Scripts/python.exe`. Úsalo desde tu
  worktree (cwd = worktree; no hay instalación editable, `src` se importa desde el cwd).
- Suite: `<python> -m pytest tests -q --ignore=tests/test_hourly_disaggregation.py`. Fallos permitidos
  (preexistentes): `test_future_events::test_get_future_events`,
  `test_holiday_homology::...christmas_profile_uses_prior_christmases` y 2 de `test_openai_analysis`.
  Cualquier otro fallo es tuyo.
- Postgres local 17 (servicio `postgresql-x64-17`, `psql` en `C:/Program Files/PostgreSQL/17/bin`). Solo bases
  temporales que borras al terminar. Si no logras autenticar, los tests de Postgres se saltan: anótalo y sigue.

## Contexto de negocio (léelo)
`docs/work/ajustes/README.md`, `REQUERIMIENTOS.md`, `PLAN.md` (Fase 1) y `LOG.md` (decisiones y hallazgos
aceptados de la Fase 1). Los documentos originales del analista no están en este PC: usa REQUERIMIENTOS.md.

## Límites obligatorios
- NO commitees ni hagas push salvo que tu prompt lo autorice explícitamente; nunca force-push, amend, ni borrar ramas.
- No modifiques `data/raw`, `models/`, `data/features` (ni en tu worktree: son copias, pero el código no debe
  escribirlas). Backtests aislados: copias en un directorio temporal, sin red. Verifica con hash antes/después.
- No commitees CSV grandes, `.env` ni secretos. El repo EPM es PÚBLICO: nada de datos sensibles en docs.
- No toques el `docs/work/ajustes/LOG.md` maestro ni `PLAN_NOCHE.md`: escribe en la carpeta de tu fase
  (`docs/work/ajustes/<fase>/`).
- Ningún cambio de valores por defecto o de algoritmo sin evidencia de backtest con holdout temporal
  (calibrar en semanas antiguas, reportar en semanas recientes no usadas para elegir).

## Cómo trabajar
1. Escribe `docs/work/ajustes/<fase>/PLAN.md` (sub-plan) ANTES de tocar código y `LOG.md` (hecho / en curso /
   falta, decisiones, resultados). Actualiza el LOG en cada hito: si te cortan, otro agente debe poder retomar
   leyendo solo esos archivos.
2. Si tu parte es grande, divídela y lanza subagentes (herramienta Agent) con prompts autocontenidos que
   apunten a estos documentos (ellos no ven tu conversación). Cada uno escribe su sub-plan y puede delegar con
   el mismo patrón. Los subagentes corren en background: cuando lances hijos y necesites sus resultados,
   termina tu turno dejando en el LOG qué esperas; te llegará la notificación o te reanudarán con SendMessage.
3. Métricas de evaluación: WAPE horario, error de energía diaria y semanal (MAPE y sesgo), error de pico
   (MW y hora), por UCP (Antioquia, Atlantico) y tipo de día (laboral/sabado/domingo/festivo). Reutiliza
   `scripts/backtest_adjustments.py` (funciones `metricas_dia`, `semanas_objetivo`, `construir_contexto`,
   proveedores `proveedor_ingenuo`, `cargar_archivo`/`proveedor_desde_curvas`, `con_cache`).
4. Al terminar deja `docs/work/ajustes/<fase>/REPORTE.md` corto: qué hiciste, archivos, resultados con números,
   decisiones sin el usuario, pendientes y riesgos.
5. Estilo: código y docs en español como el resto del módulo `src/adjustments/`; tests en `tests/`.

## Revisión (si tu prompt te pide orquestarla)
Ronda 1 siempre completa: 8 agentes SEPARADOS en paralelo, uno por ángulo (line-by-line, removed-behavior,
cross-file tracer, reuse, simplification, efficiency, altitude, conventions), más 1 agente de regresión (suite
completa, import de la API `src.api.main`, happy paths existentes, lint si existe). TODOS con `model: "sonnet"`.
Cada revisor recibe: el comando para ver el diff (`git -C <worktree> diff main...HEAD` más `git diff` y archivos
nuevos sin trackear), qué hace la feature, su ángulo, y debe reportar hallazgos con severidad
(crítico/alto/medio/bajo) y veredicto (CONFIRMED/PLAUSIBLE). Solo si un revisor reporta crítico o alto:
corregir y repetir la ronda COMPLETA (mismos 9 agentes, diff completo). Si ninguno reporta crítico/alto: NO
hay otra ronda; corrige medios/bajos (o acéptalos con una línea de justificación en el LOG) y corre una sola
verificación estándar (suite completa + import de la API). Tú no subes la severidad de un medio a alto.

## Línea base medida en este PC (2026-11-12, main 74bc100)
`242 passed, 4 skipped, 1 failed` (`test_christmas_profile_uses_prior_christmases`); los de future_events y
openai_analysis pasan o se saltan aquí. pandas 3.0.6, numpy 1.26.4, sklearn 1.9.1, xgboost 3.2.0. ~45 s.

## Regla del usuario (2026-11-13): estado documentado ANTES de cada fase y DESPUÉS de cada paso
Antes de empezar una fase/parte y apenas termines CADA paso (o sub-paso, o cada vez que un hijo reporte),
actualiza tu `LOG.md` con una sección "Estado actual" al principio: hecho / en curso / falta, archivos tocados
sin commitear, comandos para verificar, IDs de subagentes vivos y qué espera cada uno, y el siguiente paso
exacto. Objetivo: si te quedas sin tokens o te corta el límite de uso, otro agente retoma leyendo SOLO ese
archivo y el worktree. No acumules para el final.
