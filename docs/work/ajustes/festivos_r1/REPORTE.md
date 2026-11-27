# REPORTE — festivos por tipo + R1 limpia (perfil jamer)

Fecha: 2026-11-27. Plan: `PLAN.md` (con Enmienda 2026-11-27). Detalle agregado: `reports/ajustes/backtest_mesa_vs_jamer_20261127.md`.

## Decisión

| Regla | Veredicto | Acción |
|---|---|---|
| `festivo_por_tipo` | **PASA** | Activada en `perfiles.params_perfil("jamer")` (nivel y forma). `jamer_modelo` no se evaluó → apagada. |
| `r1_limpia` | NO PASA | Código disponible, default off. |
| `saltar_puente` | NO PASA | Sin cambios (flag existente, default off). |
| `ambas` | NO PASA | — |

## Números clave (muestra del PLAN: 33 semanas VF, 20 calibración / 13 prueba)

- `festivo_por_tipo`:
  - Regla 1 (Enmienda: muestra ampliada vs real, festivos fijo/semana_santa): −0,76 pp en 2022–2024 (20 días) y −1,86 pp en 2025–2026 (13 días).
  - Regla 2: pareado global −0,046 pp, IC95 [−0,131; +0,000].
  - Regla 3: laboral/sábado/domingo 0,00; festivo −0,90 pp.
  - MAPE h VF 33 semanas: 2,57 % → 2,53 % (Jamer: 2,50 %). Calibración 2,75 → 2,68. Prueba 2,29 sin cambio (sin festivos fijos ahí).
- `r1_limpia`: mejora −1,84 pp en los 138 días ampliados donde cambia R1, pero 0 días cambiados en las 13 semanas de prueba
  (no evaluable) y domingos +0,15 pp → NO PASA. Pendiente: más semanas de prueba antes de reconsiderar.
- `saltar_puente`: empeora en su segmento (+0,26 pp ampliada, +0,44 pp prueba) → descartado.
- Mesa `jamer` vs Jamer (sin cambios): empate estadístico; últimas 13 semanas Mesa 2,29 % vs Jamer 2,47 %.

## Metodología (anti-overfitting)

- Criterio fijado antes de correr; enmienda decidida por el usuario cuando la revisión mostró que las 13 semanas de prueba
  solo tenían festivos lunes (la regla de festivos no era evaluable allí).
- Corte = envío − 1 sin fuga (assert); VF solo días > corte; bootstrap pareado con semilla; umbral 6 % fijo (4/8 % informativos).
- Muestra fija 2026-04-06 … 2026-11-16 (`HASTA_PLAN`). La muestra informativa (48 semanas) no decide.

## Revisión

- Ronda 1 y 2: ángulos completos + regresión; hallazgos altos corregidos (`correcciones_r1/`, `correcciones_r2/`).
- Ronda 3: un hallazgo alto (semanas de `saltar_puente` ausentes de la muestra ampliada) y medios corregidos inline; el usuario
  pidió no correr ronda 4 → verificación estándar (suite completa) en su lugar. En la ronda 3 no reportaron los agentes de
  removed-behavior, cross-file y regresión (cortados por límite de uso); la regresión con flags off quedó cubierta en rondas 1–2 y
  por los tests `test_flags_off_*`.
- Suite final: ver LOG.md.

## Pendientes / riesgos

- Activar en producción (Railway / Sphere) requiere desplegar esta rama: no hecho, pedir aprobación.
- `jamer_modelo` sin evaluar con `festivo_por_tipo`.
- PATCH de un solo paso puede dejar el flag distinto entre nivel y forma (documentado en el contrato).
