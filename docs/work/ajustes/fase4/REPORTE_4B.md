# Reporte 4B — paso `especiales`

## Hecho
Paso `especiales` (orden 4, entre `forma` y `control`) implementado con `aplicar=false` por
defecto: detecta y deja sugerencias en `diagnostics` sin cambiar curvas. Con `aplicar=true`
ajusta solo los días especiales detectados y lo registra en `dias_ajustados`.

## Archivos
- `src/adjustments/steps/especiales.py` (nuevo): detección, evidencia por análogos o por la
  semana de referencia, aplicación `E(d) = (1−α)·E_previa + α·E_objetivo`, diagnósticos y alertas.
- `src/adjustments/pipeline.py`:
  - `STEP_ORDER = base, referencia, nivel, forma, especiales, control` (6 pasos).
  - `_MODULOS["especiales"]`; `_PASOS_QUE_MODIFICAN` incluye `especiales`.
  - `_energia_esperada` excluye los días ajustados/editados por `especiales` para no emitir
    `RECONCILIACION_ROTA`.
  - `_records_por_paso`: migración transparente de corridas viejas de 5 pasos (inserta
    `especiales` omitido con `{"omitido": True, "migrado": True}` y renumera `control` a 5 con
    `dataclasses.replace`); no persiste hasta el siguiente PATCH.
- `tests/test_adjustments_especiales.py` (nuevo).
- `tests/test_adjustments_pipeline.py`, `tests/test_adjustments_analogos_pasos.py`: índices y
  órdenes actualizados a 6 pasos.

## Contrato implementado
- `DEFAULTS = {"aplicar": False, "alpha": 0.25, "ancla": "actual"}` (sin efecto por defecto; el
  candidato analógico α 0.5 + `ancla="analogos"` NO se sostiene tras el re-backtest `modelo_k1`,
  ver `REPORTE_PASO0.md`).
- Params opcionales `analogos` y `filtros_analogos` (mismos validadores que forma/nivel); con su
  default no se guardan en el record.
- Detección: Semana Santa (`analogos.es_semana_santa`), 24/12, 31/12 y post-festivo (laboral o
  sábado cuyo día anterior es festivo, confirmado por `analogos.descriptor_dia`). Un motivo por
  día con prioridad Semana Santa > 24/12 > 31/12 > post-festivo.
- Evidencia con `analogos.seleccionar` + `_analogos` (`seleccion_dia`, `evidencia_analogos`,
  `forma_ponderada`, `filtros_efectivos`, `anios_excluidos_ctx`). Filtros por motivo: Semana
  Santa `semana_santa=true`; 24/31 dic `meses=[12]`; post-festivo la `relacion_festivo` real del
  descriptor (`despues` o `puente`) y su distancia si es `despues`; siempre `dia_semana="auto"`.
  El límite es `filtros_analogos.limite` del analista cuando viene; si no, `LIMITE_AUTO_NIVEL`.
- `ancla="analogos"`: forma ponderada de análogos normalizada y escalada a E(d).
  `ancla="actual"` (default): solo energía (evidencia de la semana de referencia del paso nivel),
  forma previa intacta.
- `alpha <= 0`: no modifica energía ni forma; el día queda `aplicado=false`.
- Días editados a mano en pasos previos (`fechas_editadas` del pipeline): se saltan, se conserva
  la curva editada y se emite `ESPECIALES_EDICION_PRESERVADA`.
- Elecciones sugeridas por IA (`ctx.extras["eventos"]`): días con texto de elecciones/votación/
  consulta popular se marcan `elecciones_ia` solo en diagnóstico y alerta `ELECCIONES_IA`; nunca
  modifican curvas. Sin `extras["eventos"]`, no pasa nada.
- Sin evidencia: el día no cambia y se emite `ESPECIALES_SIN_EVIDENCIA`. Con `aplicar=false` y
  días detectados se emite `ESPECIALES_SUGERIDOS`.
- Brecha: no se toca (solo se recorre la semana objetivo).
- Integración con control: `pipeline._energia_nivel` usa el `E_final` de `especiales` como
  baseline de los días ajustados, para no emitir `TENDENCIA_FUERA_BANDA` por el ajuste
  intencional.

## Tests y resultados (tras correcciones ronda 1)
Comando objetivo:
`python -m pytest tests/test_adjustments_especiales.py tests/test_adjustments_pipeline.py tests/test_adjustments_steps.py -q -p no:cacheprovider`
→ **184 passed**.

Suite completa:
`python -m pytest tests -q --ignore=tests/test_hourly_disaggregation.py -p no:cacheprovider`
→ **614 passed, 5 skipped, 1 failed**; el único fallo es el preexistente
`test_holiday_homology::test_christmas_profile_uses_prior_christmases` (`'Navidad' != 'Christmas Day'`).

## Correcciones de la revisión (ronda 1)
1. **Post-festivo (ALTO):** la detección y el filtro se derivan de
   `analogos.descriptor_dia(...)["relacion_festivo"]` y aceptan `despues` y `puente`. Cubre
   viernes tras jueves festivo y sábado tras viernes festivo (ambos `puente`), con tests.
2. **`alpha=0` (ALTO):** con `alpha <= 0` no se tocan curvas ni formas y el día queda
   `aplicado=false`; test de curva idéntica a la previa.
3. **Ediciones manuales (ALTO):** el pipeline pasa a `especiales.step` las `fechas_editadas` de
   los records previos; el paso las salta, conserva la curva editada y emite
   `ESPECIALES_EDICION_PRESERVADA`. Test con edición de nivel y de forma.
4. **Enclavamiento (MEDIO):** `pipeline._energia_nivel` sustituye el baseline de los días
   `dias_ajustados` (aplicado=true) por el `E_final` de `especiales`; test de pipeline con
   `aplicar=true` sin `TENDENCIA_FUERA_BANDA`.
5. **Elecciones (ALTO de alcance):** `ctx.extras["eventos"]` se usa para marcar `elecciones_ia`
   solo en diagnóstico/alerta (`ELECCIONES_IA`), nunca en curvas; tests con y sin eventos.
6. **Límite del analista (MEDIO):** `especiales` respeta `filtros_analogos.limite` cuando viene y
   solo usa `LIMITE_AUTO_NIVEL` en la búsqueda automática; test.
7. **Defaults (MEDIO):** `{"aplicar": False, "alpha": 0.25, "ancla": "actual"}`. El candidato
   α 0.5 + ancla análogos queda documentado como NO sostenido por el re-backtest `modelo_k1`.
8. **Docstrings (BAJO):** control (paso 6/orden 5), pipeline (`especiales` editable,
   `final_curves`), types y `__init__` actualizados a 6 pasos.
9. **Helper de energía (BAJO):** `data.energia_curva(valores)` es el único helper; lo usan
   `especiales` y `control`. Se eliminó la rama muerta `e_final <= 0` y la comprobación
   redundante de `curva_prev`.

## Decisiones sin el usuario
- `ancla="actual"` reutiliza `nivel.evidencias(ctx)` (homólogo/referencia) en vez de duplicar la
  lógica de la semana de referencia; sin `semana_ref_inicio` no hay evidencia y el día queda intacto.
- Si el analista pasa `filtros_analogos`, se combinan con los filtros por motivo, que tienen
  prioridad en sus claves (Semana Santa, mes, relación/distancia con festivo y `dia_semana`);
  los años excluidos de la corrida se siguen sumando siempre.
- `ESPECIALES_SIN_EVIDENCIA` se emite siempre que haya días detectados sin evidencia (también con
  `aplicar=false`), para que el analista sepa que la sugerencia no tiene respaldo.

## Riesgos / pendientes
- La migración de 5 pasos ya no depende de `rerun_from`: `pipeline.migrar_records` es pública e
  idempotente y el router la aplica en `_run_out` (GET) y en `patch_step` (`by_paso`). Una corrida
  vieja devuelve 6 pasos y un PATCH de `especiales` funciona sin otro PATCH previo.
- El ajuste por Semana Santa depende de que el histórico tenga Semanas Santas observadas fuera de
  los años excluidos; con histórico corto cae en `ESPECIALES_SIN_EVIDENCIA`.
- El candidato `alpha=0.5` + `ancla="analogos"` **no se sostiene** con el re-backtest `modelo_k1`
  (`REPORTE_PASO0.md`); no hay evidencia para activarlo. El default es
  `{"aplicar": False, "alpha": 0.25, "ancla": "actual"}`, sin efecto por defecto.
- La detección de elecciones depende de `ctx.extras["eventos"]` (IA, no verificada) y solo
  alimenta diagnóstico/alerta; no hay calendario electoral offline en el repo.
- Al activar `aplicar=true` se emite el warning `ESPECIALES_SIN_BACKTEST` (una vez, no bloquea);
  `analogos`/`filtros_analogos` con `ancla != "analogos"` se ignoran y emiten info
  `PARAMS_ANALOGOS_IGNORADOS` (no rompe clientes).
