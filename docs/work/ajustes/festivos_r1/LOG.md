# LOG — festivos por tipo + R1 limpia (perfil jamer)

Leer primero `PLAN.md` de esta carpeta (plan aprobado por el usuario 2026-11-27).

## Contexto para retomar
- Worktree: `<worktree>` = `<scratchpad>/epm_jamer` (`<scratchpad>` = carpeta temporal de la sesión de Claude Code; ver `git worktree list` desde el repo EPM),
  rama `feat/ajustes-festivos-r1` (desde `origin/feat/ajustes-replica-jamer` @ 514d1f6).
- Python: `<python>` = `<repo EPM>/.venv/Scripts/python.exe` (cwd = `<worktree>`).
- Suite: `<python> -m pytest tests -q --ignore=tests/test_hourly_disaggregation.py`; fallos preexistentes permitidos
  en `docs/work/ajustes/REGLAS_AGENTES.md`.
- Datos del backtest (FUERA del repo, no commitear): `<scratchpad>/datos_bt/` →
  `historico_antioquia.csv` (FECHA,P1..P24; 2017-01-01..2026-11-21), `festivos_antioquia.csv` (fecha,nombre),
  `envios_jamer.csv` (sesion,envio,version,fecha,P1..P24; envíos desde 2025-12-01, tipo P = semana objetivo).
- Backtest previo (referencia de lo que se debe reproducir): `<scratchpad>/bt_multi.py` y `bt_analisis.py`.

## Decisiones confirmadas con el usuario
| # | Decisión | Por qué |
|---|---|---|
| 1 | Mejorar la Mesa sin overfitting: reglas fijas, flags default off, activar solo con criterio previo | pedido explícito del usuario |
| 2 | No tocar `g` ni clima en esta fase | g no estable fuera de muestra; clima con poca data |
| 3 | Sin push ni despliegue a Railway sin preguntar | reglas del usuario |

## Progreso real
- [x] Plan aprobado, rama creada, datos exportados a CSV (2026-11-27)
- [x] Agente 1: selector (festivo_por_tipo, r1_limpia) + tests — hecho 2026-11-27, sin commit; suite: 1030 passed, 16 skipped, 1 failed (preexistente christmas). Ver `agente1/REPORTE.md`
- [x] Agente 2: scripts/backtest_mesa_vs_jamer.py + 22 tests (pasan); corrida real 229 s con código del Agente 1 a medio cambiar (cifras provisionales, re-correr); ver `agente2/REPORTE.md`
- [x] Ronda 1 de revisión cerrada; correcciones R1 (14 hallazgos) aplicadas sin commit — ver `correcciones_r1/REPORTE.md`
- [ ] Ronda 2 de revisión multi-ángulo + regresión (diff completo)
- [ ] Backtest, criterio de aceptación, defaults del perfil, REPORTE.md
- [ ] Commit en la rama (sin push)

## Hitos
- 2026-11-27: Agente 1 terminado (ver agente1/REPORTE.md): flags festivo_por_tipo, r1_limpia, r1_umbral_atipico; off = bit-idéntico; suite OK salvo preexistente.
- 2026-11-27: Agente 2 terminado (ver agente2/REPORTE.md): scripts/backtest_mesa_vs_jamer.py reproduce 2,57 vs 2,50 (33 sem). Sus veredictos son PROVISIONALES (corrió mientras Agente 1 editaba). Pendiente: re-correr tras revisión. Muestra: 48 semanas (desde 2025-12-01) vs 33 (desde 2026-04-06) — decidir.
- 2026-11-27: Ronda 1 de revisión multi-ángulo lanzada.
- 2026-11-27: Ronda 1 cerrada. Hallazgos ALTOS: (R1-A1) backtest usa 48 sem por defecto, PLAN dice 33/13 → default --desde 2026-04-06, 48 solo informativo;
  (R1-A2) homólogo H de festivo fijo puede caer en otro tipo de día (laboral/sáb/dom) que el target → F mal aplicado;
  (R1-A3) mediana de `_es_atipico` incluye días de temporada/Semana Santa → en enero salta R1 limpias.
  MEDIOS: clasificador propio de R1 saltada en el script difiere de src (usar el real/diagnósticos); N_H sin filtro de temporada/SS;
  VF evalúa días <= corte; Parte 2 KeyError sin semanas; fallback forma con forma normal si homólogo None (exigir e_h>0);
  --help UnicodeEncodeError (U+2212); CONTRATO_BACKEND sin params nuevos. Bajos varios (ver reportes). Sin regresión con flags off (suite base 946/1 fallo preexistente, cambiado 1030/1 mismo).
  → Agente de correcciones lanzado; luego ronda 2 completa.
- 2026-11-27: Correcciones de la ronda 1 aplicadas (3 altos + 7 medios + 4 bajos; ver `correcciones_r1/REPORTE.md`). Suite: 1038 passed, 16 skipped, 1 failed (preexistente christmas). `--help` OK en cp1252; humo parte 1 (3+4 semanas) OK. Backtest completo pendiente (orquestador, tras ronda 2). Default `--desde` = 2026-04-06; 48 sem solo con `--muestra-informativa`.
- 2026-11-27: Correcciones ronda 1 aplicadas (correcciones_r1/REPORTE.md); suite 1038 ok / 1 preexistente. Ronda 2 completa lanzada.
- 2026-11-27: Ronda 2 cerrada. ALTOS (altitude): regla 1 de festivos no evaluable en las 13 sem de prueba; regla 2 más estricta que el texto.
  Usuario decidió: festivos se evalúan con corte 2022–2024 / 2025–2026 de la muestra ampliada; regla 2 estricta (ver Enmienda en PLAN.md).
  MEDIOS a corregir: homólogo lunes-puente para festivo entre semana; exclusión temporada/SS aunque el target esté en el periodo;
  CONTRATO_BACKEND desactualizado (F, regla reciente_factor_festivo, motivos); detalle.motivo mezcla códigos y texto. Regresión: 0 hallazgos.
  → Agente de correcciones ronda 2; luego ronda 3 completa.
- 2026-11-27: Correcciones ronda 2 aplicadas sin commit (A-J): regla 1 de festivos = festivos fijo/semana_santa con corte 2022-2024 / 2025-2026 de la parte 2 (ambas: + mitad R1), regla 2 estricta documentada, nota calculada de festivos lunes en prueba; homólogo no lunes-puente para festivo entre semana (anios_probados = [{anio, homologo, descarte}]); r1_limpia solo excluye temporada/SS si el objetivo está fuera; tipo_festivo: lunes antes que semana_santa (solo Jue/Vie Santo); motivos snake_case + TEXTO_MOTIVO_SIN_HOMOLOGO; info_festivo() único; CONTRATO_BACKEND sincronizado; .gitignore reports/ajustes/jamer/ (ojo: `git add -f` de la carpeta igual lo añade); test del script con importlib. Suite: 1053 passed, 16 skipped, 1 failed (preexistente christmas). Humo parte 1 (3 sem) + parte 2 corta OK (12 s); --help cp1252 OK. Detalle: correcciones_r2/REPORTE.md (lo escribe el orquestador; el agente no pudo crear archivos de reporte).
- 2026-11-27: Correcciones ronda 2 aplicadas (correcciones_r2/REPORTE.md); suite 1053 ok / 1 preexistente. Ronda 3 completa lanzada.
- 2026-11-27: Ronda 3 EN CURSO (sesión cortada por límite de uso). Reportados sin ALTOS: simplificación, reuso, eficiencia, convenciones.
  Pendientes de reportar: line-by-line, removed-behavior, cross-file, altitude, regresión (relanzar si no hay resultado).
  Medios/bajos a corregir inline (no requieren nueva ronda si nadie reporta altos): reports/ajustes/backtest_mesa_vs_jamer_20261127.md
  está OBSOLETO (regenerar antes de git add -f); CONTRATO_BACKEND.md:139-141 dice que el perfil expande flags en ambos pasos (falso:
  el cliente debe enviarlos a nivel y forma); script imprime "Detalle:" sin CSV; usar data.TIPO_FESTIVO_* en vez de literales.
  SIGUIENTE: cerrar ronda 3 → backtest completo (`<python> scripts/backtest_mesa_vs_jamer.py --historico ... --festivos ... --envios ...
  --partes 1,2 --muestra-informativa`) → criterio (PLAN + Enmienda) → defaults en perfiles si pasa → REPORTE.md → commit (sin push).
- 2026-11-27: Ronda 3 — altitude reportó 1 ALTO: muestra ampliada (semanas_ampliadas) no incluye las semanas donde `saltar_puente` cambia R1 →
  su veredicto (config jamer+saltar_puente) no es confiable. Arreglo: marcar también semanas donde R1 cambia con saltar_puente=True, o dejar esa
  config solo informativa. MEDIO: fijar fin de muestra (--hasta 2026-11-16 o assert 33 semanas). Bajos: final vs VF día de envío; u04/u08 sesgo; diluir festivo_tipo.
  → Hay ALTO en ronda 3: corregir y correr ronda 4 completa (todos los ángulos + regresión).
- 2026-11-27: Ronda 3: ALTO (saltar_puente en muestra ampliada) y medios corregidos inline (semanas r1_puente, HASTA_PLAN 2026-11-16, contrato, constantes, detalle). Usuario pidió NO correr ronda 4 → verificación estándar: suite 1053 ok / 1 preexistente, --help OK. Backtest completo lanzado.
- 2026-11-27: Backtest final: festivo_por_tipo PASA (activado en perfil jamer); r1_limpia, saltar_puente, ambas NO PASA. Tests ajustados (1054 ok / 1 preexistente). REPORTE.md escrito. Commit en la rama (sin push).
