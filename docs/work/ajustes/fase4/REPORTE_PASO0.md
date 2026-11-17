# Reporte Paso 0 — re-backtest de análogos y ASR con base `modelo_k1` sin fuga

## Contexto

Se repitieron `scripts/backtest_analogos.py` y `scripts/comparar_asr.py` usando como base las curvas
`modelo_k1` de la Fase 2a (corte previo, sin fuga). No se tocó `src/adjustments/`, ni defaults del
producto, ni `data/raw`, `models/` o `data/features`. Los CSV de curvas se copiaron a `reports/ajustes/`
y no se versionan (`/reports` está en `.gitignore`).

## Archivos tocados

- `scripts/backtest_adjustments.py`: loader reutilizable `cargar_curvas_modelo_k1(ucp, dir)` y
  `semanas_modelo_k1(historico, n, semanas_corte, curvas=None)`.
- `scripts/backtest_analogos.py`: `--base modelo_k1`, `--modelo-dir`, `--base-regla`
  (`ingenuo` default | `modelo_k1` | `archivo`), `_factory`/`semanas_de` para `modelo_k1`; el `.md`
  declara la base de la regla y que `modelo_k1` es sin fuga.
- `scripts/comparar_asr.py`: `--base modelo_k1`, `--modelo-dir` y uso del loader.
- `tests/test_backtest_analogos_modelo_k1.py` (nuevo, sintético, sin red ni CSV reales): loader,
  semanas completas, `--base-regla` filtrando la base correcta y corrida mínima con `modelo_k1`.
- `reports/ajustes/backtest_analogos_20261113_modelo.md` y
  `reports/ajustes/asr_comparacion_20261113_modelo.md` (versionados con `git add -f`; los CSV grandes no).
- `docs/work/ajustes/fase4/REPORTE_PASO0.md` (este archivo).

## Comandos ejecutados

```powershell
Copy-Item ...\EPM-wt-2a\reports\ajustes\curvas_modelo_k1_Antioquia.csv reports\ajustes\
Copy-Item ...\EPM-wt-2a\reports\ajustes\curvas_modelo_k1_Atlantico.csv reports\ajustes\

.venv\Scripts\python.exe scripts\backtest_analogos.py --semanas 52 --base modelo_k1 ingenuo archivo `
  --base-regla modelo_k1 --fecha 20261113_modelo

.venv\Scripts\python.exe scripts\comparar_asr.py --historico --semanas 52 --base modelo_k1 `
  --fecha 20261113_modelo
```

- Análogos: 441 s, 36 869 filas de detalle, 11 configs, 52 semanas por UCP (26 calibración + 26 holdout)
  y 21 semanas de `archivo` (Antioquia).
- ASR: 14 012 filas, 52 semanas por UCP.

## Resultados numéricos

### Análogos — WAPE % del paso forma con el default

| base / ventana | Antioquia calibración | Antioquia holdout | Atlántico calibración | Atlántico holdout |
|---|---|---|---|---|
| `modelo_k1`, 52 sem (esta corrida) | 2.57 | 2.60 | 4.96 | 5.15 |
| `ingenuo`, 52 sem (esta corrida) | 3.58 | 3.82 | 5.93 | 5.86 |
| `ingenuo`, 104 sem (reporte previo 20261113) | 3.66 | 3.71 | 5.46 | 5.90 |

WAPE % del paso forma con el default por UCP y segmento (misma ventana de 52 semanas):

| segmento | modelo_k1 Antioquia (calib/holdout) | modelo_k1 Atlántico (calib/holdout) | ingenuo Antioquia (calib/holdout) | ingenuo Atlántico (calib/holdout) |
|---|---|---|---|---|
| todos | 2.57 / 2.60 | 4.96 / 5.15 | 3.58 / 3.82 | 5.93 / 5.86 |
| tipo:laboral | 2.50 / 2.53 | 4.73 / 5.24 | 3.25 / 3.52 | 5.63 / 5.64 |
| tipo:sabado | 2.24 / 2.53 | 5.32 / 5.10 | 2.62 / 2.94 | 5.81 / 5.71 |
| tipo:domingo | 2.83 / 2.80 | 5.31 / 4.65 | 3.41 / 3.60 | 6.51 / 5.95 |
| tipo:festivo | 4.00 / 3.51 | 6.92 / 5.48 | 12.83 / 12.33 | 10.34 / 8.96 |
| rel:puente | 2.74 / 2.91 | 6.31 / 5.20 | 3.51 / 3.45 | 6.08 / 5.49 |
| normal | 2.45 / 2.51 | 4.93 / 5.50 | 3.21 / 3.55 | 5.93 / 6.22 |

Elección en calibración (base `modelo_k1`):

- Antioquia: `nivel_analogos_a025` (2.390 % en calibración, +6.97 % vs default; holdout 2.453 % vs
  2.605 %, +5.82 %).
- Atlántico: `default` (4.962 %; ninguna variante mejora).
- Mejor variante de forma: ninguna mejora al default en ninguna UCP.
- No se corrió combinación (la forma no mejora).

Regla de cambio de default (base `modelo_k1`, ambas UCP, calibración y holdout, umbral 2 %):

- Única variante que cumple: `nivel_analogos_a025` en `rel:puente` (8/8 celdas, mejora mínima 3.97 %,
  media 8.43 %, Δ pico máx −0.10 pp).
- `n=13` días → **no concluyente** (< 15 días).
- Las candidatas del reporte previo (`nivel_analogos_a050` / `nivel_actual_a050` en festivos y Semana
  Santa) **no se sostienen** con la base real.

### ASR — WAPE % (mismas 52 semanas)

| UCP | serie | antes (`ingenuo`, 20261112) | después (`modelo_k1`, 20261113_modelo) | Δ (pp) |
|---|---|---|---|---|
| Antioquia | base | 4.05 | 2.70 | −1.35 |
| Antioquia | mesa | 3.71 | 2.59 | −1.12 |
| Antioquia | asr_anual | 5.26 | 5.26 | 0.00 |
| Atlántico | base | 6.36 | 5.52 | −0.84 |
| Atlántico | mesa | 6.00 | 5.12 | −0.88 |
| Atlántico | asr_anual | 8.24 | 8.24 | 0.00 |

- Las series `asr_*` no dependen de la base (se construyen con históricos reales): sus métricas son
  idénticas; solo cambian `base` y `mesa`.
- La mesa mejora a la base sobre el modelo real: Antioquia 2.70 → 2.59 (−0.11 pp) y Atlántico
  5.52 → 5.12 (−0.40 pp). Con `ingenuo` la mejora era −0.34 y −0.36 pp.
- El ASR sigue peor que base y mesa: 5.26 vs 2.59 (Antioquia) y 8.24 vs 5.12 (Atlántico); gana día a
  día solo 31.6 % / 26.9 % de los días frente a la base.
- Festivos: `asr_anual` 17.65 (Antioquia) y 10.91 (Atlántico) vs mesa 3.75 / 5.60; `anual_alineado`
  baja Antioquia a 9.27. Atlántico festivo (n=10) es **no concluyente** (< 15 días).
- `archivo` (Antioquia, 21 semanas) sigue siendo la base con menor WAPE (base 3.03, mesa 2.79,
  asr_anual 5.42), pero es una sola emisión con ajustes post-hoc y no es comparable en ventana.

## Hallazgos

1. Con la base real (`modelo_k1`) el default del pipeline mejora claramente al `ingenuo` en la mayoría
   de segmentos (≈1 pp en Antioquia y ≈0.8–1.0 pp en Atlántico en `todos`).
2. La recomendación previa de α alto con ancla por análogos en festivos / Semana Santa era un artefacto
   de la base ingenua (pronostica festivos con días no festivos); no se sostiene con el modelo real.
3. La única variante que cumple la regla con `modelo_k1` es `nivel_analogos_a025` en `rel:puente`, con
   n=13 días: **no concluyente**. No hay evidencia para cambiar defaults.
4. El ASR de Sphere sigue muy por detrás de base y mesa incluso sobre el modelo real; `anual_alineado`
   ayuda en festivos de Antioquia pero no cierra la brecha.
5. La mesa aporta sobre la base real, pero menos que sobre la base ingenua en Antioquia (−0.11 vs
   −0.34 pp) y de forma similar en Atlántico (−0.40 vs −0.36 pp).

## Verificación

- Hashes de directorios protegidos idénticos antes/después:
  - `data/raw`: `5A3DFD8E92DB345487BF3E58BE36588E83498ACB60CA9AEB3AF3BA4DF689B599`
  - `models`: `6F987FC2213019B0DC9335CA9433EB35B272484B04F02DACEA4EF49DE8C95787`
  - `data/features`: `AA048A7928E0F570DD9169260A7812530F12BFAE5B287885FC735645D6A7C666`
- Suite: `486 passed, 5 skipped, 4 failed`. Fallos preexistentes/documentados:
  `test_future_events::test_get_future_events` y 2 de `test_openai_analysis` (pytest-asyncio no
  instalado) y `test_holiday_homology::...christmas_profile_uses_prior_christmases`
  (`'Navidad' != 'Christmas Day'` por locale).
- `python -c "import src.api.main"` → OK.

## Pendientes y riesgos

- La corrida nueva usa 52 semanas (26+26) frente a 104 del reporte previo: menos potencia estadística y
  ventanas no idénticas; las comparaciones "antes/después" deben leerse con esa salvedad.
- Las curvas `modelo_k1` no se regeneran en este paso; si la Fase 2a cambia, hay que repetir la corrida.
- Segmentos con n < 15 días (`rel:puente` 13, festivos de Atlántico 10, Semana Santa) quedan como
  indicio, no evidencia. No se cambiaron defaults.
- La sección "antes/después" de `backtest_analogos_20261113_modelo.md` se añadió al final del `.md`;
  `backtest_analogos.py` no tiene mecanismo `secciones_manuales`, así que una regeneración futura la
  borraría (el de ASR sí conserva secciones ≥ 6).
- No se hizo push; el commit es local en `main`.
