# Fase 2b — LOG (snapshot resumible)

Leer primero `PLAN.md` de esta carpeta. Este archivo dice qué está hecho, en curso y pendiente.

## Estado actual (actualizar antes de cada parte y después de cada paso)
- Hecho: PLAN, Parte A, Parte B, Parte C (backtest + reporte `reports/ajustes/backtest_analogos_20261113.md`;
  CSV fuera de git). Decisión de defaults: NO se cambian (decisión 6).
- 2026-11-13 (coordinador nuevo, el anterior fue detenido): verificada Parte C (reporte .md presente,
  `tests/test_backtest_analogos.py` 11 passed). Los revisores lanzados por el coordinador anterior se dan por
  perdidos (sin IDs ni reportes): se relanza la ronda 1 completa.
- En curso: ronda 1 de revisión (8 ángulos + regresión, sonnet) relanzada; espera sus 9 reportes.
- Falta: fixes de hallazgos, verificación estándar, `REPORTE.md`, commit local en `feat/ajustes-fase2b` (sin push).
- Sin commitear: `src/adjustments/router.py`, `steps/forma.py`, `steps/nivel.py`, `steps/referencia.py` (mod.);
  nuevos `src/adjustments/analogos.py`, `src/adjustments/steps/_analogos.py`, `scripts/backtest_analogos.py`,
  `tests/test_adjustments_analogos.py`, `tests/test_adjustments_analogos_pasos.py`, `tests/test_backtest_analogos.py`,
  `docs/work/ajustes/fase2b/*`, reporte .md (requiere `git add -f`).
- Subagentes vivos: los 9 revisores de la ronda 1 (no escriben código).
- Siguiente paso exacto: consolidar hallazgos en "Hallazgos de revisión"; si hay crítico/alto -> corregir y
  repetir ronda completa; si no -> corregir medios/bajos, suite + import API, REPORTE.md, commit.

## Progreso real
- [x] Lectura de reglas, contrato y código existente (coordinador).
- [x] PLAN.md de la fase escrito.
- [x] Parte A (analogos.py + endpoint + tests) — hecha (subagente A, ver "Parte A — resultado"). Sin commit.
- [x] Parte B (integración en pasos) — hecha (subagente B relanzado, ver "Parte B — resultado"). Sin commit.
- [x] Parte C (backtest + reporte) — hecha (subagente C, ver "Parte C — resultado"). Sin commit.
- [ ] Ronda de revisión (8 ángulos + regresión, sonnet).
- [ ] Fixes, verificación estándar, REPORTE.md, commit local.

## Decisiones tomadas sin el usuario
1. Backtest de análogos en un script nuevo (`scripts/backtest_analogos.py`) que importa del existente, para no
   chocar con la Fase 2a que modifica `scripts/backtest_adjustments.py`.
2. Semántica de filtros ausentes/null y defaults (ver PLAN §2.1): ausente = no filtra, salvo
   `anios_excluidos` ([2020]) y `limite` (20).
3. Festivos en `forma` con `fuente="analogos"` automática: se conserva el perfil homólogo (ya es el "análogo"
   de un festivo); el mapa explícito del analista sí lo reemplaza.
4. Selección automática de análogos queda en `diagnostics` (fechas y pesos por día); la del analista en
   `params.analogos` (trazabilidad F09). La búsqueda es determinista dado el corte, así que recalcular la
   reproduce.
5. Temporada escolar aproximada (calendario A) con fechas fijas documentadas en el módulo.
6. (Coordinador, tras la Parte C) NO se cambian defaults. La forma por análogos pierde contra `promedio_n` en
   todo. La única regla cumplida (festivo y Semana Santa con nivel α 0.5) la cumplen tanto `ancla: "analogos"`
   como `"actual"`: la ganancia es de α (alcance de la Fase 2a, α por tipo de día) y en festivos la base ingenua
   da error artificial; el aporte propio del ancla por análogos es ≈ 0.8–0.9 pp, bajo el umbral. Candidato a
   revalidar con `--base modelo`: `nivel` con `ancla: "analogos"` y α 0.5 (global y en festivos).

## Resultados

### Parte A — resultado
Archivos (worktree, sin commit):
- `src/adjustments/analogos.py` (nuevo): módulo puro (sin red, no escribe archivos).
- `src/adjustments/router.py` (aditivo): 2 líneas en el docstring del módulo + sección final con
  `ClimaLoader`, `set_clima_loader`, `AnalogosRequest` y `POST /adjustments/analogos` (función `post_analogos`).
  El `import analogos` está en esa sección final (`# noqa: E402`) para no tocar el bloque de imports.
- `tests/test_adjustments_analogos.py` (nuevo): 88 tests (descriptor, filtros, búsqueda, clima, helpers,
  endpoint, y uno con datos reales de Antioquia que se salta si falta el CSV).

API pública final (firmas exactas):
```python
FILTROS_DEFAULT: Dict[str, Any]      # 13 claves del contrato; anios_excluidos=[2020], limite=20, resto None
PESOS: Dict[str, float]              # dia_semana .25, festivo .25, estacionalidad .20, recencia .10,
                                     # semana_santa .05, temporada_escolar .05, clima .10
def validar_filtros(filtros: Optional[dict]) -> dict
def descriptor_dia(d: date, festivos: Iterable[str], clima: Optional[dict] = None) -> dict
def clima_diario(df_clima: pd.DataFrame) -> Dict[str, Dict[str, float]]
def cargar_clima(ucp: str, data_dir: Optional[Path] = None) -> Dict[str, Dict[str, float]]
def puntuar(obj: dict, cand: dict) -> Tuple[float, Dict[str, float], List[str]]
def buscar_analogos(historico, festivos, fecha, fecha_corte=None, filtros=None, clima=None) -> dict
def seleccionar(historico, festivos, fecha, fecha_corte, filtros=None) -> List[Tuple[str, float, np.ndarray]]
def puntuar_fechas(historico, festivos, fecha, fechas) -> List[Tuple[str, float, Optional[np.ndarray]]]
# auxiliares públicas: slug_festivo(nombre), es_semana_santa(d), temporada_escolar(d); constantes
# MAX_DIST_FESTIVO=3, LLUVIA_UMBRAL_MM=1.0, RELACIONES, TEMPORADAS.
```
Notas para la Parte B:
- `historico`: acepta el normalizado (`fecha` datetime64 + P1..P24, p. ej. `ctx.historico`); si no lo es
  (columna FECHA, etc.) lo normaliza con `data.normalizar_historico` (más lento, ~25 ms extra).
- `festivos`: lista/set ISO (`ctx.festivos`). `fecha`/`fecha_corte`: date o ISO.
- `seleccionar`: top `limite` (default 20) de los OBSERVADOS, score desc / fecha desc, respeta filtros
  (`anios_excluidos` default [2020]); nunca la propia fecha; solo <= corte; no usa clima.
- `puntuar_fechas`: no aplica filtros ni corte (lo valida el paso); curva None si el día no está observado;
  ValueError con fechas inválidas o fuera de 1990..2100.
- `descriptor_dia(...)["clima"]` es el clima del día (el parámetro `clima` es el mapa {iso: {...}}).

Tiempos medidos (este PC):
- Sintético 2019–2025 (normalizado): búsqueda cacheada ~6–10 ms; `seleccionar` ~10 ms.
- Real Antioquia (2017-01-01..2026-07-04, 3471 días, con clima): primera búsqueda (construye índice) 0.12 s;
  cacheada 12–14 ms; `seleccionar` cacheada 9.7 ms. Objetivo < 100 ms cumplido. `cargar_clima` 0.06 s.
- La clave de caché del histórico es un hash blake2b de fechas + P1..P24 (~1–2 ms); caché LRU de 16
  entradas para calendario (años, festivos) y para observados (hash). Backtest con cortes semanales distintos
  reconstruye solo la parte "observados" (vectorizada, barata); el calendario se reutiliza por rango de años.

Verificación con datos reales (Antioquia, festivos de la librería, 2026-05-19 martes tras lunes festivo
2026-05-18 Ascensión, corte 2026-05-16):
- Sin filtros (top 5): 2024-05-14 0.9413 [martes, mayo, día después de festivo (2024-05-13), clases];
  2023-05-23 0.9267 [ídem (2023-05-22)]; 2025-06-03 0.9208 [martes, día después de festivo (2025-06-02),
  clases]; 2021-05-18 0.917 [martes, mayo, ídem (2021-05-17)]; 2024-06-04 0.8941 [martes, ídem (2024-06-03)].
- Filtros "martes de mayo después de festivo" (dia_semana/relacion/distancia auto, meses [5]): 8 candidatos;
  top 5: 2024-05-14 0.9413, 2023-05-23 0.9267, 2021-05-18 0.917, 2018-05-15 0.8891, 2022-05-31 0.8857.
- Clima real: solo 232 días (2025-12-01..2026-07-20); p_i suma ~19 000 "mm"/día (valores en cientos por hora),
  así que `lluvia` real no es informativo: todo día con clima cuenta "con lluvia". Se usa tal cual (pedido).

Suite: `330 passed, 4 skipped, 1 failed` (el preexistente `test_christmas_profile_uses_prior_christmases`);
línea base 242 passed + 88 nuevos. `import src.api.main` OK y `/adjustments/analogos` aparece en el OpenAPI.

Decisiones de la Parte A (sin el usuario):
1. `anios_excluidos: null` explícito = default [2020] (igual que ausente); `[]` = no excluye.
2. `dia_semana` int → `[int]`; listas vacías en `dia_semana`/`meses` → 422 (use null para no filtrar).
3. `clima.temp_tolerancia` ausente/null = no filtra por temperatura. El filtro de clima "está activo" solo si
   aplica algún criterio (temperatura con objetivo conocido, o `lluvia` no null); solo entonces se descartan
   candidatos sin clima. Si no aplica nada se anota en `notas`.
4. `distancia_festivo` con relación (resuelta) distinta de antes/despues se ignora y se anota.
5. `fecha_corte` de la respuesta = la pedida (o la efectiva si no llegó); la usada va en
   `filtros_aplicados.fecha_corte_efectiva`. `filtros_aplicados` trae los "auto" ya resueltos.
6. `nombre_festivo` = nombre (librería) del festivo RELACIONADO (el propio si es festivo).
7. Puente con dos festivos a igual distancia en el bloque → se reporta el anterior (coherente con
   "empate → despues").
8. `pico.hora` = periodo 1..24 (argmax + 1), igual que `scripts/backtest_adjustments.py`.
9. En el endpoint los filtros se validan justo después de `_validar_ucp` y antes de leer datos (falla rápido).
10. El clima observado del día objetivo entra al score aunque sea posterior al corte (es el endpoint del
    analista). `seleccionar`/`puntuar_fechas` no usan clima: sin fuga en el backtest.
11. Score de "festivo" en relación `puente` no distingue tipo de festivo (siempre 1.0 si ambos son puente).

### Parte B — resultado
Archivos (worktree, sin commit): `src/adjustments/steps/_analogos.py` (nuevo, lógica compartida),
`steps/forma.py`, `steps/nivel.py`, `steps/referencia.py` (aditivos: import, docstring, `FUENTES` + "analogos",
`validar_params` con `_analogos.separar` al inicio y `p.update(...)` al final, ramas nuevas en `step`, claves
nuevas al final de los diagnósticos), `tests/test_adjustments_analogos_pasos.py` (38 tests).

Params nuevos exactos (defaults no cambian nada y NO se guardan en el record; `validar_params` idempotente;
claves desconocidas siguen rechazadas):
- `forma.fuente`: `"reciente"|"referencia"|"promedio_n"|"analogos"` (default `"promedio_n"`).
- `forma.analogos` / `nivel.analogos`: `None` (default) | `{"YYYY-MM-DD": ["YYYY-MM-DD", ...]}`; ≤ 7 claves,
  ≤ 60 fechas por día, fechas en [FECHA_MIN, FECHA_MAX]; se normaliza (ISO, claves ordenadas, sin duplicados en
  orden del analista). En el `step`: claves en la semana objetivo, fechas ≤ `ctx.fecha_corte` y ≠ día
  objetivo (ValueError → 422). Lista vacía → error (omitir el día = búsqueda automática).
- `forma.filtros_analogos` / `nivel.filtros_analogos`: `None` (default = `_analogos.FILTROS_AUTO`:
  `dia_semana/relacion_festivo/distancia_festivo = "auto"`) | filtros de `analogos.validar_filtros` (se guarda
  completo, validado). Los años excluidos de la corrida se suman siempre. Forma: `limite = n`. Nivel: con
  filtros automáticos `limite = LIMITE_AUTO_NIVEL = 10`; con filtros explícitos, el `limite` del filtro.
- `nivel.ancla`: `"actual"` (default) | `"analogos"`.
- Invocación: `run_pipeline(ctx, prov, params={"forma": {"fuente": "analogos"}, "nivel": {"ancla": "analogos"}})`;
  con mapa: `{"forma": {"fuente": "analogos", "analogos": {"2026-05-19": ["2025-05-20", ...]}}}`.

Alertas nuevas: `ANALOGOS_NO_OBSERVADOS` (info; fechas del mapa sin 24 periodos, `detalle={día: [fechas]}`),
`NIVEL_ANCLA_ALTERNA` (info; días sin ratios válidos → evidencia actual). Se reutiliza `FORMA_FUENTE_ALTERNA`
con motivo `sin_analogos`. Diagnósticos nuevos: forma `dias[f].seleccion` (`analista|automatica|homologo_festivo`)
y `pesos` + `filtros_analogos` (solo con fuente analogos); nivel `ancla` y `dias[f].seleccion/analogos`
([{fecha, ancla, ratio, peso}]) solo con ancla analogos; referencia `semanas_sugeridas` siempre.

Decisiones sin el usuario:
1. **Causa del test que fallaba (bug de código, no del test):** el ancla del análogo c_a = c − lag podía caer en
   otro régimen que el ancla del objetivo (p. ej. 2025-07-01 → c_a 2025-06-24, martes post-festivo), así
   E(c)/E(c_a) ≈ 1.0 en vez de 0.9 y sesgaba el ratio (0.918). Se exige que c_a tenga la misma
   `(relacion_festivo, distancia_festivo)` que el ancla del objetivo (si no, se retrocede de a 7). Con eso el
   caso construido da 0.9005–0.901 (crecimiento de 1–2 semanas) y el test pasa con la tolerancia original.
   El ancla del objetivo no exige años excluidos (es la de siempre, cercana al corte).
2. Festivo sin entrada en el mapa con fuente analogos → perfil homólogo (`seleccion: "homologo_festivo"`).
3. `semanas_sugeridas` se calcula siempre en referencia (5 análogos automáticos por día; semanas con los 7 días
   ≤ corte y sin años excluidos; empate → semana más reciente). Cuesta ~90 ms por corrida.
4. Params de análogos con otra fuente / ancla actual: se aceptan y validan (incluido contra el contexto) pero
   no cambian el resultado.

Tiempos (Antioquia real, `construir_contexto` + `proveedor_ingenuo`, 5 semanas de 2025-12..2026-06):
`run_pipeline` default 0.20–0.44 s; forma analogos + nivel ancla analogos 0.39–0.47 s. < 1 s: sin optimizar más.
Con datos reales las 7 jornadas usan `regla: "analogos"` (7–10 análogos por día), sin alertas.

Suite: `368 passed, 4 skipped, 1 failed` (preexistente `test_christmas_profile_uses_prior_christmases`);
330 previos + 38 nuevos. `import src.api.main` OK.

### Parte C — resultado
Archivos (worktree, sin commit): `scripts/backtest_analogos.py` (nuevo; importa `backtest_adjustments.py` sin
modificarlo), `tests/test_backtest_analogos.py` (11 tests sintéticos, ~10 s), reporte
`reports/ajustes/backtest_analogos_20261113.md` y CSV `reports/ajustes/backtest_analogos_20261113.csv` (9.1 MB,
36 869 filas, NO commitear). **Ojo al commitear: `/reports` está en `.gitignore` (línea 121); el .md hay que
agregarlo con `git add -f` (igual que los `backtest_20261112*.md` ya versionados).**

Comando: `<python> scripts/backtest_analogos.py --fecha 20261113` (defaults: bases ingenuo + archivo, 104 semanas,
n ∈ {3,5,8} × temporada {sin, 45}, α ∈ {0.25, 0.5}, umbral 2 %). 655 s. El .md se puede regenerar sin recorrer:
`--desde-csv reports/ajustes/backtest_analogos_20261113.csv --fecha 20261113 --tiempo-s 655`.
Base modelo (Fase 2a): `--base modelo ingenuo --fecha <AAAAMMDD>`.
Semanas: ingenuo Antioquia cal 2024-07-01..2025-06-23 / hold 2025-06-30..2026-06-22; Atlántico cal
2023-12-25..2025-04-28 / hold 2025-05-05..2026-04-27 (52+52); archivo Antioquia 21 semanas 2026-01..05 (aparte).
Sin fuga (aserción de contexto + todo análogo usado ≤ corte); hash de data/raw, models, data/features igual
antes/después. Suite: 379 passed, 4 skipped, 1 failed (preexistente christmas); 368 + 11 nuevos.

WAPE % del paso final (forma), cal / hold:
| config | Antioquia | Atlántico | archivo Ant. |
|---|---|---|---|
| default | 3.66 / 3.71 | 5.46 / 5.90 | 2.79 |
| forma_referencia | 3.74 / 3.83 | 6.81 / 6.39 | 2.86 |
| forma_analogos_n3 (mejor de forma) | 3.72 / 3.79 | 6.37 / 6.22 | 2.82 |
| forma_analogos_n8_t45 | 3.77 / 3.87 | 6.64 / 6.24 | 2.88 |
| nivel_actual_a050 | 3.57 / 3.57 | 5.59 / 6.04 | 2.87 |
| nivel_analogos_a025 | 3.61 / 3.61 | 5.46 / 5.91 | 2.52 |
| nivel_analogos_a050 | 3.35 / 3.29 | 5.44 / 5.83 | 2.41 |

- Forma por análogos: peor que `promedio_n` en todo (Antioquia −1.7..−4.7 %, Atlántico −5..−22 %), también en
  segmentos difíciles; la ventana de temporada 45 casi no cambia nada. Usa análogos el ~95 % de los días.
- Ancla de nivel por análogos α 0.5: mejor config en calibración en ambas UCP; holdout Antioquia +11.2 %
  (3.29 vs 3.71), Atlántico +1.1 % (no llega al 2 %); archivo +13.7 %. Nunca peor globalmente. Sin combinación
  (ninguna forma mejora al default).
- Segmentos difíciles (WAPE forma default cal/hold; n días por celda): festivos Ant. 12.9/12.6 (17–18), Atl.
  7.2/9.5 (17–19); Semana Santa Ant. 5.7/8.4 (8), Atl. 6.9/8.1 (8–16); puentes 23–30 días; después de festivo
  dist 1: 11–13 días; diciembre Atl. cal solo 7 días.
- Regla PLAN §5 (ingenuo, ambas UCP, cal+hold, ≥ 2 % en nivel y forma, pico no peor): cumplen SOLO
  `nivel_analogos_a050` y `nivel_actual_a050` en `festivo` (mejora mín 10.7 % / 9.2 %) y `semana_santa`
  (13.1 % / 9.7 %, n = 8). La ganancia es de α (la ancla actual con α 0.5 también cumple; aporte propio del ancla
  por análogos ≈ +0.8–0.9 pp) y en festivos con base ingenua el error es artificial (no interpretable, igual que
  v3). Puente, después de festivo, antes, diciembre, normal: no cumplen.
- **Recomendación: no cambiar defaults.** Candidato a revalidar con `--base modelo`: α = 0.5 con
  `ancla: "analogos"` en festivos / Semana Santa (coordinar con el α por tipo de día de la Fase 2a) y, como
  candidato global, `nivel_analogos_a050` (gana fuerte en Antioquia, marginal en Atlántico). La forma por
  análogos queda como herramienta opcional del analista.

Decisiones sin el usuario (Parte C):
1. Partición: mitad más antigua = calibración (impar: la extra a holdout). Archivo = partición aparte.
2. Barrido no cartesiano: forma con nivel default, nivel con forma default, combo solo si ambas mejores mejoran
   en calibración (no ocurrió).
3. Regla para configs de nivel: exige ≥ 2 % en el WAPE de nivel Y de forma, y pico no peor en ambos.
4. Elección en calibración = mayor mejora relativa media (ingenuo, ambas UCP) del WAPE de forma.
5. Se agregó `--desde-csv` para regenerar el reporte sin recorrer y una "lectura" automática en la conclusión
   (α vs ancla; festivos con ingenuo no interpretables).

## Hallazgos de revisión
(pendiente)

## Retoma (coordinador 3)
- Se retoma tras detener coordinadores previos durante la ronda 1. Borrado `bash.exe.stackdump`.
- Plan: ronda 1 (8 ángulos + regresión, sonnet) → corregir → verificación → commit local.
- Ronda 1: 8 ángulos sin críticos/altos. Corregidos inline: semanas_sugeridas con try/except, validar_mapa_ctx solo si se usa, anios_excluidos de la corrida respetado, import router arriba, imports fusionados, var muerta, doble validar_params, lock de caché. Aceptados: clima del endpoint sin caché (medio, eficiencia; solo endpoint), clima no inyectado en pasos (medio; el mapa explícito es la vía), lectura α vs ancla del reporte (matiz documentado abajo), sin barrido k nivel, semanas_sugeridas siempre presente (documentado), duplicación anios_excluidos/_relacion/Semana Santa (bajo), ratio sin tope (bajo), redondeo/bisiesto (bajo).
