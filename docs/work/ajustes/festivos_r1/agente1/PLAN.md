# Agente 1 — sub-plan (selector: festivo_por_tipo + r1_limpia)

Parte A y B de `../PLAN.md`. Sin commit, sin tocar `scripts/backtest_mesa_vs_jamer.py`.

## Diseño
- `data.py`: se MUEVE aquí el cálculo de Pascua (`domingos_semana_santa(anio)` con lru_cache,
  `es_semana_santa(d)`) y `analogos.py` lo reutiliza (analogos importa data; al revés sería circular).
  Nuevo `tipo_festivo(d, festivos) -> "lunes" | "semana_santa" | "fijo"` (None si `d` no es festivo).
- `referencia_reciente.seleccionar_recientes(..., festivo_por_tipo=False, r1_limpia=False,
  umbral_atipico=UMBRAL_ATIPICO)`:
  - el bucle de búsqueda pasa a un helper `_buscar` (misma lógica; con flags off resultado idéntico).
  - `RecientesDia.festivo` (nuevo, None por defecto): dict con `tipo_festivo`, `homologo`,
    `normal_homologo`, `E_homologo`, `E_normal_homologo`, `factor` (None si no se aplica).
    **F viaja en el resultado del selector**; nivel lo multiplica (E = E(R1)·(1+g)·F) y forma usa la
    forma normalizada del homólogo. Así nivel y forma comparten una sola definición.
  - festivo fijo/semana_santa con flag: R1 = búsqueda de clase `normal` (respeta r1_limpia);
    H = homólogo año anterior (datos válidos, <= corte); N_H = H−7, H+7, H−14, H+14 (normal, válido,
    <= corte). Sin H/N_H (o sin R1 normal) → comportamiento actual + alerta info `FESTIVO_SIN_HOMOLOGO`.
  - r1_limpia (targets no festivos, y la búsqueda normal del festivo por tipo): salta candidatos en
    Semana Santa (`semana_santa`), 24/dic–6/ene (`temporada`) o atípicos (`atipico`,
    |E(c)/mediana(4 previos normales mismo dow) − 1| > umbral, mínimo 3 previos, previos buscados
    hasta 8 semanas atrás). Si no queda ninguno limpio → búsqueda sin limpieza + alerta `R1_SIN_LIMPIA`.
- nivel / forma: params `festivo_por_tipo`, `r1_limpia` (bool) y `r1_umbral_atipico` (float (0, 0.5])
  en DEFAULTS_RECIENTE / DEFAULTS_JAMER; se pasan al selector. Diagnóstico por día solo si el flag está
  activo (flags off → diagnósticos idénticos).
- pipeline: `_ALERTAS_R1_COMPARTIDAS` += `FESTIVO_SIN_HOMOLOGO`, `R1_SIN_LIMPIA`.

## Tests
- `tests/test_adjustments_festivos_tipo.py`, `tests/test_adjustments_r1_limpia.py` (nuevos).
- Actualizar asserts de dict exacto (nivel_jamer). No-regresión: pipeline perfil jamer con flags
  ausentes == flags False explícitos (curvas y diagnósticos).

## Estado
- [x] sub-plan
- [x] código  - [x] tests  - [x] suite completa  - [x] REPORTE.md + LOG de la fase
