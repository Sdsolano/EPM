# REPORTE — factor post-festivo (perfil jamer)

Fecha: 2026-11-27. Plan: `PLAN.md`. Detalle agregado: `reports/ajustes/backtest_post_festivo_20261127.md`.

## Decisión: NO se activa

| Variante | Rol | 2022–2024 (calibración) | 2025–2026 (prueba) | Veredicto |
|---|---|---|---|---|
| `pf_homologo` | elegida en calibración | −0,11 pp (45 d) | **+0,43 pp (28 d)** | **NO PASA** (regla 1) |
| `pf_recientes` | informativa | +0,29 pp | +0,28 pp | — |
| `pf_combinado` | informativa | +0,30 pp | +0,53 pp | — |

- Regla 2 (pareado global, 33 semanas): −0,025 pp, IC95 [−0,077; +0,017] → pasa. Regla 3: pasa.
- Informativo: en los 3 post-festivos de las 13 semanas de prueba (incl. 3/nov) la variante homólogo mejora −1,46 pp.
- Caso 3/nov/2026: Mesa actual −5,51 %, `pf_homologo` +0,77 % (F 0,9979), Jamer −1,04 %. El homólogo acierta ese día,
  pero en los 28 post-festivos de 2025–2026 empeora: el efecto post-festivo varía mucho de un año a otro
  (Todos los Santos: 1,028 / 0,948 / 0,998 / 0,992), así que un factor fijo no generaliza.

El código queda disponible con `post_festivo_factor=false` (default).

## Revisión (alcance definido por el usuario)

- Verificador: sin hallazgos críticos/altos; recálculo independiente del 3/nov coincide. Bajos aceptados: `recientes`
  no excluye post-festivos en temporada/Semana Santa (la mediana lo amortigua); tomar los 4 primeros antes de descartar
  los que no tienen vecino puede bajar de 3 ratios. Ambos irrelevantes con la regla apagada.
- Regresión: suite base 1054/1 vs cambiado 1094/1 (misma falla preexistente); flag off idéntico en sintético y en datos
  reales (semanas 2/nov y 16/nov 2026); PATCH de corridas viejas OK.

## Ideas para seguir (no hechas)
- Usar el factor solo para corregir el R1 cuando el post-festivo reciente es atípico frente a su martes normal (en vez
  de reemplazarlo siempre), o combinar con señales externas (receso escolar, clima).
