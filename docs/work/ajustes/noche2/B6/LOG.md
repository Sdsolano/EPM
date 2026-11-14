# B6 - correccion final (ronda 3: sin criticos/altos)

Hecho: M1 (confianza con ventana de 12 semanas + texto "hasta el"), M2 (clima cualitativo), M3 (rutas del PC fuera de scripts), M4 (alias
duplicados), bajos 1-8 (guardas de dias con real<=0, docstring de insights, frase de ventana festivos, jerga, plural, doble espacio,
rama muerta/`_energia_semana` dividida/`series(tablas)`/docstrings de endpoints, `CurvasNp`). Tests nuevos en `tests/test_adjustments_insights.py`.
Verificacion: suite completa (783 passed, 1 fallo aceptado: christmas), `import src.api.main` OK. Contrato: CONTRATO_B.md v1.5.

## Aceptados sin cambio
- `list_runs` pesado en Postgres: aceptable al volumen actual (limite acotado a 500).
- Sin cache de `validar_rolling`: se calcula por segmento por peticion; costo bajo.
- Concurrencia en frio del cache del historico: dos peticiones simultaneas pueden cargar dos veces; sin efecto en el resultado.
- `precision()` con ucp None: el router siempre lo rellena; la logica pura lo deja None por diseno.
- `correccion.py` se queda en `src/adjustments` (opt-in, documentado), no se mueve a scripts.
