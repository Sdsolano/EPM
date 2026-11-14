# LOG B5 (correccion tras ronda 2 de revision, Pista B noche 2)

## Estado: implementado, sin commitear. Suite completa: 778 passed, 12 skipped, 1 failed aceptado (christmas). `import src.api.main` OK.

## Hallazgo ALTO (honestidad) y decision del coordinador
Un backtest rolling sin fuga mostro que las recomendaciones de sesgo emitidas NO mejoran fuera de muestra, y `validar_rolling` se
evaluaba sobre las mismas ~8 semanas que originan la regla (seleccion sobre ventana corta, no validacion independiente). Fix:
- Recomendaciones de sesgo/nivel/forma: `prioridad` "baja", `respaldo: "tendencia_reciente"` (alertas: `"alerta_del_sistema"`),
  `validado_fuera_de_muestra: false`, texto "Tendencia reciente observada; las pruebas del sistema no confirmaron que corregirla mejore
  el pronostico. Usala como referencia, tu decides." Frase "En una prueba con semanas anteriores..." eliminada; la cifra se etiqueta
  "en las ultimas semanas (ventana con la que se eligio la regla)"; `evidencia.validacion` -> `evidencia.ventana_seleccion`.
- `ajuste` se sigue entregando; `aplicable` = hay ajuste y la corrida es borrador.
- Filtro de emision: ROLLING_MIN_DIAS 8, ROLLING_MIN_MEJORA 0.75, ROLLING_MIN_MEJORA_WAPE_PP 0.3 (constantes nombradas en insights.py).
- Resumen: tendencias NO cuentan como "alta", NO marcan el dia "revisar" y NO empujan `siguiente_paso` (solo alertas del sistema).
  `kpis.recomendaciones` ahora `{total, alta, del_sistema, tendencia_reciente}`.

## Backtest rolling (scripts/noche2_b5_backtest.py; EPM_RAW_DIR apunta a data/raw del repo principal; 364 dias con emitido y real por UCP)
Para cada semana del historial emitido, solo datos hasta el domingo previo; WAPE sobre los dias donde la regla se habria emitido.
| UCP | filtro | dias emitidos | % dias que mejoran | WAPE antes -> despues |
|---|---|---|---|---|
| Antioquia | anterior (min 4 dias, 60 %, w1<w0) | 34 | 62 % | 3.08 -> 2.70 |
| Antioquia | actual (min 8, 75 %, >=0.3 pp) | 14 | 64 % | 3.05 -> 2.60 |
| Atlantico | anterior | 20 | 25 % | 5.55 -> 7.53 |
| Atlantico | actual | 3 | 33 % | 4.57 -> 6.55 |
WAPE global sin corregir: Antioquia 2.71, Atlantico 5.46. El filtro anterior reproduce al revisor (34 dias/62 %/3.08->2.71 y 20 dias/25 %;
el WAPE "antes" de Atlantico difiere, 5.55 vs 5.03, por diferencias menores de segmentacion del calculo). Con el filtro actual se emite
en 14/364 dias (Antioquia) y 3/364 (Atlantico); aun asi la evidencia de mejora fuera de muestra no se confirma (Atlantico empeora): por eso
van como referencia de baja prioridad y sin marca de validacion.

## Medios/bajos corregidos
(a) clima `info` no es recomendacion accionable (ni marca dia, ni siguiente_paso); (b) corrida no borrador -> `ya_aprobada`;
(c) `_TEXTOS_ALERTA`: FESTIVOS_FUENTE_ALTERNA, bloqueos por curva (MOTIVOS_CURVA) y RECONCILIACION_ROTA con `{dias}`, SIN_FORMA visible;
(d) router_insights: `fecha_fin` explicito <= corte sin fuga con run_id; `_aprobadas` con setdefault (la mas reciente prevalece) fusionada con
`_aprobadas_ventana` (firma `_aprobadas(store, ucp, hasta, semanas)`); `_datos(ucp, reales, store=None, semanas, hasta)` (sin parametro aprobadas);
(e) `_loo_error(forma=True)` normaliza la energia como `_corregir_curva`; (f) `festivos_por_defecto(fechas)` sin varargs, `_articulo` y
`_prioridad_sesgo` eliminados, `ref_reales` una vez en series(), `por_semana[s] = validas` sin copia, `resumen(c, d)` sin parametro `recs`,
rama inalcanzable de `correccion._log_errores` eliminada; (g) `data.corte_semana`, `data.TIPOS_DIA` y `data.dias_semana/iso` reutilizados;
(h) `COBERTURA_MEDIDA.nota` con fecha 2026-11-14.

## Decisiones propias
- La interpretacion de "validado_fuera_de_muestra" es siempre false en sesgo (nunca se valida fuera de muestra) y null en alertas.
- `evidencia.ventana_seleccion` incluye wape_antes/despues de la ventana para que quede claro de donde sale la cifra.
- Alertas info EVENTOS_SEMANA siguen siendo recomendacion (baja) como antes; solo el clima info se excluyo.
- Corrida no borrador distinta de aprobada: texto "Esta corrida ya no esta en borrador; no requiere revision."

## Archivos
src/adjustments/{insights,router_insights,correccion,data}.py, tests/test_adjustments_insights{,_router}.py,
scripts/noche2_b5_backtest.py (nuevo), docs/work/ajustes/noche2/{CONTRATO_B.md (v1.4),B5/LOG.md}.
