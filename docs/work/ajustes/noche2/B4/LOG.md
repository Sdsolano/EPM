# LOG B4 (correccion tras ronda 1 de revision, Pista B noche 2)

## Estado: implementado, sin commitear (el coordinador repite la ronda completa de revision).

## Cambios (archivos: src/adjustments/{insights,router_insights,correccion}.py, tests/test_adjustments_insights{,_router}.py,
## scripts/noche2_b2_{datos,lib}.py, docs/work/ajustes/noche2/CONTRATO_B.md v1.3)
- A1: `_rec_sesgo` sin promesa numerica; prioridad max "media"; titulo "Tendencia reciente: ..."; `validar_rolling` en el gating;
  si no pasa NO se emite (decision: no emitir informativas para no inflar el conteo de "recomendaciones" del resumen).
- M1: sug = final * factor; delta real en MWh/% en explicacion; suprime si el pipeline ya movio el dia (final/base) en la
  direccion del sesgo >= 50 % de lo necesario, o si el signo del delta contradice el sesgo.
- M2: COBERTURA_MEDIDA corregida; ESCALA_BANDA = 1.2 propio de insights (sin importar correccion); nota de modulo en correccion.py.
- M3: cache lru(8) de curvas reales por (ruta, mtime) solo con el cargador por defecto (con cargadores inyectados no se cachea, por
  tests/demo); carga+normalizacion en threadpool; `_aprobadas(desde, hasta, limite)` por endpoint (resumen 9 semanas, precision
  semanas+1, banda/series/recomendaciones sin aprobadas).
- B-1 `Corrida.fecha_fin_sin_fuga` (= min(semana-1, fecha_corte)) para banda, series y evidencia de sesgo; `banda.muestra` lo expone.
  B-2 n_semanas_historial respeta filtros. B-3 evidencia usa fecha_corte. B-4 ELECCIONES_IA con texto. B-5 version nunca null.
- B-6: `data.energia_curva`; sin `if estado != borrador` redundante; `extend`; `_expandir_alerta(a)` sin `c`; eliminado
  `proveedor_sesgo` (no tenia uso real: la validacion rolling usa la regla incorporada, un proveedor externo la invalidaria);
  unificado `_segmento(tipo, fecha)` + `Corrida.tipo` (reemplaza `c_tipo`/`_segmento_corrida`); `holiday_name_for_date`; imports/vars
  sin usar en scripts B2 y tests (alias `I` -> `insights`).
- A3: historial estatico documentado en CONTRATO_B v1.3 y docstring de insights.py (Antioquia termina 2026-06-28, Atlantico 2026-05-03;
  refrescar con scripts/generar_historial_pronosticos.py tras regenerar k1, o persistir pronosticos aprobados: pendiente).

## Resultado con datos reales (24 semanas hacia atras desde el fin del historial emitido)
- Casos (semana x segmento) con evidencia simple: Antioquia 26 -> pasan rolling 1 (4 %); Atlantico 50 -> 8 (16 %, jueves/miercoles
  con 4-6 dias de prueba). Se suprime ~90 % de lo que antes se recomendaba; lo que pasa es poca evidencia (4-6 dias): por eso sigue
  etiquetado "no garantiza mejora".

## Decisiones propias
- Corte de evidencia en la simulacion rolling = sabado previo al lunes de cada dia (corte de emision del modelo).
- Umbrales rolling: >= 4 dias, >= 60 % de mejora, WAPE agregado menor; ventana 8 semanas (52 festivos).
- `fecha_fin` explicita en banda se respeta aunque cause fuga (el default es sin fuga).
