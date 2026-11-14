# Sesión nocturna 2 (2026-11-14): UX "wow" + mejora real de pronósticos + redespliegue

## Objetivo
Mañana el usuario presenta la demo (Railway, interna) a un cliente. Feedback actual: la Mesa de ajustes es NADA intuitiva; las gráficas semanales son barras y deben ser CURVAS (24h continuas); hay que entender de un vistazo qué pasa y qué hacer. Además: mejorar de verdad los pronósticos y agregar features que ayuden al analista. Total libertad para modificar y crear features. Sin preguntar nada. Todo documentado en este directorio.

## Principios
- Todo lo que se afirme como mejora debe estar MEDIDO por backtest sin fuga (base `modelo_k1`/sandbox, ver `reports/ajustes/`); si no mejora, no se activa por defecto y se dice. Nada de números inventados en la UI.
- Curvas, no barras: toda serie horaria (24 h) o semanal (7×24 = 168 puntos) se dibuja como línea suave continua (base vs ajustado vs referencia vs real, banda de incertidumbre opcional), eje X con días/horas legibles, tooltips, leyenda clara. Energía diaria también como línea/área, no barras.
- Intuitivo: el analista debe entender en <10 s: "qué semana, qué pronóstico, qué tan confiable, qué debo revisar, qué hago". Lenguaje en español claro, sin jerga (ratio, α, ancla → explicar en lenguaje de negocio). Flujo guiado: ver resumen → revisar alertas → ajustar (con recomendación del sistema) → aprobar.
- Aspecto producto final (sin textos de desarrollo/mock). Usar skills de diseño: frontend-design, impeccable, better-interface (revisión), emil-design-eng; respetar prefers-reduced-motion. `unslop` no.

## Pistas
- **Pista A — Front (Sphere, worktree `../Sphere-wt-n2`, rama `feat/noche2-front` desde `origin/test`)**: rediseño completo de la Mesa de ajustes + pantalla de entrada "Resumen ejecutivo" (KPIs: error reciente, semana actual, alertas, recomendaciones), gráficas a curvas, comparador base/ajustado/referencia, explicaciones en lenguaje natural, recomendación "aplicar sugerido" de un clic. Build demo: `npm run build:demo`. Criterio: `tsc -b` sin sumar errores respecto a la base, `vite build` OK, probado en navegador (Playwright/preview) con el mock y contra backend local.
- **Pista B — Backend/modelo (EPM, worktree `../EPM-wt-n2`, rama `feat/noche2-backend` desde `main`)**: (1) mejoras de pronóstico medidas por backtest (candidatas: corrección de sesgo por tipo de día/hora con ventana móvil, mezcla ponderada base+análogos+referencia con pesos aprendidos por segmento, ajuste por clima/festivos/puentes, intervalos de confianza empíricos, selección automática del mejor método por segmento; el agente puede proponer más); (2) endpoints nuevos para el front: resumen ejecutivo/insights, métricas de precisión recientes (pronóstico vs real por semana), recomendaciones por día con explicación, banda de incertidumbre; (3) `GET` de contrato documentado en `docs/work/ajustes/noche2/CONTRATO_B.md` ANTES de implementar, para que A pueda integrarlo; (4) tests. Modo demo (`src/api/demo.py`) debe seguir funcionando.
- Integración A↔B: B publica el contrato primero (primera hora); A construye contra mock del contrato y al final integra con el backend real.

## Cierre (orquestador)
Rebase y suite (`.venv/Scripts/python.exe -m pytest tests -q --ignore=tests/test_hourly_disaggregation.py -p no:cacheprovider`; fallos de entorno aceptados: christmas y 3 async), merge a `main` (EPM) y `test` (Sphere), `bash scripts/demo_build.sh`, redeploy a Railway con el procedimiento de `docs/work/ajustes/demo/ESTADO_DESPLIEGUE.md`, verificación HTTP + visual, `docs/work/ajustes/noche2/RESUMEN.md` con qué cambió, números medidos y qué decir al cliente.

## Reglas de proceso (ver REGLAS_AGENTES.md y metodología global)
Plan/sub-plan por parte en este directorio; implementación por subagentes; ronda 1 de revisión con ángulos separados (model sonnet) + regresión; solo repetir ronda completa si hay hallazgos críticos/altos; documentar estado antes de cada fase y tras cada paso; `docs/` ignorado → `git add -f`; commits en español con `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; sin amend/force. Secretos nunca en git.
