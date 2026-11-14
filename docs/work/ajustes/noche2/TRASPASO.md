# TRASPASO noche 2 (2026-11-14) — leer primero si retomas

Objetivo y principios: `PLAN.md` (mismo dir). Estado detallado: `LOG.md` maestro (docs/work/ajustes/LOG.md, final).

## Estado
- **Pista B (backend)**: TERMINADA e integrada en `main` (EPM `cb5a895`). Endpoints `/adjustments/insights/{resumen,precision,recomendaciones,banda,series}`; contrato `CONTRATO_B.md`; reporte `REPORTE_B.md`. Ninguna mejora de pronóstico se activó (no cumplió la regla); banda calibrada.
- **Pista A (front, Sphere)**: worktree `C:\Users\samue\OneDrive\Documentos\GitHub\Sphere-wt-n2`, rama `feat/noche2-front` (base `origin/test` 8a6302c). Fase 1 commiteada (`f10c971`: curvas, provider/mock de insights, lenguaje). Fase 2 (A3 resumen ejecutivo + flujo guiado, A4 curvas/lenguaje en pasos) escrita pero SIN commit al último chequeo. Coordinador agentId `ae796b7c8dabb0731` (reanudable con SendMessage). Docs: `LOG_A.md`, `A3/`, `A4/REPORTE.md`, capturas `capturas_A/` (en la rama feat/noche2-backend de EPM-wt-n2 / main).
- **NO está hecho**: merge de Sphere a `test`, `build:demo`, redeploy a Railway, `noche2/RESUMEN.md`.

## Cómo continuar (en orden)
1. En `Sphere-wt-n2`: `git status`; si hay cambios sin commitear, verificar `npx tsc -b` (no debe sumar errores sobre la base ≈178; base en `tsc_base_A.txt`) y `npx vite build`; luego `npm run build:demo` (sin carga; antes matar node huérfanos). Commit en español con `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
2. Integrar contra backend real: levantar la API EPM (checkout `EPM`, `.venv`), apuntar el front, probar el flujo (bienvenida → resumen → alertas → ajustar → aprobar) y revisar que las curvas se vean bien; NO correr E2E que escriba en `data/` del checkout principal (usar copia).
3. Ronda 1 de revisión (ángulos separados, model sonnet, + regresión + better-interface); solo repetir con críticos/altos.
4. Rebase sobre `origin/test`, `git push origin feat/noche2-front` y `git push origin feat/noche2-front:test`.
5. Redeploy Railway: seguir `docs/work/ajustes/demo/ESTADO_DESPLIEGUE.md` ("Como redesplegar"): `bash scripts/demo_build.sh` (aborta si Sphere está sucio; exige Sphere en `test` actualizado), copiar contexto mínimo a /tmp/demo_ctx, `railway link` + `railway up --no-gitignore --service app --ci`. Verificar /health, /api/demo/ucps (401 sin clave, 200 con), front 200 y visualmente en navegador.
6. Escribir `noche2/RESUMEN.md`: qué cambió, números medidos (ver REPORTE_B.md), qué decir al cliente; commit y push a main (`git add -f`, docs/ ignorado).

## Qué decir al cliente (honesto)
Las mejoras de esta noche son de experiencia (curvas, resumen ejecutivo, recomendaciones, banda de confianza calibrada), no de precisión del modelo: ningún candidato superó la regla de backtest. WAPE de prueba base→final: Antioquia 2.44→2.33, Atlántico 5.86→5.46. Limitación: el historial de precisión es estático (termina jun-2026 Antioquia, may-2026 Atlántico).

## Riesgos conocidos
Límite de uso de tokens (resetea ~12:00 America/Bogotá) cortó a los agentes varias veces: siempre releer LOG_A.md/LOG_B.md. `taskkill /IM node.exe` mata todos los node de la máquina: evitarlo. Docker no probado local; Railway construyó el Dockerfile previamente.
