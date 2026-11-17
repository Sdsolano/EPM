# LOG D2 — front demo (Sphere)

## 2026-11-13
- Leídos PLAN.md, REQUERIMIENTOS, fase3 (DESIGN, CONTRATO_BACKEND, LOG), fase4 (REPORTE_4B/4C/4D) y PLAN_D0D1.md (D1: HTTP Basic,
  estáticos con fallback SPA, `/` sirve la SPA). Ajustado el contrato del front a Basic y escrito `CONTRATO_AUTH.md`
  (pide a D1: `/api/demo/ucps`, `/api/demo/reales`, 401 sin `WWW-Authenticate`, `/` y estáticos públicos).
- Baseline Sphere: `tsc -b` = 178 errores, `vite build` OK, sin tests.
- Implementado y commiteado en Sphere `test` (8a6302c): tipos 6 pasos, paso especiales, R2, alertas IA, demo (acceso, layout, provider,
  cliente), `usuario` en run/PATCH/aprobar, `build:demo`, `.env.demo`, `base` configurable.
- Verificación: `tsc -b` 178 (= base), `vite build` y `build:demo` OK, eslint de lo tocado limpio. Prueba en navegador contra un stub
  HTTP (no el backend real): acceso con clave mala/buena, crear corrida (llega `usuario` y `aplicar_automatico`), paso especiales
  (sugiere/aplica), alertas IA, aprobación con `usuario`.
- Corregido de paso: el panel "Opciones del cálculo" se mostraba siempre abierto (CSS `display:flex` pisaba `hidden`).
- Pendiente: probar contra el backend real de D1 (D4).
