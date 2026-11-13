# Sub-plan D2 — front demo en Sphere

Plan maestro: `PLAN.md`. Contrato con el backend: `CONTRATO_AUTH.md`. Código en Sphere, rama `test`.

## Objetivo
Perfil "demo" del front: misma Mesa de ajustes, servida desde el mismo origen que el FastAPI de EPM, sin pantalla de avance
ni textos/mocks de desarrollo, con acceso por contraseña compartida y cubriendo los 6 pasos, análogos, alertas (incluida IA no
verificada), versionado y aprobación.

## Decisiones
- Interruptor de build `VITE_DEMO=true` (`.env.demo`, `npm run build:demo`). `main.tsx` carga por `import()` o la app completa
  (`AppProduccion`) o `demo/DemoApp`: el build demo no arrastra router/módulos de Sphere.
- Router por hash (`/#/acceso`, `/#/`): el backend solo necesita servir `index.html` en `/` (sin fallback SPA).
- Marco propio (`demo/DemoLayout`): barra superior con marca, único módulo "Mesa de ajustes" y usuario; sin sidebar (los otros
  módulos de Sphere dependen del backend intermedio y no existen en la demo). Se reutilizan los tokens `--mesa-*`.
- Auth: D1 eligió HTTP Basic con `DEMO_PASSWORD`. El front valida con `GET /api/demo/ucps` y manda Basic en cada llamada
  (usuario fijo `demo`); el nombre del analista va solo en el campo `usuario` de run/PATCH/approve.
- Provider `providers/ajustes/demo.ts`: habla directo con EPM (`/adjustments/*`), sin envoltorio `{success,data}`; añade
  `aplicar_automatico: true` y `usuario`. Sin "pronóstico emitido" (se oculta el interruptor).
- Paso `especiales`: tipos, etiquetas, diagnóstico (tabla por día), formulario (aplicar, α, evidencia), omitible y editable.
- R2: tabla "Recomendación por tipo de día" en nivel/forma desde `meta.extras.segmentos_paso`.
- Alertas de IA (`EVENTOS_SEMANA`, `ELECCIONES_IA`, `detalle.verificado=false`): etiqueta "Sugerido por IA, no verificado" y lista de
  eventos, en el paso y en el resumen.
- `usuario` también se envía en el provider normal (el backend intermedio lo sobrescribe).

## Hecho / falta
Ver `LOG_D2.md` y `REPORTE_D2.md`.
