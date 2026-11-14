# REPORTE A2 - datos, provider, mock y lenguaje (Sphere-wt-n2, sin commit)

Archivos nuevos (Sphere): `src/types/ajustesInsights.d.ts`, `src/providers/ajustes/insights.ts` (adaptador y degradacion),
`src/providers/ajustes/recomendaciones.ts` (dominio), `src/providers/ajustes/mock/insights.ts`, `src/providers/ajustes/__verif__/verificar.ts`,
`src/views/pages/mesa_ajustes/hooks/useInsights.ts`, `.../lenguaje.ts`, `.../components/ayuda/AyudaTermino.tsx` + `ayuda.module.css`.
Editados: `providers/ajustes/index.ts`, `demo.ts`, `mock/index.ts`, `CONTRATO_BACKEND.md`, `constants/ajustes/index.ts` (5 URLs).
No se tocaron useMesaAjustes.ts ni utils.ts.

Verificacion: tsc sin lineas nuevas vs base (178), eslint limpio, vite build y build:demo OK, script de verificacion 141 OK
(esbuild + node; ver cabecera de verificar.ts).
