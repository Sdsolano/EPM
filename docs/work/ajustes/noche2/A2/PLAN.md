# A2 - Datos, provider, mock y lenguaje de negocio (sub-plan)
Padre: ../PLAN_A.md. Contrato: ../CONTRATO_B.md. Codigo: Sphere-wt-n2 (feat/noche2-front).

Entregables y orden:
1. `src/types/ajustesInsights.d.ts` (tipos del contrato).
2. Provider: 5 funciones en `ApiAjustesClient` (real/demo/mock) devolviendo `ApiReturnAjustes<T>` (nunca lanzan);
   URLs en `constants/ajustes/index.ts`; contrato en `CONTRATO_BACKEND.md`.
3. Dominio: `src/providers/ajustes/recomendaciones.ts` (aplicarRecomendacion, aplicarTodasLasRecomendaciones).
4. Mock: `mock/insights.ts` (5 respuestas desde la corrida mock; modo "sin datos"; aplicar cambia la corrida mock).
5. Hook `mesa_ajustes/hooks/useInsights.ts`.
6. `mesa_ajustes/lenguaje.ts` + `components/ayuda/AyudaTermino.tsx`.
7. Verificacion: `src/providers/ajustes/__verif__/verificar.ts` (tsx), tsc diff, eslint, builds.
Estado: ver REPORTE.md al terminar.
