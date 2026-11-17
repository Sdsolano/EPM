# R1 - angulo conventions (Sphere-wt-n2, 8a6302c..HEAD)

Verificado: eslint sobre mesa_ajustes, providers/ajustes y demo = 0 avisos. tsc -b: ningun error en archivos del diff (los errores son preexistentes en Update_data/Valoracion). Todos los var(--mesa-*) usados estan definidos en tokens.module.css. Sin secretos (la contrasena demo se valida contra el backend, no esta en el codigo), sin archivos temporales ni bytecode commiteados, sin console.log fuera de __verif__ (script de desarrollo), sin eslint-disable/ts-ignore/any, sin emojis (solo el glifo texto de check en BarraEtapas). Textos es-CO con tildes correctas.

Hallazgos (ninguno critico/alto):

1. MEDIA/baja - CONFIRMED - components/analogos/GraficaAnalogos.tsx:10-15 y :170-184
   Colores de serie hardcodeados (COLOR_ACEPTADO "#8E9A94", etc.) en vez de los tokens --mesa-serie-* / charts/colores.ts. #8E9A94 sobre blanco da ~2.9:1 (<3:1 grafico), justo lo que charts/colores.ts documenta evitar (#9AA5A0). Ademas la leyenda usa style inline.
   Arreglo: importar de charts/colores.ts (o usar var(--mesa-serie-analogo)/#6F7C75 para "aceptado") y mover estilos de leyenda a analogos.module.css.

2. BAJA - CONFIRMED - CSS modules (ejecutivo/flujo/paso/resumen/charts .module.css)
   Hex literales sin token: #e3f3ea, #cfe5d8, #f1f8f4, #e9f5ef, #eef6f1, #eceeed, #7a1b16, #e8c98f, #e7a7a2, #b8c2bd, #6f7c75 y #fff repetido (donde existe --mesa-surface). Incumple "tokens --mesa-*" del DESIGN.md y dificulta cambio de tema.
   Arreglo: agregar tokens (--mesa-ok-bg, --mesa-ok-borde, --mesa-bloqueo-ink, etc.) en tokens.module.css y usar var(); reemplazar #fff por var(--mesa-surface).

3. BAJA - CONFIRMED - charts/colores.ts, charts/CurvaHoraria.tsx:~398, CurvaEnergiaDiaria.tsx:~741, CurvaMini, GraficaAnalogos.tsx
   Paleta duplicada como hex en TS (stroke "#fff", "#5C6B63", "#c9d2ce", "#e3e8e5"). Es comun con Recharts, pero hay deriva (p.ej. "#e3e8e5" y "#E8ECEA" no son tokens). Arreglo: leer CSS vars con var(--mesa-...) en atributos SVG (stroke="var(--mesa-surface)") o centralizar todo en colores.ts (GraficaAnalogos y los "#fff" no lo usan).

4. BAJA - CONFIRMED - charts/__previa__/Previa.tsx y providers/ajustes/__verif__/verificar.ts (dentro de src/)
   Artefactos de desarrollo en src con hex inline y console.log; no se importan (no entran al bundle), pero contaminan src y tsc/eslint los procesan. Arreglo: moverlos fuera de src (p.ej. /dev o /scripts) o documentar en README; aceptable si se deja.

5. BAJA - PLAUSIBLE - naming
   Mezcla es/en: carpetas/archivos en espanol (ejecutivo, flujo, paso) con hooks/utilidades en ingles (useInsights, hooks.ts, tipos.ts, colores.ts). Coincide con el resto de Sphere (useMesaAjustes), por lo que no se pide cambio.
