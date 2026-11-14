# R1 - better-interface (Pista A, Mesa de ajustes)

Alcance: src/views/pages/mesa_ajustes (charts, ejecutivo, flujo, paso, resumen, analogos, seleccion, stepper, ayuda) en Sphere-wt-n2, mas capturas A1..A5. Stack: React + MUI + CSS modules, tokens --mesa-*. Docs de convencion: tokens.module.css (sin guia de diseno propia). Dominios revisados: accesibilidad, layout, escritura, tipografia, color, pulido (todos inspeccionados por codigo + capturas; no se corrio navegador en vivo, no se midio contraste con herramienta).

## Hallazgos (sin critica/alta: ninguno bloquea uso, rompe lo central ni es accesibilidad grave)

| Sev | Dominio | Ubicacion | Problema | Arreglo | Estado |
| --- | --- | --- | --- | --- | --- |
| MEDIA | layout | components/resumen/resumen.module.css:242-250 + AlertasVigentes.tsx:66-76 | `.alerta` es grid `minmax(0,1fr) auto` sin apilar en movil: el boton "Ir al paso para resolverlo" no hace wrap y deja el texto de la alerta en una columna de ~130px a 375px (captura A5_11_aprobar_375). Dificulta leer el aviso, que es contenido clave de la aprobacion. | En ancho estrecho `grid-template-columns: 1fr` con el boton debajo (justify-self:start), o `@container`. | CONFIRMED |
| MEDIA | accesibilidad | components/resumen/AlertasVigentes.tsx:70 | `aria-label="Ir al paso X para resolver: ..."` no contiene el texto visible "Ir al paso para resolverlo" (WCAG 2.5.3 Label in Name, A); usuarios de dictado por voz no pueden decir lo que ven. | Quitar aria-label y dejar texto visible, o que el nombre empiece con el texto visible: "Ir al paso para resolverlo: <titulo>". | CONFIRMED |
| MEDIA | accesibilidad/color | charts/CurvaHoraria.tsx:360-372 (ReferenceArea con COLOR_DESTACADO), colores.ts:46-49 | La franja de color en la base de la grafica (violeta = dia editado, ocre = dia con alerta) no tiene leyenda visible; su significado solo esta en el tooltip (hover) y en la tabla (`title` en span no enfocable + texto sr-only). Para teclado/tactil el significado por color no se ve (1.4.1). En A5_04 una franja ocre cubre toda la semana sin explicacion. | Agregar items a LeyendaSeries ("Dia con alerta", "Dia editado") con icono/texto, o rotular la franja; reemplazar `title` por texto visible en la tabla. | CONFIRMED (leyenda ausente) |
| MEDIA | accesibilidad | charts/CurvaHoraria.tsx:336, CurvaEnergiaDiaria.tsx:169, charts.module.css:56-59 | Graficas son `role="img"` con `accessibilityLayer={false}`; el detalle hora a hora solo aparece en tooltip al pasar cursor. Hay alternativa tabular (resumen + "Ver el detalle por dia"), asi que no es bloqueante, pero el detalle por hora no es alcanzable por teclado/tactil y el `outline:none` en recharts-wrapper es inocuo solo mientras eso no sea enfocable. | Mantener la tabla como alternativa y mencionarla en el aria-label; verificar que recharts no deje el wrapper en tab order. | PLAUSIBLE |
| MEDIA | layout | resumen.module.css:69-77, paso.module.css:264/326, charts.module.css:210; capturas A3_04_ajustar_320 | A 320px las tablas (energia por dia, editor de curvas con columna "Nuevo (MW)", tabs de dias "dom 15/11 (previo)") se cortan a mitad de valor dentro de contenedores con scroll horizontal; son operables (region con tabIndex=0) pero el unico cue es la scrollbar fina, invisible en movil. El input de edicion queda parcialmente fuera de vista. | Sombra/gradiente en el borde o columnas prioritarias; en el editor, apilar Hora/Calculada/Nuevo en filas en ancho estrecho. | PLAUSIBLE (por captura) |
| BAJA | tipografia | analogos.module.css:165, :556 | `font-size: 11px` (bajo el piso de 12px) y line-height en px (analogos 167/417, paso 65/676, resumen 518) en lugar de sin unidad. | 12px minimo y line-height sin unidad. | CONFIRMED |
| BAJA | tipografia | tokens.module.css:31-32 | Escala base muy pequena para uso denso: --mesa-text-xs 12px / sm 13px en gran parte del texto de ayuda y tablas (se ve en A5_07 "Que hizo el sistema"); legible pero al limite. | Subir texto de ayuda/tablas a 13-14px. | PLAUSIBLE |
| BAJA | ui | ejecutivo.module.css:72, 447; flujo.module.css:76,102; stepper.module.css:113,128,197; seleccion.module.css:197 | Elementos de 20-24px: chips/contadores no interactivos (ok); confirmar que ninguno sea clicable. Los botones revisados tienen minHeight 32-40 (cumple 2.5.8). | Sin cambio, solo verificar. | REFUTED como defecto |
| BAJA | escritura | resumen/AlertasVigentes.tsx:73 | Mismo texto "Ir al paso para resolverlo" repetido en varias alertas; ambiguo fuera de contexto (ver hallazgo aria-label). | Incluir el nombre del paso visible: "Ir al paso Control". | CONFIRMED |
| BAJA | pulido | components/ayuda/AyudaTermino.tsx:104-120 | Popper role="tooltip" con hover y click; el panel va en portal (orden DOM tras el boton ok por aria-describedby). Esc, foco y 1.4.13 correctos. Sin hallazgo real, solo nota: abre en hover tambien con foco de puntero tactil. | Ninguno. | REFUTED |

## Cobertura
- Accesibilidad: foco visible presente en todos los modulos CSS; tabs/radiogroup con roving tabindex y flechas; Dialog de aprobacion con aria; estados vacio/carga/error con role=status/alert; reduced-motion en 15 archivos. Hallazgos: 3 (aria-label, leyenda de franja, grafica solo hover).
- Layout: Hallazgos: 2 (alertas en movil, tablas a 320px).
- Escritura: tono de negocio consistente y verbos claros en botones; solo el boton repetido. 1 hallazgo menor.
- Tipografia: tabular-nums en todos los modulos con cifras, unidades rem/px mixtas. 2 hallazgos bajos.
- Color: tokens con contraste documentado (#00703E, #5C6B63, tinta en blanco); series diferenciadas por trazo ademas de color. No se midio con herramienta: Not verified.
- Pulido: sin `transition: all`; hover con @media (hover:hover) en 8 de 9 reglas (no se identifico cual falta; PLAUSIBLE, baja).

## Verificacion
- Pasado: grep de outline:none (1, inocuo), focus-visible, prefers-reduced-motion, tabIndex, role/aria; lectura de 8 capturas (1280/375/320).
- Not verified: navegador en vivo, navegacion por teclado, contraste medido, 200% zoom, RTL.

## Veredicto
Approve (sin HIGH). Medias/bajas quedan como trabajo.
