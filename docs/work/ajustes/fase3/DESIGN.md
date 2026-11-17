# DESIGN — Mesa de ajustes (alcance: solo esta pantalla de Sphere)

Sphere no tiene DESIGN.md global; esto fija la dirección de la Mesa respetando lo existente (DM Sans, verde
`#009653`, MUI 7, fondo blanco del `MainLayout`, tarjetas `--secondaryLight #f2f2f2`).

## Producto y usuario (PRODUCT inferido)
- Usuario: analista de pronóstico de demanda de energía (EPM), experto, usa la pantalla a diario con
  números en MW/MWh. Quiere ver qué hizo el sistema en cada paso, por qué, y corregir solo lo necesario.
- Trabajo principal: revisar una semana (lunes–domingo) paso a paso, ajustar, aprobar con trazabilidad.
- Voz: técnica, precisa, en español de Colombia, sin marketing. Verbos concretos ("Aplicar", "Omitir paso",
  "Guardar curvas", "Aprobar semana"). Números con formato es-CO (miles con punto, decimales con coma).
- Anti-referencias: dashboards de tarjetas idénticas con KPIs gigantes y gradientes; emojis como iconos;
  eyebrows en mayúsculas; colores de estado usados como decoración.

## Concepto
"Mesa de trabajo" densa y calma. El elemento memorable es el **riel de pasos**: una columna (en móvil, fila
desplazable) donde cada paso muestra su estado y la energía semanal resultante con la variación frente al
paso anterior, así el analista ve cómo se mueve la semana paso a paso. Todo lo demás es silencioso.

## Tokens (en `mesa.module.css`, prefijo `--mesa-`)
Color
- `--mesa-ink: #12261C` texto principal (verde casi negro, coherente con `--dark #051910`).
- `--mesa-muted: #5C6B63` texto secundario (AA sobre blanco).
- `--mesa-surface: #FFFFFF`, `--mesa-panel: #F5F7F6` fondo de paneles, `--mesa-rule: #DDE3E0` bordes.
- `--mesa-primary: #009653` acción primaria y estado "aplicado"; `--mesa-primary-ink: #006B3B` texto verde AA.
- Estados de alerta: info `#1F5F99` (fondo `#EAF2FA`), warning `#8A5300` (fondo `#FFF4E0`),
  bloqueo `#B3261E` (fondo `#FDECEA`). Solo para alertas y estados, nunca decorativo.
- Estados de paso: aplicado (verde), editado (`#6B4FA0` violeta sobrio, fondo `#F1EDF8`), omitido (gris `#5C6B63`
  con texto tachado no; usar etiqueta), sugerido/automático como etiqueta de texto pequeña.
- Series de gráfica: previa `#9AA5A0` discontinua, resultado `#00703E` sólida 2px, real `#1F5F99` puntos+línea
  fina, curva calculada (pre-edición) `#6B4FA0` discontinua; análogos grises `#B8C2BD` 1px y promedio `#00703E`.

Tipografía: DM Sans en toda la pantalla. Escala 12 / 13 / 14 (base) / 16 / 20 / 24. Pesos 400/500/600.
Números siempre `font-variant-numeric: tabular-nums`. Sentence case en todo; sin mayúsculas sostenidas.

Espaciado: base 4px (4, 8, 12, 16, 24, 32). Radios jerárquicos: contenedor de pantalla 16px, paneles 12px,
controles 8px, chips 999px. Sombras: ninguna en paneles (borde `--mesa-rule`); solo menús/drawers usan la
sombra de MUI.

Movimiento: solo respuesta a acciones (abrir drawer, cambiar de paso: fade 150ms de opacidad); nada en
cargas repetitivas; `prefers-reduced-motion` desactiva transiciones. Carga: skeletons de MUI, no spinners
de página completa salvo el cálculo inicial (con texto "Calculando la semana…").

## Layout
```
[Mesa de ajustes]  UCP ▾  Semana (lunes) ▾  [x] Usar pronóstico emitido   [Calcular] [Crear corrida]
estado: Borrador v3 · creada por … (o "Vista previa sin guardar")
┌ riel de pasos ───────┐┌ detalle del paso seleccionado ─────────────────────────────┐
│ Base      aplicado   ││ título, estado, sugerido/automático, acciones               │
│ 52.310 MWh           ││ alertas del paso                                            │
│ Referencia aplicado  ││ diagnósticos legibles                                       │
│ Nivel     editado +1,2%│ gráfica por día (tabs lun..dom) previa vs resultado vs real │
│ Forma     aplicado   ││ editor de params / tabla de curvas (expandible)             │
│ Control   2 alertas  ││                                                             │
│ Resumen y aprobación ││                                                             │
└──────────────────────┘└─────────────────────────────────────────────────────────────┘
Buscador de análogos: drawer derecho (desde forma/nivel o desde la barra superior).
```
Alineación a la izquierda; números a la derecha en tablas. <900px: riel pasa a fila desplazable arriba.
320px sin scroll horizontal de página (las tablas de 24 periodos hacen scroll dentro de su contenedor).

## Accesibilidad
Foco visible (outline 2px `--mesa-primary` con offset 2px), objetivos >= 40px (44 en táctil), contraste AA,
errores con `role="alert"`, avisos de recálculo con `aria-live="polite"`, el riel es una lista con
`aria-current="step"`, gráficas con resumen textual (energía y pico) al lado.
