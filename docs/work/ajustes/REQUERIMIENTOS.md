# Mesa de ajustes: objetivo, requerimientos del analista y backlog

Documento vivo. Cada sesión que retome este trabajo lo lee junto con `README.md` y lo actualiza
cuando cambie una prioridad o llegue retroalimentación nueva.

## Objetivo final
**Reducir mucho el error de las predicciones semanales.** La Mesa de ajustes no es un fin en sí
misma. Tiene dos funciones:
1. Darle al analista **todas las herramientas y el paso a paso** que hoy hace a mano (procedimiento
   en "Procedimiento para llevar a cabo el pronóstico de la demanda de energía").
2. Que el sistema **haga solo gran parte de ese trabajo**: cada paso que el backtest demuestre
   confiable pasa de "sugerido" a "automático", por segmento (mercado y tipo de día).

Métrica de éxito: WAPE horario, error de energía diaria y semanal, error de pico, por mercado y tipo
de día, medidos con backtest temporal y luego contra los pronósticos reales emitidos. Metas de la
evaluación funcional: −10 % relativo de WAPE, −50 % de ajustes manuales y −40 % del tiempo de
elaboración.

## Retroalimentación del analista (recibida 2026-11-12)

### R1. El "ajuste inteligente por escenario de referencia" no está sirviendo — revisar el viernes 2026-11-13
Palabras del analista, resumidas: lo está usando en el pronóstico de esta semana y no le sirve como
quisiera. La idea es que los modelos queden lo más ajustados posible y que hagan buena parte de lo
que hace él.

- **Qué es hoy.** El botón ASR de Sphere: `Sphere/src/views/pages/pronosticos/Pronosticos.tsx`
  (`handleASR`, ~línea 5146). Calcula, hora a hora,
  `delta = (hist2 − hist3) − (pronóstico − hist1)` usando los históricos que el analista tenga
  seleccionados. Lo envía a `POST /validate-hourly-adjustments` (`src/api/main.py`), que aplica
  `HourlyAdjustmentValidator` (`src/prediction/hourly/adjustment_validator.py`): los ajustes del
  usuario con *soft constraints* de ±15 % respecto al cluster histórico.
- **Hipótesis de por qué falla** (validar el viernes con un caso real):
  - El delta depende del orden en que se seleccionaron los históricos. No hay alineación por tipo
    de día, ni por festivo, ni por el desfase entre el último dato real y la semana objetivo.
  - No usa una semana de referencia equivalente del año anterior: compara días sueltos.
  - Trabaja hora a hora: mezcla nivel y forma y puede introducir ruido en el perfil.
  - No conserva la energía del día ni revisa la coherencia con la tendencia semanal.
  - El validador "preserva el cluster", pero el cluster puede no representar el escenario (festivo,
    puente, clima).
- **Relación con la Fase 1.** Los pasos `referencia` → `nivel` → `forma` de la mesa son el
  reemplazo de diseño del ASR: separan nivel y forma, alinean por día de semana y festivo con el
  mismo desfase de corte, conservan la energía y dejan trazabilidad.
- **Preparación para el viernes:**
  1. Pedir al analista la semana que está pronosticando, su pronóstico base y su resultado final
     ajustado a mano.
  2. Correr `POST /adjustments/preview` con `base_curvas` (su pronóstico base) para esa semana, y
     también el ASR actual.
  3. Comparar las tres curvas (ASR, mesa y ajuste del analista) por día: energía, pico y forma.
     Cuando llegue el dato real, medir el error de cada una.
  4. Anotar qué decisiones toma el analista que la mesa todavía no reproduce. Esas notas son la
     entrada directa del backlog.
- **Salida esperada:** decidir si el ASR se reemplaza por la mesa en Sphere (Fase 3) o se corrige
  antes como paliativo.

### R2. Que los modelos hagan la mayor parte del trabajo del analista
- Cada paso que hoy es manual debe tener una versión automática, medida con backtest:
  - elección de la semana de referencia;
  - variación diaria (nivel);
  - curva de referencia (forma);
  - días especiales;
  - control de coherencia semanal.
- Activar la aplicación automática por segmento solo cuando el backtest lo respalde. El analista
  revisa excepciones.
- Registrar cada intervención del analista (ya se guardan `curvas_calculadas`, versiones y eventos)
  para el aprendizaje posterior (F09): detectar ajustes recurrentes y llevarlos al modelo o a reglas.

### R3. Buscador de días y escenarios análogos muy específicos
Ejemplo del pedido: "un martes de mayo después de festivo". La mesa debe permitir buscar en el
histórico días con escenarios muy similares y usarlos como referencia.

- **Filtros combinables:**
  - día de semana;
  - mes o temporada;
  - posición respecto a un festivo: día anterior o posterior, puente, a n días;
  - tipo de festivo, Semana Santa, elecciones;
  - temporada escolar o vacaciones;
  - clima similar: temperatura, lluvia, humedad;
  - rango de años y años excluidos (2020);
  - UCP o mercado.
- **Resultado:**
  - lista ordenada por similitud, explicando por qué cada día es análogo;
  - marca de datos observados frente a faltantes;
  - curvas normalizadas y energía de cada día;
  - opción de aceptar o descartar cada candidato.
- **Uso en los pasos:**
  - en `forma`, una fuente `analogos`: promedio ponderado de los días elegidos, sin suavizar picos;
  - en `nivel`, usar los análogos como ancla o ratio;
  - en `referencia`, sugerir semanas que contengan esos escenarios.
  - La selección queda registrada en la corrida (trazabilidad y F09).
- **API tentativa:** `GET /adjustments/analogos?ucp=&fecha=&filtros...`, más params nuevos en
  `forma` y `nivel` que acepten la lista de fechas elegidas.
- Corresponde a F01 (escenarios equivalentes) y F03 (selección de curvas) de la evaluación funcional.

## Backlog priorizado (actualizado 2026-11-12)
1. **Viernes 2026-11-13: revisión del ASR con el analista** (R1). Preparar la comparación ASR vs
   mesa vs ajuste manual.
2. **Fase 2a: backtest con el modelo real** y recalibración de alpha (por tipo de día), fuente de
   forma y bandas, con holdout temporal. Hoy los valores por defecto son provisionales. Bloqueos
   conocidos: `fillna(method=...)` con pandas 3 y aislar `full_update_csv`; ver `README.md`.
3. **Fase 2b: buscador de análogos** (R3) e integración en `forma`, `nivel` y `referencia`, medido
   con backtest.
4. **Fase 3: UI "Mesa de ajustes" en Sphere** (rama `test`) más rutas en el backend intermedio.
   Reemplaza al ASR si el viernes así se decide.
5. **Fase 4:**
   - paso de días especiales (Semana Santa, 24 y 31 de diciembre, elecciones);
   - alertas multivariables de clima y eventos (F06);
   - aprendizaje de intervenciones (F09);
   - aplicación automática por segmento (R2).
