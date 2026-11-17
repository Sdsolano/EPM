# Reporte: componente LLM del sistema EPM — Introspección y Supervisor

**Propósito:** documento de apoyo para un artículo aparte sobre el componente LLM del sistema de pronóstico de demanda.
**Versión:** 1.0 · Octubre 2026

> **AVISO DE ALCANCE (leer primero).**
> Este reporte mezcla dos tipos de contenido y los marca explícitamente:
>
> | Etiqueta | Significado |
> |---|---|
> | **[IMPLEMENTADO]** | Existe en el repositorio (`src/api/main.py`, `tests/test_openai_analysis.py`, `reports/performance/`). |
> | **[HIPOTÉTICO]** | Diseño, protocolo y resultados **ilustrativos** para el artículo. Las cifras de esas secciones **no provienen de experimentos reales**; deben reemplazarse por mediciones reales antes de cualquier uso fuera del contexto del paper hipotético. |
>
> El componente **Supervisor** se describe como si ya existiera, por indicación del equipo. Hoy no existe en el código.

---

## 0. Resumen ejecutivo

El LLM cumple tres roles en el sistema, de menor a mayor autonomía:

| Rol | Cuándo actúa | Qué produce | Estado |
|---|---|---|---|
| **Introspector** | Después del hecho (MAPE > 5 %) | Explicación textual de la causa del desvío | [IMPLEMENTADO] `analyze_error_with_openai` |
| **Explorador de eventos** | Antes del hecho (horizonte de pronóstico) | Calendario `{fecha: evento}` | [IMPLEMENTADO] `get_future_events_from_openai`, endpoint `/get_events` |
| **Supervisor** | Ciclo de mejora continua | Variables candidatas (features) que el modelo incorpora **solo si** mejoran el pronóstico fuera de muestra | [HIPOTÉTICO] |

Hallazgos centrales (cifras hipotéticas, §4–§7):

1. **Introspección:** 70,4 % de las explicaciones identifican la causa documentada (Hit) y 88,3 % la identifican al menos parcialmente. El 91,7 % de las afirmaciones verificables son factualmente correctas. Las explicaciones son estables ante repetición (acuerdo 0,83) y la estabilidad predice la corrección.
2. **Información web errónea:** el LLM nunca es fuente de verdad. Una variable solo entra al modelo si supera una compuerta de validación por backtest. Con 25 % de evidencia corrupta, un uso ingenuo degrada el MAPE de 2,77 % a 3,02 %; con la compuerta queda en 2,63 %.
3. **Supervisor:** las variables aceptadas reducen el MAPE medio de 3,08 % a 2,76 % (−10,3 %), con ganancia concentrada en días con eventos y sin degradación en días normales.
4. **Generalización:** al ampliar de 4 a 12 series (UCP), el supervisor mejora 10, es neutro en 2 y no degrada ninguna.

---

## 1. Arquitectura del componente LLM

```
                    ┌──────────────────────────────────────────┐
                    │        Pipeline de pronóstico base        │
                    │  features → ensemble (LGBM/XGB/RF) →      │
                    │  total diario → desagregación horaria     │
                    └───────────────┬──────────────────────────┘
                                    │ pronóstico + real (T+1)
                                    ▼
                         ┌─────────────────────┐
                         │  Detector de desvío │  MAPE mensual > 5 %  ó
                         │                     │  2 días consecutivos > 5 %
                         └─────────┬───────────┘
              ┌────────────────────┴─────────────────────┐
              ▼                                          ▼
   ┌───────────────────────┐                 ┌─────────────────────────┐
   │ INTROSPECTOR          │                 │ EXPLORADOR DE EVENTOS   │
   │ ¿Por qué falló?       │                 │ ¿Qué viene?             │
   │ web_search + texto    │                 │ web_search + JSON       │
   └───────────┬───────────┘                 └───────────┬─────────────┘
               │ hipótesis causales                      │ calendario futuro
               └────────────────┬────────────────────────┘
                                ▼
                    ┌──────────────────────────┐
                    │ SUPERVISOR [HIPOTÉTICO]  │
                    │ 1. Estructura evidencia  │
                    │ 2. Propone variables     │
                    │ 3. Verifica evidencia    │
                    │ 4. Backtest A/B          │
                    │ 5. Compuerta de promoción│
                    └────────────┬─────────────┘
                                 ▼ solo variables aprobadas
                       Registro de features → reentrenamiento
```

Principio de diseño: **el LLM propone, el backtest decide.** Ninguna salida del LLM modifica el pronóstico sin pasar por validación cuantitativa.

---

## 2. Información del modelo LLM

### 2.1 Configuración [IMPLEMENTADO]

| Aspecto | Valor |
|---|---|
| Modelo | `gpt-5-mini` |
| API | OpenAI **Responses API** (`client.responses.create`) |
| Herramienta de búsqueda | `web_search` nativa (`tools=[{"type": "web_search"}]`) |
| Entrada | `input=[{role: "user", content: ...}]` (rol de sistema inyectado como prefijo del mensaje de usuario) |
| Salida | `response.output_text` |
| Invocación | Asíncrona vía `run_in_threadpool` (el SDK es síncrono) |
| Credencial | Variable de entorno `API_KEY`; si falta, degrada con gracia (`"Análisis no disponible"` / `{}`) |
| Fallos | Cualquier excepción se captura y devuelve texto/diccionario vacío; **nunca** interrumpe el pronóstico |
| Fuentes | El modelo decide qué buscar; no hay lista blanca de dominios (limitación, ver §5.4) |

### 2.2 Dónde se invoca [IMPLEMENTADO]

| Función | Disparador | Salida |
|---|---|---|
| `analyze_error_with_openai` | `should_retrain = True` en `/analyze-deviation` | Texto (2–4 oraciones) que se concatena al campo `reason` |
| `get_future_events_from_openai` | Endpoint `POST /get_events` | `Dict[str, str]` `{"YYYY-MM-DD": "evento"}` |
| `sanitize_analysis_text` | Post-proceso del texto | Elimina citas de dominios y oraciones donde el modelo narra su búsqueda (“no pude encontrar…”, “fuentes consultadas…”) |

---

## 3. Prompts

### 3.1 Prompt de rol (común a la introspección) [IMPLEMENTADO]

```
Eres un analista experto en sistemas energéticos y demanda eléctrica en Colombia.
Proporcionas análisis concisos basados en información factual encontrada en internet.
```

### 3.2 Introspección — tres variantes según el tipo de error [IMPLEMENTADO]

La variante se elige por `error_type` ∈ {`consecutivo`, `mensual`, `ambos`}. Estructura común:

```
Eres un analista energético experto. Necesito que investigues en internet las
posibles causas de una anomalía en la demanda energética.

**Contexto:**
- UCP: {ucp}, Colombia
- {Fechas afectadas | Periodo}: ...
- Tipo de anomalía: ...

**Tarea:**
Busca en internet eventos ... que pudieron causar variaciones significativas
en la demanda de energía eléctrica.

Considera: <lista de categorías según la variante>

Proporciona un análisis conciso (máximo 2-3 oraciones) con las causas más
probables encontradas.
```

| Variante | Contexto inyectado | Categorías sugeridas | Longitud |
|---|---|---|---|
| `consecutivo` | Fechas de los 2 días con error > 5 % | Clima extremo, eventos masivos, festivos, apagones, políticos/sociales, paros | 2–3 oraciones |
| `mensual` | Mes y MAPE mensual | Clima atípico, actividad industrial, temporadas, racionamientos, demografía | 2–3 oraciones |
| `ambos` | Mes, MAPE y fechas | Conectar picos puntuales con la tendencia mensual | 3–4 oraciones |

**Observación metodológica.** Los prompts piden “las causas más probables”, lo que induce al modelo a **siempre** proponer una causa aunque no exista evidencia (sesgo de confirmación del prompt). Esto motiva la mejora del §5.5 (permitir abstención).

### 3.3 Explorador de eventos [IMPLEMENTADO]

```
Busca en internet todos los eventos que podrían afectar la demanda de energía
eléctrica en {ucp}, Colombia entre {fecha_inicio} y {fecha_fin}.

Incluye: festivos nacionales y locales, eventos masivos, partidos importantes,
eventos climáticos previstos, paros programados, elecciones, otros relevantes.

IMPORTANTE: Responde ÚNICAMENTE en formato JSON:
{ "YYYY-MM-DD": "Nombre breve del evento", ... }

REGLAS: sin links ni fuentes; nombres de máximo 50 caracteres; JSON puro.
```

Post-proceso: se retiran cercas Markdown (```` ```json ````) y se parsea con `json.loads`; ante cualquier error se devuelve `{}`.

### 3.4 Supervisor — prompts [HIPOTÉTICO]

El supervisor usa **dos llamadas encadenadas** para separar la recolección de evidencia de la propuesta de variables (reduce la tentación de “inventar” para completar un formato).

**Paso A — Estructuración de evidencia** (entrada: textos del introspector y del explorador)

```
Eres un auditor de evidencia para un sistema de pronóstico de demanda eléctrica.
Para cada hipótesis causal recibida, devuelve JSON con:
{
  "hipotesis_id": str,
  "evento": str,
  "tipo": "festivo|evento_masivo|clima|paro|mantenimiento|estructural|otro",
  "fecha_inicio": "YYYY-MM-DD", "fecha_fin": "YYYY-MM-DD",
  "ubicacion": str,
  "efecto_esperado": "aumenta|reduce|incierto",
  "horas_afectadas": [1..24] | null,
  "fuentes": [{"url": str, "fecha_publicacion": "YYYY-MM-DD", "medio": str}],
  "confianza": 0.0-1.0,
  "evidencia_suficiente": true|false
}
REGLAS: exige >= 2 fuentes independientes; si no las hay, evidencia_suficiente=false.
Prefiere responder "incierto" a especular. No uses información posterior a {fecha_corte}.
```

**Paso B — Propuesta de variables** (entrada: solo hipótesis con `evidencia_suficiente=true` + catálogo de features existentes)

```
Dado el catálogo de variables existentes {catalogo} y las hipótesis verificadas
{hipotesis}, propone como máximo {K} variables nuevas que un modelo de árboles
pueda usar. Para cada una devuelve:
{
  "nombre": str (snake_case), "definicion": str,
  "tipo": "binaria|ordinal|continua",
  "regla_de_construccion": str  // determinista y calculable ANTES del pronóstico
  "disponibilidad": "T-1d|T-7d|calendario",
  "efecto_esperado": "+|-", "horas_afectadas": [...],
  "justificacion": str
}
REGLAS: no dupliques variables del catálogo; la regla debe depender solo de
información disponible en la fecha de emisión del pronóstico.
```

---

## 4. Búsqueda y proceso

### 4.1 Proceso de búsqueda [IMPLEMENTADO]

1. El detector calcula MAPE mensual y días consecutivos con APE > 5 %.
2. Se arma el prompt con UCP, fechas y tipo de error.
3. El modelo ejecuta `web_search` de forma autónoma (consultas, lectura de páginas y síntesis ocurren dentro de una sola llamada; el código no controla ni registra las consultas individuales).
4. Se recibe `output_text`, se sanea (`sanitize_analysis_text`) y se anexa al campo `reason`.

### 4.2 Proceso completo del Supervisor [HIPOTÉTICO]

| Fase | Entrada | Operación | Salida | Criterio de paso |
|---|---|---|---|---|
| 1. Detección | Pronóstico vs real | MAPE y residuales por día/hora | Lista de desvíos | APE > 5 % |
| 2. Recolección | Desvíos + horizonte | Introspector y Explorador (web_search) | Hipótesis textuales | — |
| 3. Estructuración | Texto | Prompt Paso A | Evidencia JSON | ≥ 2 fuentes independientes |
| 4. Verificación | Evidencia | Chequeos automáticos (§5.3) | Evidencia verificada | Fecha, lugar y plausibilidad cuantitativa |
| 5. Propuesta | Evidencia verificada | Prompt Paso B | Variables candidatas | Regla causal-temporal válida (sin fuga) |
| 6. Construcción | Candidatas | Se calculan sobre todo el histórico | Columnas nuevas | Cobertura ≥ 2 ocurrencias |
| 7. Backtest A/B | Base vs base+variable | Validación de origen móvil | ΔMAPE, IC, p | Compuerta (§6.2) |
| 8. Promoción | Variables aprobadas | Registro y reentrenamiento | Nueva versión del modelo | Sin regresión en el suite |
| 9. Monitoreo | Producción | Re-evaluación mensual | Retiro de variables que dejan de aportar | ΔMAPE sostenido |

---

## 5. Introspección: validez, consistencia y qué pasa si la web está mal

### 5.1 Qué significa “correcta” y “consistente” [HIPOTÉTICO — protocolo]

Se evalúan **cuatro propiedades** distintas; una explicación puede ser fluida y aun así fallar en alguna.

| Propiedad | Pregunta | Cómo se mide |
|---|---|---|
| **Corrección causal** | ¿Identifica la causa real? | Comparación con causa documentada → Hit / Partial / Miss (3 anotadores, κ de Fleiss = 0,78) |
| **Veracidad factual** | ¿Cada afirmación es cierta? | Verificación afirmación por afirmación (fecha, lugar, hecho) |
| **Consistencia** | ¿Dice lo mismo si se repite o reformula? | 5 repeticiones, 3 paráfrasis del prompt y repetición a 7 días; acuerdo semántico por pares (embeddings + juez) |
| **Coherencia con los datos** | ¿El efecto afirmado coincide con el residual? | Signo y franja horaria del efecto vs. residual observado |

**Dataset:** 240 desvíos (APE > 5 %) de 4 UCP (ene 2025 – jun 2026), con causa documentada a partir de registros operativos, prensa y reportes climáticos. Incluye los 50 desvíos del piloto de la especificación técnica (§8.2).

### 5.2 Resultados [HIPOTÉTICO]

**Corrección causal global (n = 240)**

| Resultado | n | % |
|---|---:|---:|
| Hit | 169 | 70,4 |
| Partial | 43 | 17,9 |
| Miss | 28 | 11,7 |

Precisión (Hit) = 70,4 %; recall (Hit + Partial) = 88,3 %.

**Por tipo de causa**

| Tipo de causa | n | Hit | Hit % |
|---|---:|---:|---:|
| Festivo / puente | 48 | 45 | 94 |
| Evento masivo (deporte, concierto) | 52 | 43 | 83 |
| Clima extremo | 61 | 45 | 74 |
| Paro / manifestación | 29 | 20 | 69 |
| Falla o mantenimiento de red | 32 | 13 | 41 |
| Cambio industrial / estructural | 18 | 3 | 17 |

Lectura: el LLM funciona donde la causa **tiene huella pública** (calendario, prensa). Falla donde la causa es **privada u operativa** (nueva planta, maniobras de red). Esto delimita el alcance honesto de la herramienta.

**Veracidad factual (612 afirmaciones verificables)**

| Categoría | n | % |
|---|---:|---:|
| Correcta | 561 | 91,7 |
| Fecha o lugar erróneo | 19 | 3,1 |
| Evento inexistente (fabricado) | 12 | 2,0 |
| No verificable | 20 | 3,3 |

**Consistencia**

| Prueba | Acuerdo |
|---|---:|
| 5 repeticiones, mismo prompt | 0,83 |
| 3 paráfrasis del prompt | 0,79 |
| Repetición a 7 días (la web cambia) | 0,74 |

La consistencia **predice la corrección**: en los 170 desvíos con acuerdo ≥ 0,8, el Hit-rate es 80,6 % (137/170); en los 70 restantes es 45,7 % (32/70). Esto habilita usar el acuerdo entre repeticiones como **señal de confianza barata** (sin etiquetas).

**Coherencia con los datos.** En 212 explicaciones (Hit + Partial) con efecto declarado, el signo coincide con el residual en 201 casos (94,8 %).

### 5.3 Qué pasa si la información de la web está mal

**Taxonomía de fallas**

| Falla | Ejemplo | Detectable por |
|---|---|---|
| Evento fabricado (alucinación) | Concierto que no ocurrió | Exigir ≥ 2 fuentes independientes verificables |
| Fecha errónea | Evento real, día equivocado | Cruce con fecha de publicación y con el calendario oficial |
| Ubicación errónea | Evento en otra ciudad/UCP | Chequeo geográfico automático |
| Fuente poco fiable o sátira | Nota sin respaldo | Lista blanca de medios + reputación del dominio |
| Dato obsoleto o duplicado | Misma noticia replicada | Deduplicación por similitud |
| Omisión | Causa real sin cobertura | No detectable en origen → el LLM igual “encuentra” otra cosa |
| Sesgo del prompt | Pide “causas más probables” → siempre responde | Permitir `incierto` / abstención |

**Salvaguardas en capas** [HIPOTÉTICO]

1. **Evidencia mínima:** sin 2 fuentes independientes, la hipótesis no avanza.
2. **Verificación determinista:** fecha dentro de la ventana, ubicación dentro de la UCP, publicación previa a la emisión del pronóstico.
3. **Plausibilidad cuantitativa:** el efecto declarado (signo, horas) debe ser compatible con el residual; una explicación contradictoria se descarta.
4. **Compuerta por backtest (§6.2):** aunque la evidencia sea falsa, una variable construida sobre ella no mejora fuera de muestra y no se promueve. **Esta es la defensa final: el modelo no confía en el LLM, confía en el backtest.**
5. **Contención del daño:** variable aprobada entra con regularización y tope de influencia; el monitoreo mensual la retira si deja de aportar.

**Experimento de inyección de fallas** (Antioquia, ene–jun 2026; MAPE base = 2,77 %)

Se corrompe un porcentaje de la evidencia con cuatro tipos de error (evento fabricado, fecha desplazada ±1–3 días, región equivocada, efecto con signo invertido) y se compara el supervisor **sin compuerta** (confía en el LLM) y **con compuerta**.

| Evidencia corrupta | Sin compuerta (MAPE %) | Con compuerta (MAPE %) |
|---:|---:|---:|
| 0 % | 2,49 | 2,50 |
| 10 % | 2,71 | 2,55 |
| 25 % | 3,02 | 2,63 |
| 50 % | 3,55 | 2,74 |

Sin compuerta, a partir de ~10 % de corrupción se pierde la mejora y con 25 % el sistema queda peor que el modelo base (3,02 % vs 2,77 %). Con compuerta, el peor caso (50 %) sigue por debajo del base. **Prueba placebo:** de 1 000 variables aleatorias presentadas como “propuestas del LLM”, se aceptó el 3,1 % (cercano al α = 0,05 de la compuerta), es decir, la tasa de falsos positivos está controlada.

### 5.4 Limitaciones de la introspección

- **Correlación ≠ causalidad.** Una explicación plausible no demuestra que el evento causó el desvío; por eso la introspección se trata como **hipótesis**, no como diagnóstico.
- **Sin lista blanca de fuentes** en la implementación actual.
- **No se registran las consultas** de búsqueda ni las URL usadas, lo que limita la auditoría.
- **Fuga temporal retrospectiva:** al evaluar sobre el pasado, el LLM puede conocer eventos posteriores al corte. Mitigación: usar un modelo con fecha de corte anterior al periodo de prueba y/o **modo sombra prospectivo** de 3 meses (§7.4).

### 5.5 Mejoras propuestas

Permitir abstención explícita (“sin evidencia suficiente”), registrar consultas y URL como trazabilidad, añadir lista blanca de medios y reportar un **puntaje de confianza** derivado del acuerdo entre repeticiones.

---

## 6. Supervisor: demostración de que las variables mejoran el pronóstico

### 6.1 Variables propuestas [HIPOTÉTICO]

Ejemplos aceptadas por el supervisor tras la compuerta:

| Variable | Definición | Disponibilidad |
|---|---|---|
| `evento_masivo_flag` | 1 si hay evento con asistencia estimada > 20 000 en la UCP | T-7d |
| `ola_calor_prevista` | 1 si el pronóstico oficial supera p95 de temperatura máxima | T-1d |
| `puente_festivo_extendido` | Días hábiles intercalados entre festivo y fin de semana | Calendario |
| `paro_programado` | 1 si hay paro/manifestación anunciada oficialmente | T-1d |

Rechazada: `mantenimiento_red` (sin ocurrencias suficientes ni mejora fuera de muestra).

### 6.2 Protocolo de validación

**Diseño:** validación de **origen móvil** (rolling-origin): se entrena con datos hasta `t`, se pronostica `t+1…t+h`, se avanza y se repite. Se compara **base** vs **base + variable** sobre los **mismos** días y semillas.

**Controles contra autoengaño**

| Control | Para qué sirve |
|---|---|
| Variable **placebo** (flag aleatorio con igual prevalencia) | Descartar que cualquier columna extra “mejora” |
| Variable **desordenada** (fechas barajadas) | Descartar que importe solo la frecuencia, no el momento |
| **Oráculo** (causa verdadera conocida ex post) | Cota superior de lo alcanzable |
| Sin fuga | La regla de construcción solo usa información disponible a la fecha de emisión |

**Compuerta de promoción.** Una variable se aprueba si, en el backtest: (i) ΔMAPE < 0 con IC bootstrap pareado al 95 % que excluye 0 (o p < 0,05 en Diebold–Mariano); (ii) no empeora el MAPE en días normales en más de 0,05 pt; (iii) no empeora el desempeño por tipo de día (laboral, fin de semana, festivo); y (iv) pasa el suite completo de regresión.

### 6.3 Resultados [HIPOTÉTICO]

**Mejora global por UCP (MAPE %, base vs base + variables aprobadas)**

| UCP | Base | Con supervisor | Δ relativo |
|---|---:|---:|---:|
| Antioquia | 2,77 | 2,41 | −13,0 % |
| Atlántico | 3,12 | 2,78 | −10,9 % |
| Oriente | 3,48 | 3,15 | −9,5 % |
| Cundinamarca | 2,95 | 2,71 | −8,1 % |
| **Media** | **3,08** | **2,76** | **−10,3 %** |

(El MAPE base de Antioquia, 2,77 %, corresponde al desempeño real ene–jun 2026 reportado en `reports/performance/antioquia_ene_jun_2026/`.)

**Significancia (Antioquia):** ΔMAPE = −0,36 pt; IC bootstrap 95 % [−0,52, −0,21]; Diebold–Mariano p = 0,003.

**Ablación (Antioquia, MAPE %)**

| Configuración | MAPE | Δ vs base |
|---|---:|---:|
| Base | 2,77 | — |
| + `evento_masivo_flag` | 2,62 | −0,15 |
| + `puente_festivo_extendido` | 2,64 | −0,13 |
| + `ola_calor_prevista` | 2,66 | −0,11 |
| + `paro_programado` | 2,72 | −0,05 |
| + `mantenimiento_red` (rechazada) | 2,77 | 0,00 |
| **+ todas las aprobadas** | **2,41** | **−0,36** |
| Placebo (flag aleatorio) | 2,78 | +0,01 |
| Fechas barajadas | 2,79 | +0,02 |
| Oráculo (causa real ex post) | 2,28 | −0,49 |

Las ganancias individuales suman más que la conjunta (−0,44 vs −0,36) porque las variables se solapan (p. ej. puente y festivo). Placebo y barajado no mejoran, lo que indica que la mejora proviene del **contenido** de la variable y no de añadir columnas. El supervisor captura ~73 % de la mejora del oráculo (0,36/0,49).

**Dónde se gana.** La mejora se concentra en días con eventos (MAPE 5,9 % → 3,6 %), mientras que en días normales no hay degradación (2,62 % → 2,55 %). Por tipo de día, el beneficio es mayor en festivos, el segmento más difícil del sistema actual (MAPE real 3,74 % frente a 2,79 % en laborales).

---

## 7. Generalización: más marcadores, más evidencia

> Se interpreta “marcadores” como **series/unidades de prueba adicionales** (más UCP y mercados). Si el término se refiere a otra cosa (p. ej. indicadores o variables marcadoras), ajustar esta sección.

### 7.1 Por qué importa

Cuatro series pueden coincidir con el LLM por azar o por rasgos locales (Colombia, prensa en español, calendario conocido). Ampliar la prueba responde a: *¿la mejora se debe al método o a la elección de los casos?*

### 7.2 Diseño [HIPOTÉTICO]

| Eje | Ampliación |
|---|---|
| Unidades | De 4 a **12 UCP** con regímenes de consumo distintos (costero cálido, andino industrial, residencial) |
| Mercados externos | **2 conjuntos públicos** de otro país/idioma (p. ej. un operador norteamericano y uno europeo) para probar independencia de la geografía y el idioma |
| LLM | **3 backends** distintos para probar que el resultado no depende de un proveedor |
| Transferencia | Validación *leave-one-series-out*: reglas de variables aprobadas en 11 series, probadas en la restante |

### 7.3 Resultados [HIPOTÉTICO]

| Prueba | Resultado |
|---|---|
| 12 UCP | Mejora en **10**, neutro en **2**, empeora en **0**; significativa (p < 0,05) en 8 |
| Mejora relativa de MAPE | Mediana −8,7 % (RIC [−11,4; −5,2]) |
| 2 mercados externos | Mejora en ambos (−6,2 % y −4,8 %), menor que en Colombia por menor cobertura de eventos locales |
| 3 backends LLM | Solapamiento de variables aprobadas (Jaccard) = 0,71; ganancia dentro de ±1,5 pt entre backends |
| Leave-one-series-out | 9 de 12 series mantienen mejora significativa al excluir su propia selección de variables |

Lectura: el patrón se repite entre geografías, idiomas y proveedores, lo que sustenta que el efecto proviene del **mecanismo** (propuesta + validación) y no de una configuración afortunada. Las series neutras corresponden a UCP con baja sensibilidad a eventos públicos (demanda dominada por carga industrial base), coherente con el hallazgo del §5.2 sobre causas privadas.

### 7.4 Amenazas a la validez y modo sombra

| Amenaza | Mitigación |
|---|---|
| Fuga temporal del LLM (conoce el futuro) | Modelo con corte previo al periodo de prueba; modo **sombra prospectivo** de 3 meses (propone y se evalúa sin intervenir producción) |
| Selección de series favorables | Todas las series disponibles incluidas, sin descartes post hoc |
| Múltiples comparaciones | Corrección de Holm sobre las variables evaluadas |
| Dependencia de un proveedor | Prueba con 3 backends |
| Costo y latencia | Supervisor corre fuera de línea (ciclo mensual), no en el camino crítico del pronóstico diario |

---

## 8. Trazabilidad entre este reporte y el código

| Tema | Archivo |
|---|---|
| Prompts de introspección y eventos, llamadas OpenAI | `src/api/main.py` (`get_future_events_from_openai`, `analyze_error_with_openai`) |
| Saneamiento de texto | `src/api/main.py` (`sanitize_analysis_text`) |
| Prueba manual de la introspección | `tests/test_openai_analysis.py` |
| Especificación técnica y piloto (n = 50) | `docs/ESPECIFICACION_TECNICA_ENDPOINTS_ARTICULO_CIENTIFICO.md` §5 y §8.2 |
| Desempeño base real (Antioquia ene–jun 2026) | `reports/performance/antioquia_ene_jun_2026/RESUMEN_AJUSTADO.md` |

**Brechas conocidas entre el reporte y el código:** (1) el Supervisor, la compuerta de promoción y los chequeos de evidencia no existen aún; (2) en la implementación actual la salida del LLM solo se anexa como texto al campo `reason` o se devuelve como calendario, no se convierte en features; (3) en uno de los flujos de `/analyze-deviation` la llamada a OpenAI está comentada; (4) el piloto real documentado en la especificación usa n = 50, no n = 240.
