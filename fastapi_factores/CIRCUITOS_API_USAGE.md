# API de Curvas Típicas por Circuito

Documentación completa del endpoint `/factores/calculos/curvas-tipicas-circuitos`

## Descripción

Este endpoint procesa curvas típicas agrupadas por circuito **sin consultar base de datos**. Todo el procesamiento se realiza sobre los datos recibidos en el request.

**Características:**

- No requiere conexión a base de datos
- Agrupa automáticamente por circuito
- Selecciona N curvas más típicas usando algoritmo IQR + centralidad
- Calcula promedio aritmético de las curvas seleccionadas
- Calcula pesos normalizados garantizando suma = 1.0

---

## Endpoints Disponibles

### 1. Procesar Curvas Típicas

**URL:** `POST /factores/calculos/curvas-tipicas-circuitos`

**Content-Type:** `application/json`

---

## Request Schema

### Estructura Completa

```json
{
  "medidas": [
    {
      "codigo_rpm": "string",
      "circuito": "string",
      "ucp": "string",
      "fecha": "YYYY-MM-DD",
      "flujo": "string",
      "p1": 0.0,
      "p2": 0.0,
      ...
      "p24": 0.0
    }
  ],
  "n_max": 8
}
```

### Campos del Request

| Campo       | Tipo    | Requerido | Descripción                                                           |
| ----------- | ------- | --------- | ---------------------------------------------------------------------- |
| `medidas` | Array   | Sí       | Lista de medidas a procesar                                            |
| `n_max`   | Integer | No        | Máximo de curvas típicas por circuito (default: 8, min: 1, max: 100) |

### Campos de cada Medida

| Campo            | Tipo   | Requerido | Validación            | Descripción                              |
| ---------------- | ------ | --------- | ---------------------- | ----------------------------------------- |
| `codigo_rpm`   | String | Sí       | -                      | Código del RPM                           |
| `circuito`     | String | Sí       | -                      | Identificador del circuito                |
| `ucp`          | String | Sí       | -                      | UCP asociado                              |
| `fecha`        | String | Sí       | Formato `YYYY-MM-DD` | Fecha de la medida                        |
| `flujo`        | String | Sí       | -                      | Tipo de flujo (ej: "AE", "AS")            |
| `p1` - `p24` | Float  | Sí       | -                      | Valores de potencia para los 24 períodos |

### Validaciones y Recomendaciones

1. **Homogeneidad de flujo:** Se recomienda enviar medidas del **mismo tipo de flujo** por circuito

   - **Correcto:** Todas las medidas con `flujo: "AE"`
   - **Evitar:** Mezclar `flujo: "AE"` y `flujo: "AS"` para el mismo circuito
   - **Razón:** El algoritmo IQR puede filtrar como outliers las curvas con magnitudes muy diferentes
2. **Formato de fecha:** Debe ser estrictamente `YYYY-MM-DD`

   - Correcto: `"2025-10-02"`
   - Incorrecto: `"02/10/2025"` o `"2025/10/02"`
3. **Valores numéricos:** Todos los períodos deben ser números válidos (int o float)

---

## Response Schema

### Estructura Completa

```json
{
  "ok": true,
  "circuitos": {
    "CIRCUITO_A": {
      "curvas_seleccionadas": [
        {
          "codigo_rpm": "RPM1",
          "fecha": "2025-10-02",
          "periodos": {
            "p1": 35.5,
            "p2": 34.2,
            ...
            "p24": 38.1
          }
        }
      ],
      "promedio": {
        "p1": 35.5,
        "p2": 34.2,
        ...
        "p24": 38.1
      },
      "pesos": {
        "p1": 0.0417,
        "p2": 0.0402,
        ...
        "p24": 0.0448
      },
      "n_curvas": 3
    },
    "CIRCUITO_B": { ... }
  }
}
```

### Campos del Response

| Campo         | Tipo    | Descripción                            |
| ------------- | ------- | --------------------------------------- |
| `ok`        | Boolean | Indica si la operación fue exitosa     |
| `circuitos` | Object  | Diccionario con resultados por circuito |

### Campos por Circuito

| Campo                    | Tipo    | Descripción                                              |
| ------------------------ | ------- | --------------------------------------------------------- |
| `curvas_seleccionadas` | Array   | Lista de las N curvas típicas seleccionadas              |
| `promedio`             | Object  | Promedio aritmético de las curvas seleccionadas (p1-p24) |
| `pesos`                | Object  | Pesos normalizados (suma = 1.0) de cada período          |
| `n_curvas`             | Integer | Número de curvas seleccionadas (puede ser < n_max)       |

### Campos de cada Curva Seleccionada

| Campo          | Tipo   | Descripción                   |
| -------------- | ------ | ------------------------------ |
| `codigo_rpm` | String | Código del RPM de origen      |
| `fecha`      | String | Fecha de la medida             |
| `periodos`   | Object | Diccionario con valores p1-p24 |

---

## Algoritmo de Procesamiento

Para cada circuito, el endpoint ejecuta los siguientes pasos:

### 1. Filtrado de Outliers (IQR)

- Calcula Q1, Q3 e IQR para cada período
- Elimina curvas con valores fuera de `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`

### 2. Selección de Curvas Típicas

- Calcula distancia euclidiana entre todas las curvas (SIN normalización L2)
- Considera **forma Y nivel** (magnitud)
- Selecciona hasta N curvas con menor distancia promedio (más centrales)

### 3. Cálculo de Promedio

```
promedio[pi] = mean(curva1[pi], curva2[pi], ..., curvaN[pi])
```

### 4. Cálculo de Pesos Normalizados

```
suma_total = sum(promedio[p1], ..., promedio[p24])
peso[pi] = promedio[pi] / suma_total
```

- Se redondea a 5 decimales
- Se ajusta el valor máximo para garantizar `sum(pesos) = 1.0` exactamente

---

## Ejemplos de Uso

### Ejemplo 1: Request Simple

```json
{
  "medidas": [
    {
      "codigo_rpm": "SJU748",
      "circuito": "CIRCUITO_A",
      "ucp": "PRIMEGRID",
      "fecha": "2025-10-02",
      "flujo": "AE",
      "p1": -37.695, "p2": -36.645, "p3": -35.295, "p4": -33.51,
      "p5": -32.715, "p6": -31.59, "p7": -27.48, "p8": -28.065,
      "p9": -28.755, "p10": -29.535, "p11": -30.33, "p12": -31.89,
      "p13": -34.47, "p14": -37.845, "p15": -39.45, "p16": -39.33,
      "p17": -37.26, "p18": -35.385, "p19": -37.53, "p20": -38.145,
      "p21": -39.54, "p22": -41.04, "p23": -41.025, "p24": -39.9
    },
    {
      "codigo_rpm": "SJU748",
      "circuito": "CIRCUITO_A",
      "ucp": "PRIMEGRID",
      "fecha": "2025-10-03",
      "flujo": "AE",
      "p1": -38.31, "p2": -37.05, "p3": -35.895, "p4": -34.86,
      "p5": -34.005, "p6": -32.625, "p7": -28.77, "p8": -27.315,
      "p9": -27.225, "p10": -27.48, "p11": -27.975, "p12": -29.76,
      "p13": -32.565, "p14": -35.595, "p15": -36.255, "p16": -33.885,
      "p17": -30.735, "p18": -27.525, "p19": -25.89, "p20": -26.295,
      "p21": -25.98, "p22": -25.74, "p23": -31.8, "p24": -38.775
    }
  ],
  "n_max": 3
}
```

### Ejemplo 2: Python con requests

```python
import requests

url = "http://localhost:8000/factores/calculos/curvas-tipicas-circuitos"

payload = {
    "medidas": [
        {
            "codigo_rpm": f"RPM{i}",
            "circuito": "CIRCUITO_TEST",
            "ucp": "PRIMEGRID",
            "fecha": f"2025-10-0{i}",
            "flujo": "AE",
            **{f"p{j}": float(i * 10 + j) for j in range(1, 25)}
        }
        for i in range(1, 6)
    ],
    "n_max": 3
}

response = requests.post(url, json=payload)
data = response.json()

print(f"OK: {data['ok']}")
print(f"Circuitos procesados: {list(data['circuitos'].keys())}")

for circuito, resultado in data['circuitos'].items():
    print(f"\n{circuito}:")
    print(f"  Curvas seleccionadas: {resultado['n_curvas']}")
    print(f"  Suma de pesos: {sum(resultado['pesos'].values()):.10f}")
```

### Ejemplo 3: Python con TestClient (FastAPI)

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

response = client.post("/factores/calculos/curvas-tipicas-circuitos", json={
    "medidas": [...],
    "n_max": 8
})

assert response.status_code == 200
data = response.json()
assert data["ok"] is True
```

---

## Códigos de Error

| Código       | Descripción              | Solución                                                  |
| ------------- | ------------------------- | ---------------------------------------------------------- |
| **400** | Error de validación      | Verificar que `medidas` no esté vacío y `n_max >= 1` |
| **422** | Error de formato Pydantic | Verificar tipos de datos y formato de fecha                |
| **500** | Error interno             | Revisar logs del servidor                                  |

### Ejemplos de Errores

#### Error 400: Medidas vacías

```json
{
  "detail": "medidas no puede estar vacío"
}
```

#### Error 422: Fecha inválida

```json
{
  "detail": [
    {
      "loc": ["body", "medidas", 0, "fecha"],
      "msg": "fecha debe estar en formato YYYY-MM-DD",
      "type": "value_error"
    }
  ]
}
```

---

## Health Check

**URL:** `GET /factores/calculos/circuitos/health`

**Response:**

```json
{
  "ok": true,
  "service": "circuitos",
  "version": "1.0.0",
  "endpoint": "/factores/calculos/curvas-tipicas-circuitos"
}
```

---

## Notas Importantes

### Comportamiento con Valores en Cero

Si las curvas típicas seleccionadas tienen valores en 0 (por ejemplo, cuando se mezclan diferentes tipos de flujo), el resultado será:

- `promedio`: Todos los períodos en 0
- `pesos`: Distribución uniforme (0.04167 por período)
- `suma(pesos)`: Aproximadamente 1.0

**Esto es comportamiento esperado**, no un error. El algoritmo funciona correctamente.

### Recomendación para Producción

**Filtrar medidas antes de enviar:**

```python
# Filtrar por flujo específico
medidas_ae = [m for m in medidas_totales if m['flujo'] == 'AE']

# Enviar solo medidas homogéneas
response = requests.post(url, json={
    "medidas": medidas_ae,
    "n_max": 8
})
```

### Performance

- **Sin consultas a BD:** Tiempo de respuesta depende solo de cómputo
- **Estimado:** < 100ms para 1000 medidas
- **Límite práctico:** ~10,000 medidas por request

---

## Testing

El endpoint incluye tests completos:

```bash
# Unit tests
pytest tests/test_circuitos_service.py -v

# API tests
pytest tests/test_circuitos_endpoint.py -v

# Integration test con datos reales
python test_circuitos_primegrid.py
```

---

## Documentación Interactiva

Acceder a la documentación Swagger en:

```
http://localhost:8000/docs
```

Buscar el endpoint: **POST /factores/calculos/curvas-tipicas-circuitos**
