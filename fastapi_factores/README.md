# FastAPI Factores

Backend en FastAPI para el modulo de Factores (barras, agrupaciones, medidas y utilidades) usando la misma base de datos PostgreSQL.

## Requisitos

- Python 3.10+
- PostgreSQL accesible desde el host

## Configuracion

Definir `DATABASE_URL` con el DSN de PostgreSQL:

```bash
export DATABASE_URL="postgresql://usuario:password@host:puerto/base"
```

## Instalacion

```bash
pip install -r requirements.txt
```

## Ejecutar

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Notas

- Las consultas y operaciones se basan en `App_Code/Factores.cs` y los WebMethods de `modulofactores.aspx.cs`.
- El endpoint `/factores/medidas/faltantes` reemplaza la logica de `modalinicio`.
- El endpoint `/factores/medidas/marcar` reemplaza `ActualizarMedidas` (solo actualiza en BD).

## Endpoints principales

- Barras: `/factores/barras`, `/factores/barras/por-mc/{mc}`
- Agrupaciones: `/factores/agrupaciones`, `/factores/agrupaciones/por-barra/{id}`
- Medidas: `/factores/medidas`, `/factores/medidas/completo`, `/factores/medidas/calcular-completo`
- Rangos: `/factores/rangos`
- Utilidades: `/factores/tipo-dia/{nombre}`, `/factores/festivos`
- **Cálculos:** `/factores/calculos/clustering`, `/factores/calculos/curvas-tipicas`, `/factores/calculos/fda`, `/factores/calculos/fdp`
- **Circuitos (nuevo):** `/factores/calculos/curvas-tipicas-circuitos` - Ver [CIRCUITOS_API_USAGE.md](CIRCUITOS_API_USAGE.md)

Consulta el OpenAPI en `/docs` para ver todos los parametros.

## Módulo de Circuitos

El módulo de circuitos (`/app/circuitos/`) procesa curvas típicas agrupadas por circuito **sin consultar base de datos**.

**Características:**
- Recibe medidas directamente en el request (JSON)
- Agrupa por circuito y selecciona N curvas más típicas
- Calcula promedio y pesos normalizados (suma = 1.0)
- Independiente del módulo de factores existente

**Documentación completa:** [CIRCUITOS_API_USAGE.md](CIRCUITOS_API_USAGE.md)

**Ejemplo rápido:**
```bash
curl -X POST "http://localhost:8000/factores/calculos/curvas-tipicas-circuitos" \
  -H "Content-Type: application/json" \
  -d '{
    "medidas": [
      {
        "codigo_rpm": "RPM1",
        "circuito": "CIRCUITO_A",
        "ucp": "PRIMEGRID",
        "fecha": "2025-10-02",
        "flujo": "AE",
        "p1": 10.0, "p2": 20.0, ..., "p24": 30.0
      }
    ],
    "n_max": 8
  }'
```
