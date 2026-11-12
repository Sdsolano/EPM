# Scraping XM IDO 


## Diagnóstico del sitio

| Característica | Valor |
|---|---|
| Framework | ASP.NET WebForms (ASPX) |
| Renderizado | Server-side + AJAX dinámico (UpdatePanel) |
| Autenticación | No requerida (datos públicos) |
| URL base | `https://ido.xm.com.co` |
| Problema principal | Los datos se inyectan via AJAX tras cargar la página; `requests` solo ve HTML vacío |

### URLs relevantes

```
# IDO completo (incluye sección Demanda No Atendida Versión)
https://ido.xm.com.co/Views/Ido/ido

# Eventos DNA no programada (filtrable por fecha)
https://ido.xm.com.co/Views/Eventos/demanda.aspx?q=noprogramada

# Eventos DNA programada
https://ido.xm.com.co/Views/Eventos/demanda.aspx?q=programada

# DNA acumulada
https://ido.xm.com.co/Views/Eventos/demanda.aspx?q=acumulada

# Limitación de suministro
https://ido.xm.com.co/Views/Eventos/demanda.aspx?q=limitacion
```

### Columnas de datos disponibles

| Campo | Descripción |
|---|---|
| Área | Zona/región afectada |
| Fecha inicial | Inicio del evento DNA |
| Fecha final | Fin del evento DNA |
| MWh | Energía no atendida |
| Descripción | Causa o descripción del evento |
| Subestación | Subestación involucrada |
| Municipio | Municipio afectado |

---

## Comparación de herramientas

| Herramienta | JavaScript | POST ASPX | Velocidad | Dependencia | Recomendado para |
|---|---|---|---|---|---|
| `requests` solo | ❌ | ✅ (con `__VIEWSTATE`) | Muy rápido | Incluida | Solo si el POST devuelve HTML completo |
| `requests` + `BeautifulSoup` | ❌ | ✅ | Rápido | Ligera (`beautifulsoup4`) | Parsear HTML estático post-POST |
| **`Playwright`** | ✅ | ✅ | Medio | Pesada (Chromium) | **Recomendado — maneja AJAX y UpdatePanel** |
| `Selenium` | ✅ | ✅ | Lento | Pesada (WebDriver) | Alternativa a Playwright |
| `mechanize` | ❌ | ✅ | Rápido | Ligera | Formularios WebForms sin JS |
| `httpx` + `parsel` | ❌ | ✅ | Rápido | Ligera | Igual que requests+BS pero async |

**Veredicto:** Playwright es la única opción que garantiza ver los datos porque el sitio usa `UpdatePanel` (AJAX de ASP.NET) para inyectar las filas de la tabla sin recargar la página.

---

## Opción A — Playwright (recomendada)

### 1. Instalación

```bash
pip install playwright beautifulsoup4
playwright install chromium
```

### 2. Script base para extraer DNA

```python
import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from datetime import date

async def get_demanda_no_atendida(
    fecha_inicio: str,   # "YYYY-MM-DD"
    fecha_fin: str,      # "YYYY-MM-DD"
    tipo: str = "noprogramada"  # noprogramada | programada | acumulada | limitacion
) -> list[dict]:
    """
    Extrae eventos de Demanda No Atendida del portal XM IDO.

    Returns:
        Lista de dicts con keys: area, fecha_inicial, fecha_final,
        mwh, descripcion, subestacion, municipio
    """
    url = f"https://ido.xm.com.co/Views/Eventos/demanda.aspx?q={tipo}"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Navegar a la página
        await page.goto(url, wait_until="networkidle")

        # Llenar fecha inicial
        await page.fill('input[id*="fechaInicial"], input[name*="fechaInicial"]', fecha_inicio)

        # Llenar fecha final
        await page.fill('input[id*="fechaFinal"], input[name*="fechaFinal"]', fecha_fin)

        # Hacer clic en el botón de búsqueda/filtrar
        await page.click('input[type="button"][value*="Buscar"], button:has-text("Buscar")')

        # Esperar a que el UpdatePanel termine (AJAX)
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(2)  # margen extra para UpdatePanel

        # Extraer HTML de la tabla de resultados
        html = await page.content()
        await browser.close()

    return _parse_dna_table(html)


def _parse_dna_table(html: str) -> list[dict]:
    """Parsea la tabla de DNA del HTML renderizado."""
    soup = BeautifulSoup(html, "html.parser")

    # La tabla de resultados suele tener class 'grid' o 'GridView'
    tabla = soup.find("table", class_=lambda c: c and ("grid" in c.lower() or "gridview" in c.lower()))
    if not tabla:
        # Fallback: buscar la tabla con más filas
        tablas = soup.find_all("table")
        tabla = max(tablas, key=lambda t: len(t.find_all("tr")), default=None)

    if not tabla:
        return []

    filas = tabla.find_all("tr")[1:]  # saltar header
    resultados = []

    for fila in filas:
        celdas = [td.get_text(strip=True) for td in fila.find_all("td")]
        if len(celdas) >= 5:
            resultados.append({
                "area": celdas[0],
                "fecha_inicial": celdas[1],
                "fecha_final": celdas[2],
                "mwh": celdas[3],
                "descripcion": celdas[4],
                "subestacion": celdas[5] if len(celdas) > 5 else "",
                "municipio": celdas[6] if len(celdas) > 6 else "",
            })

    return resultados


# Uso:
if __name__ == "__main__":
    eventos = asyncio.run(get_demanda_no_atendida("2025-01-15", "2025-01-15"))
    for e in eventos:
        print(e)
```

### 3. Inspeccionar los selectores correctos (paso crítico)

Los `id` de los inputs cambian entre versiones ASP.NET. Para encontrar los exactos:

1. Abrir Chrome → ir a la URL de eventos
2. `F12` → Inspector → click en el campo de fecha
3. Copiar el `id` o `name` del `<input>`
4. Hacer la búsqueda y en la pestaña **Network** filtrar por `XHR`
5. Ver el `__VIEWSTATE` en el payload del POST resultante

```python
# Para depurar selectores con Playwright:
async with async_playwright() as p:
    browser = await p.chromium.launch(headless=False)  # visible
    page = await browser.new_page()
    await page.goto("https://ido.xm.com.co/Views/Eventos/demanda.aspx?q=noprogramada")
    await page.pause()  # abre el inspector de Playwright
```

---

## Opción B — requests + BeautifulSoup (si el POST funciona sin JS)

Funciona solo si ASP.NET devuelve la tabla completa en la respuesta del POST (no siempre es el caso con UpdatePanel).

### 1. Instalación

```bash
pip install requests beautifulsoup4
```

### 2. Script

```python
import requests
from bs4 import BeautifulSoup

def get_dna_via_post(fecha_inicio: str, fecha_fin: str) -> list[dict]:
    session = requests.Session()
    url = "https://ido.xm.com.co/Views/Eventos/demanda.aspx?q=noprogramada"

    # Paso 1: GET para obtener __VIEWSTATE
    r = session.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    viewstate = soup.find("input", {"name": "__VIEWSTATE"})
    eventval = soup.find("input", {"name": "__EVENTVALIDATION"})
    viewstategenerator = soup.find("input", {"name": "__VIEWSTATEGENERATOR"})

    # Paso 2: POST con los parámetros de fecha
    # NOTA: los nombres exactos de los campos deben verificarse con DevTools
    payload = {
        "__VIEWSTATE": viewstate["value"] if viewstate else "",
        "__EVENTVALIDATION": eventval["value"] if eventval else "",
        "__VIEWSTATEGENERATOR": viewstategenerator["value"] if viewstategenerator else "",
        "__EVENTTARGET": "",
        "__EVENTARGUMENT": "",
        "ctl00$ContentPlaceHolder1$txtFechaInicial": fecha_inicio,
        "ctl00$ContentPlaceHolder1$txtFechaFinal": fecha_fin,
        "ctl00$ContentPlaceHolder1$btnBuscar": "Buscar",
    }

    r2 = session.post(url, data=payload)
    return _parse_dna_table(r2.text)
```

> **Advertencia:** Si el sitio usa `ScriptManager` con `UpdatePanel`, el POST devuelve un payload especial (`|updatePanel|...|`) que necesita parsing adicional, no HTML puro.

### 3. Parsear respuesta de UpdatePanel (si aplica)

```python
def parse_update_panel_response(raw: str) -> str:
    """Extrae el HTML del bloque UpdatePanel de la respuesta AJAX."""
    # Formato: longitud|tipo|id|contenido|
    partes = raw.split("|")
    html_chunks = []
    i = 0
    while i < len(partes) - 3:
        if partes[i + 1] == "updatePanel":
            longitud = int(partes[i])
            contenido = partes[i + 3][:longitud]
            html_chunks.append(contenido)
        i += 1
    return "\n".join(html_chunks)
```

---

## Integración con el endpoint `/analyze-deviation`

Una vez que tengas la función `get_demanda_no_atendida`, intégrala en `analyze_deviation_with_openai` en `src/api/main.py`:

```python
async def analyze_deviation_with_openai(ucp, fecha, mape):
    # 1. Obtener datos DNA de XM IDO
    try:
        dna_eventos = await asyncio.to_thread(
            get_dna_via_playwright,  # o get_dna_via_post
            fecha_inicio=fecha,
            fecha_fin=fecha
        )
        if dna_eventos:
            dna_texto = "\n".join([
                f"- {e['area']}: {e['mwh']} MWh no atendidos "
                f"({e['fecha_inicial']} → {e['fecha_final']}) "
                f"| {e['descripcion']} | {e['subestacion']}"
                for e in dna_eventos
            ])
        else:
            dna_texto = "No se encontraron eventos de DNA en XM IDO para esta fecha."
    except Exception as ex:
        dna_texto = f"No fue posible consultar XM IDO: {ex}"

    # 2. Construir prompt con datos reales
    prompt = f"""...
**Datos reales de Demanda No Atendida (XM IDO):**
{dna_texto}

**Tarea:** Analiza si los eventos DNA encontrados explican el desvío...
"""
```

---

## Paso a paso para identificar los selectores correctos

Este es el paso más importante antes de escribir el scraper definitivo.

1. Abrir `https://ido.xm.com.co/Views/Eventos/demanda.aspx?q=noprogramada` en Chrome
2. Abrir DevTools → pestaña **Network** → filtrar `XHR` o `Fetch`
3. En la página, llenar las fechas y hacer clic en **Buscar**
4. En Network verás una petición POST o la misma URL con payload
5. Hacer clic en esa petición → pestaña **Payload** → copiar los nombres de los campos
6. Pestaña **Response** → ver si devuelve HTML puro o el formato UpdatePanel (`|updatePanel|...`)
7. Si es UpdatePanel → usar Playwright (Opción A)
8. Si es HTML puro → usar requests+BS4 (Opción B, más simple)

```
Checklist campo a campo:
[ ] Nombre del campo fecha inicial (ej: ctl00$CP1$txtFechaInicial)
[ ] Nombre del campo fecha final
[ ] Nombre del botón submit
[ ] ¿La respuesta es HTML o UpdatePanel?
[ ] ¿Hay ScriptManager activo en la página?
[ ] ¿Los datos están en un GridView o tabla simple?
```

---

## Dependencias a agregar en `requirements.txt`

```txt
# Para Opción A (recomendada):
playwright>=1.40.0
beautifulsoup4>=4.12.0

# Para Opción B:
beautifulsoup4>=4.12.0
# requests ya está incluido
```

Después de instalar Playwright, ejecutar una vez:
```bash
playwright install chromium
```
