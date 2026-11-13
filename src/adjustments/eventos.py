"""
Eventos sugeridos por IA para una semana (F06, sub-paso 4D).

Consulta a OpenAI con búsqueda web (`get_future_events_from_openai` del API
delega aquí) y cachea el resultado en disco por `(UCP, semana)` para no repetir
la llamada. Es una fuente **no verificada**: solo alimenta alertas informativas
de `steps/control.py`; nunca modifica curvas ni bloqueos.

Contrato
--------
- `consultar_openai(ucp, inicio, fin, *, client=None)`: síncrona; replica el
  prompt y la llamada `client.responses.create(model="gpt-5-mini",
  tools=[{"type": "web_search"}])`; limpia markdown y parsea JSON. Sin
  `API_KEY` (o sin `openai`) devuelve `{}`. `client` es inyectable para tests.
- `obtener_eventos(ucp, inicio, fin, *, consultar=None, cache_dir=None,
  ttl_dias=7, ttl_vacios_dias=1)`: asíncrona; cache en `.cache/eventos_ajustes`
  (o `AJUSTES_EVENTOS_CACHE_DIR`, si está vacío se usa el default) con JSON
  `{ucp, inicio, fin, generado, eventos}`, con clave exacta `(ucp, inicio, fin)`
  y archivo `{ucp}_{inicio}_{fin}.json`. Una respuesta vacía también se cachea,
  pero con `ttl_vacios_dias` (corto, reintenta pronto). Cualquier fallo del
  cliente devuelve la cache vieja si existe o `{}`, sin lanzar; un fallo al
  guardar la cache no descarta los eventos frescos. La llamada síncrona se
  ejecuta con `asyncio.to_thread`.
- `eventos_de_semana(ucp, lunes, ...)`: rango lunes..domingo.

Saneo: claves `YYYY-MM-DD` válidas, valores `str` no vacíos truncados a 120
caracteres y solo fechas dentro de `[inicio, fin]` (sin margen en la salida).
No importa fastapi.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.adjustments.types import parse_date

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR_DEFAULT = ROOT / ".cache" / "eventos_ajustes"
ENV_CACHE_DIR = "AJUSTES_EVENTOS_CACHE_DIR"
MAX_LEN_EVENTO = 120
_MODELO = "gpt-5-mini"
_RE_FECHA = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RE_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


# ---------------------------------------------------------------------------
# Consulta síncrona a OpenAI
# ---------------------------------------------------------------------------

def _prompt(ucp: str, inicio: str, fin: str) -> str:
    return f"""Busca en internet todos los eventos que podrían afectar la demanda de energía eléctrica en {ucp}, Colombia entre {inicio} y {fin}.

Incluye:
- Festivos nacionales y locales
- Eventos masivos (conciertos, festivales, ferias)
- Partidos de fútbol importantes
- Eventos climáticos previstos
- Paros o manifestaciones programadas
- Elecciones o eventos políticos
- Cualquier otro evento relevante

IMPORTANTE: Responde ÚNICAMENTE en formato JSON así:
{{
  "YYYY-MM-DD": "Nombre breve del evento",
  "YYYY-MM-DD": "Otro evento"
}}

REGLAS:
- NO incluyas links, URLs ni referencias a fuentes
- Solo nombres cortos y concisos (máximo 50 caracteres)
- NO agregues explicaciones adicionales
- Solo el JSON puro"""


def _limpiar_markdown(texto: str) -> str:
    """Quita cercas de código (```json ... ```) que a veces agrega el modelo."""
    t = (texto or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _iso(valor: Any) -> Optional[str]:
    d = _fecha(valor)
    return d.isoformat() if d else None


def _fecha(valor: Any) -> Optional[date]:
    try:
        return parse_date(valor)
    except (TypeError, ValueError):
        return None


def consultar_openai(
    ucp: str,
    inicio: Any,
    fin: Any,
    *,
    client: Any = None,
) -> Dict[str, Any]:
    """
    Consulta OpenAI (web search) y devuelve `{fecha ISO: evento}` crudo.

    Sin `client` exige `API_KEY` en el entorno; si falta (o `openai` no está
    instalado) devuelve `{}`. Nunca lanza: cualquier error → `{}`.
    """
    inicio_s, fin_s = _iso(inicio), _iso(fin)
    if not inicio_s or not fin_s:
        return {}

    if client is None:
        api_key = os.getenv("API_KEY")
        if not api_key:
            logger.warning("API_KEY no encontrada; se omiten los eventos sugeridos por IA")
            return {}
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - depende del entorno
            logger.warning("openai no disponible; se omiten los eventos: %s", exc)
            return {}
        try:
            client = OpenAI(api_key=api_key)
        except Exception as exc:
            logger.warning("No se pudo crear el cliente OpenAI: %s", exc)
            return {}

    try:
        response = client.responses.create(
            model=_MODELO,
            input=[{"role": "user", "content": _prompt(ucp, inicio_s, fin_s)}],
            tools=[{"type": "web_search"}],
        )
        texto = _limpiar_markdown(getattr(response, "output_text", "") or "")
        if not texto:
            return {}
        datos = json.loads(texto)
        return dict(datos) if isinstance(datos, dict) else {}
    except Exception as exc:
        logger.warning("Error consultando eventos en OpenAI: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Saneo y cache en disco
# ---------------------------------------------------------------------------

def _sanear(eventos: Any, inicio: date, fin: date) -> Dict[str, str]:
    """Solo `YYYY-MM-DD` válidos, valores str ≤ 120 y fechas dentro de [inicio, fin]."""
    if not isinstance(eventos, dict):
        return {}
    limpio: Dict[str, str] = {}
    for clave, valor in eventos.items():
        if not isinstance(clave, str) or not _RE_FECHA.match(clave):
            continue
        try:
            d = date.fromisoformat(clave)
        except ValueError:
            continue
        if not (inicio <= d <= fin):
            continue
        if not isinstance(valor, str):
            continue
        texto = valor.strip()
        if not texto:
            continue
        limpio[d.isoformat()] = texto[:MAX_LEN_EVENTO]
    return dict(sorted(limpio.items()))


def _slug(valor: Any) -> str:
    s = _RE_SLUG.sub("_", str(valor or "")).strip("._-")
    return (s or "ucp")[:80]


def _dir_cache(cache_dir: Any = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    return Path(os.getenv(ENV_CACHE_DIR) or str(CACHE_DIR_DEFAULT))


def _ruta_cache(ucp: str, inicio: date, fin: date, cache_dir: Any = None) -> Path:
    return _dir_cache(cache_dir) / f"{_slug(ucp)}_{inicio.isoformat()}_{fin.isoformat()}.json"


def _leer_cache(ruta: Path, ucp: str, inicio: date, fin: date) -> Optional[Dict[str, Any]]:
    """Contenido de la cache si es del mismo (UCP, inicio, fin); None si no sirve."""
    try:
        if not ruta.is_file():
            return None
        data = json.loads(ruta.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Cache de eventos ilegible (%s): %s", ruta, exc)
        return None
    if not isinstance(data, dict):
        return None
    if (str(data.get("ucp")) != str(ucp)
            or str(data.get("inicio")) != inicio.isoformat()
            or str(data.get("fin")) != fin.isoformat()):
        return None
    return data


def _fresca(data: Dict[str, Any], ttl_dias: Optional[float]) -> bool:
    if ttl_dias is None:
        return True
    try:
        generado = datetime.fromisoformat(str(data.get("generado")))
    except (TypeError, ValueError):
        return False
    if generado.tzinfo is None:
        generado = generado.replace(tzinfo=timezone.utc)
    edad = datetime.now(timezone.utc) - generado
    return edad <= timedelta(days=float(ttl_dias))


def _escribir_cache(ruta: Path, ucp: str, inicio: date, fin: date,
                    eventos: Dict[str, str]) -> None:
    data = {
        "ucp": str(ucp),
        "inicio": inicio.isoformat(),
        "fin": fin.isoformat(),
        "generado": datetime.now(timezone.utc).isoformat(),
        "eventos": eventos,
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".tmp")
    temporal.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(temporal, ruta)


# ---------------------------------------------------------------------------
# API asíncrona
# ---------------------------------------------------------------------------

async def obtener_eventos(
    ucp: str,
    inicio: Any,
    fin: Any,
    *,
    consultar: Optional[Callable[[str, str, str], Any]] = None,
    cache_dir: Any = None,
    ttl_dias: Optional[float] = 7,
    ttl_vacios_dias: Optional[float] = 1,
) -> Dict[str, str]:
    """
    Eventos de `[inicio, fin]` con cache exacta por `(ucp, inicio, fin)`.

    Si la cache está fresca se reutiliza (un resultado vacío usa
    `ttl_vacios_dias`, más corto, para permitir reintentar pronto). Si no, se
    llama `consultar` (default `consultar_openai`) en un hilo. Un fallo de la
    consulta devuelve la cache vieja o `{}`; un fallo al guardar no descarta los
    eventos obtenidos. Solo deja warnings en el log.
    """
    inicio_d, fin_d = _fecha(inicio), _fecha(fin)
    if inicio_d is None or fin_d is None or fin_d < inicio_d:
        return {}
    ruta = _ruta_cache(ucp, inicio_d, fin_d, cache_dir)
    previa = _leer_cache(ruta, ucp, inicio_d, fin_d)

    if previa is not None:
        ttl = ttl_vacios_dias if not previa.get("eventos") else ttl_dias
        if _fresca(previa, ttl):
            return _sanear(previa.get("eventos"), inicio_d, fin_d)

    consultar = consultar or consultar_openai
    try:
        crudo = await asyncio.to_thread(
            consultar, ucp, inicio_d.isoformat(), fin_d.isoformat()
        )
    except Exception as exc:
        logger.warning("No se pudieron obtener eventos para %s (%s..%s): %s",
                       ucp, inicio_d, fin_d, exc)
        if previa is not None:
            return _sanear(previa.get("eventos"), inicio_d, fin_d)
        return {}

    eventos = _sanear(crudo, inicio_d, fin_d)
    try:
        _escribir_cache(ruta, ucp, inicio_d, fin_d, eventos)
    except Exception as exc:
        logger.warning("No se pudo guardar la cache de eventos (%s): %s", ruta, exc)
    return eventos


async def eventos_de_semana(
    ucp: str,
    lunes: Any,
    *,
    consultar: Optional[Callable[[str, str, str], Any]] = None,
    cache_dir: Any = None,
    ttl_dias: Optional[float] = 7,
) -> Dict[str, str]:
    """Eventos de la semana que empieza en `lunes` (lunes..domingo)."""
    inicio = _fecha(lunes)
    if inicio is None:
        return {}
    return await obtener_eventos(
        ucp,
        inicio,
        inicio + timedelta(days=6),
        consultar=consultar,
        cache_dir=cache_dir,
        ttl_dias=ttl_dias,
    )
