"""
Buscador de días análogos (R3): días históricos parecidos a un día objetivo.

Funciones puras (sin red, sin escribir archivos). Las usan el endpoint
`POST /adjustments/analogos` (router) y los pasos forma/nivel/referencia
(`seleccionar`, `puntuar_fechas`).

Descriptor de un día (`descriptor_dia`)
---------------------------------------
- `dia_semana` (0=lunes .. 6=domingo), `mes`, `anio`, `dia_del_anio`.
- `festivo`: la fecha está en el set de festivos (de la UCP / `ctx.festivos`).
- Relación con festivo (`relacion_festivo`), en este orden de prioridad:
  1. `festivo`: el día es festivo.
  2. `puente`: el día no es festivo y pertenece a un bloque contiguo de días no
     laborables (sábado, domingo, festivo) que contiene al menos un festivo
     (p. ej. sábado y domingo antes de un lunes festivo), o es un día laboral
     entre dos días no laborables con al menos un festivo adyacente ("sándwich").
  3. `antes` / `despues`: hay un festivo a <= `MAX_DIST_FESTIVO` (3) días de
     calendario; se usa el más cercano y en empate gana `despues`.
     `distancia_festivo` = días a ese festivo (1..3).
  4. `ninguna` (`distancia_festivo` = None).
  `festivo_relacionado` = fecha ISO del festivo usado (None en `ninguna`).
- `tipo_festivo`: slug del festivo RELACIONADO (minúsculas, sin tildes, sin
  "(observado)", espacios → "_", p. ej. `dia_de_san_jose`). Jueves y Viernes
  Santo → `semana_santa`; festivo de la UCP sin nombre en la librería
  `holidays` → `festivo_ucp`. `nombre_festivo` = nombre (librería, español) del
  festivo relacionado (el propio si el día es festivo).
- `semana_santa`: Domingo de Ramos .. Domingo de Resurrección (Pascua por
  `dateutil.easter`).
- `temporada_escolar` (APROXIMADA, calendario A de Colombia, fechas fijas):
  `vacaciones` si 1..25 de enero, Semana Santa (lunes..domingo), 15 de junio ..
  10 de julio, la semana (lunes..domingo) del lunes festivo de octubre (receso
  escolar) o diciembre completo; si no, `clases`.
- `clima` (opcional): `{"temp_media", "lluvia_mm", "humedad_media"}` del día.

Clima
-----
`cargar_clima` lee `data/raw/{ucp}/clima_new.csv` (columnas `fecha, periodo,
p_t, p_h, p_v, p_i`; p_t temperatura °C, p_h humedad %, p_i precipitación mm).
El clima se usa TAL CUAL viene del archivo: solo se descartan valores no
finitos y, si un (fecha, periodo) está repetido, se promedia. Días con < 20
periodos válidos quedan sin clima. `temp_media` = media de p_t, `humedad_media`
= media de p_h, `lluvia_mm` = suma de p_i de los periodos presentes. El archivo
real trae valores raros (p_i en cientos de mm por hora al final del archivo):
no se corrigen aquí. "Con lluvia" = `lluvia_mm >= LLUVIA_UMBRAL_MM`.

Score (0..1, explicable)
------------------------
`score = Σ w·s / Σ w` sobre los criterios presentes, redondeado a 4 decimales:
- día de semana `PESOS["dia_semana"]` = 0.25: mismo día 1.0; ambos laborales
  (lunes a viernes) 0.4; si no 0.
- festivo / relación 0.25: misma relación → 1.0, salvo: `festivo` con distinto
  `tipo_festivo` 0.6; `antes`/`despues` con distinta distancia 0.6. Distinta
  relación 0.
- estacionalidad 0.20: `max(0, 1 − dist_días_del_año/60)` (distancia circular).
- recencia 0.10: `exp(−años/4)` (años entre objetivo y candidato).
- Semana Santa 0.05 y temporada escolar 0.05: 1 si coinciden.
- clima 0.10: solo si objetivo y candidato tienen temperatura:
  `max(0, 1 − |Δt|/3)`; si no, se omite y el resto se renormaliza.
`componentes` = aporte de cada criterio al score (`w·s/Σw`). `razones` = solo
los criterios que coinciden, en español legible.

Filtros (`validar_filtros`, ver docs/work/ajustes/fase2b/PLAN.md §2.1)
---------------------------------------------------------------------
Ausente o null = no filtra, salvo `anios_excluidos` (ausente/null → [2020];
`[]` = no excluye) y `limite` (ausente/null → 20; 1..366). Los candidatos son
los días entre el primer día del histórico y la fecha de corte (nunca el propio
objetivo); los que no tienen 24 periodos válidos salen con `observado: False`.

Rendimiento: el calendario de descriptores (por rango de años y festivos) y las
curvas observadas (por histórico) se cachean a nivel de módulo; la búsqueda es
vectorizada (numpy).
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import unicodedata
from collections import OrderedDict
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dateutil.easter import easter

from src.adjustments import data
from src.adjustments.steps._params import FECHA_MAX, FECHA_MIN, _corto, entero, lista_enteros, numero
from src.adjustments.types import parse_date, to_jsonable
from src.prediction.holiday_homology import colombia_holiday_map

PASO = "filtros"
MAX_DIST_FESTIVO = 3
MIN_PERIODOS_CLIMA = 20
LLUVIA_UMBRAL_MM = 1.0
VENTANA_ESTACIONAL_DIAS = 60.0
RECENCIA_ANIOS = 4.0
CLIMA_DELTA_T = 3.0
MAX_ITEMS_LISTA = 100
MAX_LEN_TIPO = 120

PESOS: Dict[str, float] = {
    "dia_semana": 0.25,
    "festivo": 0.25,
    "estacionalidad": 0.20,
    "recencia": 0.10,
    "semana_santa": 0.05,
    "temporada_escolar": 0.05,
    "clima": 0.10,
}

RELACIONES = ("festivo", "puente", "antes", "despues", "ninguna")
_REL_CODIGO = {r: i for i, r in enumerate(RELACIONES)}
TEMPORADAS = ("clases", "vacaciones")
TIPO_SEMANA_SANTA = "semana_santa"
TIPO_FESTIVO_UCP = "festivo_ucp"

DIAS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
MESES = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre")

FILTROS_DEFAULT: Dict[str, Any] = {
    "dia_semana": None,
    "meses": None,
    "temporada_dias": None,
    "relacion_festivo": None,
    "distancia_festivo": None,
    "tipo_festivo": None,
    "semana_santa": None,
    "temporada_escolar": None,
    "clima": None,
    "anio_min": None,
    "anio_max": None,
    "anios_excluidos": list(data.ANIOS_EXCLUIDOS_DEFAULT),
    "limite": 20,
}
_CLIMA_CLAVES = ("temp_tolerancia", "lluvia", "temp_media")

_EPOCH = date(1970, 1, 1)


def _ep(d: date) -> int:
    return (d - _EPOCH).days


def _de_ep(n: int) -> date:
    return _EPOCH + timedelta(days=int(n))


# ---------------------------------------------------------------------------
# Validación de filtros
# ---------------------------------------------------------------------------

def _bool_o_none(clave: str, v: Any) -> Optional[bool]:
    if v is None or isinstance(v, (bool, np.bool_)):
        return None if v is None else bool(v)
    raise ValueError(f"{PASO}.{clave} debe ser true, false o null, llegó {_corto(v)}")


def _lista_no_vacia(clave: str, v: Any, minimo: int, maximo: int) -> List[int]:
    valores = lista_enteros(PASO, clave, v, minimo=minimo, maximo=maximo, max_items=MAX_ITEMS_LISTA)
    if not valores:
        raise ValueError(f"{PASO}.{clave} no puede ser una lista vacía (use null para no filtrar)")
    return valores


def validar_filtros(filtros: Optional[dict]) -> dict:
    """
    Completa defaults y valida los filtros de búsqueda. Lanza ValueError (→ 422)
    ante claves desconocidas, tipos o rangos inválidos. Devuelve un dict nuevo
    con todas las claves de `FILTROS_DEFAULT` (los "auto" se conservan: se
    resuelven contra el día objetivo en `buscar_analogos`).
    """
    if filtros is None:
        filtros = {}
    if not isinstance(filtros, dict):
        raise ValueError(f"filtros debe ser un objeto, llegó {type(filtros).__name__}")
    desconocidas = sorted(str(k) for k in filtros if k not in FILTROS_DEFAULT)
    if desconocidas:
        raise ValueError(f"filtros desconocidos: {desconocidas}; válidos: {sorted(FILTROS_DEFAULT)}")
    f = {**FILTROS_DEFAULT, **filtros}
    out: Dict[str, Any] = {}

    v = f["dia_semana"]
    if v is None or v == "auto":
        out["dia_semana"] = v
    elif isinstance(v, (list, tuple)):
        out["dia_semana"] = _lista_no_vacia("dia_semana", v, 0, 6)
    else:
        out["dia_semana"] = [entero(PASO, "dia_semana", v, minimo=0, maximo=6)]

    out["meses"] = None if f["meses"] is None else _lista_no_vacia("meses", f["meses"], 1, 12)
    out["temporada_dias"] = (None if f["temporada_dias"] is None
                             else entero(PASO, "temporada_dias", f["temporada_dias"], minimo=1, maximo=183))

    v = f["relacion_festivo"]
    if v is not None and (not isinstance(v, str) or v not in ("auto", *RELACIONES)):
        raise ValueError(f"{PASO}.relacion_festivo debe ser una de {('auto', *RELACIONES)} o null, "
                         f"llegó {_corto(v)}")
    out["relacion_festivo"] = v

    v = f["distancia_festivo"]
    out["distancia_festivo"] = (v if v is None or v == "auto"
                                else entero(PASO, "distancia_festivo", v, minimo=1, maximo=7))

    v = f["tipo_festivo"]
    if v is None or v == "auto":
        out["tipo_festivo"] = v
    else:
        if not isinstance(v, str) or not v.strip() or len(v) > MAX_LEN_TIPO:
            raise ValueError(f"{PASO}.tipo_festivo debe ser un texto no vacío (<= {MAX_LEN_TIPO}), "
                             f"\"auto\" o null, llegó {_corto(v)}")
        slug = slug_festivo(v)
        if not slug:
            raise ValueError(f"{PASO}.tipo_festivo inválido: {_corto(v)}")
        out["tipo_festivo"] = slug

    out["semana_santa"] = _bool_o_none("semana_santa", f["semana_santa"])

    v = f["temporada_escolar"]
    if v is not None and (not isinstance(v, str) or v not in TEMPORADAS):
        raise ValueError(f"{PASO}.temporada_escolar debe ser una de {TEMPORADAS} o null, llegó {_corto(v)}")
    out["temporada_escolar"] = v

    out["clima"] = _validar_clima(f["clima"])

    for clave in ("anio_min", "anio_max"):
        out[clave] = (None if f[clave] is None
                      else entero(PASO, clave, f[clave], minimo=FECHA_MIN.year, maximo=FECHA_MAX.year))
    if out["anio_min"] is not None and out["anio_max"] is not None and out["anio_min"] > out["anio_max"]:
        raise ValueError(f"{PASO}.anio_min ({out['anio_min']}) no puede ser mayor que anio_max ({out['anio_max']})")

    v = f["anios_excluidos"]
    out["anios_excluidos"] = (list(data.ANIOS_EXCLUIDOS_DEFAULT) if v is None else
                              lista_enteros(PASO, "anios_excluidos", v, minimo=FECHA_MIN.year,
                                            maximo=FECHA_MAX.year, max_items=MAX_ITEMS_LISTA))
    out["limite"] = (FILTROS_DEFAULT["limite"] if f["limite"] is None
                     else entero(PASO, "limite", f["limite"], minimo=1, maximo=366))
    return out


def _validar_clima(v: Any) -> Optional[Dict[str, Any]]:
    if v is None:
        return None
    if not isinstance(v, dict):
        raise ValueError(f"{PASO}.clima debe ser un objeto o null, llegó {_corto(v)}")
    desconocidas = sorted(str(k) for k in v if k not in _CLIMA_CLAVES)
    if desconocidas:
        raise ValueError(f"{PASO}.clima: claves desconocidas {desconocidas}; válidas: {list(_CLIMA_CLAVES)}")
    tol = v.get("temp_tolerancia")
    temp = v.get("temp_media")
    return {
        "temp_tolerancia": (None if tol is None else numero(PASO, "clima.temp_tolerancia", tol, minimo=0.0,
                                                             maximo=50.0, minimo_exclusivo=True)),
        "lluvia": _bool_o_none("clima.lluvia", v.get("lluvia")),
        "temp_media": (None if temp is None else numero(PASO, "clima.temp_media", temp, minimo=-50.0,
                                                         maximo=60.0)),
    }


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------

def slug_festivo(nombre: Optional[str]) -> Optional[str]:
    """Slug de un nombre de festivo (sin tildes, sin "(observado)"); Jueves/Viernes Santo → semana_santa."""
    if nombre is None:
        return None
    s = unicodedata.normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"\((observado|observed)\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if not s:
        return None
    if s in (TIPO_SEMANA_SANTA,) or "jueves_santo" in s or "viernes_santo" in s:
        return TIPO_SEMANA_SANTA
    return s


def _nombre_festivo(d: date) -> Optional[str]:
    return colombia_holiday_map(d.year).get(d.isoformat())


def _tipo_festivo(d: date) -> str:
    return slug_festivo(_nombre_festivo(d)) or TIPO_FESTIVO_UCP


@lru_cache(maxsize=256)
def _calendario_anio(anio: int) -> Tuple[date, date, Optional[date]]:
    """(domingo de Ramos, domingo de Pascua, lunes festivo de octubre o None) del año."""
    pascua = easter(anio)
    octubre = sorted(date.fromisoformat(k) for k in colombia_holiday_map(anio)
                     if k[5:7] == "10" and date.fromisoformat(k).weekday() == 0)
    return pascua - timedelta(days=7), pascua, (octubre[0] if octubre else None)


def es_semana_santa(d: date) -> bool:
    ramos, pascua, _ = _calendario_anio(d.year)
    return ramos <= d <= pascua


def temporada_escolar(d: date) -> str:
    """Aproximación del calendario A de Colombia (ver docstring del módulo)."""
    _, pascua, lunes_oct = _calendario_anio(d.year)
    vacaciones = (
        d.month == 12
        or (d.month == 1 and d.day <= 25)
        or pascua - timedelta(days=6) <= d <= pascua
        or date(d.year, 6, 15) <= d <= date(d.year, 7, 10)
        or (lunes_oct is not None and lunes_oct <= d <= lunes_oct + timedelta(days=6))
    )
    return "vacaciones" if vacaciones else "clases"


def _relacion(d: date, festivos: set) -> Tuple[str, Optional[int], Optional[date]]:
    """(relacion_festivo, distancia_festivo, festivo_relacionado). Ver docstring del módulo."""

    def fest(x: date) -> bool:
        return x.isoformat() in festivos

    def no_laborable(x: date) -> bool:
        return x.weekday() >= 5 or fest(x)

    if fest(d):
        return "festivo", None, d
    if no_laborable(d):  # fin de semana: bloque contiguo de no laborables
        izq = d
        while no_laborable(izq - timedelta(days=1)) and (d - izq).days < 14:
            izq -= timedelta(days=1)
        der = d
        while no_laborable(der + timedelta(days=1)) and (der - d).days < 14:
            der += timedelta(days=1)
        en_bloque = [izq + timedelta(days=k) for k in range((der - izq).days + 1)]
        festivos_bloque = [x for x in en_bloque if fest(x)]
        if festivos_bloque:
            # el más cercano; empate → el anterior (consistente con "despues")
            rel = min(festivos_bloque, key=lambda x: (abs((x - d).days), x > d))
            return "puente", None, rel
    else:
        antes, despues = d - timedelta(days=1), d + timedelta(days=1)
        if no_laborable(antes) and no_laborable(despues) and (fest(antes) or fest(despues)):
            return "puente", None, antes if fest(antes) else despues
    for k in range(1, MAX_DIST_FESTIVO + 1):
        if fest(d - timedelta(days=k)):
            return "despues", k, d - timedelta(days=k)
        if fest(d + timedelta(days=k)):
            return "antes", k, d + timedelta(days=k)
    return "ninguna", None, None


def _descriptor_base(d: date, festivos: set) -> Dict[str, Any]:
    rel, dist, fest_rel = _relacion(d, festivos)
    return {
        "dia_semana": d.weekday(),
        "mes": d.month,
        "anio": d.year,
        "dia_del_anio": d.timetuple().tm_yday,
        "festivo": rel == "festivo",
        "relacion_festivo": rel,
        "distancia_festivo": dist,
        "festivo_relacionado": fest_rel.isoformat() if fest_rel else None,
        "tipo_festivo": _tipo_festivo(fest_rel) if fest_rel else None,
        "nombre_festivo": _nombre_festivo(fest_rel) if fest_rel else None,
        "semana_santa": es_semana_santa(d),
        "temporada_escolar": temporada_escolar(d),
    }


def _a_set(festivos: Iterable[str]) -> set:
    if isinstance(festivos, (set, frozenset)):
        return festivos
    return {str(f)[:10] for f in festivos or ()}


def descriptor_dia(d: date, festivos: Iterable[str], clima: Optional[dict] = None) -> dict:
    """
    Descriptor de `d` (ver docstring del módulo). `festivos`: ISO de la UCP.
    `clima`: mapa {iso: {...}} (salida de `clima_diario` / `cargar_clima`); la
    clave `clima` del descriptor es el clima de ese día o None.
    """
    d = parse_date(d)
    desc = dict(_descriptor_base(d, _a_set(festivos)))
    desc["clima"] = _clima_de(clima, d.isoformat())
    return desc


def _clima_de(clima: Optional[dict], iso_: str) -> Optional[dict]:
    if not clima:
        return None
    c = clima.get(iso_)
    return dict(c) if c else None


# ---------------------------------------------------------------------------
# Clima
# ---------------------------------------------------------------------------

def clima_diario(df_clima: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """{"YYYY-MM-DD": {"temp_media", "lluvia_mm", "humedad_media"}} desde el clima horario (tal cual)."""
    columnas = ["fecha", "periodo", "p_t", "p_h", "p_i"]
    if df_clima is None or df_clima.empty or any(c not in df_clima.columns for c in columnas):
        return {}
    df = df_clima[columnas].copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.strftime("%Y-%m-%d")
    for c in ("periodo", "p_t", "p_h", "p_i"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    finitos = np.isfinite(df[["periodo", "p_t", "p_h", "p_i"]].to_numpy(dtype=float)).all(axis=1)
    df = df[finitos & df["fecha"].notna() & df["periodo"].between(1, 24)]
    if df.empty:
        return {}
    por_periodo = df.groupby(["fecha", "periodo"], as_index=False)[["p_t", "p_h", "p_i"]].mean()
    g = por_periodo.groupby("fecha").agg(n=("periodo", "size"), temp_media=("p_t", "mean"),
                                         humedad_media=("p_h", "mean"), lluvia_mm=("p_i", "sum"))
    g = g[g["n"] >= MIN_PERIODOS_CLIMA]
    return {
        str(f): {"temp_media": float(r.temp_media), "lluvia_mm": float(r.lluvia_mm),
                 "humedad_media": float(r.humedad_media)}
        for f, r in g.iterrows()
    }


def cargar_clima(ucp: str, data_dir: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    """Clima diario de `data/raw/{ucp}/clima_new.csv` si existe; si no, {}. Solo lectura."""
    raiz = Path(data_dir or data.RAW_DATA_DIR).resolve()
    ruta = (raiz / str(ucp) / "clima_new.csv").resolve()
    if not ruta.is_relative_to(raiz) or not ruta.is_file():
        return {}
    return clima_diario(pd.read_csv(ruta))


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def _dist_circular(a: Any, b: Any) -> Any:
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    return np.minimum(d, 365.0 - d)


def _temp(desc: dict) -> Optional[float]:
    c = desc.get("clima") or {}
    t = c.get("temp_media")
    return float(t) if t is not None and math.isfinite(float(t)) else None


def puntuar(obj: dict, cand: dict) -> Tuple[float, Dict[str, float], List[str]]:
    """
    (score 0..1, componentes {criterio: aporte}, razones) de `cand` frente a
    `obj` (ambos descriptores de `descriptor_dia`). Misma aritmética que la
    búsqueda vectorizada.
    """
    s: Dict[str, float] = {}
    razones: List[str] = []
    mismo_dia = obj["dia_semana"] == cand["dia_semana"]
    s["dia_semana"] = 1.0 if mismo_dia else (0.4 if obj["dia_semana"] < 5 and cand["dia_semana"] < 5 else 0.0)
    if mismo_dia:
        razones.append(DIAS[cand["dia_semana"]])
    if obj["mes"] == cand["mes"]:
        razones.append(MESES[cand["mes"] - 1])

    rel = obj["relacion_festivo"]
    if rel != cand["relacion_festivo"]:
        s["festivo"] = 0.0
    elif rel == "festivo":
        s["festivo"] = 1.0 if obj.get("tipo_festivo") == cand.get("tipo_festivo") else 0.6
    elif rel in ("antes", "despues"):
        s["festivo"] = 1.0 if obj.get("distancia_festivo") == cand.get("distancia_festivo") else 0.6
    else:
        s["festivo"] = 1.0
    if rel == cand["relacion_festivo"] and rel != "ninguna":
        razones.append(_razon_festivo(cand))

    dist_doy = float(_dist_circular(obj["dia_del_anio"], cand["dia_del_anio"]))
    s["estacionalidad"] = float(np.maximum(0.0, 1.0 - dist_doy / VENTANA_ESTACIONAL_DIAS))
    anios = abs(_ep(_fecha_desc(obj)) - _ep(_fecha_desc(cand))) / 365.25
    s["recencia"] = float(np.exp(-anios / RECENCIA_ANIOS))
    s["semana_santa"] = 1.0 if obj["semana_santa"] == cand["semana_santa"] else 0.0
    if obj["semana_santa"] and cand["semana_santa"]:
        razones.append("Semana Santa")
    s["temporada_escolar"] = 1.0 if obj["temporada_escolar"] == cand["temporada_escolar"] else 0.0
    if s["temporada_escolar"]:
        razones.append("temporada de clases" if cand["temporada_escolar"] == "clases" else "vacaciones escolares")

    t_obj, t_cand = _temp(obj), _temp(cand)
    if t_obj is not None and t_cand is not None:
        s["clima"] = float(np.maximum(0.0, 1.0 - abs(t_obj - t_cand) / CLIMA_DELTA_T))
        if s["clima"] >= 0.5:
            razones.append(f"temperatura similar ({t_cand:.1f} °C)")
    lluvia_obj = (obj.get("clima") or {}).get("lluvia_mm")
    lluvia_cand = (cand.get("clima") or {}).get("lluvia_mm")
    if lluvia_obj is not None and lluvia_cand is not None and lluvia_obj >= LLUVIA_UMBRAL_MM \
            and lluvia_cand >= LLUVIA_UMBRAL_MM:
        razones.append("con lluvia")

    total_w = sum(PESOS[k] for k in s)
    acumulado = 0.0
    for k in _ORDEN_SUMA:
        if k in s:
            acumulado = acumulado + PESOS[k] * s[k]
    score = round(acumulado / total_w, 4)
    componentes = {k: round(PESOS[k] * s[k] / total_w, 4) for k in s}
    return score, componentes, razones


_ORDEN_SUMA = ("dia_semana", "festivo", "estacionalidad", "recencia", "semana_santa", "temporada_escolar", "clima")


def _fecha_desc(desc: dict) -> date:
    f = desc.get("fecha")
    if f is not None:
        return parse_date(f)
    return date(int(desc["anio"]), 1, 1) + timedelta(days=int(desc["dia_del_anio"]) - 1)


def _razon_festivo(cand: dict) -> str:
    rel, dist, f = cand["relacion_festivo"], cand.get("distancia_festivo"), cand.get("festivo_relacionado")
    if rel == "festivo":
        return f"festivo: {cand.get('nombre_festivo') or 'festivo de la UCP'}"
    if rel == "puente":
        return f"puente festivo ({f})"
    palabra = "después" if rel == "despues" else "antes"
    if dist == 1:
        return f"día {palabra} de festivo ({f})"
    return f"{dist} días {palabra} de festivo ({f})"


# ---------------------------------------------------------------------------
# Índice (cacheado) y búsqueda
# ---------------------------------------------------------------------------

class _Calendario:
    """Descriptores base de todos los días de [anio0, anio1] para un set de festivos."""

    def __init__(self, anio0: int, anio1: int, festivos: set) -> None:
        self.inicio = _ep(date(anio0, 1, 1))
        fin = date(anio1, 12, 31)
        n = _ep(fin) - self.inicio + 1
        self.descs: List[Dict[str, Any]] = []
        for k in range(n):
            d = _de_ep(self.inicio + k)
            desc = _descriptor_base(d, festivos)
            desc["fecha"] = d.isoformat()
            self.descs.append(desc)
        self.ep = np.arange(self.inicio, self.inicio + n, dtype=np.int64)
        self.dow = np.array([x["dia_semana"] for x in self.descs], dtype=np.int64)
        self.mes = np.array([x["mes"] for x in self.descs], dtype=np.int64)
        self.anio = np.array([x["anio"] for x in self.descs], dtype=np.int64)
        self.doy = np.array([x["dia_del_anio"] for x in self.descs], dtype=float)
        self.rel = np.array([_REL_CODIGO[x["relacion_festivo"]] for x in self.descs], dtype=np.int64)
        self.dist = np.array([x["distancia_festivo"] or 0 for x in self.descs], dtype=np.int64)
        self.tipo = np.array([x["tipo_festivo"] or "" for x in self.descs], dtype=object)
        self.ss = np.array([x["semana_santa"] for x in self.descs], dtype=bool)
        self.vac = np.array([x["temporada_escolar"] == "vacaciones" for x in self.descs], dtype=bool)


class _Observado:
    """Curvas observadas del histórico alineadas a un calendario."""

    def __init__(self, historico: pd.DataFrame, cal: _Calendario) -> None:
        n = len(cal.ep)
        self.curvas = np.full((n, 24), np.nan)
        self.ok = np.zeros(n, dtype=bool)
        validas = historico[data.filas_validas(historico)]
        ep = _ep_serie(validas["fecha"])
        pos = ep - cal.inicio
        dentro = (pos >= 0) & (pos < n)
        self.curvas[pos[dentro]] = validas[data.PERIODOS].to_numpy(dtype=float)[dentro]
        self.ok[pos[dentro]] = True
        self.energia = self.curvas.sum(axis=1)
        todas = _ep_serie(historico["fecha"])
        self.primer_ep = int(todas.min()) if len(todas) else None
        self.ultimo_real_ep = int(ep.max()) if len(ep) else None


def _ep_serie(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s).to_numpy().astype("datetime64[D]").astype(np.int64)


_CACHE_MAX = 16
_cache_cal: "OrderedDict[Any, _Calendario]" = OrderedDict()
_cache_obs: "OrderedDict[Any, _Observado]" = OrderedDict()


_cache_lock = threading.Lock()


def _cache_get(cache: OrderedDict, clave: Any, crear) -> Any:
    with _cache_lock:
        if clave in cache:
            cache.move_to_end(clave)
            return cache[clave]
        valor = crear()
        cache[clave] = valor
        while len(cache) > _CACHE_MAX:
            cache.popitem(last=False)
        return valor


def _normalizado(historico: pd.DataFrame) -> pd.DataFrame:
    if historico is None or not isinstance(historico, pd.DataFrame):
        raise ValueError("historico debe ser un DataFrame")
    if ("fecha" in historico.columns and pd.api.types.is_datetime64_any_dtype(historico["fecha"])
            and all(p in historico.columns for p in data.PERIODOS)):
        return historico
    return data.normalizar_historico(historico)


def _clave_hist(historico: pd.DataFrame) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(_ep_serie(historico["fecha"]).tobytes())
    h.update(np.ascontiguousarray(historico[data.PERIODOS].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()


def _indice(historico: pd.DataFrame, festivos: set) -> Tuple[_Calendario, _Observado]:
    if historico.empty:
        raise ValueError("El histórico está vacío")
    fechas = historico["fecha"]
    anio0, anio1 = int(fechas.min().year), int(fechas.max().year)
    clave_fest = tuple(sorted(festivos))
    cal = _cache_get(_cache_cal, (anio0, anio1, clave_fest), lambda: _Calendario(anio0, anio1, festivos))
    obs = _cache_get(_cache_obs, (_clave_hist(historico), anio0, anio1), lambda: _Observado(historico, cal))
    return cal, obs


def _resolver(f: Dict[str, Any], obj: Dict[str, Any], notas: List[str]) -> Dict[str, Any]:
    """Resuelve los "auto" contra el descriptor objetivo."""
    r = dict(f)
    if r["dia_semana"] == "auto":
        r["dia_semana"] = [obj["dia_semana"]]
    if r["relacion_festivo"] == "auto":
        r["relacion_festivo"] = obj["relacion_festivo"]
    if r["distancia_festivo"] == "auto":
        r["distancia_festivo"] = obj["distancia_festivo"]
    if r["relacion_festivo"] not in ("antes", "despues") and r["distancia_festivo"] is not None:
        notas.append("distancia_festivo solo filtra con relacion_festivo antes/despues: no se aplica")
        r["distancia_festivo"] = None
    if r["tipo_festivo"] == "auto":
        r["tipo_festivo"] = obj["tipo_festivo"]
        if r["tipo_festivo"] is None:
            notas.append("tipo_festivo auto: el día objetivo no tiene festivo relacionado; no se filtra")
    return r


def _clima_arrays(clima: Optional[dict], cal: _Calendario) -> Tuple[np.ndarray, np.ndarray]:
    n = len(cal.ep)
    temp = np.full(n, np.nan)
    lluvia = np.full(n, np.nan)
    for iso_, c in (clima or {}).items():
        try:
            pos = _ep(date.fromisoformat(str(iso_)[:10])) - cal.inicio
        except ValueError:
            continue
        if 0 <= pos < n and c:
            t, ll = c.get("temp_media"), c.get("lluvia_mm")
            temp[pos] = np.nan if t is None else float(t)
            lluvia[pos] = np.nan if ll is None else float(ll)
    return temp, lluvia


def _buscar(historico: pd.DataFrame, festivos: Iterable[str], fecha: Any, fecha_corte: Any,
            filtros: Optional[dict], clima: Optional[dict]) -> Dict[str, Any]:
    """Núcleo: filtra y puntúa (vectorizado). Devuelve índices ordenados y lo necesario para la salida."""
    objetivo = parse_date(fecha)
    if objetivo is None or not FECHA_MIN <= objetivo <= FECHA_MAX:
        raise ValueError(f"fecha fuera de rango ({FECHA_MIN.isoformat()}..{FECHA_MAX.isoformat()})")
    corte_pedido = parse_date(fecha_corte)
    if corte_pedido is not None and not FECHA_MIN <= corte_pedido <= FECHA_MAX:
        raise ValueError(f"fecha_corte fuera de rango ({FECHA_MIN.isoformat()}..{FECHA_MAX.isoformat()})")
    f = validar_filtros(filtros)
    hist = _normalizado(historico)
    fest = _a_set(festivos)
    cal, obs = _indice(hist, fest)
    if obs.ultimo_real_ep is None:
        raise ValueError("El histórico no tiene días con 24 periodos válidos")
    notas: List[str] = []
    ultimo_real = _de_ep(obs.ultimo_real_ep)
    if corte_pedido is None:
        corte = ultimo_real
    elif corte_pedido > ultimo_real:
        corte = ultimo_real
        notas.append(f"fecha_corte {corte_pedido.isoformat()} posterior al último dato real; "
                     f"se usa {ultimo_real.isoformat()}")
    else:
        corte = corte_pedido

    obj = descriptor_dia(objetivo, fest, clima)
    obj["fecha"] = objetivo.isoformat()
    r = _resolver(f, obj, notas)

    temp, lluvia = _clima_arrays(clima, cal)
    c = r["clima"]
    t_obj = _temp(obj)
    clima_aplicado = None
    if c is not None:
        t_objetivo = c["temp_media"] if c["temp_media"] is not None else t_obj
        if c["temp_media"] is not None:
            t_obj = float(c["temp_media"])
        usa_temp = c["temp_tolerancia"] is not None and t_objetivo is not None
        if c["temp_tolerancia"] is not None and t_objetivo is None:
            notas.append("sin temperatura del día objetivo (ni pronosticada ni observada): "
                         "no se aplica el filtro de temperatura")
        clima_aplicado = {"temp_tolerancia": c["temp_tolerancia"], "lluvia": c["lluvia"],
                          "temp_media": t_objetivo if usa_temp else None}
        r["clima"] = clima_aplicado
    if t_obj is not None:
        obj["clima"] = {**(obj.get("clima") or {}), "temp_media": t_obj}

    # --- máscara ---
    ep_obj = _ep(objetivo)
    m = (cal.ep >= (obs.primer_ep if obs.primer_ep is not None else cal.inicio)) & (cal.ep <= _ep(corte))
    m &= cal.ep != ep_obj
    if r["anios_excluidos"]:
        m &= ~np.isin(cal.anio, r["anios_excluidos"])
    if r["anio_min"] is not None:
        m &= cal.anio >= r["anio_min"]
    if r["anio_max"] is not None:
        m &= cal.anio <= r["anio_max"]
    if r["dia_semana"] is not None:
        m &= np.isin(cal.dow, r["dia_semana"])
    if r["meses"] is not None:
        m &= np.isin(cal.mes, r["meses"])
    dist_doy = _dist_circular(cal.doy, obj["dia_del_anio"])
    if r["temporada_dias"] is not None:
        m &= dist_doy <= r["temporada_dias"]
    if r["relacion_festivo"] is not None:
        m &= cal.rel == _REL_CODIGO[r["relacion_festivo"]]
    if r["distancia_festivo"] is not None:
        m &= cal.dist == r["distancia_festivo"]
    if r["tipo_festivo"] is not None:
        m &= cal.tipo == r["tipo_festivo"]
    if r["semana_santa"] is not None:
        m &= cal.ss == r["semana_santa"]
    if r["temporada_escolar"] is not None:
        m &= cal.vac == (r["temporada_escolar"] == "vacaciones")
    if clima_aplicado is not None:
        activo = False
        if clima_aplicado["temp_media"] is not None:
            m &= np.isfinite(temp) & (np.abs(temp - clima_aplicado["temp_media"])
                                      <= clima_aplicado["temp_tolerancia"])
            activo = True
        if clima_aplicado["lluvia"] is not None:
            con_lluvia = np.isfinite(lluvia) & (lluvia >= LLUVIA_UMBRAL_MM)
            sin_lluvia = np.isfinite(lluvia) & (lluvia < LLUVIA_UMBRAL_MM)
            m &= con_lluvia if clima_aplicado["lluvia"] else sin_lluvia
            activo = True
        if not activo:
            notas.append("filtro de clima sin criterios aplicables: no se filtra por clima")

    idx = np.flatnonzero(m)
    scores = _scores_vec(obj, cal, idx, dist_doy[idx], temp[idx], ep_obj)
    orden = np.lexsort((-cal.ep[idx], -scores))
    r["fecha_corte_efectiva"] = corte.isoformat()
    r["notas"] = notas
    return {"obj": obj, "filtros": r, "cal": cal, "obs": obs, "idx": idx[orden], "scores": scores[orden],
            "temp": temp, "lluvia": lluvia, "corte": corte, "corte_pedido": corte_pedido}


def _scores_vec(obj: dict, cal: _Calendario, idx: np.ndarray, dist_doy: np.ndarray, temp: np.ndarray,
                ep_obj: int) -> np.ndarray:
    """Misma aritmética que `puntuar`, vectorizada."""
    dow = cal.dow[idx]
    s_dow = np.where(dow == obj["dia_semana"], 1.0, np.where((dow < 5) & (obj["dia_semana"] < 5), 0.4, 0.0))
    rel = obj["relacion_festivo"]
    misma = cal.rel[idx] == _REL_CODIGO[rel]
    if rel == "festivo":
        s_rel = np.where(cal.tipo[idx] == (obj["tipo_festivo"] or ""), 1.0, 0.6)
    elif rel in ("antes", "despues"):
        s_rel = np.where(cal.dist[idx] == (obj["distancia_festivo"] or 0), 1.0, 0.6)
    else:
        s_rel = np.ones(len(idx))
    s_rel = np.where(misma, s_rel, 0.0)
    s_est = np.maximum(0.0, 1.0 - dist_doy / VENTANA_ESTACIONAL_DIAS)
    s_rec = np.exp(-(np.abs(cal.ep[idx] - ep_obj) / 365.25) / RECENCIA_ANIOS)
    s_ss = np.where(cal.ss[idx] == obj["semana_santa"], 1.0, 0.0)
    s_esc = np.where(cal.vac[idx] == (obj["temporada_escolar"] == "vacaciones"), 1.0, 0.0)
    acumulado = (np.zeros(len(idx)) + PESOS["dia_semana"] * s_dow + PESOS["festivo"] * s_rel
                 + PESOS["estacionalidad"] * s_est + PESOS["recencia"] * s_rec
                 + PESOS["semana_santa"] * s_ss + PESOS["temporada_escolar"] * s_esc)
    base_w = (PESOS["dia_semana"] + PESOS["festivo"] + PESOS["estacionalidad"] + PESOS["recencia"]
              + PESOS["semana_santa"] + PESOS["temporada_escolar"])
    t_obj = _temp(obj)
    if t_obj is not None:
        con_t = np.isfinite(temp)
        s_cl = np.maximum(0.0, 1.0 - np.abs(np.where(con_t, temp, t_obj) - t_obj) / CLIMA_DELTA_T)
        acumulado = np.where(con_t, acumulado + PESOS["clima"] * s_cl, acumulado)
        total_w = np.where(con_t, base_w + PESOS["clima"], base_w)
    else:
        total_w = base_w
    return np.round(acumulado / total_w, 4)


def _desc_candidato(res: Dict[str, Any], i: int) -> Dict[str, Any]:
    desc = dict(res["cal"].descs[i])
    t, ll = res["temp"][i], res["lluvia"][i]
    desc["clima"] = None if not math.isfinite(t) else {"temp_media": float(t),
                                                       "lluvia_mm": float(ll) if math.isfinite(ll) else None}
    return desc


def _candidato(res: Dict[str, Any], i: int, clima: Optional[dict]) -> Dict[str, Any]:
    obs = res["obs"]
    desc = _desc_candidato(res, i)
    completo = _clima_de(clima, desc["fecha"])
    score, componentes, razones = puntuar(res["obj"], desc)
    if completo is not None:
        desc["clima"] = completo
    out: Dict[str, Any] = {"fecha": desc["fecha"], "score": score, "razones": razones,
                           "observado": bool(obs.ok[i])}
    if obs.ok[i]:
        curva = obs.curvas[i]
        energia = float(obs.energia[i])
        hora = int(np.argmax(curva))
        out.update({
            "energia": energia,
            "pico": {"mw": float(curva[hora]), "hora": hora + 1},
            "curva": curva.tolist(),
            "curva_normalizada": (curva / energia).tolist() if energia > 0 else None,
        })
    else:
        out.update({"energia": None, "pico": None, "curva": None, "curva_normalizada": None})
    out["clima"] = (None if desc["clima"] is None else
                    {"temp_media": desc["clima"].get("temp_media"), "lluvia_mm": desc["clima"].get("lluvia_mm")})
    out["descriptor"] = {k: v for k, v in desc.items() if k != "fecha"}
    out["componentes"] = componentes
    return out


def _resumen(candidatos: List[Dict[str, Any]]) -> Dict[str, Any]:
    obs = [c for c in candidatos if c["observado"] and c["curva_normalizada"] is not None]
    resumen: Dict[str, Any] = {"n_candidatos": len(candidatos), "n_observados": len(obs),
                               "curva_promedio_normalizada": None, "energia_promedio": None,
                               "ponderado_por_score": True}
    if not obs:
        return resumen
    pesos = np.array([c["score"] for c in obs], dtype=float)
    if not np.isfinite(pesos).all() or pesos.sum() <= 0:
        pesos = np.ones(len(obs))
    formas = np.array([c["curva_normalizada"] for c in obs], dtype=float)
    energias = np.array([c["energia"] for c in obs], dtype=float)
    resumen["curva_promedio_normalizada"] = (pesos @ formas / pesos.sum()).tolist()
    resumen["energia_promedio"] = float(pesos @ energias / pesos.sum())
    return resumen


def buscar_analogos(historico: pd.DataFrame, festivos: Iterable[str], fecha: Any, fecha_corte: Any = None,
                    filtros: Optional[dict] = None, clima: Optional[dict] = None) -> dict:
    """
    Busca días análogos a `fecha` en el histórico (solo días <= corte, nunca la
    propia fecha). Devuelve el contrato de `POST /adjustments/analogos` sin
    "ucp": `fecha`, `fecha_corte`, `descriptor_objetivo`, `filtros_aplicados`,
    `candidatos`, `resumen` (JSON-safe). ValueError ante filtros/fechas inválidos.
    """
    res = _buscar(historico, festivos, fecha, fecha_corte, filtros, clima)
    limite = res["filtros"]["limite"]
    candidatos = [_candidato(res, int(i), clima) for i in res["idx"][:limite]]
    candidatos.sort(key=lambda c: (-c["score"], -_ep(date.fromisoformat(c["fecha"]))))
    obj = {k: v for k, v in res["obj"].items() if k != "fecha"}
    salida = {
        "fecha": res["obj"]["fecha"],
        "fecha_corte": (res["corte_pedido"] or res["corte"]).isoformat(),
        "descriptor_objetivo": obj,
        "filtros_aplicados": res["filtros"],
        "candidatos": candidatos,
        "resumen": _resumen(candidatos),
    }
    return to_jsonable(salida)


def seleccionar(historico: pd.DataFrame, festivos: Iterable[str], fecha: Any, fecha_corte: Any,
                filtros: Optional[dict] = None) -> List[Tuple[str, float, np.ndarray]]:
    """
    Para los pasos: [(fecha_iso, score, curva_24)] de los `limite` mejores
    análogos OBSERVADOS (24 periodos válidos), ordenados por score desc y
    fecha desc. Respeta los filtros (incl. `anios_excluidos`, default [2020]).
    """
    res = _buscar(historico, festivos, fecha, fecha_corte, filtros, None)
    obs = res["obs"]
    limite = res["filtros"]["limite"]
    out: List[Tuple[str, float, np.ndarray]] = []
    for i, score in zip(res["idx"], res["scores"]):
        if obs.ok[i]:
            out.append((_de_ep(res["cal"].ep[i]).isoformat(), float(score), obs.curvas[i].copy()))
            if len(out) >= limite:
                break
    return out


def puntuar_fechas(historico: pd.DataFrame, festivos: Iterable[str], fecha: Any,
                   fechas: Sequence[Any]) -> List[Tuple[str, float, Optional[np.ndarray]]]:
    """
    Para la selección explícita del analista: [(fecha_iso, score, curva_24 o
    None si el día no está observado)] en el orden recibido. No aplica filtros
    ni corte (eso lo valida el paso). ValueError ante fechas inválidas.
    """
    objetivo = parse_date(fecha)
    fest = _a_set(festivos)
    obj = descriptor_dia(objetivo, fest)
    obj["fecha"] = objetivo.isoformat()
    hist = _normalizado(historico)
    curvas = data.curvas_reales(hist)
    out: List[Tuple[str, float, Optional[np.ndarray]]] = []
    for f in fechas:
        try:
            d = parse_date(f)
        except (TypeError, ValueError):
            raise ValueError(f"fecha análoga inválida: {_corto(f)}") from None
        if d is None or not FECHA_MIN <= d <= FECHA_MAX:
            raise ValueError(f"fecha análoga fuera de rango: {_corto(f)}")
        cand = descriptor_dia(d, fest)
        cand["fecha"] = d.isoformat()
        score, _, _ = puntuar(obj, cand)
        curva = curvas.get(d.isoformat())
        out.append((d.isoformat(), score, None if curva is None else np.asarray(curva, dtype=float).copy()))
    return out
