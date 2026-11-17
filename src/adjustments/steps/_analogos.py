"""
Lógica compartida de días análogos (R3, Fase 2b) para los pasos forma, nivel y
referencia. Usa el buscador `src.adjustments.analogos` (`seleccionar`,
`puntuar_fechas`); aquí solo vive la validación de los params nuevos, la
selección por día (mapa del analista o búsqueda automática), los pesos y el
ancla de nivel por análogos.

Params nuevos (opcionales; con su default no cambian nada y NO se guardan en el
record, así los records y salidas por defecto quedan idénticos a la Fase 1):
- `analogos`: None o `{"YYYY-MM-DD" (día de la semana objetivo): ["YYYY-MM-DD", ...]}`
  (a lo sumo 7 claves y `MAX_FECHAS_POR_DIA` fechas por día, en
  [FECHA_MIN, FECHA_MAX]). Se normaliza a fechas ISO, claves ordenadas y listas
  sin duplicados (orden del analista). Que las claves estén en la semana
  objetivo y que las fechas sean <= `ctx.fecha_corte` y distintas del día
  objetivo se valida en el `step` (`validar_mapa_ctx`, necesita el contexto).
- `filtros_analogos`: None (filtros automáticos `FILTROS_AUTO`) o un objeto de
  filtros del buscador (`analogos.validar_filtros`; se guarda completo,
  validado). Los años excluidos de la corrida (`ctx.extras["anios_excluidos"]`)
  se suman siempre a los del filtro.
- `ancla` (solo nivel): "actual" (default) | "analogos".
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from src.adjustments import analogos as buscador
from src.adjustments.data import (
    ANIOS_EXCLUIDOS_DEFAULT,
    ancla_mismo_dow,
    dias_semana,
    energia_diaria,
    iso,
    lunes_de,
)
from src.adjustments.steps import _params
from src.adjustments.types import Alert, RunContext, parse_date

ANCLAS = ("actual", "analogos")
MAX_FECHAS_POR_DIA = 60
MAX_DIAS_MAPA = 7
# Filtros automáticos: mismo día de semana y misma relación/distancia con festivo
# que el día objetivo (el score ya pondera estacionalidad, recencia, etc.).
FILTROS_AUTO: Dict[str, Any] = {"dia_semana": "auto", "relacion_festivo": "auto", "distancia_festivo": "auto"}
LIMITE_AUTO_NIVEL = 10  # análogos por día para el ancla de nivel (búsqueda automática)
LIMITE_SUGERENCIAS = 5  # análogos por día para `semanas_sugeridas` (referencia)
TOP_SEMANAS = 5

DEFAULTS_FORMA: Dict[str, Any] = {"analogos": None, "filtros_analogos": None}
DEFAULTS_NIVEL: Dict[str, Any] = {"ancla": "actual", "analogos": None, "filtros_analogos": None}


# ---------------------------------------------------------------------------
# Validación de params
# ---------------------------------------------------------------------------

def _fecha(paso: str, clave: str, v: Any) -> date:
    try:
        if not isinstance(v, (str, date)):
            raise TypeError
        d = parse_date(v)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{paso}.{clave} debe ser una fecha YYYY-MM-DD, llegó {_params._corto(v)}") from None
    if not _params.FECHA_MIN <= d <= _params.FECHA_MAX:
        raise ValueError(
            f"{paso}.{clave} debe estar entre {_params.FECHA_MIN} y {_params.FECHA_MAX}, llegó {d.isoformat()}"
        )
    return d


def validar_mapa(paso: str, mapa: Any) -> Optional[Dict[str, List[str]]]:
    """None o {fecha ISO: [fechas ISO]} normalizado (sin contexto)."""
    if mapa is None:
        return None
    if not isinstance(mapa, dict):
        raise ValueError(f"{paso}.analogos debe ser un objeto {{fecha: [fechas]}} o null, "
                         f"llegó {_params._corto(mapa)}")
    if len(mapa) > MAX_DIAS_MAPA:
        raise ValueError(f"{paso}.analogos admite a lo sumo {MAX_DIAS_MAPA} días, llegaron {len(mapa)}")
    out: Dict[str, List[str]] = {}
    for k, fechas in mapa.items():
        dia = iso(_fecha(paso, "analogos (día objetivo)", k))
        if dia in out:
            raise ValueError(f"{paso}.analogos repite el día {dia}")
        if not isinstance(fechas, (list, tuple)):
            raise ValueError(f"{paso}.analogos[{dia}] debe ser una lista de fechas, llegó {_params._corto(fechas)}")
        if not fechas:
            raise ValueError(f"{paso}.analogos[{dia}] no puede ser una lista vacía "
                             "(omita el día para usar la búsqueda automática)")
        if len(fechas) > MAX_FECHAS_POR_DIA:
            raise ValueError(f"{paso}.analogos[{dia}] admite a lo sumo {MAX_FECHAS_POR_DIA} fechas, "
                             f"llegaron {len(fechas)}")
        lista: List[str] = []
        for f in fechas:
            x = iso(_fecha(paso, f"analogos[{dia}][]", f))
            if x not in lista:
                lista.append(x)
        out[dia] = lista
    return {k: out[k] for k in sorted(out)}


def validar_filtros(paso: str, filtros: Any) -> Optional[Dict[str, Any]]:
    if filtros is None:
        return None
    try:
        return buscador.validar_filtros(filtros)
    except ValueError as e:
        raise ValueError(f"{paso}.filtros_analogos: {e}") from None


def separar(paso: str, params: Any, defaults: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Separa y valida los params de análogos de `params` → (resto, validados).
    `validados` solo trae los que difieren del default (idempotente). Si
    `params` no es dict lo devuelve tal cual (lo rechaza `_params.completar`).
    """
    if not isinstance(params, dict):
        return params, {}
    resto = {k: v for k, v in params.items() if k not in defaults}
    p = {**defaults, **{k: v for k, v in params.items() if k in defaults}}
    out: Dict[str, Any] = {}
    if "ancla" in defaults:
        out["ancla"] = _params.opcion(paso, "ancla", p["ancla"], ANCLAS)
    out["analogos"] = validar_mapa(paso, p["analogos"])
    out["filtros_analogos"] = validar_filtros(paso, p["filtros_analogos"])
    return resto, {k: v for k, v in out.items() if v != defaults[k]}


def validar_mapa_ctx(paso: str, mapa: Optional[Dict[str, List[str]]], ctx: RunContext) -> None:
    """Claves dentro de la semana objetivo; fechas <= corte y distintas del día objetivo."""
    if not mapa:
        return
    semana = {iso(d) for d in dias_semana(ctx.semana_inicio)}
    corte = iso(ctx.fecha_corte)
    for dia, fechas in mapa.items():
        if dia not in semana:
            raise ValueError(f"{paso}.analogos: el día {dia} no está en la semana objetivo "
                             f"({min(semana)}..{max(semana)})")
        for f in fechas:
            if f > corte:
                raise ValueError(f"{paso}.analogos[{dia}]: {f} es posterior a la fecha de corte {corte}")
            if f == dia:
                raise ValueError(f"{paso}.analogos[{dia}]: el día objetivo no puede ser su propio análogo")


# ---------------------------------------------------------------------------
# Selección
# ---------------------------------------------------------------------------

def anios_excluidos_ctx(ctx: RunContext) -> Set[int]:
    valor = ctx.extras.get("anios_excluidos")
    return {int(a) for a in (ANIOS_EXCLUIDOS_DEFAULT if valor is None else valor)}


def filtros_efectivos(filtros: Optional[Dict[str, Any]], excluidos: Iterable[int],
                      limite: Optional[int] = None) -> Dict[str, Any]:
    """Filtros del analista (o `FILTROS_AUTO`) + años excluidos de la corrida (+ límite si llega)."""
    f = dict(FILTROS_AUTO if filtros is None else filtros)
    base = f.get("anios_excluidos")
    base_set = set() if base is None else set(base)
    if base_set == set(ANIOS_EXCLUIDOS_DEFAULT):
        base_set = set()  # valor por defecto = "no especificado": manda el set de la corrida
    f["anios_excluidos"] = sorted(base_set | set(excluidos))
    if limite is not None:
        f["limite"] = int(limite)
    return f


def seleccion_dia(
    ctx: RunContext, d: date, mapa: Optional[Dict[str, List[str]]], filtros: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Análogos de un día: {"seleccion": "analista"|"automatica",
    "items": [(fecha, score, curva_24)] solo observados, "omitidas": [fechas no observadas]}.
    """
    f = iso(d)
    if mapa and f in mapa:
        puntuados = buscador.puntuar_fechas(ctx.historico, ctx.festivos, d, mapa[f])
        items = [(x, s, c) for x, s, c in puntuados if c is not None]
        omitidas = [x for x, _, c in puntuados if c is None]
        return {"seleccion": "analista", "items": items, "omitidas": omitidas}
    items = buscador.seleccionar(ctx.historico, ctx.festivos, d, ctx.fecha_corte, filtros)
    return {"seleccion": "automatica", "items": items, "omitidas": []}


def pesos(scores: List[float]) -> np.ndarray:
    """Pesos normalizados por score (uniformes si no suman > 0)."""
    w = np.asarray(scores, dtype=float)
    if not len(w) or not np.isfinite(w).all() or w.sum() <= 0:
        return np.full(len(w), 1.0 / max(1, len(w)))
    return w / w.sum()


def forma_ponderada(items: List[Tuple[str, float, np.ndarray]]) -> Tuple[Optional[np.ndarray], List[str], List[float]]:
    """Promedio ponderado por score de las formas normalizadas → (forma, fechas, pesos)."""
    usados = []
    for fecha, score, curva in items:
        total = float(np.sum(curva))
        if np.isfinite(total) and total > 0:
            usados.append((fecha, score, np.asarray(curva, dtype=float) / total))
    if not usados:
        return None, [], []
    w = pesos([s for _, s, _ in usados])
    forma = np.sum([wi * fz for wi, (_, _, fz) in zip(w, usados)], axis=0)
    return forma / forma.sum(), [x for x, _, _ in usados], [round(float(x), 6) for x in w]


def alerta_no_observadas(no_observadas: Dict[str, List[str]]) -> Optional[Alert]:
    if not no_observadas:
        return None
    return Alert(
        nivel="info",
        codigo="ANALOGOS_NO_OBSERVADOS",
        mensaje="Algunas fechas análogas elegidas no tienen 24 periodos reales y se omitieron",
        fechas=sorted(no_observadas),
        detalle=no_observadas,
    )


# ---------------------------------------------------------------------------
# Nivel: ancla por análogos
# ---------------------------------------------------------------------------

def _ancla_ok(x: date, energia: Dict[str, float], festivos: set, excluidos: Set[int]) -> bool:
    return x.year not in excluidos and energia.get(iso(x), 0) > 0 and iso(x) not in festivos


def _relacion(x: date, festivos: set) -> Tuple[Any, Any]:
    """(relacion_festivo, distancia_festivo) de `x` según el buscador."""
    desc = buscador.descriptor_dia(x, festivos)
    return desc["relacion_festivo"], desc["distancia_festivo"]


def evidencia_analogos(
    ctx: RunContext,
    d: date,
    sel: Dict[str, Any],
    energia: Dict[str, float],
    festivos: set,
    excluidos: Set[int],
    max_semanas_extra: int,
) -> Optional[Dict[str, Any]]:
    """
    E*(d) por análogos: a = ancla actual de d (último mismo día de semana <=
    corte, observada, no festiva; retrocede de a 7 días hasta
    `max_semanas_extra`), lag = d − a; para cada análogo c, c_a = c − lag
    (observada, no festiva, fuera de años excluidos y con la MISMA relación con
    festivo que a, p. ej. un martes normal y no uno post-festivo; si no,
    retrocede de a 7). Así E(c)/E(c_a) mide lo mismo que E(d)/E(a).
    ratio = Σ score·E(c)/E(c_a) / Σ score; E* = E(a)·ratio. None sin ratios válidos.
    """
    primera = ancla_mismo_dow(d, ctx.fecha_corte)
    ancla = next(
        (primera - timedelta(days=7 * k) for k in range(max_semanas_extra + 1)
         if _ancla_ok(primera - timedelta(days=7 * k), energia, festivos, set())),
        None,
    )
    if ancla is None:
        return None
    lag = (d - ancla).days
    rel_ancla = _relacion(ancla, festivos)
    usados: List[Dict[str, Any]] = []
    for fecha, score, _ in sel["items"]:
        c = date.fromisoformat(fecha)
        e_c = energia.get(fecha, 0.0)
        if c > ctx.fecha_corte or not e_c > 0:
            continue
        c_a = next(
            (c - timedelta(days=lag + 7 * k) for k in range(max_semanas_extra + 1)
             if _ancla_ok(c - timedelta(days=lag + 7 * k), energia, festivos, excluidos)
             and _relacion(c - timedelta(days=lag + 7 * k), festivos) == rel_ancla),
            None,
        )
        if c_a is None:
            continue
        usados.append({"fecha": fecha, "ancla": iso(c_a), "ratio": e_c / energia[iso(c_a)], "score": score})
    if not usados:
        return None
    w = pesos([u["score"] for u in usados])
    ratio = float(sum(wi * u["ratio"] for wi, u in zip(w, usados)))
    for wi, u in zip(w, usados):
        u["peso"] = round(float(wi), 6)
        del u["score"]
    e_ancla = energia[iso(ancla)]
    return {
        "ancla": iso(ancla),
        "dia_ref": None,
        "ancla_ref": None,
        "lag_dias": lag,
        "ratio": ratio,
        "E_ancla": e_ancla,
        "E_objetivo": e_ancla * ratio,
        "regla": "analogos",
        "seleccion": sel["seleccion"],
        "analogos": usados,
    }


def evidencias_analogos(
    ctx: RunContext, p: Dict[str, Any], actuales: Dict[str, Optional[Dict[str, Any]]], max_semanas_extra: int
) -> Tuple[Dict[str, Optional[Dict[str, Any]]], List[str], Dict[str, List[str]]]:
    """
    Reemplaza la evidencia de cada día por la de análogos; sin ratios válidos se
    conserva la actual (semana de referencia). → (evidencias, días alternos, no observadas).
    """
    mapa = p.get("analogos")
    validar_mapa_ctx("nivel", mapa, ctx)
    excluidos = anios_excluidos_ctx(ctx)
    filtros = filtros_efectivos(p.get("filtros_analogos"), excluidos,
                                None if p.get("filtros_analogos") is not None else LIMITE_AUTO_NIVEL)
    energia = energia_diaria(ctx.historico)
    festivos = set(ctx.festivos)
    out: Dict[str, Optional[Dict[str, Any]]] = dict(actuales)
    alternos: List[str] = []
    no_obs: Dict[str, List[str]] = {}
    for d in dias_semana(ctx.semana_inicio):
        f = iso(d)
        sel = seleccion_dia(ctx, d, mapa, filtros)
        if sel["omitidas"]:
            no_obs[f] = sel["omitidas"]
        ev = evidencia_analogos(ctx, d, sel, energia, festivos, excluidos, max_semanas_extra)
        if ev is None:
            alternos.append(f)
        else:
            out[f] = ev
    return out, alternos, no_obs


# ---------------------------------------------------------------------------
# Referencia: semanas sugeridas
# ---------------------------------------------------------------------------

def semanas_sugeridas(ctx: RunContext, excluidos: Set[int]) -> List[Dict[str, Any]]:
    """
    Top `TOP_SEMANAS` semanas históricas (lunes) por suma de scores de los días
    objetivo que cubren con su mejor análogo dentro de esa semana. Solo semanas
    con los 7 días <= corte y sin días en años excluidos.
    """
    filtros = filtros_efectivos(None, excluidos, LIMITE_SUGERENCIAS)
    por_semana: Dict[date, Dict[str, Tuple[float, str]]] = {}
    for d in dias_semana(ctx.semana_inicio):
        try:
            items = buscador.seleccionar(ctx.historico, ctx.festivos, d, ctx.fecha_corte, filtros)
        except ValueError:
            return []  # histórico vacío o sin días válidos: no hay sugerencias
        for fecha, score, _ in items:
            c = date.fromisoformat(fecha)
            lunes = lunes_de(c)
            if lunes + timedelta(days=6) > ctx.fecha_corte or any(x.year in excluidos for x in dias_semana(lunes)):
                continue
            dias = por_semana.setdefault(lunes, {})
            if iso(d) not in dias or score > dias[iso(d)][0]:
                dias[iso(d)] = (score, fecha)
    ranking = sorted(
        ((round(sum(s for s, _ in dias.values()), 4), lunes, dias) for lunes, dias in por_semana.items()),
        key=lambda t: (-t[0], -t[1].toordinal()),
    )
    return [
        {"semana_inicio": iso(lunes), "score": score,
         "dias": {k: dias[k][1] for k in sorted(dias)}}
        for score, lunes, dias in ranking[:TOP_SEMANAS]
    ]
