"""Aprendizaje de intervenciones (F09): diagnóstico de solo lectura.

Compara, para cada día editado, la curva **calculada** (`curvas_calculadas`,
persistida desde R1-C) contra la **editada** (`curves`), por UCP × paso × tipo
de día. No reentrena ni cambia reglas: solo agrega las diferencias para ver
intervenciones recurrentes del analista.

Entrada: iterable de `{"meta": {...}, "records": [StepRecord, ...]}` (formato
de `store.get_run`/`store.list_runs`). Los registros viejos sin
`curvas_calculadas` se ignoran y se cuentan en `omitidos_sin_calculada`; la
función nunca lanza por ellos.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

from src.adjustments import data
from src.adjustments.types import parse_date, to_jsonable

DIAS_SEMANA = ("lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo")
MAX_FECHAS = 10
DECIMALES = 6

FestivosPorRun = Union[Mapping[str, Sequence[str]], Callable[[Dict[str, Any]], Sequence[str]], None]


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Attr o clave de un dict (los records pueden venir como StepRecord o como dict)."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _estado(rec: Any) -> str:
    estado = _field(rec, "estado")
    return str(getattr(estado, "value", estado))


def _energia(valores: Any) -> Optional[float]:
    """
    Suma de los periodos almacenados, o None si la curva no es numérica/finita.

    Difiere a propósito de `data.energia_curva`: las curvas guardadas en records
    viejos no siempre tienen 24 periodos ni son no negativas, y este diagnóstico
    debe agregarlas igual mientras sean finitas; `energia_curva` las rechazaría.
    """
    if valores is None:
        return None
    try:
        if len(valores) == 0:  # ndarray: `if not valores` lanzaría con >1 elemento
            return None
    except TypeError:
        return None
    try:
        arr = np.asarray(valores, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    return float(arr.sum())


def _ronda(x: float) -> float:
    return round(float(x), DECIMALES)


def _pct(x: float) -> float:
    """Fracción (0.10) → porcentaje (10.0) redondeado, coherente con la API."""
    return _ronda(float(x) * 100.0)


def _festivos_de(meta: Dict[str, Any], festivos_por_run: FestivosPorRun) -> List[str]:
    """Festivos del run: por callback/mapping o los guardados en `meta.extras.festivos`."""
    if callable(festivos_por_run):
        try:
            return [str(f)[:10] for f in (festivos_por_run(meta) or [])]
        except Exception:  # noqa: BLE001 — un loader roto no debe tumbar el diagnóstico
            return []
    if isinstance(festivos_por_run, Mapping):
        rid = meta.get("id")
        for clave in (rid, str(rid)):
            if clave in festivos_por_run:
                return [str(f)[:10] for f in (festivos_por_run[clave] or [])]
    extras = _as_dict(meta.get("extras"))
    return [str(f)[:10] for f in (extras.get("festivos") or [])]


def _stats(casos: List[Dict[str, Any]], min_casos: Optional[int] = None) -> Dict[str, Any]:
    n = len(casos)
    if n == 0:
        out: Dict[str, Any] = {
            "n_casos": 0, "n_dias": 0, "n_corridas": 0,
            "delta_pct_medio": None, "delta_pct_mediano": None, "pct_positivo": None,
            "p50_abs": None, "p90_abs": None,
        }
        if min_casos is not None:
            out["recurrente"] = False
        return out
    deltas = np.asarray([c["delta_pct"] for c in casos], dtype=float)
    absd = np.abs(deltas)
    out = {
        "n_casos": n,
        "n_dias": len({c["fecha"] for c in casos}),
        "n_corridas": len({c["run_id"] for c in casos}),
        "delta_pct_medio": _pct(deltas.mean()),
        "delta_pct_mediano": _pct(np.median(deltas)),
        "pct_positivo": _pct((deltas > 0).mean()),
        "p50_abs": _pct(np.percentile(absd, 50)),
        "p90_abs": _pct(np.percentile(absd, 90)),
    }
    if min_casos is not None:
        out["recurrente"] = out["n_dias"] >= int(min_casos)
    return out


def _fechas_top(casos: List[Dict[str, Any]]) -> List[str]:
    """Hasta 10 fechas con mayor |delta| (desempate por fecha)."""
    por_fecha: Dict[str, float] = {}
    for c in casos:
        f = c["fecha"]
        por_fecha[f] = max(por_fecha.get(f, 0.0), abs(c["delta_pct"]))
    orden = sorted(por_fecha.items(), key=lambda kv: (-kv[1], kv[0]))
    return [f for f, _ in orden[:MAX_FECHAS]]


def _dia_semana(fecha: date) -> str:
    return DIAS_SEMANA[fecha.weekday()]


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def resumir(runs: Iterable[Dict[str, Any]], festivos_por_run: FestivosPorRun = None,
            min_casos: int = 3) -> Dict[str, Any]:
    """
    Agrega las ediciones de las corridas dadas (solo lectura, nunca lanza por
    records viejos).

    Devuelve:
    - `grupos`: lista por (ucp, paso, tipo_dia) con `n_dias`, `n_corridas`,
      `delta_pct_medio`, `delta_pct_mediano`, `pct_positivo`, `p50_abs`,
      `p90_abs`, `fechas` (top 10 por |delta|) y `recurrente`
      (`n_dias >= min_casos`; los grupos con menos días quedan `False`).
      Las métricas de porcentaje (`delta_pct_*`, `p50_abs`, `p90_abs`) y
      `pct_positivo` se devuelven en **porcentaje ×100** (0.10 → 10.0), como el
      resto de la API; `pct_positivo` va de 0 a 100.
    - `resumen_global` y `por_dia_semana` con las mismas métricas y el mismo
      `recurrente` (`n_dias >= min_casos`).
    - `omitidos_sin_calculada`: días editados sin `curvas_calculadas` (records
      viejos) y `min_casos` usado.
    """
    min_casos = max(1, int(min_casos))
    casos: List[Dict[str, Any]] = []
    omitidos = 0

    for indice, run in enumerate(runs or []):
        meta = _as_dict(_field(run, "meta"))
        records = _field(run, "records") or []
        rid = meta.get("id")
        run_id = str(rid) if rid is not None else f"run-{indice}"
        ucp = meta.get("ucp")
        festivos = set(_festivos_de(meta, festivos_por_run))

        for rec in records:
            if _estado(rec) != "editado":
                continue
            result = _field(rec, "result") or {}
            diag = _as_dict(_field(result, "diagnostics"))
            curvas = _as_dict(_field(result, "curves"))
            calculadas = _as_dict(_field(result, "curvas_calculadas"))
            fechas = [str(f)[:10] for f in (diag.get("fechas_editadas") or [])]
            if not fechas:
                fechas = sorted(set(calculadas) & set(curvas))
            paso = _field(rec, "paso")

            for fecha in fechas:
                calculada = calculadas.get(fecha)
                if calculada is None:
                    omitidos += 1
                    continue
                editada = curvas.get(fecha)
                if editada is None:
                    continue
                e_calc = _energia(calculada)
                e_edit = _energia(editada)
                if e_calc is None or e_edit is None or e_calc == 0:
                    continue
                delta_pct = e_edit / e_calc - 1.0
                if not np.isfinite(delta_pct):
                    continue
                try:
                    fecha_date = parse_date(fecha)
                except (TypeError, ValueError):
                    continue
                casos.append({
                    "ucp": ucp,
                    "paso": paso,
                    "tipo_dia": data.tipo_dia(fecha_date, festivos),
                    "dia_semana": _dia_semana(fecha_date),
                    "fecha": fecha,
                    "run_id": run_id,
                    "delta_pct": float(delta_pct),
                })

    por_grupo: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for c in casos:
        por_grupo[(c["ucp"], c["paso"], c["tipo_dia"])].append(c)

    grupos = []
    for (ucp, paso, tipo_dia) in sorted(por_grupo, key=lambda k: tuple(str(x) for x in k)):
        grupo_casos = por_grupo[(ucp, paso, tipo_dia)]
        grupos.append({
            "ucp": ucp,
            "paso": paso,
            "tipo_dia": tipo_dia,
            **_stats(grupo_casos, min_casos=min_casos),
            "fechas": _fechas_top(grupo_casos),
        })

    por_dia: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in casos:
        por_dia[c["dia_semana"]].append(c)
    por_dia_semana = {
        dia: _stats(por_dia[dia], min_casos=min_casos) for dia in DIAS_SEMANA if dia in por_dia
    }

    return {
        "grupos": grupos,
        "resumen_global": _stats(casos, min_casos=min_casos),
        "por_dia_semana": por_dia_semana,
        "omitidos_sin_calculada": omitidos,
        "min_casos": min_casos,
    }


def a_dict(resumen: Any) -> Any:
    """Garantiza JSON estricto (numpy/fechas/NaN) para la respuesta del endpoint."""
    return to_jsonable(resumen)
