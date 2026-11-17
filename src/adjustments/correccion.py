"""
Correcciones y bandas empíricas sobre pronósticos, medidas por backtest sin fuga (noche 2, parte B2).

OPT-IN, NO CABLEADO AL PIPELINE POR DISEÑO. Su `tablas_banda` es el estimador experimental de B2 (log-error, pool de horas,
ventana móvil), distinto del que sirve la API en `insights.tablas_banda` (error relativo simple con todo el historial previo,
escala 1.2); las coberturas de abajo corresponden a ESTE estimador, no al de insights (ver `insights.COBERTURA_MEDIDA`).

Funciones puras (sin I/O, sin estado) que solo usan errores de días con fecha <= `fecha_corte`:

- `tablas_banda` / `bandas_empiricas` / `aplicar_bandas`: bandas p10-p90 empíricas por tipo de día x hora.
  Es lo único de este módulo recomendado para producción: no cambia el pronóstico (solo dice su incertidumbre).
- `corregir_sesgo`: corrección multiplicativa por segmento x hora con ventana móvil (OPT-IN).
- `corregir_clima`: ajuste de energía diaria por anomalía de temperatura (OPT-IN, EXPERIMENTAL).

Veredicto del backtest (docs/work/ajustes/noche2/B2/REPORTE.md; base `modelo_k1`, 52 semanas por UCP,
31 de calibración / 21 de prueba, WAPE horario):
- `corregir_sesgo` NO mejora de forma consistente (Antioquia ~0 en prueba, Atlántico +0.15..+0.20 pp): no está
  activada en ningún paso del pipeline y no debe activarse por defecto.
- `corregir_clima` mejora Atlántico (cal -0.18 / prueba -0.11 pp) y Antioquia en prueba (-0.08 pp), pero Antioquia no tiene clima
  en calibración y el clima del backtest es OBSERVADO (optimista): requiere un pronóstico de temperatura real.
- Bandas: cobertura nominal 80 %; con `escala` = 1.2 (elegida en calibración) la cobertura en prueba fue ~91 % (Antioquia) y
  ~80 % (Atlántico). La cobertura depende del régimen de error: si el error sube, la banda subcubre.

Convenciones: curvas = {"YYYY-MM-DD": [P1..P24]} (MWh por periodo). `festivos` = lista de fechas ISO. Los días con curva
inválida (no 24 valores, no finitos, negativos o real <= 0) se ignoran como evidencia.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from src.adjustments import data
from src.adjustments.steps import _params
from src.adjustments.types import Curves, parse_date, validar_curva


DEFAULTS_BANDA: Dict[str, Any] = {
    "q_bajo": 0.10,
    "q_alto": 0.90,
    "ventana_semanas": 26,
    "ventana_festivo_semanas": 52,
    "pool_horas": 3,        # junta las horas h-3..h+3 para tener más casos por hora
    "minimo_casos": 2,      # días mínimos del tipo de día; si no, respaldo con todos los tipos
    "escala": 1.2,          # factor sobre los log-cuantiles (calibrado: cobertura de calibración ≈ 80 %)
}

DEFAULTS_SESGO: Dict[str, Any] = {
    "segmento": "tipo",          # "global" | "tipo" | "dow"
    "ventana_semanas": 13,
    "ventana_festivo_semanas": 52,
    "k0": 3.0,                   # shrinkage: factor = 1 + n/(n+k0)·g·(razón − 1)
    "g": 0.5,
    "minimo_casos": 3,
    "modo": "hora",              # "hora" (segmento×hora) | "energia" (un factor por día) | "forma" (hora, conserva energía)
    "suavizado_horas": 0,
}

DEFAULTS_CLIMA: Dict[str, Any] = {
    "ventana_semanas": 26,
    "k0": 5.0,
    "minimo_casos": 10,
    "g": 1.0,
    "dias_previos": 28,
    "solo_laboral": True,
}

_MIN_CASOS_ABSOLUTO = 3  # menos días de evidencia que esto: no hay banda


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _fecha(valor: Any) -> date:
    d = parse_date(valor)
    if d is None:
        raise ValueError(f"fecha inválida: {valor!r}")
    return d


def _validar_banda(params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    p = _params.completar("banda", DEFAULTS_BANDA, dict(params) if params is not None else None)
    p["q_bajo"] = _params.numero("banda", "q_bajo", p["q_bajo"], minimo=0.0, maximo=0.5, minimo_exclusivo=False)
    p["q_alto"] = _params.numero("banda", "q_alto", p["q_alto"], minimo=0.5, maximo=1.0)
    p["ventana_semanas"] = _params.entero("banda", "ventana_semanas", p["ventana_semanas"], minimo=1, maximo=520)
    p["ventana_festivo_semanas"] = _params.entero("banda", "ventana_festivo_semanas", p["ventana_festivo_semanas"],
                                                  minimo=1, maximo=520)
    p["pool_horas"] = _params.entero("banda", "pool_horas", p["pool_horas"], minimo=0, maximo=12)
    p["minimo_casos"] = _params.entero("banda", "minimo_casos", p["minimo_casos"], minimo=1, maximo=1000)
    p["escala"] = _params.numero("banda", "escala", p["escala"], minimo=0.0, maximo=10.0)
    return p


def _validar_sesgo(params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    p = _params.completar("sesgo", DEFAULTS_SESGO, dict(params) if params is not None else None)
    p["segmento"] = _params.opcion("sesgo", "segmento", p["segmento"], ("global", "tipo", "dow"))
    p["modo"] = _params.opcion("sesgo", "modo", p["modo"], ("hora", "energia", "forma"))
    p["ventana_semanas"] = _params.entero("sesgo", "ventana_semanas", p["ventana_semanas"], minimo=1, maximo=520)
    p["ventana_festivo_semanas"] = _params.entero("sesgo", "ventana_festivo_semanas", p["ventana_festivo_semanas"],
                                                  minimo=1, maximo=520)
    p["k0"] = _params.numero("sesgo", "k0", p["k0"], minimo=0.0, maximo=1000.0)
    p["g"] = _params.numero("sesgo", "g", p["g"], minimo=0.0, maximo=1.0)
    p["minimo_casos"] = _params.entero("sesgo", "minimo_casos", p["minimo_casos"], minimo=1, maximo=1000)
    p["suavizado_horas"] = _params.entero("sesgo", "suavizado_horas", p["suavizado_horas"], minimo=0, maximo=12)
    return p


def _validar_clima(params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    p = _params.completar("clima", DEFAULTS_CLIMA, dict(params) if params is not None else None)
    p["ventana_semanas"] = _params.entero("clima", "ventana_semanas", p["ventana_semanas"], minimo=1, maximo=520)
    p["k0"] = _params.numero("clima", "k0", p["k0"], minimo=0.0, maximo=1e6)
    p["minimo_casos"] = _params.entero("clima", "minimo_casos", p["minimo_casos"], minimo=2, maximo=10000)
    p["g"] = _params.numero("clima", "g", p["g"], minimo=0.0, maximo=1.0)
    p["dias_previos"] = _params.entero("clima", "dias_previos", p["dias_previos"], minimo=2, maximo=365)
    if not isinstance(p["solo_laboral"], bool):
        raise ValueError("clima.solo_laboral debe ser booleano")
    return p


def _como_array(valores: Any) -> Optional[np.ndarray]:
    """Curva válida de 24 valores → ndarray; None si es inválida."""
    if validar_curva(valores) is not None:
        return None
    return np.asarray(valores, dtype=float)


def _pares(
    historico_pronosticos: Mapping[str, Any],
    historico_real: Mapping[str, Any],
    fecha_corte: date,
) -> Dict[date, Tuple[np.ndarray, np.ndarray]]:
    """{fecha: (pronóstico, real)} de días con ambos válidos, real > 0 y fecha <= fecha_corte (sin fuga)."""
    reales = {str(k)[:10]: v for k, v in historico_real.items()}
    out: Dict[date, Tuple[np.ndarray, np.ndarray]] = {}
    for f, pron in historico_pronosticos.items():
        try:
            d = _fecha(f)
        except ValueError:
            continue
        if d > fecha_corte:
            continue
        p, r = _como_array(pron), _como_array(reales.get(d.isoformat()))
        if p is None or r is None or (r <= 0).any() or (p <= 0).any():
            continue
        out[d] = (p, r)
    return out


def _segmento_dow(d: date, festivos: set) -> str:
    """Segmento por día de semana; los festivos van aparte ("festivo")."""
    if d.isoformat() in festivos:
        return data.TIPO_FESTIVO
    return str(d.weekday())


def _etiqueta(d: date, festivos: set, segmento: str) -> str:
    if segmento == "global":
        return "global"
    if segmento == "dow":
        return _segmento_dow(d, festivos)
    return data.tipo_dia(d, festivos)


def _ventana(pares: Dict[date, Any], fecha_corte: date, semanas: int) -> List[date]:
    """Fechas con evidencia en (fecha_corte − 7·semanas, fecha_corte]."""
    ini = fecha_corte - timedelta(days=7 * semanas)
    return [d for d in pares if ini < d <= fecha_corte]


# ---------------------------------------------------------------------------
# Bandas empíricas
# ---------------------------------------------------------------------------

def _log_errores(pares: Dict[date, Tuple[np.ndarray, np.ndarray]], fechas: Iterable[date]) -> np.ndarray:
    """Matriz [n, 24] de log(real/pronóstico); el llamador garantiza al menos un día."""
    return np.vstack([np.log(pares[d][1] / pares[d][0]) for d in fechas])


def _cuantiles_horarios(E: np.ndarray, q_bajo: float, q_alto: float, pool: int) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.zeros(24)
    hi = np.zeros(24)
    for h in range(24):
        hs = [x for x in range(h - pool, h + pool + 1) if 0 <= x < 24]
        v = E[:, hs].ravel()
        lo[h], hi[h] = np.quantile(v, q_bajo), np.quantile(v, q_alto)
    return lo, hi


def tablas_banda(
    historico_pronosticos: Mapping[str, Any],
    historico_real: Mapping[str, Any],
    fecha_corte: Any,
    festivos: Iterable[str],
    params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Tabla de la banda empírica por tipo de día: errores relativos log(real/pronóstico) de los días con fecha <= `fecha_corte`
    (ventana móvil), cuantiles q_bajo/q_alto por hora (agrupando ±`pool_horas`), multiplicados por `escala`.

    Devuelve `{tipo_dia: {"n_dias", "respaldo", "inferior": [24], "superior": [24]}}` con `inferior/superior` como
    multiplicadores de la curva (curva·inferior, curva·superior). Un tipo con menos de `minimo_casos` días usa como
    respaldo todos los tipos de día (`respaldo`=True); sin al menos 3 días en total el tipo queda con `inferior/superior`=None.
    """
    p = _validar_banda(params)
    corte = _fecha(fecha_corte)
    fest = {str(f)[:10] for f in festivos}
    pares = _pares(historico_pronosticos, historico_real, corte)
    todos = _ventana(pares, corte, p["ventana_festivo_semanas"])
    out: Dict[str, Dict[str, Any]] = {}
    for tipo in data.TIPOS_DIA:
        semanas = p["ventana_festivo_semanas"] if tipo == data.TIPO_FESTIVO else p["ventana_semanas"]
        fechas = [d for d in _ventana(pares, corte, semanas) if data.tipo_dia(d, fest) == tipo]
        respaldo = False
        n_propio = len(fechas)
        if n_propio < p["minimo_casos"]:
            fechas, respaldo = todos, True
        if len(fechas) < _MIN_CASOS_ABSOLUTO:
            out[tipo] = {"n_dias": n_propio, "respaldo": respaldo, "inferior": None, "superior": None}
            continue
        lo, hi = _cuantiles_horarios(_log_errores(pares, fechas), p["q_bajo"], p["q_alto"], p["pool_horas"])
        out[tipo] = {"n_dias": n_propio, "respaldo": respaldo,
                     "inferior": np.exp(lo * p["escala"]).tolist(), "superior": np.exp(hi * p["escala"]).tolist()}
    return out


def aplicar_bandas(
    curvas: Curves,
    tablas: Mapping[str, Mapping[str, Any]],
    festivos: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Aplica `tablas_banda` a curvas: `{fecha: {"tipo_dia", "n_dias", "respaldo", "inferior": [24], "superior": [24]}}`.
    Las fechas con curva inválida o tipo sin banda se omiten.
    """
    fest = {str(f)[:10] for f in festivos}
    out: Dict[str, Dict[str, Any]] = {}
    for f, curva in curvas.items():
        c = _como_array(curva)
        if c is None:
            continue
        d = _fecha(f)
        tipo = data.tipo_dia(d, fest)
        t = tablas.get(tipo)
        if not t or t.get("inferior") is None:
            continue
        out[d.isoformat()] = {
            "tipo_dia": tipo, "n_dias": t["n_dias"], "respaldo": bool(t["respaldo"]),
            "inferior": (c * np.asarray(t["inferior"])).tolist(), "superior": (c * np.asarray(t["superior"])).tolist(),
        }
    return out


def bandas_empiricas(
    curvas: Curves,
    historico_pronosticos: Mapping[str, Any],
    historico_real: Mapping[str, Any],
    fecha_corte: Any,
    festivos: Iterable[str],
    params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Atajo: `aplicar_bandas(curvas, tablas_banda(...), festivos)`. Sin fuga: solo errores con fecha <= `fecha_corte`."""
    festivos = list(festivos)
    return aplicar_bandas(curvas, tablas_banda(historico_pronosticos, historico_real, fecha_corte, festivos, params),
                          festivos)


# ---------------------------------------------------------------------------
# Corrección de sesgo (opt-in; no mejora de forma consistente en el backtest)
# ---------------------------------------------------------------------------

def _suavizar(f: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return f
    pad = np.pad(f, k, mode="edge")
    return np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="valid")


def corregir_sesgo(
    curvas: Curves,
    historico_pronosticos: Mapping[str, Any],
    historico_real: Mapping[str, Any],
    fecha_corte: Any,
    festivos: Iterable[str],
    params: Optional[Mapping[str, Any]] = None,
) -> Curves:
    """
    Corrección multiplicativa por segmento x hora con ventana móvil y shrinkage hacia 1 (sin fuga: solo errores con fecha
    <= `fecha_corte`).

    Para cada día de `curvas` de segmento s: razón_h = Σ real_h / Σ pronóstico_h sobre los días de s en la ventana;
    factor_h = 1 + n/(n+k0)·g·(razón_h − 1); curva·factor. Con evidencia < `minimo_casos` el día queda igual.
    `segmento`: "tipo" (laboral/sábado/domingo/festivo), "dow" (día de semana; festivo aparte) o "global".
    `modo`: "hora" (factor por hora), "energia" (un factor por día = razón de energías) o "forma" (por hora pero
    conservando la energía del día).

    ADVERTENCIA: no activar por defecto. Backtest (modelo_k1): mejora calibración pero no prueba (Atlántico +0.15 pp WAPE).
    `historico_pronosticos` debe contener lo que el modelo/pipeline pronosticó en su momento (misma salida que se corrige).
    """
    p = _validar_sesgo(params)
    corte = _fecha(fecha_corte)
    fest = {str(f)[:10] for f in festivos}
    pares = _pares(historico_pronosticos, historico_real, corte)
    etiquetas = {d: _etiqueta(d, fest, p["segmento"]) for d in pares}
    cache: Dict[str, Optional[np.ndarray]] = {}
    out: Curves = {}
    for f, curva in curvas.items():
        c = _como_array(curva)
        if c is None:
            out[f] = list(curva) if isinstance(curva, (list, tuple)) else curva
            continue
        d = _fecha(f)
        seg = _etiqueta(d, fest, p["segmento"])
        if seg not in cache:
            semanas = p["ventana_festivo_semanas"] if seg == data.TIPO_FESTIVO else p["ventana_semanas"]
            fechas = [x for x in _ventana(pares, corte, semanas) if etiquetas[x] == seg]
            cache[seg] = _factor(pares, fechas, p) if len(fechas) >= p["minimo_casos"] else None
        factor = cache[seg]
        if factor is None:
            out[f] = c.tolist()
            continue
        v = c * factor
        if p["modo"] == "forma":
            v = v * (c.sum() / v.sum())
        out[f] = v.tolist()
    return out


def _factor(pares: Dict[date, Tuple[np.ndarray, np.ndarray]], fechas: List[date], p: Dict[str, Any]) -> np.ndarray:
    P = np.vstack([pares[d][0] for d in fechas])
    R = np.vstack([pares[d][1] for d in fechas])
    n = len(fechas)
    razon = np.full(24, R.sum() / P.sum()) if p["modo"] == "energia" else R.sum(0) / P.sum(0)
    lam = n / (n + p["k0"]) * p["g"]
    return _suavizar(1 + lam * (razon - 1), p["suavizado_horas"])


# ---------------------------------------------------------------------------
# Clima (opt-in, experimental)
# ---------------------------------------------------------------------------

def anomalia_temperatura(clima_dia: Mapping[str, float], d: date, dias_previos: int = 28) -> Optional[float]:
    """
    T media del día `d` − T media de los `dias_previos` días hasta el corte de su semana (sábado anterior).
    None si falta la T del día o hay menos de la mitad de los días previos.
    """
    T = {str(k)[:10]: float(v) for k, v in clima_dia.items() if v is not None and math.isfinite(float(v))}
    if d.isoformat() not in T:
        return None
    corte = data.corte_semana(d)
    previos = [T[(corte - timedelta(days=i)).isoformat()] for i in range(dias_previos)
               if (corte - timedelta(days=i)).isoformat() in T]
    if len(previos) < max(1, dias_previos // 2):
        return None
    return T[d.isoformat()] - float(np.mean(previos))


def corregir_clima(
    curvas: Curves,
    historico_pronosticos: Mapping[str, Any],
    historico_real: Mapping[str, Any],
    clima_dia: Mapping[str, float],
    fecha_corte: Any,
    festivos: Iterable[str],
    params: Optional[Mapping[str, Any]] = None,
) -> Curves:
    """
    Regresión móvil de log(E_real/E_pronóstico) del día sobre la anomalía de temperatura (`clima_dia` = T media diaria
    {fecha: °C}); multiplica la curva del día por exp(g·b·(dT − dT̄)). Solo pendiente (no corrige el sesgo medio); entrena con
    días con fecha <= `fecha_corte` y dT disponible (ridge con `k0`); con menos de `minimo_casos` el día queda igual.

    EXPERIMENTAL: el backtest usó temperatura OBSERVADA del día objetivo (optimista); en producción hay que entregar un
    pronóstico de temperatura, cuyo error no está medido. Antioquia solo tiene clima desde 2025-12-01 (no calibrable).
    """
    p = _validar_clima(params)
    corte = _fecha(fecha_corte)
    fest = {str(f)[:10] for f in festivos}
    pares = _pares(historico_pronosticos, historico_real, corte)
    xs: List[float] = []
    es: List[float] = []
    for d in _ventana(pares, corte, p["ventana_semanas"]):
        if p["solo_laboral"] and data.tipo_dia(d, fest) != data.TIPO_LABORAL:
            continue
        x = anomalia_temperatura(clima_dia, d, p["dias_previos"])
        if x is None:
            continue
        pr, re = pares[d]
        xs.append(x)
        es.append(math.log(re.sum() / pr.sum()))
    out: Curves = {f: (list(v) if isinstance(v, (list, tuple)) else v) for f, v in curvas.items()}
    if len(xs) < p["minimo_casos"]:
        return out
    x, e = np.asarray(xs), np.asarray(es)
    xm, em = x.mean(), e.mean()
    vx = float(((x - xm) ** 2).sum())
    if vx <= 1e-9:
        return out
    b = float(((x - xm) * (e - em)).sum() / (vx + p["k0"]))
    for f, curva in curvas.items():
        c = _como_array(curva)
        if c is None:
            continue
        d = _fecha(f)
        if p["solo_laboral"] and data.tipo_dia(d, fest) != data.TIPO_LABORAL:
            continue
        dT = anomalia_temperatura(clima_dia, d, p["dias_previos"])
        if dT is None:
            continue
        out[f] = (c * math.exp(p["g"] * b * (dT - xm))).tolist()
    return out
