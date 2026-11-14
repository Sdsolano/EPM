"""
Insights de la mesa de ajustes: lógica pura (sin FastAPI) detrás de `/adjustments/insights/*`.
Contrato: docs/work/ajustes/noche2/CONTRATO_B.md.

Fuentes de datos
- Pronósticos emitidos históricos: `precision_data/{ucp}_k1.csv` (fecha, semana_inicio, P1..P24; generado por
  `scripts/generar_historial_pronosticos.py`). Se leen con caché por (ruta, mtime).
- Real: `data.curvas_reales(histórico)`.
- Corridas aprobadas del store (opcional): prevalecen sobre el CSV en `precision`.

Convenciones: porcentajes en puntos porcentuales ×100 (3.2 = 3.2 %), energía en MWh/día, potencia en MW, hora
0..23 = periodo P1..P24. Valores no calculables = None. Nada lanza por falta de historial: se degrada a
None / [] / "sin_datos".

Umbrales (documentados aquí, no son parámetros del modelo):
- Confianza: WAPE reciente ≤ 3 % alta, ≤ 6 % media, > 6 % baja (`UMBRAL_CONFIANZA_ALTA/MEDIA`).
- Recomendaciones de sesgo (tendencia reciente, NO una predicción ni una mejora respaldada): n ≥ 6 días, |sesgo| ≥ 0.5 %,
  mismo signo en ≥ 65 % de los días, la corrección reduce el error medido sin mirar el día evaluado (leave-one-out) Y la
  misma regla, simulada con corte creciente en las últimas ~8 semanas (52 para festivos), actuó en ≥ ROLLING_MIN_DIAS días,
  redujo el error en ≥ 75 % de ellos y mejoró el WAPE del segmento ≥ 0.3 pp (`validar_rolling`). OJO: esa simulación se
  evalúa sobre las mismas semanas que originan la regla (selección sobre ventana corta), NO es validación independiente;
  un backtest sin fuga (B5) mostró que no mejora fuera de muestra (Antioquia 3.08→2.71 por miércoles; Atlántico 5.03→7.13).
  Por eso se emiten con prioridad "baja", `respaldo: "tendencia_reciente"` y `validado_fuera_de_muestra: false`; solo las
  alertas del sistema (`respaldo: "alerta_del_sistema"`) cuentan como recomendación accionable en el resumen.
- Banda: cuantiles empíricos p10/p50/p90 con ≥ 5 casos por tipo de día; festivo con < 10 casos usa domingo. Con run_id, la
  muestra de errores termina en min(semana − 1 día, fecha_corte de la corrida): sin fuga.
- Historial estático: `precision_data/*_k1.csv` es una foto (Antioquia termina 2026-06-28, Atlántico 2026-05-03). La evidencia
  de sesgo caduca con `MAX_ANTIGUEDAD_DIAS`; para refrescarla, correr `scripts/generar_historial_pronosticos.py` tras regenerar
  las curvas k1 (o persistir los pronósticos de las corridas aprobadas: pendiente).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.adjustments import data
from src.adjustments.types import MOTIVOS_CURVA, Alert, StepRecord, parse_date, validar_curva

logger = logging.getLogger(__name__)

PRECISION_DIR = Path(__file__).resolve().parent / "precision_data"
PERIODOS = data.PERIODOS

UMBRAL_CONFIANZA_ALTA = 3.0
UMBRAL_CONFIANZA_MEDIA = 6.0
MIN_N_RECOMENDACION = 6
MIN_EFECTO_PCT = 0.5
MIN_CONSISTENCIA = 0.65
MEJORA_FORMA_PP = 0.3          # la forma por hora debe mejorar ≥ 0.3 pp el error frente a solo nivel
OCURRENCIAS_DIA_SEMANA = 8     # últimos N mismos días de semana (no festivos)
OCURRENCIAS_FESTIVO = 20
MAX_ANTIGUEDAD_DIAS = 140      # la evidencia de sesgo no puede terminar más de 20 semanas antes de la semana objetivo
MIN_N_BANDA = 5
MIN_FESTIVOS_BANDA = 10
NIVEL_CONFIANZA_BANDA = 0.8
ROLLING_SEMANAS = 8            # ventana de la validación fuera de muestra (días de semana)
ROLLING_SEMANAS_FESTIVO = 52   # los festivos son pocos: ventana más larga
ROLLING_MIN_DIAS = 8           # días mínimos en los que la regla habría actuado en la ventana de selección
ROLLING_MIN_MEJORA = 0.75      # fracción mínima de días de la ventana en que la corrección redujo el error
ROLLING_MIN_MEJORA_WAPE_PP = 0.3  # el WAPE del segmento (ventana de selección) debe bajar al menos 0.3 pp
RESPALDO_TENDENCIA = "tendencia_reciente"
RESPALDO_ALERTA = "alerta_del_sistema"
AVISO_TENDENCIA = ("Tendencia reciente observada; las pruebas del sistema no confirmaron que corregirla mejore el "
                   "pronóstico. Úsala como referencia, tú decides.")
# Factor de calibración de la banda: los cuantiles p10/p90 del error relativo se multiplican por este factor (p50 no). Sin
# él la banda de 80 % subcubre; con 1.2 (elegido en la calibración de B2) cubre más. Constante propia de este módulo: no se
# importa de `correccion` (opt-in, otro estimador).
ESCALA_BANDA = 1.2
COBERTURA_MEDIDA = {
    "nominal": 0.8,
    "antioquia_prueba": 0.876,
    "atlantico_prueba": 0.802,
    "antioquia_sin_escala": 0.813,
    "atlantico_sin_escala": 0.732,
    "ancho_medio_relativo_pct": {"antioquia": 9.9, "atlantico": 20.1},
    "nota": "Medida el 2026-11-14 con esta implementación (error relativo simple, escala 1.2 sobre p10/p90, banda con todo el "
            "historial previo a cada semana) en backtest sin fuga de 21 semanas: Antioquia 87.6 %, Atlántico 80.2 % (sin escala "
            "81.3 % / 73.2 %). Mínimos por segmento: festivo Antioquia 71 % (7 días), sábado Antioquia 82 %. No es una garantía.",
}
SEMANAS_CONFIANZA = 8
TENDENCIA_UMBRAL_PP = 0.3
PASOS_PARCHE = ("forma", "especiales")  # pasos editables donde un día editado se respeta hasta el final

TIPOS_DIA = data.TIPOS_DIA
DIAS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
DIAS_ABREV = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
MESES_ABREV = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
NOMBRES_UCP = {"Atlantico": "Atlántico"}
ETIQUETA_TIPO = {"laboral": "Laboral", "sabado": "Sábado", "domingo": "Domingo", "festivo": "Festivo"}
_PRIORIDAD_ORD = {"alta": 0, "media": 1, "baja": 2}
_NIVEL_ORD = {"critico": 0, "warning": 1, "info": 2}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _r(x: Any, nd: int = 1) -> Optional[float]:
    """float redondeado, o None si no es finito."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


def _lista(arr: Any, nd: int = 1) -> List[Optional[float]]:
    return [_r(v, nd) for v in np.asarray(arr, dtype=float).ravel().tolist()]


def wape(pronostico: Any, real: Any) -> Optional[float]:
    """100·Σ|f−r|/Σr (None si Σr no es positiva)."""
    f, r = np.asarray(pronostico, dtype=float), np.asarray(real, dtype=float)
    den = float(np.sum(r))
    if not math.isfinite(den) or den <= 0:
        return None
    return float(np.sum(np.abs(f - r)) / den * 100)


def _sesgo(pronostico: Any, real: Any) -> Optional[float]:
    """100·(Σf−Σr)/Σr: negativo = el pronóstico se quedó corto."""
    den = float(np.sum(np.asarray(real, dtype=float)))
    if not math.isfinite(den) or den <= 0:
        return None
    return float((np.sum(np.asarray(pronostico, dtype=float)) - den) / den * 100)


def fecha_larga(d: date) -> str:
    return f"{d.day} {MESES_ABREV[d.month - 1]}"


def rango_semana_texto(lunes: date) -> str:
    dom = lunes + timedelta(days=6)
    if lunes.month == dom.month:
        return f"{lunes.day} al {fecha_larga(dom)}"
    return f"{fecha_larga(lunes)} al {fecha_larga(dom)}"


def nombre_dia(fecha: Any) -> str:
    return DIAS_ES[parse_date(fecha).weekday()]


def nombre_ucp(ucp: str) -> str:
    return NOMBRES_UCP.get(ucp, ucp)


# ---------------------------------------------------------------------------
# Pronósticos emitidos (CSV compacto, caché por ruta+mtime)
# ---------------------------------------------------------------------------

Curvas = Dict[str, np.ndarray]


@lru_cache(maxsize=16)
def _leer_csv(ruta: str, mtime: float) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    df = pd.read_csv(ruta)
    faltan = [c for c in ("fecha", "semana_inicio", *PERIODOS) if c not in df.columns]
    if faltan:
        raise ValueError(f"{ruta}: faltan columnas {faltan}")
    fechas = pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d").tolist()
    semanas = pd.to_datetime(df["semana_inicio"]).dt.strftime("%Y-%m-%d").tolist()
    valores = df[PERIODOS].to_numpy(dtype=float)
    curvas: Curvas = {}
    semana_de: Dict[str, str] = {}
    for f, s, v in zip(fechas, semanas, valores):
        if np.isfinite(v).all() and (v > 0).all():
            v = v.copy()
            v.setflags(write=False)
            curvas[f] = v
            semana_de[f] = s
    return curvas, semana_de


def cargar_emitidos(ucp: str, directorio: Optional[Path] = None) -> Tuple[Curvas, Dict[str, str]]:
    """({fecha: array(24)}, {fecha: semana_inicio}) del historial emitido; vacío si no hay archivo. No mutar."""
    ruta = Path(directorio or PRECISION_DIR) / f"{ucp}_k1.csv"
    try:
        return _leer_csv(str(ruta), ruta.stat().st_mtime)
    except FileNotFoundError:
        return {}, {}
    except (ValueError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        logger.warning("[insights] no se pudo leer %s: %s", ruta, e)
        return {}, {}


@lru_cache(maxsize=8)
def _festivos_anios(anio_ini: int, anio_fin: int) -> frozenset:
    return frozenset(data.festivos_colombia(range(anio_ini, anio_fin + 1)))


def festivos_por_defecto(fechas_iso: Iterable[str]) -> Set[str]:
    """Festivos oficiales (librería, sin red) que cubren las fechas dadas (±1 año)."""
    anios = [int(f[:4]) for f in fechas_iso]
    if not anios:
        anios = [date.today().year]
    return set(_festivos_anios(min(anios) - 1, max(anios) + 1))


@dataclass
class Datos:
    """Todo lo que necesitan los cálculos: emitidos, reales, festivos y corridas aprobadas."""
    emitidos: Curvas = field(default_factory=dict)
    semana_de: Dict[str, str] = field(default_factory=dict)
    reales: Curvas = field(default_factory=dict)
    festivos: Set[str] = field(default_factory=set)
    aprobadas: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)  # semana → {fecha: curva}

    def tipo(self, fecha: str) -> str:
        return data.tipo_dia(fecha, self.festivos)


def _pares(d: Datos, desde: Optional[str] = None, hasta: Optional[str] = None) -> List[str]:
    fs = sorted(set(d.emitidos) & set(d.reales))
    return [f for f in fs if (desde is None or f >= desde) and (hasta is None or f <= hasta)]


# ---------------------------------------------------------------------------
# Precisión
# ---------------------------------------------------------------------------

def _semanas_completas(d: Datos) -> Dict[str, Tuple[Curvas, str]]:
    """{lunes: ({fecha: pronóstico}, fuente)} solo con 7 días de pronóstico y 7 de real."""
    por_semana: Dict[str, Curvas] = {}
    for f, s in d.semana_de.items():
        if f in d.emitidos:
            por_semana.setdefault(s, {})[f] = d.emitidos[f]
    fuentes = {s: "modelo_historico" for s in por_semana}
    for s, curvas in d.aprobadas.items():
        validas = {f: np.asarray(v, dtype=float) for f, v in curvas.items()
                   if validar_curva(v) is None and np.all(np.asarray(v, dtype=float) > 0)}
        if validas:
            por_semana[s] = validas
            fuentes[s] = "corrida_aprobada"
    out = {}
    for s, curvas in por_semana.items():
        fechas = [data.iso(x) for x in data.dias_semana(parse_date(s))]
        if all(f in curvas and f in d.reales for f in fechas):
            out[s] = ({f: curvas[f] for f in fechas}, fuentes[s])
    return out


def _tendencia(wapes: List[float]) -> str:
    if len(wapes) < 4:
        return "sin_datos"
    mitad = len(wapes) // 2
    antes, despues = float(np.mean(wapes[:mitad])), float(np.mean(wapes[-mitad:]))
    if despues < antes - TENDENCIA_UMBRAL_PP:
        return "mejorando"
    if despues > antes + TENDENCIA_UMBRAL_PP:
        return "empeorando"
    return "estable"


def precision(d: Datos, semanas: int = 12, hasta: Optional[date] = None) -> Dict[str, Any]:
    """Pronóstico emitido vs real por semana/día/tipo de día/hora (contrato §2)."""
    completas = _semanas_completas(d)
    lunes = sorted(s for s in completas if hasta is None or s <= hasta.isoformat())[-max(1, semanas):]
    vacio_h = [None] * 24
    if not lunes:
        return {"ucp": None, "desde": None, "hasta": None,
                "resumen": {"semanas": 0, "wape_pct": None, "mape_energia_semana_pct": None,
                            "sesgo_energia_pct": None, "mejor_semana": None, "peor_semana": None,
                            "tendencia": "sin_datos"},
                "semanas": [], "por_tipo_dia": [], "por_hora": {"sesgo_pct": vacio_h, "wape_pct": vacio_h}}

    out_semanas, wapes, errs_energia = [], [], []
    todos_f, todos_r, tipos = [], [], []
    for s in lunes:
        curvas, fuente = completas[s]
        fechas = sorted(curvas)
        F = np.vstack([curvas[f] for f in fechas])
        R = np.vstack([d.reales[f] for f in fechas])
        dias = []
        for i, f in enumerate(fechas):
            tipo = d.tipo(f)
            dias.append({
                "fecha": f, "dia": nombre_dia(f), "tipo_dia": tipo,
                "wape_pct": _r(wape(F[i], R[i])), "error_energia_pct": _r(_sesgo(F[i], R[i])),
                "energia_pronosticada_mwh": _r(F[i].sum()), "energia_real_mwh": _r(R[i].sum()),
                "pico_pronosticado_mw": _r(F[i].max()), "pico_real_mw": _r(R[i].max()),
                "hora_pico_pronosticada": int(F[i].argmax()), "hora_pico_real": int(R[i].argmax()),
            })
            tipos.append(tipo)
        w = wape(F, R)
        e = _sesgo(F, R)
        wapes.append(w)
        errs_energia.append(e)
        todos_f.append(F)
        todos_r.append(R)
        out_semanas.append({
            "semana_inicio": s, "fuente": fuente, "wape_pct": _r(w),
            "energia_pronosticada_mwh": _r(F.sum()), "energia_real_mwh": _r(R.sum()),
            "error_energia_pct": _r(e),
            "pico_error_pct": _r((F.max() - R.max()) / R.max() * 100 if R.max() > 0 else None),
            "dias_con_datos": len(fechas), "dias": dias,
        })

    F, R = np.vstack(todos_f), np.vstack(todos_r)
    wv = [w for w in wapes if w is not None]
    mejor = lunes[int(np.argmin([w if w is not None else np.inf for w in wapes]))]
    peor = lunes[int(np.argmax([w if w is not None else -np.inf for w in wapes]))]
    por_tipo = []
    tipos_arr = np.array(tipos)
    for t in TIPOS_DIA:
        m = tipos_arr == t
        if m.any():
            por_tipo.append({"tipo_dia": t, "wape_pct": _r(wape(F[m], R[m])),
                             "sesgo_energia_pct": _r(_sesgo(F[m], R[m])), "n_dias": int(m.sum())})
    sesgo_h = [_r((F[:, h].sum() - R[:, h].sum()) / R[:, h].sum() * 100) if R[:, h].sum() > 0 else None
               for h in range(24)]
    wape_h = [_r(wape(F[:, h], R[:, h])) for h in range(24)]
    return {
        "ucp": None, "desde": lunes[0], "hasta": lunes[-1],
        "resumen": {
            "semanas": len(lunes), "wape_pct": _r(wape(F, R)),
            "mape_energia_semana_pct": _r(np.mean([abs(e) for e in errs_energia if e is not None]))
            if any(e is not None for e in errs_energia) else None,
            "sesgo_energia_pct": _r(_sesgo(F, R)), "mejor_semana": mejor, "peor_semana": peor,
            "tendencia": _tendencia(wv),
        },
        "semanas": out_semanas, "por_tipo_dia": por_tipo,
        "por_hora": {"sesgo_pct": sesgo_h, "wape_pct": wape_h},
    }


def nivel_confianza(wape_pct: Optional[float]) -> str:
    if wape_pct is None:
        return "sin_datos"
    if wape_pct <= UMBRAL_CONFIANZA_ALTA:
        return "alta"
    if wape_pct <= UMBRAL_CONFIANZA_MEDIA:
        return "media"
    return "baja"


# ---------------------------------------------------------------------------
# Banda empírica
# ---------------------------------------------------------------------------

def tablas_banda(d: Datos, fecha_inicio: Optional[str] = None, fecha_fin: Optional[str] = None,
                 ) -> Dict[str, Dict[str, Any]]:
    """
    {tipo_dia: {"n_dias", "p10","p50","p90" (arrays de 24 en fracción, o None), "nota"?}} con cuantiles (p10/p90 × ESCALA_BANDA) del
    error relativo real/pronóstico − 1. Festivo con < 10 casos usa la banda de domingo (con nota).
    """
    por_tipo: Dict[str, List[np.ndarray]] = {t: [] for t in TIPOS_DIA}
    for f in _pares(d, fecha_inicio, fecha_fin):
        por_tipo[d.tipo(f)].append(d.reales[f] / d.emitidos[f] - 1.0)
    out: Dict[str, Dict[str, Any]] = {}
    for t, errs in por_tipo.items():
        n = len(errs)
        if n >= MIN_N_BANDA:
            E = np.vstack(errs)
            p10, p50, p90 = (np.percentile(E, q, axis=0) for q in (10, 50, 90))
            # Calibración: ensancha los extremos (sobre 0, como B2); nunca cruza la mediana.
            p10 = np.minimum(p10 * ESCALA_BANDA, p50)
            p90 = np.maximum(p90 * ESCALA_BANDA, p50)
            out[t] = {"n_dias": n, "p10": p10, "p50": p50, "p90": p90}
        else:
            out[t] = {"n_dias": n, "p10": None, "p50": None, "p90": None}
    fest = out[data.TIPO_FESTIVO]
    dom = out[data.TIPO_DOMINGO]
    if fest["n_dias"] < MIN_FESTIVOS_BANDA and dom["p10"] is not None:
        out[data.TIPO_FESTIVO] = {**dom, "n_dias": fest["n_dias"],
                                  "nota": f"Pocos festivos con historial ({fest['n_dias']}); se usa la banda de domingo."}
    return out


def banda_json(tablas: Dict[str, Dict[str, Any]], d: Datos, tipo: Optional[str] = None,
               fecha_inicio: Optional[str] = None, fecha_fin: Optional[str] = None) -> Dict[str, Any]:
    """JSON de la banda; `n_semanas_historial` cuenta las semanas de la muestra realmente usada (filtros y tipo)."""
    por_tipo = {}
    for t in TIPOS_DIA:
        if tipo is not None and t != tipo:
            continue
        x = tablas[t]
        item: Dict[str, Any] = {"n_dias": x["n_dias"]}
        for k in ("p10", "p50", "p90"):
            item[f"{k}_pct"] = _lista(x[k] * 100, 2) if x[k] is not None else None
        if "nota" in x:
            item["nota"] = x["nota"]
        por_tipo[t] = item
    semanas = {d.semana_de[f] for f in _pares(d, fecha_inicio, fecha_fin)
               if f in d.semana_de and (tipo is None or d.tipo(f) == tipo)}
    return {"nivel_confianza": NIVEL_CONFIANZA_BANDA, "metodo": "cuantiles_empiricos_error_relativo",
            "escala_calibracion": ESCALA_BANDA, "cobertura_medida": dict(COBERTURA_MEDIDA),
            "n_semanas_historial": len(semanas), "por_tipo_dia": por_tipo}


def aplicar_banda(curva: Any, tipo: str, tablas: Dict[str, Dict[str, Any]]
                  ) -> Optional[Tuple[List[Optional[float]], List[Optional[float]]]]:
    """(inferior, superior) en MW para una curva de 24 valores; None si no hay banda o la curva no es válida."""
    x = tablas.get(tipo)
    if x is None or x["p10"] is None or validar_curva(curva) is not None:
        return None
    c = np.asarray(curva, dtype=float)
    return _lista(c * (1 + x["p10"]), 2), _lista(c * (1 + x["p90"]), 2)


# ---------------------------------------------------------------------------
# Sesgo por segmento
# ---------------------------------------------------------------------------

def _segmento(tipo: str, fecha: Any) -> str:
    """Segmento de sesgo: "festivo" o el día de semana (no festivo). Único lugar donde se define."""
    return data.TIPO_FESTIVO if tipo == data.TIPO_FESTIVO else nombre_dia(fecha)


def _loo_error(F: np.ndarray, R: np.ndarray, forma: bool) -> float:
    """WAPE medio por día de corregir F con factores estimados SIN el día evaluado (leave-one-out)."""
    errs = []
    n = len(F)
    for i in range(n):
        m = np.arange(n) != i
        if forma:  # misma normalización de energía que `_corregir_curva`
            g = np.mean(R[m] / F[m], axis=0)
            g = g * (F[i].sum() / float((F[i] * g).sum()))
        else:
            en = (F[m].sum(axis=1) - R[m].sum(axis=1)) / R[m].sum(axis=1)
            g = 1.0 / (1.0 + float(np.mean(en)))
        errs.append(wape(F[i] * g, R[i]))
    return float(np.mean([e for e in errs if e is not None]))


def sesgo_por_segmento(d: Datos, hasta: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Sesgo reciente del pronóstico emitido por segmento (día de semana no festivo o "festivo"), sobre los
    últimos N días emitidos con real. Cada segmento: n, sesgo_pct (negativo = corto), consistencia (fracción de
    días con el mismo signo del sesgo), error_antes_pct / error_nivel_pct / error_forma_pct (WAPE medio por día,
    corrección estimada sin el día evaluado), factor_nivel (≈ real/pronóstico) y factores_hora (24).
    """
    seg_fechas: Dict[str, List[str]] = {}
    for f in _pares(d, hasta=hasta):
        seg_fechas.setdefault(_segmento(d.tipo(f), f), []).append(f)
    out: Dict[str, Dict[str, Any]] = {}
    for seg, fechas in seg_fechas.items():
        sg = _stats_segmento(d, fechas[-_ocurrencias(seg):])
        if sg is not None:
            out[seg] = sg
    return out


def _ocurrencias(seg: str) -> int:
    return OCURRENCIAS_FESTIVO if seg == data.TIPO_FESTIVO else OCURRENCIAS_DIA_SEMANA


def _stats_segmento(d: Datos, fechas: List[str]) -> Optional[Dict[str, Any]]:
    """Estadísticas de sesgo de un conjunto de días emitidos con real (None si hay < 3)."""
    n = len(fechas)
    if n < 3:
        return None
    F = np.vstack([d.emitidos[f] for f in fechas])
    R = np.vstack([d.reales[f] for f in fechas])
    en = (F.sum(axis=1) - R.sum(axis=1)) / R.sum(axis=1)
    s = float(np.mean(en))
    signo = np.sign(s)
    consistencia = float(np.mean(np.sign(en) == signo)) if signo != 0 else 0.0
    antes = float(np.mean([wape(F[i], R[i]) for i in range(n)]))
    return {
        "n": n, "sesgo_pct": s * 100, "consistencia": consistencia, "fechas": fechas,
        "error_antes_pct": antes,
        "error_nivel_pct": _loo_error(F, R, False) if n >= 4 else None,
        "error_forma_pct": _loo_error(F, R, True) if n >= 4 else None,
        "factor_nivel": 1.0 / (1.0 + s),
        "factores_hora": np.mean(R / F, axis=0),
    }


# ---------------------------------------------------------------------------
# Vista de una corrida guardada
# ---------------------------------------------------------------------------

@dataclass
class Corrida:
    ucp: str
    semana_inicio: date
    estado: str
    version: int
    base: Dict[str, List[Optional[float]]]
    final: Dict[str, List[Optional[float]]]
    alertas: List[Tuple[str, Alert]]            # (paso, alerta)
    editadas: Set[str]                          # fechas con edición manual en algún paso
    estados_paso: Dict[str, str]
    ref_inicio: Optional[date]
    festivos: Set[str]
    fecha_corte: Optional[date] = None          # último dato real con el que se emitió la corrida

    @property
    def fechas(self) -> List[str]:
        return [data.iso(x) for x in data.dias_semana(self.semana_inicio)]

    def tipo(self, fecha: str) -> str:
        """laboral / sabado / domingo / festivo de una fecha según los festivos de la corrida."""
        return data.tipo_dia(fecha, self.festivos)

    @property
    def fecha_fin_sin_fuga(self) -> str:
        """Último día con real que se puede usar como evidencia sin fuga: min(semana − 1, fecha_corte)."""
        corte = self.semana_inicio - timedelta(days=1)
        if self.fecha_corte is not None:
            corte = min(corte, self.fecha_corte)
        return corte.isoformat()


def corrida_desde(meta: Dict[str, Any], records: List[StepRecord], final: Dict[str, Any]) -> Corrida:
    por_paso = {r.paso: r for r in records}
    alertas = [(r.paso, a) for r in records for a in r.result.alerts]
    editadas: Set[str] = set()
    for r in records:
        if str(getattr(r.estado, "value", r.estado)) == "editado":
            editadas.update(str(f)[:10] for f in (r.result.diagnostics.get("fechas_editadas") or []))
    ref = None
    if "referencia" in por_paso:
        try:
            ref = parse_date(por_paso["referencia"].result.diagnostics.get("semana_ref_inicio"))
        except (TypeError, ValueError):
            ref = None
    base = por_paso["base"].result.curves if "base" in por_paso else {}
    extras = meta.get("extras") or {}
    semana = parse_date(meta["semana_inicio"])
    festivos = {str(f)[:10] for f in (extras.get("festivos") or [])}
    if not festivos:  # corridas sin festivos guardados: calendario oficial (sin red)
        festivos = festivos_por_defecto([data.iso(x) for x in data.dias_semana(semana)])
    try:
        corte = parse_date(meta.get("fecha_corte"))
    except (TypeError, ValueError):
        corte = None
    try:
        version = int(meta.get("version") or 1)
    except (TypeError, ValueError):
        version = 1
    return Corrida(
        ucp=meta["ucp"], semana_inicio=semana, estado=str(meta.get("estado") or "borrador"),
        version=version, base=dict(base), final=dict(final), alertas=alertas, editadas=editadas,
        estados_paso={r.paso: str(getattr(r.estado, "value", r.estado)) for r in records},
        ref_inicio=ref, festivos=festivos, fecha_corte=corte,
    )


def _curva_valida(curvas: Dict[str, Any], f: str) -> Optional[np.ndarray]:
    c = curvas.get(f)
    return np.asarray(c, dtype=float) if validar_curva(c) is None else None


# ---------------------------------------------------------------------------
# Textos de alertas en lenguaje de negocio
# ---------------------------------------------------------------------------

# código → (título, texto). El texto puede usar {dias}. Códigos informativos técnicos: no se muestran.
_TEXTOS_ALERTA: Dict[str, Tuple[str, str]] = {
    "TENDENCIA_FUERA_BANDA": (
        "La energía de la semana se aleja de lo esperado",
        "La energía total de la semana quedó {desvio} % distinta de la que calculó el sistema. "
        "Revisa los días señalados ({dias}) antes de aprobar."),
    "TENDENCIA_REFERENCIA_FUERA_BANDA": (
        "La tendencia se aleja de la del año anterior",
        "La evolución de la semana no se parece a la de la semana de comparación. Revisa los días señalados ({dias})."),
    "SIN_EVIDENCIA_NIVEL": (
        "Sin datos comparables para ajustar el nivel",
        "No hay una semana comparable con datos para esos días ({dias}); se dejó el nivel del modelo. "
        "Revísalos con tu criterio."),
    "CURVA_BASE_NULA": (
        "Hay días sin pronóstico válido", "El pronóstico de {dias} no tiene valores útiles; no se pudo ajustar."),
    "FESTIVOS_FUENTE_ALTERNA": (
        "Calendario de festivos de respaldo",
        "No se pudo consultar el calendario de festivos en línea; se usó el calendario oficial guardado. "
        "Confirma que los festivos de la semana sean los correctos."),
    "BRECHA_INVALIDA": (
        "Faltan datos entre el último dato real y la semana",
        "Los días estimados antes de la semana ({dias}) vienen incompletos; no impide aprobar, pero la "
        "comparación con la semana previa queda incompleta."),
    "BASE_INCOMPLETA": (
        "El modelo no entregó todos los días", "Faltan pronósticos del modelo para {dias}."),
    "ESPECIALES_SIN_BACKTEST": (
        "Ajuste de días especiales sin comprobación histórica",
        "Los días especiales se ajustaron sin poder comprobarlo con semanas pasadas; revisa el resultado antes de aprobar."),
    "ESPECIALES_SUGERIDOS": (
        "Hay días especiales con ajuste sugerido", "El sistema propone un ajuste para {dias}."),
    "RECONCILIACION_ROTA": (
        "El resultado no cuadra con el pronóstico", "Las curvas finales de {dias} no coinciden con la energía calculada; no se puede aprobar."),
    "REFERENCIA_ANIO_EXCLUIDO": (
        "Semana de comparación poco confiable", "La semana usada para comparar cae en un año atípico; úsala con cautela."),
    "REFERENCIA_SIN_DATOS": (
        "Sin semana de comparación", "No hay datos de una semana comparable; el ajuste se apoya solo en el modelo."),
    "REFERENCIA_SIN_HOMOLOGO": (
        "Festivo sin día equivalente en años anteriores", "No se encontró un festivo equivalente en años pasados; se comparó con la misma semana del año anterior."),
    "ELECCIONES_IA": (
        "Posibles elecciones o votación",
        "Se detectaron días con posibles elecciones ({dias}); no están verificados y no se ajustó ninguna curva. "
        "Confírmalos antes de tocar el pronóstico."),
}
# Bloqueos por curva inválida (códigos de `types.MOTIVOS_CURVA`): título del motivo + los días afectados.
_TEXTOS_ALERTA.update({
    motivo: (desc, f"{desc[0].upper()}{desc[1:]}: {{dias}}. Corrígelos antes de aprobar la semana.")
    for motivo, desc in MOTIVOS_CURVA.items()})
_INFO_OCULTAS = {"PARAMS_ANALOGOS_IGNORADOS", "NIVEL_ANCLA_ALTERNA", "FORMA_FUENTE_ALTERNA",
                 "REFERENCIA_AJUSTADA_A_LUNES", "ESPECIALES_EDICION_PRESERVADA", "ANALOGOS_NO_OBSERVADOS",
                 "SIN_ENERGIA_NIVEL", "SIN_CONTROL_TENDENCIA", "ESPECIALES_SIN_EVIDENCIA"}


def _lista_dias(fechas: List[str]) -> str:
    if not fechas:
        return "la semana"
    nombres = [f"{nombre_dia(f)} {parse_date(f).day}" for f in fechas[:4]]
    return ", ".join(nombres) + (f" y {len(fechas) - 4} más" if len(fechas) > 4 else "")


def _nombre_festivo(fecha: str) -> Optional[str]:
    try:
        from src.prediction.holiday_homology import holiday_name_for_date
        nombre = holiday_name_for_date(fecha)
    except Exception:  # noqa: BLE001 — el nombre es solo decorativo
        return None
    return nombre.replace(" (observado)", "") if nombre else None


def _nivel_insight(a: Alert) -> str:
    return "critico" if a.nivel == "bloqueo" else a.nivel


def _expandir_alerta(a: Alert) -> List[Dict[str, Any]]:
    """Alerta del pipeline → insights (uno o varios) con texto de negocio."""
    fechas = [str(f)[:10] for f in a.fechas]
    if a.codigo == "FESTIVO_EN_SEMANA":
        out = []
        for f in fechas:
            nombre = _nombre_festivo(f)
            dia = nombre_dia(f)
            out.append({"id": f"festivo-{f}", "nivel": "warning", "codigo": a.codigo, "categoria": "festivo",
                        "titulo": f"{dia.capitalize()} festivo" + (f" ({nombre})" if nombre else ""),
                        "texto": f"El {dia} es festivo: se espera menos demanda de lo normal. "
                                 "Revisa que el pronóstico baje.", "fechas": [f]})
        return out
    if a.codigo == "EVENTOS_SEMANA":
        ev = (a.detalle or {}).get("eventos") or {}
        det = "; ".join(f"{nombre_dia(f)} {parse_date(f).day}: {t}" for f, t in sorted(ev.items()))
        return [{"id": "eventos-semana", "nivel": "info", "codigo": a.codigo, "categoria": "alerta",
                 "titulo": "Posibles eventos que afectan la demanda",
                 "texto": "Se detectaron eventos que no están verificados; confírmalos antes de ajustar. " + det,
                 "fechas": fechas}]
    if a.codigo == "CLIMA_OBSERVADO":
        det = a.detalle or {}
        partes = []
        if det.get("temp_media") is not None:
            partes.append(f"temperatura media de {det['temp_media']:.1f} °C")
        if det.get("lluvia_mm") is not None:
            partes.append(f"lluvia de {det['lluvia_mm']:.0f} mm")
        extremo = a.nivel == "warning"
        return [{"id": "clima-semana", "nivel": a.nivel, "codigo": a.codigo, "categoria": "clima",
                 "titulo": "Clima fuera de lo habitual" if extremo else "Clima de la semana",
                 "texto": ("Se registró " + " y ".join(partes) + ("; es más extremo de lo normal y puede mover la demanda."
                                                                  if extremo else ".")) if partes else a.mensaje,
                 "fechas": fechas}]
    if a.nivel == "info" and a.codigo in _INFO_OCULTAS:
        return []
    titulo, texto = _TEXTOS_ALERTA.get(a.codigo, ("Aviso del sistema", a.mensaje))
    desvio = (a.detalle or {}).get("desvio_pp")
    texto = texto.replace("{dias}", _lista_dias(fechas)).replace(
        "{desvio}", f"{desvio:+.1f}" if isinstance(desvio, (int, float)) else "")
    return [{"id": f"{a.codigo.lower()}" + (f"-{fechas[0]}" if fechas else ""), "nivel": _nivel_insight(a),
             "codigo": a.codigo, "categoria": "alerta", "titulo": titulo, "texto": texto, "fechas": fechas}]


def insights_de_alertas(c: Corrida) -> List[Dict[str, Any]]:
    vistos: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for _paso, a in c.alertas:
        for i in _expandir_alerta(a):
            if i["id"] not in vistos:
                vistos.add(i["id"])
                out.append(i)
    out.sort(key=lambda i: (_NIVEL_ORD.get(i["nivel"], 3), i["fechas"][0] if i["fechas"] else ""))
    return out


# ---------------------------------------------------------------------------
# Recomendaciones
# ---------------------------------------------------------------------------

def _plural(seg: str) -> str:
    return {"sábado": "sábados", "domingo": "domingos", "festivo": "festivos"}.get(seg, seg)


def _paso_parche(c: Corrida) -> Optional[str]:
    for p in PASOS_PARCHE:
        if c.estados_paso.get(p) in ("aplicado", "editado"):
            return p
    return None


def _hay_evidencia(sg: Dict[str, Any]) -> bool:
    """Sesgo medido suficiente: n, efecto, consistencia de signo y mejora del error sin mirar el día evaluado."""
    return not (sg["n"] < MIN_N_RECOMENDACION or abs(sg["sesgo_pct"]) < MIN_EFECTO_PCT
                or sg["consistencia"] < MIN_CONSISTENCIA
                or sg["error_nivel_pct"] is None or sg["error_nivel_pct"] >= sg["error_antes_pct"])


def _categoria_regla(sg: Dict[str, Any]) -> str:
    """"forma" si el reparto por horas mejora ≥ MEJORA_FORMA_PP frente a solo nivel; si no "nivel"."""
    d_forma, d_nivel, antes = sg["error_forma_pct"], sg["error_nivel_pct"], sg["error_antes_pct"]
    return "forma" if d_forma is not None and d_forma <= d_nivel - MEJORA_FORMA_PP and d_forma < antes else "nivel"


def _corregir_curva(sg: Dict[str, Any], curva: np.ndarray, cat: str) -> np.ndarray:
    """Aplica la regla de corrección sobre una curva: forma = factores por hora con la misma energía; nivel = factor único."""
    if cat == "forma":
        sug = curva * np.asarray(sg["factores_hora"], dtype=float)
        return sug * (float(curva.sum()) / float(sug.sum()))
    return curva * sg["factor_nivel"]


def validar_rolling(d: Datos, seg: str, hasta: str) -> Dict[str, Any]:
    """
    Simulación de la regla de sesgo de un segmento sobre la ventana con la que se elige. Para cada día del segmento en las últimas
    `ROLLING_SEMANAS` semanas (52 para festivos) hasta `hasta`, estima el sesgo SOLO con días del segmento hasta el sábado
    previo a su semana (corte de emisión del modelo), aplica la regla si habría tenido evidencia, y compara el error del día.
    `pasa` = actuó en ≥ ROLLING_MIN_DIAS días, redujo el error en ≥ ROLLING_MIN_MEJORA de ellos y bajó el WAPE agregado del
    segmento ≥ ROLLING_MIN_MEJORA_WAPE_PP. NO es validación independiente: la ventana es la misma que origina la regla
    (selección sobre una ventana corta); sirve solo como filtro de emisión, no como evidencia de mejora futura.
    """
    semanas = ROLLING_SEMANAS_FESTIVO if seg == data.TIPO_FESTIVO else ROLLING_SEMANAS
    fechas_seg = [f for f in _pares(d, hasta=hasta) if _segmento(d.tipo(f), f) == seg]
    limite = (parse_date(hasta) - timedelta(weeks=semanas)).isoformat()
    F_antes: List[np.ndarray] = []
    F_desp: List[np.ndarray] = []
    R_ok: List[np.ndarray] = []
    mejoras = 0
    for i, f in enumerate(fechas_seg):
        if f <= limite:
            continue
        corte = data.corte_semana(parse_date(f)).isoformat()  # sábado anterior al lunes de su semana
        previas = [g for g in fechas_seg[:i] if g <= corte][-_ocurrencias(seg):]
        sg = _stats_segmento(d, previas)
        if sg is None or not _hay_evidencia(sg):
            continue
        F, R = np.asarray(d.emitidos[f], dtype=float), np.asarray(d.reales[f], dtype=float)
        corr = _corregir_curva(sg, F, _categoria_regla(sg))
        e0, e1 = wape(F, R), wape(corr, R)
        if e0 is None or e1 is None:
            continue
        mejoras += int(e1 < e0)
        F_antes.append(F)
        F_desp.append(corr)
        R_ok.append(R)
    n = len(R_ok)
    out: Dict[str, Any] = {"dias_evaluados": n, "dias_mejora": mejoras, "pct_dias_mejora": None,
                           "wape_antes_pct": None, "wape_despues_pct": None, "pasa": False}
    if n == 0:
        return out
    R_all = np.vstack(R_ok)
    w0, w1 = wape(np.vstack(F_antes), R_all), wape(np.vstack(F_desp), R_all)
    out.update(pct_dias_mejora=mejoras / n, wape_antes_pct=_r(w0, 2), wape_despues_pct=_r(w1, 2))
    out["pasa"] = bool(n >= ROLLING_MIN_DIAS and mejoras / n >= ROLLING_MIN_MEJORA
                       and w0 is not None and w1 is not None and w0 - w1 >= ROLLING_MIN_MEJORA_WAPE_PP)
    return out


def _rec_sesgo(c: Corrida, f: str, seg: str, sg: Dict[str, Any], paso: Optional[str],
               val: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Tendencia reciente del segmento (ya con evidencia y filtro rolling) como referencia para el analista: prioridad "baja",
    `respaldo: "tendencia_reciente"`, `validado_fuera_de_muestra: false`. La corrección se calcula sobre lo que hoy muestra
    `final`; el delta explicado es el real. Sin promesa de mejora (las pruebas no la confirmaron).
    """
    n, s = sg["n"], sg["sesgo_pct"]
    final = _curva_valida(c.final, f)
    base = _curva_valida(c.base, f)
    if final is None:
        return None
    e_final = float(final.sum())
    cat = _categoria_regla(sg)
    dia = nombre_dia(f)
    ref_dia = "festivo" if seg == data.TIPO_FESTIVO else dia
    festivo = seg == data.TIPO_FESTIVO
    pl = _plural(ref_dia)
    sug = _corregir_curva(sg, final, cat)
    if cat == "forma":
        g = np.asarray(sg["factores_hora"], dtype=float)
        desv = g / (np.sum(g * final) / e_final) - 1.0
        h = int(np.argmax(np.abs(desv)))
        titulo = ("Tendencia reciente: el reparto por horas de los festivos se desvió" if festivo
                  else f"Tendencia reciente: el reparto por horas del {ref_dia} se desvió")
        observado = (f"en los últimos {n} {pl} el reparto por horas del pronóstico no coincidió con lo real, "
                     f"sobre todo a las {h:02d}:00 (el pronóstico quedó {abs(desv[h]) * 100:.1f} % "
                     f"{'corto' if desv[h] > 0 else 'alto'})")
    else:
        if base is not None and base.sum() > 0:
            movido = e_final / float(base.sum()) - 1.0
            necesario = sg["factor_nivel"] - 1.0
            if movido * necesario > 0 and abs(movido) >= 0.5 * abs(necesario):
                return None  # el pipeline o el analista ya movieron el día en esa dirección
        corto = s < 0
        titulo = (f"Tendencia reciente: los festivos quedaron {'cortos' if corto else 'altos'}" if festivo
                  else f"Tendencia reciente: el {ref_dia} quedó {'corto' if corto else 'alto'}")
        observado = (f"en los últimos {n} {pl} el pronóstico se quedó {abs(s):.1f} % por "
                     f"{'debajo' if corto else 'encima'} de lo real")
    sug = np.round(sug, 2)
    if not np.all(np.isfinite(sug)) or np.any(sug <= 0):
        return None
    delta_mwh = float(sug.sum()) - e_final
    delta_pct = delta_mwh / e_final * 100
    if cat == "nivel":
        if abs(delta_pct) < MIN_EFECTO_PCT or delta_mwh * s > 0:  # sin efecto, o el signo contradice el sesgo medido
            return None
        resumen = f"{'Subir' if delta_mwh > 0 else 'Bajar'} el {ref_dia} {abs(delta_pct):.1f} % ({delta_mwh:+.0f} MWh)"
        accion = f"Sugerencia: {resumen[0].lower() + resumen[1:]} sobre lo que muestra hoy."
    else:
        resumen = f"Redistribuir el {ref_dia} por horas (misma energía)"
        accion = "Sugerencia: redistribuir el día por horas sin cambiar su energía."
    explicacion = (f"{AVISO_TENDENCIA} Observado: {observado}. {accion} En las últimas semanas (ventana con la que se "
                   f"eligió la regla) habría reducido el error en {val['dias_mejora']} de {val['dias_evaluados']} días.")
    ajuste = None
    if paso is not None:
        ajuste = {"resumen": resumen, "delta_energia_pct": _r(delta_pct), "delta_energia_mwh": _r(delta_mwh, 0),
                  "curva_sugerida": [float(x) for x in sug],
                  "patch": {"paso": paso,
                            "body": {"estado": "editado", "curvas": {f: [float(x) for x in sug]},
                                     "motivo": f"Tendencia reciente del {ref_dia} (referencia, no validada fuera de muestra)", "usuario": None}}}
    return {
        "id": f"sesgo-{f}", "fecha": f, "dia": dia, "tipo_dia": c.tipo(f), "prioridad": "baja",
        "respaldo": RESPALDO_TENDENCIA, "validado_fuera_de_muestra": False,
        "categoria": cat, "titulo": titulo, "explicacion": explicacion,
        "evidencia": {"n": n, "sesgo_pct": _r(s), "fuente": "historial",
                      "ventana_seleccion": {"dias_evaluados": val["dias_evaluados"], "dias_mejora": val["dias_mejora"],
                                            "pct_dias_mejora": _r(val["pct_dias_mejora"] * 100, 0),
                                            "wape_antes_pct": val["wape_antes_pct"],
                                            "wape_despues_pct": val["wape_despues_pct"]}},
        "impacto_esperado": f"Basado en lo ocurrido en los últimos {n} {pl}; no garantiza mejora.",
        "aplicable": ajuste is not None and c.estado == "borrador", "ajuste": ajuste,
    }


def _recs_alertas(c: Corrida) -> List[Dict[str, Any]]:
    """Alertas del pipeline que piden criterio humano (warning/bloqueo y festivos/eventos). El clima `info` no es accionable."""
    out = []
    for i in insights_de_alertas(c):
        if i["nivel"] == "info" and (i["categoria"] == "clima"
                                     or (i["categoria"] == "alerta" and i["codigo"] != "EVENTOS_SEMANA")):
            continue
        f = i["fechas"][0] if i["fechas"] else c.semana_inicio.isoformat()
        prioridad = "alta" if i["nivel"] == "critico" else ("media" if i["nivel"] == "warning" else "baja")
        out.append({
            "id": i["id"], "fecha": f, "dia": nombre_dia(f), "tipo_dia": c.tipo(f), "prioridad": prioridad,
            "categoria": i["categoria"], "titulo": i["titulo"], "explicacion": i["texto"],
            "respaldo": RESPALDO_ALERTA, "validado_fuera_de_muestra": None,
            "evidencia": {"n": None, "sesgo_pct": None, "fuente": "alerta_del_sistema"},
            "impacto_esperado": None, "aplicable": False, "ajuste": None,
        })
    return out


def recomendaciones(c: Corrida, d: Datos) -> Dict[str, Any]:
    """
    Recomendaciones por día (contrato §3): alertas reales del pipeline (`respaldo: "alerta_del_sistema"`, accionables) y,
    con evidencia estricta, tendencias recientes de sesgo (`respaldo: "tendencia_reciente"`, prioridad baja, referencia no
    validada fuera de muestra, nunca automáticas).
    """
    paso = _paso_parche(c)
    ultimo_real = max(d.reales) if d.reales else None
    corte = c.fecha_fin_sin_fuga
    hasta = min(corte, ultimo_real) if ultimo_real else corte
    sesgos = sesgo_por_segmento(d, hasta=hasta) if d.emitidos and d.reales else {}
    validaciones: Dict[str, Dict[str, Any]] = {}
    recs, resueltas = [], []
    for f in c.fechas:
        seg = _segmento(c.tipo(f), f)
        sg = sesgos.get(seg)
        if not sg or not _hay_evidencia(sg):
            continue
        if parse_date(sg["fechas"][-1]) < c.semana_inicio - timedelta(days=MAX_ANTIGUEDAD_DIAS):
            continue  # evidencia vieja: no se recomienda
        if seg not in validaciones:
            validaciones[seg] = validar_rolling(d, seg, hasta)
        val = validaciones[seg]
        if not val["pasa"]:
            continue  # ni siquiera en la ventana de selección es consistente: no se emite
        if f in c.editadas:  # el día ya se editó: con evidencia, la recomendación cuenta como resuelta
            resueltas.append(f"sesgo-{f}")
            continue
        rec = _rec_sesgo(c, f, seg, sg, paso, val)
        if rec is not None:
            recs.append(rec)
    recs.extend(_recs_alertas(c))
    recs.sort(key=lambda r: (_PRIORIDAD_ORD[r["prioridad"]], r["fecha"], r["id"]))
    return {"recomendaciones": recs, "ya_resueltas": resueltas}


# ---------------------------------------------------------------------------
# Resumen ejecutivo
# ---------------------------------------------------------------------------

def _energia_semana(c: Corrida, d: Datos, lunes: date, usar_final: bool) -> Optional[float]:
    """Energía de una semana: real si existe, si no el pronóstico de la corrida (brecha); None si falta algún día."""
    total = 0.0
    for dia in data.dias_semana(lunes):
        f = data.iso(dia)
        if usar_final:
            e = data.energia_curva(c.final.get(f))
        else:
            e = float(d.reales[f].sum()) if f in d.reales else data.energia_curva(c.final.get(f)) or data.energia_curva(c.base.get(f))
        if e is None:
            return None
        total += e
    return total


def _titulo_semana(c: Corrida, energia: Optional[float], variacion: Optional[float]) -> str:
    base = f"Semana del {rango_semana_texto(c.semana_inicio)}"
    if energia is None:
        return f"{base}: sin pronóstico completo todavía"
    t = f"{base}: demanda prevista {energia / 1000:.1f} GWh"
    if variacion is None:
        return t
    if abs(variacion) < 0.05:
        return t + ", igual que la semana pasada"
    return t + f", un {abs(variacion):.1f} % {'más' if variacion > 0 else 'menos'} que la semana pasada"


def resumen(c: Corrida, d: Datos) -> Dict[str, Any]:
    """Resumen ejecutivo (contrato §1)."""
    recs = recomendaciones(c, d)
    prec = precision(d, semanas=SEMANAS_CONFIANZA, hasta=c.semana_inicio - timedelta(days=7))
    rp = prec["resumen"]
    wape_r = rp["wape_pct"]
    nivel = nivel_confianza(wape_r)
    if nivel == "sin_datos":
        texto_conf = "Todavía no hay pronósticos anteriores comparables para medir qué tan confiable es."
    else:
        texto_conf = (f"Los pronósticos de las últimas {rp['semanas']} semanas se equivocaron en promedio {wape_r:.1f} %.")

    e_sem = _energia_semana(c, d, c.semana_inicio, True)
    e_ant = _energia_semana(c, d, c.semana_inicio - timedelta(days=7), False)
    variacion = _r((e_sem / e_ant - 1) * 100) if e_sem is not None and e_ant else None

    pico = (None, None, None)
    for f in c.fechas:
        cv = _curva_valida(c.final, f)
        if cv is not None and (pico[0] is None or cv.max() > pico[0]):
            pico = (float(cv.max()), f, int(cv.argmax()))

    n_por_fecha: Dict[str, int] = {}
    sistema_por_fecha: Dict[str, int] = {}
    for r in recs["recomendaciones"]:
        n_por_fecha[r["fecha"]] = n_por_fecha.get(r["fecha"], 0) + 1
        if r["respaldo"] == RESPALDO_ALERTA:
            sistema_por_fecha[r["fecha"]] = sistema_por_fecha.get(r["fecha"], 0) + 1
    ins = insights_de_alertas(c)
    nivel_dia: Dict[str, str] = {}
    for i in ins:
        for f in i["fechas"]:
            if i["nivel"] == "critico":
                nivel_dia[f] = "critico"
            elif i["nivel"] == "warning" and nivel_dia.get(f) != "critico":
                nivel_dia[f] = "revisar"
    dias = []
    for f in c.fechas:
        cv, bv = _curva_valida(c.final, f), _curva_valida(c.base, f)
        ajuste = _r((cv.sum() / bv.sum() - 1) * 100) if cv is not None and bv is not None and bv.sum() > 0 else None
        estado = nivel_dia.get(f) or ("revisar" if sistema_por_fecha.get(f) else "ok")
        dias.append({"fecha": f, "dia": nombre_dia(f), "tipo_dia": c.tipo(f),
                     "energia_mwh": _r(cv.sum()) if cv is not None else None,
                     "pico_mw": _r(cv.max()) if cv is not None else None,
                     "pico_hora": int(cv.argmax()) if cv is not None else None,
                     "ajuste_pct": ajuste, "estado": estado, "n_recomendaciones": n_por_fecha.get(f, 0)})

    niveles = [i["nivel"] for i in ins]
    n_crit, n_warn = niveles.count("critico"), niveles.count("warning")
    n_recs = len(recs["recomendaciones"])
    n_sistema = sum(1 for r in recs["recomendaciones"] if r["respaldo"] == RESPALDO_ALERTA)
    n_alta = sum(1 for r in recs["recomendaciones"] if r["prioridad"] == "alta")
    if c.estado != "borrador":  # aprobada, reemplazada...: ya no hay nada que revisar en esta corrida
        paso = {"accion": "ya_aprobada",
                "texto": "Esta semana ya está aprobada." if c.estado == "aprobada"
                else "Esta corrida ya no está en borrador; no requiere revisión."}
    elif n_crit:
        paso = {"accion": "revisar_alertas", "texto": f"Hay {n_crit} problema(s) que impiden aprobar: revísalos primero."}
    elif n_sistema:  # las tendencias recientes (baja prioridad, sin respaldo) no empujan este paso por sí solas
        paso = {"accion": "revisar_recomendaciones",
                "texto": f"Revisa {'la recomendación' if n_sistema == 1 else f'las {n_sistema} recomendaciones'} y aprueba la semana."}
    elif n_warn:
        paso = {"accion": "revisar_alertas", "texto": "Revisa las alertas y aprueba la semana."}
    else:
        paso = {"accion": "aprobar", "texto": "No hay nada pendiente de revisar: puedes aprobar la semana."}
    return {
        "ucp": c.ucp, "nombre_ucp": nombre_ucp(c.ucp),
        "semana_inicio": c.semana_inicio.isoformat(), "semana_fin": (c.semana_inicio + timedelta(days=6)).isoformat(),
        "estado": c.estado, "version": c.version, "titulo": _titulo_semana(c, e_sem, variacion),
        "confianza": {"nivel": nivel, "texto": texto_conf, "wape_reciente_pct": wape_r,
                      "semanas_evaluadas": rp["semanas"]},
        "kpis": {
            "energia_semana_mwh": _r(e_sem), "energia_semana_anterior_mwh": _r(e_ant),
            "variacion_vs_anterior_pct": variacion,
            "pico_mw": _r(pico[0]), "pico_fecha": pico[1], "pico_hora": pico[2],
            "error_reciente_pct": wape_r, "error_reciente_tendencia": rp["tendencia"],
            "alertas": {"total": len(ins), "warning": n_warn, "info": niveles.count("info"), "critico": n_crit},
            "recomendaciones": {"total": n_recs, "alta": n_alta, "del_sistema": n_sistema,
                                "tendencia_reciente": n_recs - n_sistema},
        },
        "dias": dias,
        "insights": [{k: i[k] for k in ("id", "nivel", "titulo", "texto", "fechas")} for i in ins],
        "siguiente_paso": paso,
    }


# ---------------------------------------------------------------------------
# Series (168 puntos)
# ---------------------------------------------------------------------------

def _serie(curvas: Dict[str, Any], fechas: List[str]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for f in fechas:
        v = curvas.get(f)
        out.extend(_lista(v, 2) if v is not None and validar_curva(v) is None else [None] * 24)
    return out


def banda_corrida(c: Corrida, tablas: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Banda por día de la semana sobre las curvas finales (omite días sin curva válida o sin banda)."""
    out = {}
    for f in c.fechas:
        tipo = c.tipo(f)
        b = aplicar_banda(c.final.get(f), tipo, tablas)
        if b is not None:
            out[f] = {"tipo_dia": tipo, "inferior": b[0], "superior": b[1]}
    return out


def series(c: Corrida, d: Datos, tablas: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Series continuas de la semana (contrato §5)."""
    fechas = c.fechas
    tablas = tablas if tablas is not None else tablas_banda(d, fecha_fin=c.fecha_fin_sin_fuga)
    banda = banda_corrida(c, tablas)
    inf = [v for f in fechas for v in (banda[f]["inferior"] if f in banda else [None] * 24)]
    sup = [v for f in fechas for v in (banda[f]["superior"] if f in banda else [None] * 24)]
    hay_banda = bool(banda)

    ref = None
    ref_fechas: List[str] = []
    ref_reales: Dict[str, np.ndarray] = {}
    if c.ref_inicio is not None:
        ref_fechas = [data.iso(x) for x in data.dias_semana(c.ref_inicio)]
        ref_reales = {f: d.reales[f] for f in ref_fechas if f in d.reales}
        ref = _serie(ref_reales, ref_fechas)
        if all(v is None for v in ref):
            ref = None

    def energias(curvas: Dict[str, Any], fs: List[str]) -> List[Optional[float]]:
        return [_r(data.energia_curva(curvas[f])) if f in curvas else None for f in fs]

    etiquetas = []
    for i, f in enumerate(fechas):
        dt = parse_date(f)
        etiquetas.append({"i": i * 24, "texto": f"{DIAS_ABREV[dt.weekday()]} {dt.day}"})
    dias = []
    for f in fechas:
        tipo = c.tipo(f)
        nombre = _nombre_festivo(f) if tipo == data.TIPO_FESTIVO else None
        dias.append({"fecha": f, "tipo_dia": tipo, "editado": f in c.editadas,
                     "etiqueta": f"Festivo: {nombre}" if nombre else ETIQUETA_TIPO[tipo]})
    real_dict = {f: d.reales[f] for f in fechas if f in d.reales}
    return {
        "ucp": c.ucp, "semana_inicio": c.semana_inicio.isoformat(), "unidad": "MW", "puntos": 168,
        "fechas": fechas, "etiquetas_eje": etiquetas,
        "series": {
            "base": _serie(c.base, fechas), "ajustado": _serie(c.final, fechas), "referencia": ref,
            "referencia_semana_inicio": c.ref_inicio.isoformat() if c.ref_inicio and ref is not None else None,
            "real": _serie(real_dict, fechas),
            "banda_inferior": inf if hay_banda else None, "banda_superior": sup if hay_banda else None,
        },
        "energia_diaria": {
            "fechas": fechas, "base": energias(c.base, fechas), "ajustado": energias(c.final, fechas),
            "referencia": energias(ref_reales, ref_fechas) if ref_fechas else [None] * 7,
            "real": energias(real_dict, fechas),
        },
        "dias": dias,
    }
