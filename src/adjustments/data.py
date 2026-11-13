"""
Datos para el pipeline de ajustes: histórico real, fecha de corte, energía
diaria, tipo de día y festivos.

Una sola lectura de `data/raw/{ucp}/datos.csv` por corrida (`cargar_historico`).
Festivos: calendario oficial Colombia (`colombia_holiday_map`, librería
`holidays`, sin red). No se usa `CalendarClassifier` de `calculate_base_curves`
porque consulta la API externa de festivos y el pipeline debe ser reproducible
y funcionar sin red (tests, backtest).
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.adjustments.types import Curves, parse_date
from src.prediction.holiday_homology import colombia_holiday_map

PERIODOS: List[str] = [f"P{i}" for i in range(1, 25)]
VARIABLE_DEMANDA = "Demanda_Real"
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

TIPO_LABORAL = "laboral"
TIPO_SABADO = "sabado"
TIPO_DOMINGO = "domingo"
TIPO_FESTIVO = "festivo"

# Años anómalos que nunca se usan como referencia (pandemia). Única fuente: la
# importan referencia (DEFAULTS) y nivel (fallback si ctx no trae la lista).
ANIOS_EXCLUIDOS_DEFAULT = (2020,)


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def normalizar_historico(df: pd.DataFrame, ucp: Optional[str] = None) -> pd.DataFrame:
    """
    Normaliza un DataFrame de demanda a columnas `fecha` (datetime64), P1..P24, TOTAL.

    - Renombra FECHA→fecha.
    - Filtra por UCP y por VARIABLE == Demanda_Real si esas columnas existen.
    - TOTAL se recalcula como suma de P1..P24 (consistente con las curvas).
    - Ordena por fecha y elimina fechas duplicadas (se queda con la última).
    """
    df = df.copy()
    if "FECHA" in df.columns and "fecha" not in df.columns:
        df = df.rename(columns={"FECHA": "fecha"})
    if ucp is not None and "UCP" in df.columns:
        df = df[df["UCP"] == ucp]
    if "VARIABLE" in df.columns and (df["VARIABLE"] == VARIABLE_DEMANDA).any():
        df = df[df["VARIABLE"] == VARIABLE_DEMANDA]

    faltantes = [c for c in ["fecha", *PERIODOS] if c not in df.columns]
    if faltantes:
        raise ValueError(f"Histórico sin columnas requeridas: {faltantes}")

    df = df[["fecha", *PERIODOS]].copy()
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
    df[PERIODOS] = df[PERIODOS].apply(pd.to_numeric, errors="coerce").astype(float)
    df["TOTAL"] = df[PERIODOS].sum(axis=1, min_count=24)
    df = (
        df.sort_values("fecha")
        .drop_duplicates(subset="fecha", keep="last")
        .reset_index(drop=True)
    )
    return df


def cargar_historico(ucp: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Lee `data/raw/{ucp}/datos.csv` (una sola vez) y lo normaliza."""
    path = Path(data_dir or RAW_DATA_DIR) / ucp / "datos.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe histórico para UCP={ucp}: {path}")
    return normalizar_historico(pd.read_csv(path), ucp=ucp)


def filas_validas(historico: pd.DataFrame) -> pd.Series:
    """Máscara de filas con los 24 periodos finitos y no negativos."""
    valores = historico[PERIODOS].to_numpy(dtype=float)
    return pd.Series(
        np.isfinite(valores).all(axis=1) & (np.nan_to_num(valores, nan=-1.0) >= 0).all(axis=1),
        index=historico.index,
    )


def calcular_fecha_corte(historico: pd.DataFrame, limite: date) -> date:
    """
    Último día con los 24 periodos válidos y fecha <= `limite`.

    `limite` normalmente es fecha_emision - 1 (ver `build_context`).
    """
    limite_ts = pd.Timestamp(limite)
    candidatos = historico[filas_validas(historico) & (historico["fecha"] <= limite_ts)]
    if candidatos.empty:
        raise ValueError(f"No hay días con 24 periodos válidos en o antes de {limite}")
    return candidatos["fecha"].max().date()


def recortar(historico: pd.DataFrame, fecha_corte: date) -> pd.DataFrame:
    """Solo filas válidas con fecha <= fecha_corte (sin fuga de información)."""
    mask = filas_validas(historico) & (historico["fecha"] <= pd.Timestamp(fecha_corte))
    return historico[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Índices y energía
# ---------------------------------------------------------------------------

def curvas_reales(historico: pd.DataFrame) -> Dict[str, np.ndarray]:
    """{fecha_iso: array(24)} de las filas válidas del histórico."""
    df = historico[filas_validas(historico)]
    fechas = df["fecha"].dt.strftime("%Y-%m-%d").tolist()
    valores = df[PERIODOS].to_numpy(dtype=float)
    return dict(zip(fechas, valores))


def energia_diaria(historico: pd.DataFrame) -> Dict[str, float]:
    """{fecha_iso: energía del día (suma P1..P24, MWh)} de las filas válidas."""
    return {f: float(v.sum()) for f, v in curvas_reales(historico).items()}


def energia_curvas(curves: Curves) -> Dict[str, float]:
    """{fecha_iso: suma de los periodos} de unas curvas (None → NaN, nunca lanza por None)."""
    return {f: float(np.sum(np.asarray(v, dtype=float))) for f, v in curves.items()}


# ---------------------------------------------------------------------------
# Calendario
# ---------------------------------------------------------------------------

def festivos_colombia(anios: Iterable[int]) -> List[str]:
    """Festivos oficiales de Colombia (ISO) para los años dados, ordenados."""
    fechas: set = set()
    for anio in anios:
        fechas.update(colombia_holiday_map(int(anio)).keys())
    return sorted(fechas)


def tipo_dia(fecha, festivos: Iterable[str]) -> str:
    """laboral / sabado / domingo / festivo (el festivo tiene prioridad)."""
    d = parse_date(fecha)
    festivos_set = festivos if isinstance(festivos, (set, frozenset)) else set(festivos)
    if d.isoformat() in festivos_set:
        return TIPO_FESTIVO
    if d.weekday() == 5:
        return TIPO_SABADO
    if d.weekday() == 6:
        return TIPO_DOMINGO
    return TIPO_LABORAL


def dias_semana(lunes: date) -> List[date]:
    """Los 7 días (lunes..domingo) de la semana que empieza en `lunes`."""
    return [lunes + timedelta(days=i) for i in range(7)]


def ancla_mismo_dow(d: date, corte: date) -> date:
    """
    Último día con el mismo día de semana que `d`, estrictamente anterior a `d`
    y <= `corte`: d − 7·max(1, ceil((d − corte)/7)). Regla común de nivel
    (ancla actual), forma (días comparables) y la base ingenua del backtest.
    """
    k = max(1, math.ceil((d - corte).days / 7))
    return d - timedelta(days=7 * k)


def lunes_de(fecha: date) -> date:
    return fecha - timedelta(days=fecha.weekday())


def iso(d: date) -> str:
    return d.isoformat()
