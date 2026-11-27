"""Datos sintéticos compartidos por los tests del módulo de ajustes (sin CSV ni modelos)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.adjustments.data import PERIODOS, festivos_colombia, tipo_dia

CRECIMIENTO_ANUAL = 0.03
NIVEL_TIPO = {"laboral": 24_000.0, "sabado": 21_000.0, "domingo": 18_000.0, "festivo": 17_000.0}


def forma_tipo(tipo: str) -> np.ndarray:
    """Forma horaria normalizada por tipo de día (pico a las 20 h, distinta por tipo)."""
    h = np.arange(24)
    desfase = {"laboral": 0.0, "sabado": 1.0, "domingo": 2.0, "festivo": 2.5}[tipo]
    v = 1.0 + 0.3 * np.sin((h - 6 - desfase) / 24 * 2 * np.pi) + 0.25 * np.exp(-((h - 19) ** 2) / 4)
    return v / v.sum()


def energia_real(d: date, festivos) -> float:
    anios = (d - date(2020, 1, 1)).days / 365.0
    return NIVEL_TIPO[tipo_dia(d, festivos)] * (1 + CRECIMIENTO_ANUAL) ** anios


def historico_sintetico(inicio: date = date(2021, 1, 1), fin: date = date(2025, 12, 31)) -> pd.DataFrame:
    festivos = set(festivos_colombia(range(inicio.year, fin.year + 1)))
    filas = []
    d = inicio
    while d <= fin:
        curva = forma_tipo(tipo_dia(d, festivos)) * energia_real(d, festivos)
        filas.append({"FECHA": d.isoformat(), **{p: float(v) for p, v in zip(PERIODOS, curva)}})
        d += timedelta(days=1)
    df = pd.DataFrame(filas)
    df["TOTAL"] = df[PERIODOS].sum(axis=1)
    return df


class ProveedorPlano:
    """base_provider falso: curvas planas de `mw` por periodo; cuenta llamadas."""

    def __init__(self, mw: float = 1000.0, faltar=None):
        self.mw = mw
        self.faltar = set(faltar or [])
        self.llamadas = []

    def __call__(self, ucp, end_date, n_days):
        self.llamadas.append((ucp, end_date, n_days))
        out = {}
        for i in range(1, n_days + 1):
            f = (end_date + timedelta(days=i)).isoformat()
            if f not in self.faltar:
                out[f] = [self.mw] * 24
        return out


# ---------------------------------------------------------------- selector de R1 (curvas planas)

def curvas_planas(
    festivos,
    desde: date,
    hasta: date,
    factores=None,
    excluir=(),
    mw_normal: float = 100.0,
    mw_festivo: float = 80.0,
    mw_semana_santa=None,
):
    """
    {iso: array(24)} plano: `mw_normal` MW los días no festivos, `mw_festivo` los festivos (`mw_semana_santa` los
    festivos de Semana Santa, si se da); `factores` {iso: k} multiplica ese día; `excluir` = días sin datos.
    """
    from src.adjustments.data import tipo_festivo

    factores = factores or {}
    out, d = {}, desde
    while d <= hasta:
        f = d.isoformat()
        if f not in excluir:
            base = mw_normal
            if f in festivos:
                es_ss = mw_semana_santa is not None and tipo_festivo(f, festivos) == "semana_santa"
                base = mw_semana_santa if es_ss else mw_festivo
            out[f] = np.full(24, base * factores.get(f, 1.0))
        d += timedelta(days=1)
    return out


def seleccionar(objetivo, corte, curvas, festivos, n: int = 1, **kw):
    """`referencia_reciente.seleccionar_recientes` con argumentos posicionales cortos (tests del selector)."""
    from src.adjustments.referencia_reciente import seleccionar_recientes

    return seleccionar_recientes(objetivo, curvas, corte, festivos, n, **kw)


def fechas(*isos):
    """['2026-01-14', ...] → [date, ...]."""
    return [date.fromisoformat(x) for x in isos]
