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
