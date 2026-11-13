"""Compatibilidad con pandas 3: relleno de features en `FeatureEngineer` (ffill y luego 0)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.feature_engineering import FeatureEngineer


def _df():
    return pd.DataFrame({
        "FECHA": pd.date_range("2026-01-01", periods=5, freq="D"),
        "f1": [np.nan, 1.0, np.nan, np.nan, 4.0],   # NaN inicial → 0; huecos → último válido
        "f2": [2.0, np.nan, 3.0, np.nan, np.nan],
        "f3": [np.nan, np.nan, np.nan, np.nan, np.nan],  # todo NaN → 0
        "TOTAL": [10.0, np.nan, 12.0, 13.0, 14.0],  # target: no se rellena
        "P1": [1.0, 2.0, np.nan, 4.0, 5.0],         # periodos: no se rellenan
        "ignorada": [1, 2, 3, 4, 5],                # no es feature ni target → se descarta
    })


def test_relleno_ffill_luego_cero():
    fe = FeatureEngineer()
    fe.feature_names = ["f1", "f2", "f3", "no_existe"]
    out = fe.get_feature_importance_ready_df(_df())

    assert out["f1"].tolist() == [0.0, 1.0, 1.0, 1.0, 4.0]
    assert out["f2"].tolist() == [2.0, 2.0, 3.0, 3.0, 3.0]
    assert out["f3"].tolist() == [0.0] * 5
    # target y periodos horarios quedan intactos (NaN incluidos)
    assert np.isnan(out["TOTAL"].iloc[1]) and out["TOTAL"].iloc[0] == 10.0
    assert np.isnan(out["P1"].iloc[2])
    assert "ignorada" not in out.columns and "no_existe" not in out.columns
    assert list(out.columns) == ["f1", "f2", "f3", "P1", "TOTAL", "FECHA"]


def test_relleno_equivale_a_ffill_fillna():
    """Mismo resultado que la expresión explícita `.ffill().fillna(0)` (la de pandas < 3 era equivalente)."""
    df = _df()
    fe = FeatureEngineer()
    fe.feature_names = ["f1", "f2", "f3"]
    out = fe.get_feature_importance_ready_df(df)
    esperado = df[["f1", "f2", "f3"]].ffill().fillna(0)
    pd.testing.assert_frame_equal(out[["f1", "f2", "f3"]], esperado)
    # no muta la entrada
    assert np.isnan(df["f1"].iloc[0])
