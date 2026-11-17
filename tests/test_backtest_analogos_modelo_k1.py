"""Tests sintéticos de la base `modelo_k1` en scripts/backtest_analogos.py (sin red ni CSV reales)."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import backtest_adjustments as bt  # noqa: E402
import backtest_analogos as ba  # noqa: E402
from tests.ajustes_sinteticos import historico_sintetico  # noqa: E402


def _escribir_curvas(path: Path, ucp: str, semanas, historico: pd.DataFrame) -> None:
    hist = bt.data.normalizar_historico(historico, ucp=ucp)
    reales = bt.data.curvas_reales(hist)
    filas = []
    for lunes in semanas:
        corte = lunes - timedelta(days=2)
        for i in range(1, 9):
            fecha = (corte + timedelta(days=i)).isoformat()
            curva = reales[fecha]
            filas.append({
                "ucp": ucp, "k": 1, "semana_inicio": lunes.isoformat(), "corte": corte.isoformat(),
                "origen": corte.isoformat(), "fecha": fecha,
                **{f"P{j + 1}": float(v) for j, v in enumerate(curva)},
            })
    pd.DataFrame(filas).to_csv(path, index=False)


def test_cargar_curvas_modelo_k1(tmp_path):
    hist = historico_sintetico(date(2025, 1, 1), date(2025, 8, 31))
    semanas = [date(2025, 6, 30), date(2025, 7, 7)]
    _escribir_curvas(tmp_path / "curvas_modelo_k1_Antioquia.csv", "Antioquia", semanas, hist)

    curvas, semanas_corte = bt.cargar_curvas_modelo_k1("Antioquia", tmp_path)

    assert semanas_corte == {"2025-06-30": date(2025, 6, 28), "2025-07-07": date(2025, 7, 5)}
    assert len(curvas) == 15  # 2 semanas x 8 días menos el domingo compartido
    assert len(curvas["2025-06-30"]) == 24
    assert all(isinstance(v, float) for v in curvas["2025-06-30"])


def test_cargar_curvas_modelo_k1_sin_archivo(tmp_path):
    with pytest.raises(FileNotFoundError):
        bt.cargar_curvas_modelo_k1("Antioquia", tmp_path)


def test_semanas_modelo_k1_solo_rango_completo(tmp_path):
    hist = historico_sintetico(date(2025, 1, 1), date(2025, 8, 31))
    semanas = [date(2025, 6, 30), date(2025, 7, 7), date(2025, 7, 14)]
    path = tmp_path / "curvas_modelo_k1_Antioquia.csv"
    _escribir_curvas(path, "Antioquia", semanas, hist)

    df = pd.read_csv(path)
    df = df[df["fecha"] != "2025-07-09"]  # rompe la semana del 2025-07-07
    df.to_csv(path, index=False)
    curvas, semanas_corte = bt.cargar_curvas_modelo_k1("Antioquia", tmp_path)

    completas = bt.semanas_modelo_k1(hist, 10, semanas_corte, curvas)
    assert completas == [date(2025, 6, 30), date(2025, 7, 14)]
    assert bt.semanas_modelo_k1(hist, 1, semanas_corte, curvas) == [date(2025, 7, 14)]


def _comp_base(base: str, mejora: float) -> pd.DataFrame:
    filas = []
    for ucp in ("Antioquia", "Atlantico"):
        for part in ba.PARTICIONES:
            filas.append({
                "segmento": "todos", "base": base, "ucp": ucp, "particion": part,
                "config": "forma_analogos_n3", "paso": "forma", "n_dias": 10,
                "mejora_rel_pct": mejora, "delta_pico_pp": -0.1,
            })
    return pd.DataFrame(filas)


def test_base_regla_filtra_la_base_correcta():
    cfgs = ba.configs_barrido([3], [0], [0.25])
    comp = pd.concat([_comp_base("ingenuo", 0.5), _comp_base("modelo_k1", 5.0)], ignore_index=True)

    con_modelo = ba.evaluar_regla(comp, ["Antioquia", "Atlantico"], 2.0, cfgs, base_regla="modelo_k1")
    fila = con_modelo[con_modelo["config"] == "forma_analogos_n3"].iloc[0]
    assert bool(fila["cumple_regla"]) and fila["celdas"] == "4/4"

    con_ingenuo = ba.evaluar_regla(comp, ["Antioquia", "Atlantico"], 2.0, cfgs, base_regla="ingenuo")
    fila = con_ingenuo[con_ingenuo["config"] == "forma_analogos_n3"].iloc[0]
    assert not bool(fila["cumple_regla"])


def test_corrida_minima_con_modelo_k1(tmp_path):
    hist = historico_sintetico(date(2023, 1, 1), date(2025, 6, 29))
    semanas = [date(2025, 6, 16), date(2025, 6, 23)]
    _escribir_curvas(tmp_path / "curvas_modelo_k1_Antioquia.csv", "Antioquia", semanas, hist)

    factory, extra = ba._factory("modelo_k1", ba.ARCHIVO_DEFAULT, tmp_path)
    semanas_ok, parts = ba.semanas_de("modelo_k1", hist, 2, extra, "Antioquia")
    assert semanas_ok == semanas

    cfgs = [c for c in ba.configs_barrido([3], [0], [0.25])
            if c.nombre in ("default", "forma_analogos_n3")]
    df = ba.correr("Antioquia", hist, semanas_ok, parts, factory, "modelo_k1", cfgs)

    assert not df.empty
    assert set(df["base"]) == {"modelo_k1"}
    assert (df["fecha_corte"] < df["semana_inicio"]).all()
    assert set(df["particion"]) == {"calibracion", "holdout"}
