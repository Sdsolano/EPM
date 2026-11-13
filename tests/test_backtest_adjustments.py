"""Test mínimo del backtest de ajustes (datos sintéticos, base ingenua, sin red)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.adjustments.data import PERIODOS
from tests.ajustes_sinteticos import historico_sintetico

ROOT = Path(__file__).resolve().parent.parent


def _cargar_script():
    spec = importlib.util.spec_from_file_location(
        "backtest_adjustments", ROOT / "scripts" / "backtest_adjustments.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["backtest_adjustments"] = mod
    spec.loader.exec_module(mod)
    return mod


bt = _cargar_script()
UCP = "Sintetica"
CONFIGS = [(0.0, "reciente"), bt.CONFIG_DEFAULT, (1.0, "promedio_n")]  # incluye el default (0.25, promedio_n)


@pytest.fixture(scope="module")
def historico():
    return historico_sintetico(date(2022, 1, 1), date(2025, 6, 30))


@pytest.fixture(scope="module")
def detalle(historico):
    from src.adjustments import data

    semanas = bt.semanas_objetivo(data.normalizar_historico(historico), 3)
    factory = lambda ucp, hist: bt.proveedor_ingenuo(hist)
    return bt.correr_backtest(UCP, historico, semanas, factory, "ingenuo", CONFIGS)


def test_semanas_objetivo_completas(historico):
    from src.adjustments import data

    semanas = bt.semanas_objetivo(data.normalizar_historico(historico), 3)
    assert len(semanas) == 3
    assert all(l.weekday() == 0 for l in semanas)
    assert semanas[-1] == date(2025, 6, 23)  # 2025-06-30 es lunes sin domingo siguiente


def test_proveedor_ingenuo_usa_misma_semana_anterior(historico):
    from src.adjustments import data

    h = data.normalizar_historico(historico)
    corte = date(2025, 6, 21)  # sábado
    prov = bt.proveedor_ingenuo(h[h["fecha"] <= pd.Timestamp(corte)])
    curvas = prov(UCP, corte, 8)  # domingo de brecha + semana objetivo
    reales = data.curvas_reales(h)
    assert sorted(curvas) == [(corte + timedelta(days=i)).isoformat() for i in range(1, 9)]
    # lunes objetivo (corte+2) = lunes anterior (corte-5); domingo objetivo = domingo de la brecha -7
    assert np.allclose(curvas["2025-06-23"], reales["2025-06-16"])
    assert np.allclose(curvas["2025-06-29"], reales["2025-06-15"])


def test_detalle_estructura_y_metricas(detalle):
    assert not detalle.empty
    # 3 semanas x 7 días x 3 pasos x 3 configs
    assert len(detalle) == 3 * 7 * 3 * len(CONFIGS)
    assert set(detalle["paso"]) == {"base", "nivel", "forma"}
    assert (detalle["fecha_corte"] < detalle["semana_inicio"]).all()
    # corte = sábado anterior a la semana objetivo
    cortes = pd.to_datetime(detalle["fecha_corte"])
    assert (cortes.dt.weekday == 5).all()
    assert ((pd.to_datetime(detalle["semana_inicio"]) - cortes).dt.days == 2).all()
    # forma conserva la energía de nivel
    piv = detalle.pivot_table(index=["semana_inicio", "fecha", "alpha", "fuente"], columns="paso",
                              values="energia_pred")
    assert np.allclose(piv["forma"], piv["nivel"], rtol=1e-9)


def test_nivel_mejora_sobre_base_ingenua_en_sintetico(detalle):
    # Con crecimiento anual constante y semana ref limpia, alpha=1 debe reducir el error de energía.
    r = bt.resumen_por(detalle[detalle["paso"] == "nivel"], ["alpha"])
    mape = dict(zip(r["alpha"], r["mape_energia_dia_pct"]))
    assert mape[1.0] <= mape[0.0] + 1e-9


def test_sin_fuga_datos_futuros(historico):
    """Alterar los datos posteriores al corte no cambia ninguna curva pronosticada."""
    from src.adjustments import data

    h = data.normalizar_historico(historico)
    lunes = date(2025, 6, 16)
    reales = data.curvas_reales(h)
    factory = lambda ucp, hist: bt.proveedor_ingenuo(hist)
    a = bt.evaluar_semana(UCP, lunes, h, reales, factory, "ingenuo", CONFIGS)

    h2 = h.copy()
    futuro = h2["fecha"] > pd.Timestamp(lunes - timedelta(days=2))
    h2.loc[futuro, PERIODOS] = h2.loc[futuro, PERIODOS] * 3.0
    h2["TOTAL"] = h2[PERIODOS].sum(axis=1)
    b = bt.evaluar_semana(UCP, lunes, h2, reales, factory, "ingenuo", CONFIGS)

    pa = pd.DataFrame(a)["energia_pred"].to_numpy()
    pb = pd.DataFrame(b)["energia_pred"].to_numpy()
    assert np.allclose(pa, pb)


def test_resumenes_y_reporte(detalle, tmp_path):
    assert not bt.resumen_semanal(detalle, ["paso"]).empty
    assert not bt.mejores_configs(detalle).empty
    alerta = bt.evaluar_alerta(detalle)
    assert {"precision", "recall", "tasa_base"} <= set(alerta.columns)
    md = tmp_path / "r.md"
    bt.escribir_reporte(detalle, md, {
        "fecha": "20990101", "csv_nombre": "x.csv", "bases": ["ingenuo"], "nota_base": "test",
        "semanas": 3, "alphas": [0.0, 0.25, 1.0], "fuentes": ["reciente", "promedio_n"],
        "limitaciones": "-", "recomendacion": bt.recomendacion_auto(detalle),
    })
    texto = md.read_text(encoding="utf-8")
    assert "Resultado por paso" in texto and "TENDENCIA_FUERA_BANDA" in texto


def test_sin_ediciones_enclavamiento_no_alerta(detalle):
    """Sin ediciones del analista la energía final = la del paso nivel → desvío 0, sin alerta."""
    assert not detalle["alerta_tendencia"].any()
    assert np.allclose(detalle["desvio_pp"].astype(float), 0.0, atol=1e-9)
    assert "diferencia_tendencia_pp" in detalle.columns
    assert not bt.evaluar_alerta(detalle, "alerta_tendencia_referencia").empty


def test_alerta_info_referencia_columnas_y_evaluacion(detalle):
    assert {"alerta_tendencia_referencia_info", "banda_referencia_pp"} <= set(detalle.columns)
    assert (detalle["banda_referencia_pp"] == 2.0).all()
    ev = bt.evaluar_alerta(detalle, "alerta_tendencia_referencia_info", "banda_referencia_pp")
    assert not ev.empty and {"precision", "tasa_base"} <= set(ev.columns)


def test_secciones_manuales_se_preservan(tmp_path):
    md = tmp_path / "r.md"
    assert bt.secciones_manuales(md) == ""
    md.write_text("# t\n\n## 5. Limitaciones\n- x\n\n## 6. A mano\ntexto\n\n## 7. Otra\nmás\n", encoding="utf-8")
    manual = bt.secciones_manuales(md)
    assert manual.startswith("## 6. A mano") and "## 7. Otra" in manual and "Limitaciones" not in manual


def test_nota_festivos_marca_ingenuo_no_interpretable():
    filas = []
    for base in ("ingenuo", "archivo"):
        for paso, err in (("base", 16.0), ("nivel", 12.0), ("forma", 11.0)):
            for alpha, fuente, es_def in ((0.25, "promedio_n", True), (1.0, "reciente", False)):
                filas.append({"base": base, "ucp": "X", "tipo_dia": "festivo", "paso": paso,
                              "alpha": alpha, "fuente": fuente, "es_default": es_def,
                              "abs_err_horario": err if es_def else err / 2, "energia_real": 100.0,
                              "energia_pred": 100.0, "pico_pred": 1.0, "pico_real": 1.0,
                              "hora_pico_pred": 1, "hora_pico_real": 1})
    nota = bt.nota_festivos(pd.DataFrame(filas))
    assert "ingenuo / X: NO interpretable" in nota and "16.00 %" in nota
    assert "archivo / X: no concluyente" in nota
    assert "pendiente de un backtest con el modelo real" in nota
