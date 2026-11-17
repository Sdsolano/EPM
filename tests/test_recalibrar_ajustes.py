"""Tests de scripts/recalibrar_ajustes.py (rápidos, sin red, sin entrenar: datos sintéticos)."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.adjustments import data
from tests.ajustes_sinteticos import historico_sintetico

ROOT = Path(__file__).resolve().parent.parent


def _cargar_script():
    spec = importlib.util.spec_from_file_location("recalibrar_ajustes", ROOT / "scripts" / "recalibrar_ajustes.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["recalibrar_ajustes"] = mod
    spec.loader.exec_module(mod)
    return mod


ra = _cargar_script()
UCP = "Sintetica"


# ---------------------------------------------------------------------------
# Holdout, rolling origin y configs
# ---------------------------------------------------------------------------

def test_particion_60_40_ordenada():
    semanas = [(date(2025, 1, 6) + timedelta(weeks=i)).isoformat() for i in range(52)]
    p = ra.particion_semanas(list(reversed(semanas)))
    cal = [s for s in semanas if p[s] == "calibracion"]
    pru = [s for s in semanas if p[s] == "prueba"]
    assert len(cal) == 31 and len(pru) == 21
    assert max(cal) < min(pru)  # calibración = semanas antiguas


def test_origenes_rolling():
    assert ra.origenes_rolling(5, 1) == [0, 1, 2, 3, 4]
    assert ra.origenes_rolling(10, 4) == [0, 0, 0, 0, 4, 4, 4, 4, 8, 8]
    with pytest.raises(ValueError):
        ra.origenes_rolling(3, 0)


def test_grid_y_default():
    grid = ra.grid_configs()
    assert len(grid) == 5 * (2 + len(ra.NS_PROMEDIO))
    dflt = [c for c in grid if c["es_default"]]
    assert len(dflt) == 1 and dflt[0]["config"] == ra.config_default()["config"]
    assert len({c["config"] for c in grid}) == len(grid)  # nombres únicos


def test_config_con_alpha_por_tipo_no_es_default_y_valida():
    from src.adjustments.steps import nivel

    c = ra.config(0.25, "promedio_n", 4, {"festivo": 1.0, "laboral": 0.5})
    assert not c["es_default"]
    assert c["config"] == "a0.25|promedio_n4|apt(laboral=0.5,festivo=1)"
    assert json.loads(c["alpha_por_tipo"]) == {"festivo": 1.0, "laboral": 0.5}
    p = nivel.validar_params(c["params"]["nivel"])  # el pipeline lo acepta
    assert p["alpha_por_tipo"] == {"laboral": 0.5, "festivo": 1.0}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def test_cache_ida_y_vuelta(tmp_path):
    curvas = {"2025-06-29": list(range(24)), "2025-06-30": [1.5] * 24}
    filas = ra.filas_cache(UCP, 1, date(2025, 6, 30), date(2025, 6, 28), date(2025, 6, 28), curvas)
    path = ra.ruta_cache(1, UCP, tmp_path)
    pd.DataFrame(filas, columns=ra.COLS_CACHE).to_csv(path, index=False)
    otra = ra.ruta_cache(1, "Otra", tmp_path)
    pd.DataFrame(ra.filas_cache("Otra", 1, date(2025, 6, 30), date(2025, 6, 28), date(2025, 6, 28), curvas),
                 columns=ra.COLS_CACHE).to_csv(otra, index=False)
    cache = ra.leer_cache_k(1, tmp_path)
    assert set(cache["ucp"]) == {UCP, "Otra"}
    por_sem = ra.curvas_por_semana(cache, UCP)
    assert list(por_sem) == ["2025-06-30"]
    assert por_sem["2025-06-30"]["2025-06-29"] == [float(i) for i in range(24)]
    assert ra.leer_cache_k(4, tmp_path).empty


# ---------------------------------------------------------------------------
# Backtest sobre curvas cacheadas (sintético)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def historico():
    return data.normalizar_historico(historico_sintetico(date(2022, 1, 1), date(2025, 6, 30)))


def _cache_sintetico(historico, semanas, factor=1.03):
    """Base 'modelo' = real × factor cacheado por semana (como el cache del sandbox)."""
    reales = data.curvas_reales(historico)
    filas = []
    for l in semanas:
        corte = ra.corte_de_semana(historico, l)
        curvas = {(corte + timedelta(days=i)).isoformat(): (reales[(corte + timedelta(days=i)).isoformat()] * factor)
                  .tolist() for i in range(1, 9)}
        filas += ra.filas_cache(UCP, 1, l, corte, corte, curvas)
    return pd.DataFrame(filas, columns=ra.COLS_CACHE)


def test_correr_base_modelo_desde_cache(historico):
    semanas = ra.ba.semanas_objetivo(historico, 3)
    cache = _cache_sintetico(historico, semanas[:2])  # la 3.ª semana no está en cache → se omite
    configs = [ra.config_default(), ra.config(0.0, "reciente"), ra.config(0.25, "promedio_n", 4, {"domingo": 0.0})]
    df = ra.correr_base("modelo_k1", UCP, semanas, configs, historico, cache=cache)
    assert df["semana_inicio"].nunique() == 2
    assert len(df) == 2 * 7 * 3 * len(configs)
    base = df[(df["paso"] == "base") & df["es_default"]]
    assert ra.wape(base) == pytest.approx(3.0, abs=1e-6)  # base = real × 1.03
    # α = 0 no cambia la energía: nivel == base
    a0 = df[df["config"] == "a0|reciente"]
    assert ra.wape(a0[a0["paso"] == "nivel"]) == pytest.approx(ra.wape(a0[a0["paso"] == "base"]))
    # la config con alpha_por_tipo corrió de verdad y conserva el parámetro
    apt = df[df["alpha_por_tipo"] != ""]
    assert not apt.empty and json.loads(apt["alpha_por_tipo"].iloc[0]) == {"domingo": 0.0}
    # sin fuga: corte antes de la semana
    assert (pd.to_datetime(df["fecha_corte"]) < pd.to_datetime(df["semana_inicio"])).all()


# ---------------------------------------------------------------------------
# Selección y métricas (tablas sintéticas)
# ---------------------------------------------------------------------------

def _filas(ucp, semana, cfg, tipo, err, paso="forma", n_dias=1, part="calibracion", dif=None, real=100.0):
    return [{"ucp": ucp, "semana_inicio": semana, "config": cfg["config"], "alpha": cfg["alpha"],
             "fuente": cfg["fuente"], "n": cfg["n"], "alpha_por_tipo": cfg["alpha_por_tipo"],
             "es_default": cfg["es_default"], "paso": paso, "tipo_dia": tipo, "particion": part,
             "energia_real": real, "energia_pred": real * (1 + err), "abs_err_horario": abs(err) * real,
             "diferencia_tendencia_pp": dif, "fecha": f"{semana}-{i}"} for i in range(n_dias)]


def test_elegir_alpha_por_tipo_respeta_minimo_festivos():
    filas = []
    for a, err_lab, err_fes in ((0.0, 0.05, 0.01), (0.5, 0.02, 0.10), (1.0, 0.03, 0.20)):
        c = ra.config(a, "reciente")
        filas += _filas(UCP, "2025-01-06", c, "laboral", err_lab, n_dias=20)
        filas += _filas(UCP, "2025-01-06", c, "festivo", err_fes, n_dias=3)
        filas += _filas(UCP, "2025-01-06", c, "domingo", 0.02 + a, n_dias=5)
    df = pd.DataFrame(filas)
    apt, tabla = ra.elegir_alpha_por_tipo(df, "reciente", None, alpha_global=0.0)
    assert apt == {"laboral": 0.5}  # domingo: mejor α=0 = global → se omite; festivo: 3 días < 10
    apt2, _ = ra.elegir_alpha_por_tipo(df, "reciente", None, alpha_global=0.5, min_festivos=2)
    assert apt2 == {"festivo": 0.0, "domingo": 0.0}
    assert set(tabla["tipo_dia"]) == {"laboral", "festivo", "domingo"}


def test_evaluar_y_elegir_banda():
    c = ra.config_default()
    filas = []
    # 4 semanas: (dif tendencia, error semanal)
    for i, (dif, err) in enumerate(((3.0, 0.035), (0.5, 0.005), (2.2, 0.001), (0.2, 0.022))):
        filas += _filas(UCP, f"2025-01-{6 + 7 * i:02d}", c, "laboral", err, dif=dif)
    tabla = ra.evaluar_banda(pd.DataFrame(filas), bandas=(1.0, 2.0, 2.5))
    b1 = tabla[tabla["banda_pp"] == 1.0].iloc[0]
    # banda 1: alertas = semanas 0 y 2; eventos (|err|>1 %) = semanas 0 y 3
    assert (b1["alertas"], b1["eventos"], b1["TP"]) == (2, 2, 1)
    assert b1["precision"] == pytest.approx(0.5) and b1["tasa_base"] == pytest.approx(0.5)
    b25 = tabla[tabla["banda_pp"] == 2.5].iloc[0]
    assert (b25["alertas"], b25["TP"]) == (1, 1)  # semana 0: dif 3 > 2.5 y err 3.5 % > 2.5
    assert ra.elegir_banda(tabla) == 2.5  # mayor lift


def test_regla_default_y_comparar():
    dflt, otra = ra.config_default(), ra.config(0.5, "referencia")
    filas = []
    for ucp, e_d, e_o in (("A", 0.03, 0.02), ("B", 0.04, 0.035)):
        for part in ("calibracion", "prueba"):
            filas += _filas(ucp, "2025-01-06", dflt, "laboral", e_d, part=part)
            filas += _filas(ucp, "2025-01-06", otra, "laboral", e_o, part=part)
    df = pd.DataFrame(filas).assign(base="modelo_k1")
    comp = ra.comparar_configs(df, {"default": dflt["config"], "x": otra["config"]})
    r = ra.regla_default(comp, "x")
    assert r["cumple"] and r["mejora_wape_todas"] and r["mape_sem_ok"]
    assert not ra.regla_default(comp, "default")["cumple"]
    # si en una UCP empeora, no cumple
    df.loc[(df["ucp"] == "B") & (df["config"] == otra["config"]), "abs_err_horario"] = 10.0
    comp2 = ra.comparar_configs(df, {"default": dflt["config"], "x": otra["config"]})
    assert not ra.regla_default(comp2, "x")["cumple"]


def test_r2_segmentos_clasifica():
    c = ra.config_default()
    filas = []
    for part, n_sem in (("calibracion", 3), ("prueba", 3)):
        for s in range(n_sem):
            sem = f"{part}-{s}"
            # laboral: nivel mejora siempre; domingo: nivel empeora siempre
            filas += _filas(UCP, sem, c, "laboral", 0.05, paso="base", n_dias=5, part=part)
            filas += _filas(UCP, sem, c, "laboral", 0.03, paso="nivel", n_dias=5, part=part)
            filas += _filas(UCP, sem, c, "laboral", 0.03, paso="forma", n_dias=5, part=part)
            filas += _filas(UCP, sem, c, "domingo", 0.02, paso="base", n_dias=1, part=part)
            filas += _filas(UCP, sem, c, "domingo", 0.04, paso="nivel", n_dias=1, part=part)
            filas += _filas(UCP, sem, c, "domingo", 0.04, paso="forma", n_dias=1, part=part)
    r2 = ra.r2_segmentos(pd.DataFrame(filas)).set_index(["tipo_dia", "paso"])
    assert r2.loc[("laboral", "nivel"), "recomendacion"] == "automático"
    assert r2.loc[("domingo", "nivel"), "recomendacion"] == "no aplicar"
    assert r2.loc[("laboral", "forma"), "recomendacion"] == "sugerido"  # neutro (Δ = 0)
