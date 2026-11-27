"""Tests de `scripts/backtest_jamer.py` (fase E6): métricas puras con fixtures sintéticos, sin red.

La prueba de humo con los datos reales de la réplica (`reports/ajustes/jamer/`, ignorados por git) se salta si
no existen; no escribe `data/raw` ni usa red.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.ajustes_sinteticos import historico_sintetico

ROOT = Path(__file__).resolve().parent.parent


def _cargar_script():
    spec = importlib.util.spec_from_file_location("backtest_jamer", ROOT / "scripts" / "backtest_jamer.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["backtest_jamer"] = mod
    spec.loader.exec_module(mod)
    return mod


bj = _cargar_script()
FECHAS = ["2026-01-05", "2026-01-06", "2026-01-07"]


def _curvas(escala=1.0, ruido=0.0, seed=0):
    rng = np.random.default_rng(seed)
    h = np.arange(24)
    forma = 1000 + 300 * np.sin((h - 6) / 24 * 2 * np.pi)
    return {f: list(forma * escala + rng.normal(0, ruido, 24)) for f in FECHAS}


# ---------------------------------------------------------------------------
# métricas puras
# ---------------------------------------------------------------------------

def test_metricas_periodo_identicas_son_cero_y_corr_uno():
    v = _curvas()
    m = bj.metricas_periodo(v, v, FECHAS)
    assert m["mae"] == m["p95"] == m["max"] == 0.0
    assert m["sesgo_max_abs"] == 0.0 and m["rmse_forma_pct"] == pytest.approx(0.0, abs=1e-12)
    assert m["corr_min"] == pytest.approx(1.0)


def test_metricas_periodo_offset_plano_da_sesgo_y_mae_igual():
    v = _curvas()
    p = {f: [x + 10.0 for x in c] for f, c in v.items()}
    m = bj.metricas_periodo(p, v, FECHAS)
    assert m["mae"] == pytest.approx(10.0) and m["max"] == pytest.approx(10.0) and m["p95"] == pytest.approx(10.0)
    assert m["sesgo_dia"] == [pytest.approx(10.0)] * 3 and m["sesgo_max_abs"] == pytest.approx(10.0)
    assert m["corr_min"] == pytest.approx(1.0)  # un offset no cambia la correlación


def test_metricas_forma_ignora_el_nivel():
    v = _curvas()
    p = {f: [x * 1.05 for x in c] for f, c in v.items()}  # mismo formato, +5 % de nivel
    m = bj.metricas_periodo(p, v, FECHAS)
    assert m["mae"] > 10  # error de nivel visible
    assert m["rmse_forma_pct"] == pytest.approx(0.0, abs=1e-9)  # la forma tras igualar Σ es perfecta


def test_error_energia_signo_y_conteos():
    v = _curvas()
    p = {FECHAS[0]: [x * 1.004 for x in v[FECHAS[0]]], FECHAS[1]: [x * 0.98 for x in v[FECHAS[1]]],
         FECHAS[2]: list(v[FECHAS[2]])}
    e = bj.error_energia(p, v, FECHAS)
    assert e["dia_pct"][0] == pytest.approx(0.4) and e["dia_pct"][1] == pytest.approx(-2.0)
    assert e["n_le_1"] == 2 and e["n_le_05"] == 2
    assert e["mape_pct"] == pytest.approx((0.4 + 2.0 + 0.0) / 3)


def test_crecimiento_pct_y_referencia_invalida():
    assert bj.crecimiento_pct(110.0, 100.0) == pytest.approx(10.0)
    assert bj.crecimiento_pct(110.0, 0.0) is None
    assert bj.crecimiento_pct(110.0, float("nan")) is None


def test_veredicto_y_criterio():
    assert bj.veredicto(True) == "pasa" and bj.veredicto(False) == "falla" and bj.veredicto(None) == "no evaluable"
    c = bj.criterio("X1", "algo", "3", "≤ 4", True)
    assert c["id"] == "X1" and c["veredicto"] == "pasa"


# ---------------------------------------------------------------------------
# generalización: semanas, partición y comparación pareada
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hist_2026():
    from src.adjustments import data

    return data.normalizar_historico(historico_sintetico(date(2024, 1, 1), date(2026, 7, 4)), ucp="Sintetica")


def test_semanas_ene_jun_son_lunes_completos_de_2026(hist_2026):
    semanas = bj.semanas_ene_jun(hist_2026)
    assert semanas and all(s.weekday() == 0 and s.year == 2026 and 1 <= s.month <= 6 for s in semanas)
    assert semanas[0] == date(2026, 1, 5) and semanas[-1] == date(2026, 6, 22)
    assert len(semanas) == 25


def test_semanas_ene_jun_omite_semana_con_hueco(hist_2026):
    hueco = hist_2026[hist_2026["fecha"] != pd.Timestamp("2026-03-11")]
    semanas = bj.semanas_ene_jun(hueco)
    assert date(2026, 3, 9) not in semanas  # la semana con un día faltante
    # las vecinas tienen sus 7 días y su sábado de corte: siguen
    assert date(2026, 3, 2) in semanas and date(2026, 3, 16) in semanas


def test_partir_calibracion_60_40_orden_temporal():
    semanas = [date(2026, 1, 5).fromordinal(date(2026, 1, 5).toordinal() + 7 * i) for i in range(10)]
    cal, rep = bj.partir_calibracion(semanas[::-1])  # entrada desordenada
    assert len(cal) == 6 and len(rep) == 4
    assert max(cal) < min(rep)


def test_partir_calibracion_pocas_semanas():
    cal, rep = bj.partir_calibracion([date(2026, 1, 5), date(2026, 1, 12)])
    assert len(cal) == 1 and len(rep) == 1


def _df_gen(valores):
    """valores: {config: [wape_semana1, ...]} -> DataFrame con una fila por semana (energía real 1000)."""
    filas = []
    for cfg, ws in valores.items():
        for i, w in enumerate(ws):
            filas.append({"ucp": "U", "semana_inicio": f"2026-01-{5 + 7 * i:02d}", "config": cfg, "tipo_dia": "laboral",
                          "abs_err_horario": w * 10.0, "energia_real": 1000.0, "energia_pred": 1000.0})
    return pd.DataFrame(filas)


def test_wape_y_comparacion_pareada():
    df = _df_gen({"a": [2.0, 2.0, 2.0, 2.0], "b": [3.0, 3.0, 3.0, 3.0]})
    assert bj.wape(df[df["config"] == "a"]) == pytest.approx(2.0)
    r = bj.comparacion_pareada(df, "a", "b")
    assert r["n"] == 4 and r["dif_media_pp"] == pytest.approx(-1.0) and r["a_favor"] == 4
    assert r["ic90"][1] < 0  # sin variabilidad: el IC queda en −1


def test_comparacion_pareada_sin_comunes():
    df = _df_gen({"a": [2.0]})
    r = bj.comparacion_pareada(df, "a", "zzz")
    assert r["n"] == 0 and np.isnan(r["dif_media_pp"])


def test_elegir_calibracion_toma_menor_wape_entre_candidatas():
    df = _df_gen({"jamer": [3.0, 3.0], "jamer_k15": [2.5, 2.5], "mesa_hoy": [1.0, 1.0]})
    assert bj.elegir_calibracion(df) == "jamer_k15"  # mesa_hoy no es candidata aunque sea la mejor
    assert bj.elegir_calibracion(df.iloc[0:0]) is None


# ---------------------------------------------------------------------------
# integración sintética de una semana (pipeline real, base ingenua, sin red)
# ---------------------------------------------------------------------------

def test_evaluar_semana_gen_sintetica_trae_todas_las_configs_y_sin_fuga(hist_2026):
    from src.adjustments import data

    reales = data.curvas_reales(hist_2026)
    festivos = data.festivos_colombia(range(2023, 2028))
    configs = {k: v for k, v in bj.configs_generalizacion().items() if k in ("mesa_hoy", "jamer")}
    filas = bj.evaluar_semana_gen("Sintetica", date(2026, 5, 11), hist_2026, reales, festivos, configs)
    df = pd.DataFrame(filas)
    assert set(df["config"]) == {"ingenuo", "mesa_hoy", "jamer"}
    assert len(df) == 3 * 7
    assert (df["energia_real"] > 0).all() and np.isfinite(df["abs_err_horario"]).all()
    # La base ingenua es exactamente el paso base: no hay modelo -> el perfil usa E_base = ΣR1 (w no evaluable)
    por = df.pivot_table(index="fecha", columns="config", values="energia_pred")
    assert (por["ingenuo"] > 0).all()


# ---------------------------------------------------------------------------
# humo con los datos reales de la réplica (ignorados por git; se salta si faltan)
# ---------------------------------------------------------------------------

JAMER_DIR = ROOT / "reports" / "ajustes" / "jamer"
HAY_DATOS = all((JAMER_DIR / n).exists() for n in ("historico_replica.csv", "base_sintetica_23_29_nov.csv", "verdad_jamer.csv", "fixture_replica.json"))


@pytest.mark.skipif(not HAY_DATOS, reason="datos de la réplica (reports/ajustes/jamer) no disponibles")
def test_replica_humo_con_datos_reales():
    res = bj.evaluar_replica(bj.cargar_insumos_replica())
    por_id = {c["id"]: c for c in res["criterios"]}
    # criterios deterministas de la réplica que deben cumplirse (A, B, E, G, F1a, F2, F3)
    for cid in ("A1", "A2", "A2b", "A3", "A4", "B1", "B2", "B3", "B4", "E1", "F1a", "F2", "F3", "G1", "G2"):
        assert por_id[cid]["veredicto"] == "pasa", por_id[cid]
    # los que hoy fallan quedan reportados, no ocultos (C2 P95; tramos con base sintética)
    assert por_id["C7"]["veredicto"] in ("pasa", "falla")
    e = res["energia"]
    assert e["jamer"]["mape_pct"] < e["mesa_hoy"]["mape_pct"] < e["crudo"]["mape_pct"]
    # reproducibilidad: dos evaluaciones dan lo mismo
    res2 = bj.evaluar_replica(bj.cargar_insumos_replica())
    assert res["energia"]["jamer"]["dia_pct"] == res2["energia"]["jamer"]["dia_pct"]


@pytest.mark.skipif(not HAY_DATOS, reason="datos de la réplica no disponibles")
def test_replica_no_modifica_entradas_y_escribe_md_sin_curvas(tmp_path):
    insumos = bj.cargar_insumos_replica()
    antes = {f: list(v) for f, v in insumos["verdad"].items()}
    res = bj.evaluar_replica(insumos)
    assert insumos["verdad"] == antes
    md = tmp_path / "r.md"
    bj.escribir_reporte(md, {"fecha": "20261120", "detalle": "x.csv"}, res, None)
    txt = md.read_text(encoding="utf-8")
    assert "SINTÉTICA" in txt and "pasa" in txt and "falla" in txt
    # sin demanda real por periodo: ninguna fila de números con 24 valores tipo curva
    assert not any(len([t for t in linea.split("|") if t.strip().replace(".", "").isdigit()]) >= 20
                   for linea in txt.splitlines())


def test_banda_de_saltos_viene_del_perfil_no_se_repite_el_40():
    from src.adjustments import perfiles

    bj = _cargar_script()
    assert bj.BANDA_SALTO_REPLICA_MW == perfiles.BANDA_SALTO_MW
    assert bj.saltos_finales.__defaults__ == (perfiles.BANDA_SALTO_MW,)
