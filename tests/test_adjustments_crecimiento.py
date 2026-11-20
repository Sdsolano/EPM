"""Tests de crecimiento interanual reciente Y y g dinámico (sintéticos, sin red)."""

from datetime import date, timedelta

import numpy as np
import pytest

from src.adjustments.crecimiento import (
    calcular_g,
    crecimiento_interanual,
    g_crudo,
    validar_params_g,
)
from src.adjustments.data import festivos_colombia

FESTIVOS = set(festivos_colombia(range(2024, 2028)))
CORTE = date(2026, 11, 14)


def _curvas(factor=1.10, desde=date(2025, 9, 1), hasta=CORTE, base=1000.0):
    """2025 plano = base; 2026 = base·factor (cada día de 2026 = día−364 × factor)."""
    out, d = {}, desde
    while d <= hasta:
        v = base if d.year == 2025 or d < date(2025, 12, 31) else base * factor
        out[d.isoformat()] = np.full(24, v / 24)
        d += timedelta(days=1)
    for k in list(out):
        if k >= "2026-01-01":
            out[k] = np.full(24, base * factor / 24)
    return out


def test_Y_recupera_factor_y_excluye_festivos():
    c = _curvas(1.10)
    r = crecimiento_interanual(c, CORTE, FESTIVOS)
    assert r.fallback is False
    assert r.Y == pytest.approx(0.10, abs=1e-9)
    # ventana 28 d (18/10-14/11): festivos 2026 (02/11) y post (03/11) fuera; también sus pares
    assert r.n_pares < 28
    assert r.ventana == ("2026-10-18", "2026-11-14")


def test_Y_excluye_dia_cuyo_par_es_festivo():
    c = _curvas(1.10)
    base = crecimiento_interanual(c, CORTE, FESTIVOS)
    # 2025-11-03 es festivo (Todos los Santos 2025) -> su par 2026-11-02/03 ya excluido; 2025-10-13 festivo -> par 2026-10-12 festivo
    # contaminamos un día normal: debe cambiar Y; contaminar un festivo no debe cambiarlo
    c2 = dict(c)
    c2["2026-11-02"] = np.full(24, 999999.0)  # festivo 2026
    assert crecimiento_interanual(c2, CORTE, FESTIVOS).Y == pytest.approx(base.Y)
    c3 = dict(c)
    c3["2026-11-03"] = np.full(24, 999999.0)  # post-festivo
    assert crecimiento_interanual(c3, CORTE, FESTIVOS).Y == pytest.approx(base.Y)
    c4 = dict(c)
    c4["2026-11-10"] = np.full(24, 999999.0)  # normal
    assert crecimiento_interanual(c4, CORTE, FESTIVOS).Y != pytest.approx(base.Y)


def test_min_pares_y_fallback():
    c = _curvas()
    r = crecimiento_interanual(c, CORTE, FESTIVOS, min_pares=40)
    assert r.fallback and r.Y is None
    vacio = crecimiento_interanual({}, CORTE, FESTIVOS)
    assert vacio.fallback and vacio.n_pares == 0 and vacio.Y is None


def test_ventana_14_vs_28():
    c = _curvas()
    r14 = crecimiento_interanual(c, CORTE, FESTIVOS, ventana_dias=14, min_pares=7)
    r28 = crecimiento_interanual(c, CORTE, FESTIVOS, ventana_dias=28)
    assert r14.ventana[0] == "2026-11-01" and r14.n_pares < r28.n_pares


def test_pares_invalidos_se_omiten():
    c = _curvas()
    c["2026-11-10"] = np.append(np.full(23, 1.0), np.nan)
    r = crecimiento_interanual(c, CORTE, FESTIVOS)
    assert r.Y == pytest.approx(0.10, abs=1e-9)


def test_g_crudo_formula():
    assert g_crudo(0.10, 14) == pytest.approx(1.10 ** (14 / 364) - 1)
    assert g_crudo(0.0, 21) == 0.0
    assert g_crudo(0.10, 14, s_sem=1.01) == pytest.approx(1.01 * 1.10 ** (14 / 364) - 1)
    with pytest.raises(ValueError):
        g_crudo(-1.0, 7)


def test_calcular_g_dinamico_orden_de_magnitud_jamer():
    # Y ≈ +9 %, lag 14-21 d -> +0.34..+0.5 %
    for lag in (14, 21):
        r = calcular_g(0.09, lag)
        assert 0.003 < r.g_usado < 0.006
        assert not r.acotado and r.alertas == []


def test_calcular_g_clip_g_k_y_alerta():
    r = calcular_g(0.30, 14, g_max=0.005)
    assert r.g_usado == 0.005 and r.acotado
    assert [a.codigo for a in r.alertas] == ["G_ACOTADO"]
    assert r.g_raw == pytest.approx(1.30 ** (14 / 364) - 1)
    r2 = calcular_g(0.09, 14, g_k=1.5)
    assert r2.g_usado == pytest.approx(1.5 * (1.09 ** (14 / 364) - 1))
    r3 = calcular_g(-0.5, 14)
    assert r3.g_usado == -0.01 and r3.acotado


def test_calcular_g_sin_evidencia():
    r = calcular_g(None, 14, g_fijo=0.004, fecha="2026-11-24")
    assert r.g_usado == 0.004 and r.g_raw == 0.004
    assert [a.codigo for a in r.alertas] == ["G_SIN_EVIDENCIA"]
    assert r.alertas[0].fechas == ["2026-11-24"]
    # sin evidencia + clip -> ambas
    r2 = calcular_g(None, 14, g_fijo=0.05)
    assert [a.codigo for a in r2.alertas] == ["G_SIN_EVIDENCIA", "G_ACOTADO"]


def test_calcular_g_fijo_no_alerta_sin_evidencia():
    r = calcular_g(None, 14, fuente="fijo", g_fijo=0.007)
    assert r.g_usado == 0.007 and r.alertas == []


def test_calcular_g_estacional_y_errores():
    r = calcular_g(0.09, 14, fuente="interanual_estacional", s_sem=1.002)
    assert r.g_raw == pytest.approx(1.002 * 1.09 ** (14 / 364) - 1)
    with pytest.raises(ValueError):
        calcular_g(0.09, 14, fuente="interanual_estacional")
    with pytest.raises(ValueError):
        calcular_g(0.09, 14, fuente="otra")
    with pytest.raises(ValueError):
        calcular_g(0.09, 14, g_min=0.02, g_max=0.0)


def test_validar_params_g():
    p = validar_params_g(None)
    assert p["g_fuente"] == "interanual" and p["g_ventana_dias"] == 28
    assert validar_params_g({"g_k": 1.5, "otro": 1})["g_k"] == 1.5
    for malo in (
        {"g_fuente": "x"},
        {"g_k": -1},
        {"g_min": 0.1, "g_max": 0.0},
        {"g_ventana_dias": 10},
        {"g_min_pares": 0},
        {"g_fijo": float("nan")},
        {"g_fijo": True},
    ):
        with pytest.raises(ValueError):
            validar_params_g(malo)


def test_validar_params_g_cotas_maximas_y_mensajes_acotados():
    """E2: cotas máximas con los helpers de _params (ningún valor desborda bucles ni potencias)."""
    ok = validar_params_g({"g_ventana_dias": 364, "g_min_pares": 364, "g_k": 1000.0, "g_fijo": 0.5, "g_min": -0.5, "g_max": 0.5})
    assert ok["g_ventana_dias"] == 364 and ok["g_k"] == 1000.0
    for malo in (
        {"g_ventana_dias": 371},          # > 364
        {"g_ventana_dias": 10 ** 400},    # entero gigante: se rechaza por rango, no desborda
        {"g_min_pares": 365},
        {"g_min_pares": 10 ** 400},
        {"g_k": 1000.1},
        {"g_k": float("inf")},
        {"g_fijo": 0.6},
        {"g_min": -0.6},
        {"g_max": 10 ** 400},
        {"g_ventana_dias": "28"},
        {"g_min_pares": 14.5},
    ):
        with pytest.raises(ValueError) as e:
            validar_params_g(malo)
        assert len(str(e.value)) < 200  # mensaje acotado (repr corto)
    assert validar_params_g({"g_min_pares": 14.0})["g_min_pares"] == 14  # float entero se acepta como el resto de pasos


def test_energia_de_la_ventana_usa_helper_unico_de_curva_valida():
    """E7: una curva con negativos / NaN / de 23 periodos no genera pares (igual que antes de unificar)."""
    from src.adjustments.crecimiento import crecimiento_interanual

    festivos = set()
    curvas = {}
    d = date(2025, 1, 1)
    while d <= date(2026, 3, 1):
        curvas[d.isoformat()] = [100.0] * 24
        d += timedelta(days=1)
    corte = date(2026, 3, 1)
    sano = crecimiento_interanual(curvas, corte, festivos, ventana_dias=14, min_pares=1)
    assert sano.n_pares == 14 and sano.Y == pytest.approx(0.0)
    curvas["2026-02-28"] = [100.0] * 23            # 23 periodos
    curvas["2026-02-27"] = [100.0] * 23 + [float("nan")]
    curvas["2026-02-26"] = [100.0] * 23 + [-1.0]
    malo = crecimiento_interanual(curvas, corte, festivos, ventana_dias=14, min_pares=1)
    assert malo.n_pares == 11
