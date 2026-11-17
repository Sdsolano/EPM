"""Reproducción del ASR de Sphere (`src/adjustments/asr.py`) y equivalencia con /validate-hourly-adjustments."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.adjustments import asr
from src.adjustments.data import festivos_colombia

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Redondeos JS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x, esperado", [(2.5, 3), (-2.5, -2), (2.4999, 2), (-2.5001, -3), (0.5, 1), (-0.5, 0)])
def test_math_round_como_js(x, esperado):
    assert asr.math_round(x) == esperado


@pytest.mark.parametrize("x, esperado", [
    (2.25, 2.3),     # mitad exacta en binario → hacia arriba
    (-2.25, -2.3),   # simétrico (toFixed trabaja con |x|)
    (1.45, 1.4),     # 1.45 es 1.4499999… en binario → (1.45).toFixed(1) === "1.4"
    (1.05, 1.1),     # 1.0500000000000000444 → "1.1"
    (1234.56, 1234.6),
])
def test_to_fixed_1_como_js(x, esperado):
    assert asr.to_fixed_1(x) == esperado


# ---------------------------------------------------------------------------
# Cálculo de ajustes (handleASR)
# ---------------------------------------------------------------------------

def _curva(nivel: float, ruido: int = 0) -> list:
    rng = np.random.default_rng(ruido)
    return (nivel * (1 + 0.2 * np.sin(np.arange(24) / 24 * 2 * np.pi)) + rng.normal(0, 3, 24)).tolist()


H_RECIENTE = asr.Historico("2026-11-09", _curva(1000, 1))
H_ANIO_D = asr.Historico("2025-11-17", _curva(950, 2))
H_ANIO_ANCLA = asr.Historico("2025-11-10", _curva(940, 3))
PRED = _curva(1010, 4)


def test_objetivo_es_h1_mas_desvi2_e_independiente_de_la_base():
    sel = [H_RECIENTE, H_ANIO_D, H_ANIO_ANCLA]  # clic de arriba abajo = fecha descendente
    for pred in (PRED, _curva(1500, 9)):
        aj = asr.ajustes_asr(pred, sel)
        objetivo = np.asarray([asr.to_fixed_1(p) for p in pred]) + np.asarray(aj)
        esperado = np.asarray(H_RECIENTE.curva) + (np.asarray(H_ANIO_D.curva) - np.asarray(H_ANIO_ANCLA.curva))
        # solo difiere por redondeos (toFixed(1) de pred y desvi2, Math.round del delta)
        assert np.max(np.abs(objetivo - esperado)) <= 0.5 + 0.05 + 0.05 + 1e-9


def test_ajustes_son_enteros_con_math_round():
    h1 = asr.Historico("2026-01-08", [10.0] * 24)
    h2 = asr.Historico("2026-01-07", [12.5] * 24)
    h3 = asr.Historico("2026-01-06", [10.0] * 24)
    # desvi2 = 2.5; desvi1 = pred − h1 = 0 → delta = 2.5 → Math.round = 3 (Python round daría 2)
    assert asr.ajustes_asr([10.0] * 24, [h1, h2, h3]) == [3] * 24
    # delta = −2.5 → Math.round = −2
    assert asr.ajustes_asr([15.0] * 24, [h1, h2, h3]) == [-2] * 24


def test_h1_es_el_mas_reciente_aunque_se_marque_ultimo():
    """historicosCombinados se ordena por fecha descendente; desvi2 congelado usa el orden de clic."""
    sel_asc = [H_ANIO_ANCLA, H_ANIO_D, H_RECIENTE]  # clic de abajo arriba
    aj = asr.ajustes_asr(PRED, sel_asc)
    objetivo = np.asarray([asr.to_fixed_1(p) for p in PRED]) + np.asarray(aj)
    # h1 = 2026-11-09 (el más reciente); desvi2 congelado = clic[1] − clic[2] = h2 − h1 → objetivo ≈ h2
    assert np.max(np.abs(objetivo - np.asarray(H_ANIO_D.curva))) <= 0.6 + 1e-9
    # sin desvi2 congelado (h2 − h3 del orden por fecha) sí da h1 + (h2 − h3)
    aj2 = asr.ajustes_asr(PRED, sel_asc, congelar_desvi2=False)
    assert aj2 == asr.ajustes_asr(PRED, [H_RECIENTE, H_ANIO_D, H_ANIO_ANCLA], congelar_desvi2=False)


def test_desvi2_congelado_pone_reales_antes_que_historicos_pronostico():
    hp = asr.Historico("2026-11-10", _curva(2000, 7), tipo="pronostico")
    sel = [hp, H_RECIENTE, H_ANIO_D, H_ANIO_ANCLA]
    orden = asr.orden_congelado(sel)
    assert [h.fecha for h in orden] == ["2026-11-09", "2025-11-17", "2025-11-10", "2026-11-10"]
    d2 = asr.desvi2_congelado(sel)
    assert d2 == [asr.to_fixed_1(a - b) for a, b in zip(H_ANIO_D.curva, H_ANIO_ANCLA.curva)]
    # pero h1 (orden por fecha, mezclando tipos) es el histórico-pronóstico 2026-11-10
    assert asr.ordenar_por_fecha(sel)[0].fecha == "2026-11-10"


def test_menos_de_tres_historicos_falla():
    with pytest.raises(ValueError):
        asr.ajustes_asr(PRED, [H_RECIENTE, H_ANIO_D])


def test_none_en_curvas_es_cero():
    h = [asr.Historico(f"2026-01-0{i}", [None] * 24) for i in (1, 2, 3)]
    assert asr.ajustes_asr([None] * 24, h) == [0] * 24


def test_payload_como_front():
    p = asr.payload_asr("Antioquia", "2026-11-16", PRED, [H_RECIENTE, H_ANIO_D, H_ANIO_ANCLA])
    assert p["tipo_dia"] == "laboral"
    assert p["predicciones_actuales"] == [asr.to_fixed_1(v) for v in PRED]
    assert all(isinstance(a, int) for a in p["ajustes_solicitados"])


def test_fechas_escenario():
    d, corte = date(2026, 5, 24), date(2026, 5, 16)  # domingo con corte sábado → ancla d − 14
    assert asr.fechas_escenario("anual", d, corte) == [date(2026, 5, 10), date(2025, 5, 25), date(2025, 5, 11)]
    assert asr.fechas_escenario("anual_desfase", d, corte)[2] == date(2025, 5, 18)
    assert asr.fechas_escenario("anual_alineado", d, corte, -357) == [
        date(2026, 5, 10), date(2025, 6, 1), date(2025, 5, 18)]
    assert asr.fechas_escenario("anual_clic_asc", d, corte) == [date(2025, 5, 11), date(2025, 5, 25), date(2026, 5, 10)]
    assert asr.fechas_escenario("recientes", d, corte) == [date(2026, 5, 10), date(2026, 5, 3), date(2026, 4, 26)]
    with pytest.raises(ValueError):
        asr.fechas_escenario("otro", d, corte)


# ---------------------------------------------------------------------------
# Equivalencia con el endpoint (motor falso, siempre corre)
# ---------------------------------------------------------------------------

def _perfil(desfase: float) -> np.ndarray:
    v = 1.0 + 0.25 * np.sin((np.arange(24) - 6 - desfase) / 24 * 2 * np.pi)
    return v / v.sum()


class _Normal:
    is_fitted = True
    cluster_by_dayofweek = {i: i for i in range(7)}
    cluster_profiles = pd.DataFrame([_perfil(i) for i in range(7)])


class _Especial:
    is_fitted = True
    cluster_by_date = {"12-25": 1}
    cluster_profiles = pd.DataFrame([_perfil(3), _perfil(4)])
    average_holiday_profile = _perfil(5)
    festivos = {"2025-12-25", "2026-01-12"}

    def is_special_day(self, fecha):
        return fecha.strftime("%m-%d") in asr.DIAS_MUY_ESPECIALES or fecha.strftime("%Y-%m-%d") in self.festivos


class MotorFalso:
    def __init__(self, *args, **kwargs):
        self.normal_disaggregator = _Normal()
        self.special_disaggregator = _Especial()


@pytest.fixture(scope="module")
def api_main():
    return pytest.importorskip("src.api.main")


@pytest.fixture
def cliente_falso(api_main, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api_main, "HourlyDisaggregationEngine", MotorFalso)
    return TestClient(api_main.app)


def _seleccion_fuerte():
    """Históricos que piden un cambio grande de forma (para que el validador recorte)."""
    h1 = asr.Historico("2026-01-05", (1000 * _perfil(0) * 24).tolist())
    h2 = asr.Historico("2025-01-06", (900 * _perfil(8) * 24).tolist())
    h3 = asr.Historico("2025-01-05", (900 * _perfil(0) * 24).tolist())
    return [h1, h2, h3]


@pytest.mark.parametrize("fecha", ["2026-01-14", "2025-12-25", "2026-01-12", "2025-12-24"])
def test_asr_igual_al_endpoint_motor_falso(cliente_falso, fecha):
    pred = (1050 * _perfil(1) * 24).tolist()
    sel = _seleccion_fuerte()
    payload = asr.payload_asr("Antioquia", fecha, pred, sel)
    r = cliente_falso.post("/validate-hourly-adjustments", json=payload)
    assert r.status_code == 200, r.text
    api = r.json()
    motor = MotorFalso()
    local = asr.validar_payload(motor, payload)
    assert np.allclose(api["valores_ajustados"], local["valores_ajustados"], atol=1e-9)
    assert api["metadata"]["cluster_id"] == local["metadata"]["cluster_id"]
    assert api["metadata"]["metodo"] == local["metadata"]["metodo"]
    res = asr.asr_dia("Antioquia", fecha, pred, sel, motor=motor)
    assert np.allclose(res.final, api["valores_ajustados"], atol=1e-9)
    assert res.periodos_modificados == api["metadata"]["periodos_modificados"]
    # con la lista de festivos (sin API) da lo mismo
    res2 = asr.asr_dia("Antioquia", fecha, pred, sel, motor=motor, festivos=_Especial.festivos)
    assert np.allclose(res2.final, res.final)


def test_tipo_dia_no_cambia_la_respuesta_del_endpoint(cliente_falso):
    pred = (1050 * _perfil(1) * 24).tolist()
    sel = _seleccion_fuerte()
    salidas = []
    for tipo in ("laboral", "festivo", "weekend"):
        payload = asr.payload_asr("Antioquia", "2026-01-14", pred, sel, tipo_dia=tipo)
        r = cliente_falso.post("/validate-hourly-adjustments", json=payload)
        assert r.status_code == 200
        salidas.append(r.json()["valores_ajustados"])
    assert salidas[0] == salidas[1] == salidas[2]


def test_validador_recorta_y_sin_validador_es_el_objetivo():
    motor = MotorFalso()
    pred = (1050 * _perfil(1) * 24).tolist()
    sel = _seleccion_fuerte()
    con = asr.asr_dia("Antioquia", "2026-01-14", pred, sel, motor=motor)
    sin = asr.asr_dia("Antioquia", "2026-01-14", pred, sel, validador=False)
    assert sin.final == sin.objetivo and sin.cluster_id is None
    assert con.periodos_modificados > 0 and not np.allclose(con.final, sin.final)
    assert con.h_fechas == ["2026-01-05", "2025-01-06", "2025-01-05"]
    assert con.desvi2_fechas == ["2025-01-06", "2025-01-05"]


# ---------------------------------------------------------------------------
# Equivalencia con el endpoint usando los modelos reales (si están)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (ROOT / "models" / "Antioquia" / "hourly_disaggregator.pkl").exists(),
                    reason="sin models/Antioquia")
@pytest.mark.parametrize("fecha", ["2026-03-10", "2026-03-23", "2025-12-24"])  # laboral, festivo, muy especial
def test_asr_igual_al_endpoint_modelos_reales(api_main, monkeypatch, fecha):
    from fastapi.testclient import TestClient

    from src.prediction.hourly import calendar_utils

    festivos = set(festivos_colombia([2025, 2026]))
    # sin red: los festivos del endpoint salen de la misma lista que usa la reproducción
    monkeypatch.setattr(calendar_utils.CalendarClassifier, "is_holiday",
                        lambda self, d: pd.Timestamp(d).strftime("%Y-%m-%d") in festivos)
    cliente = TestClient(api_main.app)
    pred = (1400 * _perfil(1) * 24).tolist()
    sel = _seleccion_fuerte()
    payload = asr.payload_asr("Antioquia", fecha, pred, sel)
    r = cliente.post("/validate-hourly-adjustments", json=payload)
    assert r.status_code == 200, r.text
    api = r.json()
    res = asr.asr_dia("Antioquia", fecha, pred, sel, festivos=festivos)
    assert np.allclose(res.final, api["valores_ajustados"], atol=1e-9)
    assert res.cluster_id == api["metadata"]["cluster_id"]
    assert res.metodo == api["metadata"]["metodo"]


# ---------------------------------------------------------------------------
# Paridad de es_dia_especial (sin red) con SpecialDaysDisaggregator.is_special_day, y caché del motor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fecha", ["2025-12-24", "2026-01-02", "2026-01-12", "2026-03-23", "2026-03-10",
                                   "2026-12-25", "2026-05-01"])
def test_es_dia_especial_igual_a_is_special_day(fecha):
    from src.prediction.hourly.special_days import SpecialDaysDisaggregator

    festivos = set(festivos_colombia([2025, 2026]))
    esp = SpecialDaysDisaggregator()
    esp.is_fitted = True
    esp.calendar_classifier.is_holiday = lambda d: pd.Timestamp(d).strftime("%Y-%m-%d") in festivos

    class Motor:
        special_disaggregator = esp

    ts = pd.Timestamp(fecha)
    assert asr.es_dia_especial(Motor(), ts, festivos) == bool(esp.is_special_day(ts))
    assert asr.es_dia_especial(Motor(), ts, None) == bool(esp.is_special_day(ts))


def test_valores_no_finitos_son_valueerror():
    with pytest.raises(ValueError):
        asr.ajustes_asr([float("nan")] * 24, [H_RECIENTE, H_ANIO_D, H_ANIO_ANCLA])
    with pytest.raises(ValueError):
        asr.payload_asr("Antioquia", "2026-11-16", [float("inf")] * 24, [H_RECIENTE, H_ANIO_D, H_ANIO_ANCLA])


def test_cargar_motor_cachea_y_recarga_si_cambian_los_pkl(tmp_path, monkeypatch):
    import os

    import src.prediction.hourly as hourly

    cargas = []

    class Motor:
        def __init__(self, **kwargs):
            cargas.append(kwargs)

    monkeypatch.setattr(hourly, "HourlyDisaggregationEngine", Motor)
    monkeypatch.setattr(asr, "_motores", {})
    (tmp_path / "Sintetica").mkdir()
    pkl = tmp_path / "Sintetica" / "hourly_disaggregator.pkl"
    pkl.write_bytes(b"x")
    m1 = asr.cargar_motor("Sintetica", tmp_path)
    assert asr.cargar_motor("Sintetica", tmp_path) is m1 and len(cargas) == 1
    os.utime(pkl, (pkl.stat().st_atime, pkl.stat().st_mtime + 10))  # "reentrenado"
    m2 = asr.cargar_motor("Sintetica", tmp_path)
    assert m2 is not m1 and len(cargas) == 2
    assert cargas[0]["models_dir"] == str(tmp_path / "Sintetica")
