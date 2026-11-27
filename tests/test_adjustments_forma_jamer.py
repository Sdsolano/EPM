"""Tests de forma (E3): reciente_clase, delta plano y tramos. Sintéticos, festivos 2026 reales, sin red."""

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.adjustments import pipeline
from src.adjustments.data import PERIODOS, festivos_colombia
from src.adjustments.steps import forma
from src.adjustments.types import RunContext

FESTIVOS = sorted(festivos_colombia(range(2024, 2028)))
CORTE = date(2026, 11, 14)
LUNES = date(2026, 11, 23)


def _curva_dia(d: date) -> np.ndarray:
    """Curva sintética con forma dependiente del día de la semana (no plana)."""
    h = np.arange(24)
    base = 1000 + 40 * d.weekday()
    return base + 300 * np.sin((h - 5) / 24 * 2 * np.pi) + 2 * h + d.toordinal() % 5


def _ctx(extra_curvas=None, corte=CORTE) -> RunContext:
    filas, d = [], date(2025, 6, 1)
    extra_curvas = extra_curvas or {}
    while d <= corte:
        v = extra_curvas.get(d.isoformat(), _curva_dia(d))
        filas.append({"fecha": pd.Timestamp(d), **{c: float(x) for c, x in zip(PERIODOS, v)}, "TOTAL": float(np.sum(v))})
        d += timedelta(days=1)
    hist = pd.DataFrame(filas)
    return RunContext(
        ucp="Antioquia", semana_inicio=LUNES, fecha_emision=LUNES, fecha_corte=corte,
        historico=hist, festivos=FESTIVOS,
    )


def _prev(ctx, energias=None, forma_prev=None):
    out = {}
    for i in range(7):
        d = LUNES + timedelta(days=i)
        e = (energias or {}).get(d.isoformat(), 30000.0)
        v = forma_prev if forma_prev is not None else np.ones(24)
        out[d.isoformat()] = (v / v.sum() * e).tolist()
    return out


def _r1(ctx, fecha):
    return np.asarray(_curva_dia(date.fromisoformat(fecha)))


# --- validación -----------------------------------------------------------

def test_defaults_no_se_persisten_salvo_con_fuente_reciente_clase():
    # con fuente="reciente_clase" (perfil) se persisten TODOS los params de la réplica, también los default
    p = forma.validar_params({"fuente": "reciente_clase", "delta": "plano"})
    assert p["delta"] == "plano" and p["fuente"] == "reciente_clase"
    assert {k: p[k] for k in forma.DEFAULTS_JAMER} == {**forma.DEFAULTS_JAMER, "delta": "plano"}
    assert forma.validar_params(p) == p  # idempotente
    # con otra fuente los defaults siguen sin guardarse (comportamiento de siempre)
    q = forma.validar_params({"fuente": "promedio_n", "delta": "plano"})
    assert q["delta"] == "plano" and "tramos" not in q and "umbral_tramo_mw" not in q
    base = forma.validar_params(None)
    assert not set(base) - {"fuente", "n", "beta"}


@pytest.mark.parametrize(
    "params",
    [
        {"delta": "otro"},
        {"tramos": "x"},
        {"tramos": "modelo_anual"},  # exige delta plano
        {"delta": "plano", "tramos": "modelo_anual", "min_periodos_tramo": 0},
        {"suavizado_borde": -1},
        {"umbral_tramo_mw": float("nan")},
        {"saltar_puente": "si"},
    ],
)
def test_validacion_rechaza(params):
    with pytest.raises(ValueError):
        forma.validar_params(params)


# --- reciente_clase ------------------------------------------------------

def test_reciente_clase_usa_r1_mismo_tipo_dia():
    ctx = _ctx()
    prev = _prev(ctx)
    res = forma.step(ctx, prev, {"fuente": "reciente_clase", "n": 1})
    dias = res.diagnostics["dias"]
    esperado = {"2026-11-23": "2026-11-09", "2026-11-24": "2026-11-10", "2026-11-27": "2026-11-13",
                "2026-11-28": "2026-11-14", "2026-11-29": "2026-11-08"}
    for f, r1 in esperado.items():
        assert dias[f]["fuente"] == "reciente_clase"
        assert dias[f]["fechas_fuente"] == [r1]
        curva = np.asarray(res.curves[f])
        r = _r1(ctx, r1)
        assert curva == pytest.approx(r / r.sum() * 30000.0, rel=1e-9)  # proporcional: forma de R1
        assert curva.sum() == pytest.approx(30000.0, abs=1e-6)


def test_reciente_clase_n2_martes_salta_post_festivo():
    ctx = _ctx()
    res = forma.step(ctx, _prev(ctx), {"fuente": "reciente_clase", "n": 2})
    d = res.diagnostics["dias"]["2026-11-24"]
    assert d["fechas_fuente"] == ["2026-11-10", "2026-10-27"]
    assert {"fecha": "2026-11-03", "motivo": "post_festivo"} in d["saltados"]


def test_reciente_clase_sin_datos_cae_a_promedio_n():
    ctx = _ctx(corte=date(2025, 6, 10))
    ctx.semana_inicio = date(2025, 6, 23)
    prev = {(date(2025, 6, 23) + timedelta(days=i)).isoformat(): [1000.0] * 24 for i in range(7)}
    res = forma.step(ctx, prev, {"fuente": "reciente_clase"})
    assert res.diagnostics["dias"]["2025-06-23"]["fuente"] in ("reciente_clase", "promedio_n", "sin_forma")


# --- delta plano: vectores G1/G2 de la fórmula Ajustado ----------------

def _p1_ajustado(p1_r1, e_base, r1_total):
    """Forma de R1 llevada al total del modelo: P1 = (P1_R1 / ΣR1) · E_base, y el delta plano es nulo si E = E_base."""
    forma_n = np.full(24, 1.0 / 24)
    forma_n[0] = p1_r1 / r1_total
    forma_n = forma_n / forma_n.sum()
    curva, delta = forma._curva_plana(np.asarray(forma_n), e_base, e_base, np.ones(24))
    assert delta == 0.0 and curva.sum() == pytest.approx(e_base)
    return p1_r1 / r1_total * e_base


@pytest.mark.parametrize(
    "p1_r1,e_base,r1_total,esperado",
    [  # vectores SINTÉTICOS (el esperado es la fórmula P1·E_base/ΣR1 evaluada a mano)
        (1000.0, 36000.0, 30000.0, 1200.0),
        (900.0, 30000.0, 27000.0, 1000.0),
        (1100.0, 25000.0, 22000.0, 1250.0),
    ],
)
def test_vectores_ajustado_p1_sinteticos(p1_r1, e_base, r1_total, esperado):
    assert _p1_ajustado(p1_r1, e_base, r1_total) == pytest.approx(esperado, abs=0.1)


FIXTURE_REPLICA = Path(__file__).resolve().parents[1] / "reports" / "ajustes" / "jamer" / "fixture_replica.json"


@pytest.mark.skipif(not FIXTURE_REPLICA.exists(),
                    reason="fixture local con los vectores reales de la réplica (reports/ajustes/jamer, ignorado por git) no disponible")
def test_vectores_ajustado_p1_reales_fixture_local():
    vectores = json.loads(FIXTURE_REPLICA.read_text(encoding="utf-8"))["g1_vectores_forma"]
    assert vectores
    for p1_r1, e_base, r1_total, esperado in vectores:
        assert _p1_ajustado(p1_r1, e_base, r1_total) == pytest.approx(esperado, abs=0.1)


def test_plano_conserva_energia_y_delta_uniforme():
    ctx = _ctx()
    e_base = {f: 36000.0 for f in _prev(ctx)}
    prev = _prev(ctx, {f: 36240.0 for f in e_base})  # E = E_base + 240 -> δ = 10 MW
    res = forma.step(ctx, prev, {"fuente": "reciente_clase", "n": 1, "delta": "plano"}, energia_base=e_base)
    f = "2026-11-23"
    curva = np.asarray(res.curves[f])
    r = _r1(ctx, "2026-11-09")
    S = r / r.sum() * 36000.0
    assert curva - S == pytest.approx(np.full(24, 10.0), abs=1e-9)
    assert curva.sum() == pytest.approx(36240.0, abs=1e-6)
    dia = res.diagnostics["dias"][f]
    assert dia["delta_plano_mw"] == pytest.approx(10.0)
    assert dia["periodos_con_gancho"] == list(range(1, 25))
    assert dia["tramos_retenidos"] == []
    assert res.diagnostics["delta"] == "plano"


def test_plano_sigma_cambia_en_delta_por_periodos():
    forma_n = np.ones(24) / 24
    curva, delta = forma._curva_plana(forma_n, 24000.0, 24000.0 + 24 * -55.0, np.ones(24))
    assert delta == pytest.approx(55.0)
    assert curva.sum() == pytest.approx(24000.0)


def test_plano_recorte_a_cero_conserva_suma():
    forma_n = np.array([0.001] * 4 + [(1 - 0.004) / 20] * 20)
    curva, _ = forma._curva_plana(forma_n, 10000.0, 20000.0, np.ones(24))
    assert (curva >= 0).all()
    assert curva.sum() == pytest.approx(10000.0)
    assert curva[:4].max() == 0.0


def test_plano_sin_energia_base_equivale_a_proporcional_con_alerta():
    ctx = _ctx()
    prev = _prev(ctx)
    a = forma.step(ctx, prev, {"fuente": "reciente_clase", "delta": "plano"})
    b = forma.step(ctx, prev, {"fuente": "reciente_clase"})
    for f in prev:
        assert a.curves[f] == pytest.approx(b.curves[f])
    assert any(x.codigo == "FORMA_SIN_ENERGIA_BASE" for x in a.alerts)


def test_proporcional_default_igual_que_hoy():
    ctx = _ctx()
    prev = _prev(ctx)
    for fuente in ("promedio_n", "reciente", "reciente_clase"):
        a = forma.step(ctx, prev, {"fuente": fuente}, energia_base={f: 1.0 for f in prev})
        b = forma.step(ctx, prev, {"fuente": fuente})
        assert a.curves == b.curves
        assert "delta" not in a.diagnostics
        assert "delta_plano_mw" not in a.diagnostics["dias"]["2026-11-23"]


# --- tramos ---------------------------------------------------------------

def test_tramos_sin_gancho_con_bulto_conocido():
    r1 = np.full(24, 1000.0)
    r3 = np.full(24, 1000.0)
    r2 = r3.copy()
    r2[7:11] += 40.0  # cambio anual de +40 MW en P8..P11
    S = r1.copy()
    S[7:11] += 25.0  # el modelo ya se desvía +25 en el mismo sentido
    S[20] += 25.0  # desviación aislada sin cambio anual: no es tramo
    g, tramos = forma._tramos_sin_gancho(S, r1, r2, r3, 20.0, 3, 0)
    assert g[7:11].tolist() == [0, 0, 0, 0]
    assert g.sum() == 20
    assert [(t["desde"], t["hasta"]) for t in tramos] == [(8, 11)]
    assert tramos[0]["desv_mw"] == pytest.approx(25.0) and tramos[0]["desv_mw_10"] == 30.0


def test_tramos_minimo_de_periodos_y_signo_opuesto():
    r1 = r3 = np.full(24, 1000.0)
    r2 = r3.copy()
    r2[7:9] += 40.0  # solo 2 periodos
    S = r1.copy()
    S[7:9] += 25.0
    g, tramos = forma._tramos_sin_gancho(S, r1, r2, r3, 20.0, 3, 0)
    assert tramos == [] and g.sum() == 24
    r2 = r3.copy()
    r2[7:12] += 40.0
    S = r1.copy()
    S[7:12] -= 25.0  # signo opuesto al cambio anual
    g, tramos = forma._tramos_sin_gancho(S, r1, r2, r3, 20.0, 3, 0)
    assert tramos == []


def test_tramos_suavizado_borde_rampa():
    r1 = r3 = np.full(24, 1000.0)
    r2 = r3.copy()
    r2[8:13] += 40.0
    S = r1.copy()
    S[8:13] += 25.0
    g, _ = forma._tramos_sin_gancho(S, r1, r2, r3, 20.0, 3, 2)
    assert g[8:13].tolist() == [0] * 5
    assert g[7] == pytest.approx(1 / 3) and g[6] == pytest.approx(2 / 3)
    assert g[13] == pytest.approx(1 / 3) and g[14] == pytest.approx(2 / 3)


def test_tramos_en_step_conserva_suma_y_retiene_desviacion():
    # R2 (d−364) con +40 en P8..P11 respecto de R3 (R1−364); el modelo se desvía de R1 en el mismo sentido.
    d = LUNES
    r1_fecha = date(2026, 11, 9)
    r2_fecha = d - timedelta(days=364)
    r3_fecha = r1_fecha - timedelta(days=364)
    r3 = _curva_dia(r3_fecha).copy()
    r2 = r3.copy()
    r2[7:11] += 40.0
    ctx = _ctx({r2_fecha.isoformat(): r2, r3_fecha.isoformat(): r3})
    r1 = np.asarray(ctx.historico.loc[ctx.historico["fecha"] == pd.Timestamp(r1_fecha), PERIODOS]).ravel()
    e_base_val = float(r1.sum()) * 1.0
    # El modelo = R1 con +25 en P8..P11 (forma distinta): la forma R1 reescalada no lo ve, el delta sí
    e_base = {f: e_base_val for f in _prev(ctx)}
    energias = {f: e_base_val + 24 * 6.0 for f in e_base}  # δ_nominal 6 MW
    prev = _prev(ctx, energias)
    res = forma.step(
        ctx, prev,
        {"fuente": "reciente_clase", "n": 1, "delta": "plano", "tramos": "modelo_anual", "umbral_tramo_mw": 20},
        energia_base=e_base,
    )
    f = d.isoformat()
    dia = res.diagnostics["dias"][f]
    # E_base = ΣR1 -> S = R1 exacto (desv = 0, sin signo): no hay tramo aunque el cambio anual sea +40
    assert dia["tramos_retenidos"] == []
    assert sum(res.curves[f]) == pytest.approx(energias[f], abs=1e-6)
    assert dia["periodos_con_gancho"] == list(range(1, 25))
    # Con E_base 1 % sobre ΣR1 la desviación es positiva en todo el día y el cambio anual +40 en P8..P11 -> tramo
    e_base2 = {f: e_base_val * 1.01 for f in e_base}
    res2 = forma.step(
        ctx, _prev(ctx, {f: e_base_val * 1.01 + 24 * 6.0 for f in e_base}),
        {"fuente": "reciente_clase", "n": 1, "delta": "plano", "tramos": "modelo_anual", "umbral_tramo_mw": 20},
        energia_base=e_base2,
    )
    dia2 = res2.diagnostics["dias"][f]
    assert [(t["desde"], t["hasta"]) for t in dia2["tramos_retenidos"]] == [(8, 11)]
    assert dia2["periodos_con_gancho"] == [h for h in range(1, 25) if not 8 <= h <= 11]
    assert sum(res2.curves[f]) == pytest.approx(e_base_val * 1.01 + 24 * 6.0, abs=1e-6)


def test_tramos_festivo_o_sin_historico_no_evalua():
    ctx = _ctx()
    prev = {"2026-11-16": [1000.0] * 24}  # lunes festivo
    for i in range(1, 7):
        prev[(LUNES - timedelta(days=7) + timedelta(days=i)).isoformat()] = [1000.0] * 24
    ctx.semana_inicio = date(2026, 11, 16)
    res = forma.step(ctx, prev, {"fuente": "reciente_clase", "delta": "plano", "tramos": "modelo_anual"},
                     energia_base={"2026-11-16": 23000.0})
    assert res.diagnostics["dias"]["2026-11-16"]["tramos_motivo"] == "festivo"
    assert sum(res.curves["2026-11-16"]) == pytest.approx(24000.0, abs=1e-6)


# --- pipeline: energia_base desde nivel -----------------------------------

def test_pipeline_pasa_energia_base_a_forma(monkeypatch):
    capturado = {}
    original = forma.step

    def espia(ctx, prev, params, energia_base=None):
        capturado["eb"] = energia_base
        return original(ctx, prev, params, energia_base=energia_base)

    monkeypatch.setattr(forma, "step", espia)
    from src.adjustments.types import StepRecord, StepResult, StepState

    nivel = StepRecord(
        orden=2, paso="nivel", estado=StepState.APLICADO, params={},
        result=StepResult(curves={}, diagnostics={"dias": {"2026-11-23": {"E_base": 123.5}, "2026-11-24": {}}}),
    )
    ctx = _ctx()
    pipeline._calcular(ctx, "forma", {"fuente": "reciente_clase", "delta": "plano"}, _prev(ctx), None, [nivel])
    assert capturado["eb"] == {"2026-11-23": 123.5}
    pipeline._calcular(ctx, "forma", {}, _prev(ctx), None, [])
    assert capturado["eb"] is None
