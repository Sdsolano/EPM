"""Tests de src/adjustments/correccion.py (bandas empíricas, corrección de sesgo y de clima)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from src.adjustments import correccion as C, data

FORMA = np.array([0.8, 0.75, 0.72, 0.72, 0.75, 0.85, 0.95, 1.05, 1.1, 1.12, 1.12, 1.1,
                  1.08, 1.08, 1.1, 1.12, 1.12, 1.15, 1.2, 1.18, 1.1, 1.0, 0.92, 0.85]) * 100.0
INICIO = date(2026, 1, 5)  # lunes
CORTE = date(2026, 5, 2)   # sábado
FESTIVO = date(2026, 5, 4)  # lunes festivo (futuro respecto al corte)


def _fechas(n_dias=120, inicio=INICIO):
    return [inicio + timedelta(days=i) for i in range(n_dias)]


def _historia(sesgo=None, ruido=0.0, seed=0, n_dias=120):
    """Pronóstico y real sintéticos: real = pronóstico · sesgo(fecha) · exp(ruido)."""
    rng = np.random.default_rng(seed)
    pron, real = {}, {}
    for d in _fechas(n_dias):
        nivel = 1.0 + 0.02 * d.weekday()
        p = FORMA * nivel
        s = 1.0 if sesgo is None else sesgo(d)
        r = p * s * np.exp(rng.normal(0, ruido, 24)) if ruido else p * s
        pron[d.isoformat()] = p.tolist()
        real[d.isoformat()] = r.tolist()
    return pron, real


def _curvas(fechas):
    return {d.isoformat(): (FORMA * (1.0 + 0.02 * d.weekday())).tolist() for d in fechas}


# ---------------------------------------------------------------------------
# Bandas
# ---------------------------------------------------------------------------

def test_banda_contiene_la_curva_y_es_ordenada():
    pron, real = _historia(ruido=0.03, seed=1)
    fechas = [CORTE + timedelta(days=i) for i in range(1, 8)]
    b = C.bandas_empiricas(_curvas(fechas), pron, real, CORTE, [])
    assert set(b) == {d.isoformat() for d in fechas}
    for f, x in b.items():
        c = np.asarray(_curvas([date.fromisoformat(f)])[f])
        assert len(x["inferior"]) == len(x["superior"]) == 24
        assert (np.asarray(x["inferior"]) < c).all() and (c < np.asarray(x["superior"])).all()
        assert x["tipo_dia"] in data.TIPOS_DIA


def test_banda_sin_fuga_ignora_reales_posteriores_al_corte():
    pron, real = _historia(ruido=0.03, seed=2, n_dias=140)
    futuro = {f: (np.asarray(v) * 3).tolist() if date.fromisoformat(f) > CORTE else v for f, v in real.items()}
    fechas = [CORTE + timedelta(days=i) for i in range(1, 8)]
    a = C.bandas_empiricas(_curvas(fechas), pron, real, CORTE, [])
    b = C.bandas_empiricas(_curvas(fechas), pron, futuro, CORTE, [])
    assert a == b


def test_banda_cobertura_cercana_a_la_nominal_sin_escala():
    pron, real = _historia(ruido=0.03, seed=3, n_dias=26 * 7 + 60)
    corte = INICIO + timedelta(days=26 * 7 + 30)
    tablas = C.tablas_banda(pron, real, corte, [], {"escala": 1.0})
    dentro = total = 0
    for i in range(1, 30):
        d = corte + timedelta(days=i)
        if d.isoformat() not in real:
            break
        t = tablas[C.data.tipo_dia(d, [])]
        r, p = np.asarray(real[d.isoformat()]), np.asarray(pron[d.isoformat()])
        dentro += int(((r >= p * np.asarray(t["inferior"])) & (r <= p * np.asarray(t["superior"]))).sum())
        total += 24
    assert 0.65 <= dentro / total <= 0.95  # nominal 80 %


def test_escala_ensancha_la_banda():
    pron, real = _historia(ruido=0.03, seed=4)
    t1 = C.tablas_banda(pron, real, CORTE, [], {"escala": 1.0})["laboral"]
    t2 = C.tablas_banda(pron, real, CORTE, [], {"escala": 1.5})["laboral"]
    assert (np.asarray(t2["superior"]) > np.asarray(t1["superior"])).all()
    assert (np.asarray(t2["inferior"]) < np.asarray(t1["inferior"])).all()


def test_banda_festivo_sin_historial_usa_respaldo_de_todos_los_tipos():
    pron, real = _historia(ruido=0.03, seed=5)
    t = C.tablas_banda(pron, real, CORTE, [])
    assert t["festivo"]["n_dias"] == 0 and t["festivo"]["respaldo"] is True
    assert t["festivo"]["inferior"] is not None
    assert t["laboral"]["respaldo"] is False
    b = C.bandas_empiricas(_curvas([FESTIVO]), pron, real, CORTE, [FESTIVO.isoformat()])
    assert b[FESTIVO.isoformat()]["tipo_dia"] == "festivo" and b[FESTIVO.isoformat()]["respaldo"] is True


def test_banda_sin_historial_no_inventa_banda():
    t = C.tablas_banda({}, {}, CORTE, [])
    assert all(x["inferior"] is None for x in t.values())
    assert C.bandas_empiricas(_curvas([CORTE + timedelta(days=1)]), {}, {}, CORTE, []) == {}


def test_banda_omite_curvas_invalidas():
    pron, real = _historia(ruido=0.03, seed=6)
    curvas = _curvas([CORTE + timedelta(days=1)])
    curvas["2026-05-04"] = [1.0] * 23
    assert set(C.bandas_empiricas(curvas, pron, real, CORTE, [])) == {"2026-05-03"}


def test_banda_params_invalidos():
    with pytest.raises(ValueError):
        C.tablas_banda({}, {}, CORTE, [], {"escalaa": 1})
    with pytest.raises(ValueError):
        C.tablas_banda({}, {}, CORTE, [], {"q_bajo": 0.9})
    with pytest.raises(ValueError):
        C.tablas_banda({}, {}, CORTE, [], {"escala": float("nan")})


# ---------------------------------------------------------------------------
# Sesgo
# ---------------------------------------------------------------------------

def test_sesgo_constante_se_corrige_con_g1_k0_0():
    pron, real = _historia(sesgo=lambda d: 1.05)
    fechas = [CORTE + timedelta(days=i) for i in range(1, 8)]
    curvas = _curvas(fechas)
    out = C.corregir_sesgo(curvas, pron, real, CORTE, [], {"g": 1.0, "k0": 0.0})
    for f, v in out.items():
        np.testing.assert_allclose(v, np.asarray(curvas[f]) * 1.05, rtol=1e-9)


def test_sesgo_shrinkage_y_g_atenúan_el_factor():
    pron, real = _historia(sesgo=lambda d: 1.05)
    curvas = _curvas([CORTE + timedelta(days=2)])
    f = next(iter(curvas))
    fuerte = C.corregir_sesgo(curvas, pron, real, CORTE, [], {"g": 1.0, "k0": 0.0})[f]
    suave = C.corregir_sesgo(curvas, pron, real, CORTE, [], {"g": 0.5, "k0": 3.0})[f]
    base = np.asarray(curvas[f])
    assert (np.asarray(base) < np.asarray(suave)).all() and (np.asarray(suave) < np.asarray(fuerte)).all()


def test_sesgo_sin_fuga():
    pron, real = _historia(sesgo=lambda d: 1.05, n_dias=140)
    futuro = {f: (np.asarray(v) * 2).tolist() if date.fromisoformat(f) > CORTE else v for f, v in real.items()}
    curvas = _curvas([CORTE + timedelta(days=i) for i in range(1, 8)])
    assert C.corregir_sesgo(curvas, pron, real, CORTE, []) == C.corregir_sesgo(curvas, pron, futuro, CORTE, [])


def test_sesgo_por_tipo_de_dia_no_mezcla_segmentos():
    # solo los sábados están subestimados en 10 %
    pron, real = _historia(sesgo=lambda d: 1.10 if d.weekday() == 5 else 1.0)
    sab, lun = CORTE + timedelta(days=7), CORTE + timedelta(days=9)  # 2026-05-09 sábado, 2026-05-11 lunes
    assert sab.weekday() == 5 and lun.weekday() == 0
    curvas = _curvas([sab, lun])
    out = C.corregir_sesgo(curvas, pron, real, CORTE, [], {"g": 1.0, "k0": 0.0})
    np.testing.assert_allclose(out[sab.isoformat()], np.asarray(curvas[sab.isoformat()]) * 1.10, rtol=1e-9)
    np.testing.assert_allclose(out[lun.isoformat()], curvas[lun.isoformat()], rtol=1e-9)


def test_sesgo_segmento_dow_separa_dias_de_semana():
    pron, real = _historia(sesgo=lambda d: 1.08 if d.weekday() == 1 else 1.0)
    mar, mie = CORTE + timedelta(days=3), CORTE + timedelta(days=4)  # martes y miércoles
    assert mar.weekday() == 1
    curvas = _curvas([mar, mie])
    out = C.corregir_sesgo(curvas, pron, real, CORTE, [], {"segmento": "dow", "g": 1.0, "k0": 0.0})
    np.testing.assert_allclose(out[mar.isoformat()], np.asarray(curvas[mar.isoformat()]) * 1.08, rtol=1e-9)
    np.testing.assert_allclose(out[mie.isoformat()], curvas[mie.isoformat()], rtol=1e-9)


def test_sesgo_pocos_casos_deja_igual():
    pron, real = _historia(sesgo=lambda d: 1.05)
    curvas = _curvas([FESTIVO])  # sin festivos en el historial
    out = C.corregir_sesgo(curvas, pron, real, CORTE, [FESTIVO.isoformat()])
    assert out == {FESTIVO.isoformat(): curvas[FESTIVO.isoformat()]}


def test_sesgo_modo_energia_conserva_la_forma_y_forma_conserva_la_energia():
    pron, real = _historia(sesgo=lambda d: 1.05)
    curvas = _curvas([CORTE + timedelta(days=2)])
    f = next(iter(curvas))
    base = np.asarray(curvas[f])
    en = np.asarray(C.corregir_sesgo(curvas, pron, real, CORTE, [], {"modo": "energia", "g": 1.0, "k0": 0.0})[f])
    np.testing.assert_allclose(en / base, np.full(24, 1.05), rtol=1e-9)
    fo = np.asarray(C.corregir_sesgo(curvas, pron, real, CORTE, [], {"modo": "forma", "g": 1.0, "k0": 0.0})[f])
    assert fo.sum() == pytest.approx(base.sum())


def test_sesgo_pasa_curvas_invalidas_sin_tocarlas():
    pron, real = _historia(sesgo=lambda d: 1.05)
    curvas = {"2026-05-04": [1.0] * 5}
    assert C.corregir_sesgo(curvas, pron, real, CORTE, []) == {"2026-05-04": [1.0] * 5}


def test_sesgo_params_invalidos():
    with pytest.raises(ValueError):
        C.corregir_sesgo({}, {}, {}, CORTE, [], {"segmento": "mes"})
    with pytest.raises(ValueError):
        C.corregir_sesgo({}, {}, {}, CORTE, [], {"g": 2})
    with pytest.raises(ValueError):
        C.corregir_sesgo({}, {}, {}, CORTE, [], {"nada": 1})


# ---------------------------------------------------------------------------
# Clima
# ---------------------------------------------------------------------------

def _clima(n_dias=140, inicio=INICIO - timedelta(days=40), seed=0):
    rng = np.random.default_rng(seed)
    return {(inicio + timedelta(days=i)).isoformat(): float(27 + rng.normal(0, 2)) for i in range(n_dias + 40)}


def test_anomalia_temperatura_usa_corte_de_la_semana():
    clima = {(date(2026, 4, 1) + timedelta(days=i)).isoformat(): 20.0 for i in range(60)}
    d = date(2026, 5, 6)  # miércoles; su corte es el sábado 2026-05-02
    clima[d.isoformat()] = 25.0
    assert C.anomalia_temperatura(clima, d) == pytest.approx(5.0)
    assert C.anomalia_temperatura({}, d) is None
    assert C.anomalia_temperatura({d.isoformat(): 25.0}, d) is None  # sin días previos


def test_clima_corrige_la_energia_segun_la_anomalia():
    clima = _clima()
    pron, real = {}, {}
    for d in _fechas(140):
        a = C.anomalia_temperatura(clima, d)
        p = FORMA * (1.0 + 0.02 * d.weekday())
        pron[d.isoformat()] = p.tolist()
        real[d.isoformat()] = (p * np.exp(0.02 * (a if a is not None else 0.0))).tolist()
    d = CORTE + timedelta(days=3)  # miércoles
    a = C.anomalia_temperatura(clima, d)
    curvas = _curvas([d])
    out = np.asarray(C.corregir_clima(curvas, pron, real, clima, CORTE, [], {"k0": 0.0})[d.isoformat()])
    verdadero = np.asarray(curvas[d.isoformat()]) * np.exp(0.02 * a)
    sin_corregir = np.asarray(curvas[d.isoformat()])
    assert abs(out.sum() - verdadero.sum()) < abs(sin_corregir.sum() - verdadero.sum())


def test_clima_sin_evidencia_o_sin_clima_deja_igual_y_no_toca_fines_de_semana():
    pron, real = _historia(sesgo=lambda d: 1.0)
    d = CORTE + timedelta(days=3)
    curvas = _curvas([d, CORTE + timedelta(days=1)])  # miércoles y domingo
    assert C.corregir_clima(curvas, pron, real, {}, CORTE, []) == curvas
    out = C.corregir_clima(curvas, pron, real, _clima(), CORTE, [])
    assert out[(CORTE + timedelta(days=1)).isoformat()] == curvas[(CORTE + timedelta(days=1)).isoformat()]


def test_clima_sin_fuga():
    clima = _clima()
    pron, real = _historia(sesgo=lambda d: 1.0 + 0.01 * ((d.toordinal() % 5) - 2), n_dias=140)
    futuro = {f: (np.asarray(v) * 2).tolist() if date.fromisoformat(f) > CORTE else v for f, v in real.items()}
    curvas = _curvas([CORTE + timedelta(days=3)])
    assert C.corregir_clima(curvas, pron, real, clima, CORTE, []) == C.corregir_clima(curvas, pron, futuro, clima, CORTE, [])


def test_clima_params_invalidos():
    with pytest.raises(ValueError):
        C.corregir_clima({}, {}, {}, {}, CORTE, [], {"solo_laboral": 1})
    with pytest.raises(ValueError):
        C.corregir_clima({}, {}, {}, {}, CORTE, [], {"minimo_casos": 1})
