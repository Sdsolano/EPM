"""Tests de clase_dia y seleccionar_recientes (sintéticos, festivos 2026 reales, sin red)."""

from datetime import date, timedelta

import numpy as np
import pytest

from src.adjustments.data import clase_dia, festivos_colombia
from src.adjustments.referencia_reciente import seleccionar_recientes

FESTIVOS = set(festivos_colombia(range(2024, 2028)))
CORTE = date(2026, 11, 14)


def _curvas(desde=date(2026, 8, 1), hasta=CORTE, excluir=()):
    out, d = {}, desde
    while d <= hasta:
        if d.isoformat() not in excluir:
            out[d.isoformat()] = np.full(24, 100.0 + d.toordinal() % 7)
        d += timedelta(days=1)
    return out


def _f(*iso):
    return [date.fromisoformat(x) for x in iso]


def test_clase_dia_basico():
    assert clase_dia("2026-11-02", FESTIVOS) == "festivo"
    assert clase_dia("2026-11-03", FESTIVOS) == "post_festivo"
    assert clase_dia("2026-11-17", FESTIVOS) == "post_festivo"
    assert clase_dia("2026-11-10", FESTIVOS) == "normal"
    # sábado y domingo previos a lunes festivo: normal salvo puente=True
    assert clase_dia("2026-11-01", FESTIVOS) == "normal"
    assert clase_dia("2026-11-15", FESTIVOS) == "normal"
    assert clase_dia("2026-11-01", FESTIVOS, puente=True) == "puente"
    assert clase_dia("2026-11-14", FESTIVOS, puente=True) == "puente"
    assert clase_dia("2026-11-08", FESTIVOS, puente=True) == "normal"
    # un sábado después de festivo de viernes no es post_festivo
    assert clase_dia(date(2026, 11, 7), {"2026-11-06"}) == "normal"


@pytest.mark.parametrize(
    "objetivo,esperado",
    [
        ("2026-11-23", ["2026-11-09"]),
        ("2026-11-24", ["2026-11-10"]),
        ("2026-11-25", ["2026-11-11"]),
        ("2026-11-26", ["2026-11-12"]),
        ("2026-11-27", ["2026-11-13"]),
        ("2026-11-28", ["2026-11-14"]),
        ("2026-11-29", ["2026-11-08"]),
    ],
)
def test_semana_jamer(objetivo, esperado):
    r = seleccionar_recientes(objetivo, _curvas(), CORTE, FESTIVOS)
    assert r.fechas == _f(*esperado)
    assert r.clase == "normal"
    assert r.alertas == []


def test_martes_n2_salta_post_festivo():
    r = seleccionar_recientes("2026-11-24", _curvas(), CORTE, FESTIVOS, n=2)
    assert r.fechas == _f("2026-11-10", "2026-10-27")
    assert {"fecha": "2026-11-03", "motivo": "post_festivo"} in r.saltados


def test_lunes_normal_no_usa_lunes_festivo():
    # corte 22/11: ancla de lun 30/11 = 23/11? no (> corte); k=2 -> 16/11 festivo, se salta -> 09/11
    corte = date(2026, 11, 22)
    r = seleccionar_recientes("2026-11-30", _curvas(hasta=corte), corte, FESTIVOS)
    assert r.fechas == _f("2026-11-09")
    assert {"fecha": "2026-11-16", "motivo": "festivo"} in r.saltados


def test_objetivo_festivo_y_post_festivo():
    assert seleccionar_recientes("2026-11-16", _curvas(), CORTE, FESTIVOS).fechas == _f("2026-11-02")
    assert seleccionar_recientes("2026-11-17", _curvas(), CORTE, FESTIVOS).fechas == _f("2026-11-03")


def test_sin_datos_salta_y_registra():
    r = seleccionar_recientes("2026-11-24", _curvas(excluir={"2026-11-10"}), CORTE, FESTIVOS)
    assert r.fechas == _f("2026-10-27")
    assert {"fecha": "2026-11-10", "motivo": "sin datos"} in r.saltados


def test_curva_incompleta_no_cuenta():
    c = _curvas()
    c["2026-11-10"] = np.append(np.full(23, 100.0), np.nan)
    assert seleccionar_recientes("2026-11-24", c, CORTE, FESTIVOS).fechas == _f("2026-10-27")


def test_max_semanas_y_fallback_normal_con_alerta():
    # objetivo post-festivo sin ningún post-festivo en ventana corta: cae al último normal
    c = _curvas(desde=date(2026, 11, 4))  # sin 03/11 ni 02/11
    r = seleccionar_recientes("2026-11-17", c, CORTE, FESTIVOS, max_semanas=2)
    assert r.clase == "post_festivo"
    assert r.fechas == _f("2026-11-10")
    assert [a.codigo for a in r.alertas] == ["R1_SIN_MISMA_CLASE"]


def test_festivo_sin_historico_usa_homologo_del_anio_anterior():
    # objetivo 16/11/2026 (Independencia Cartagena), sin ningún festivo reciente con datos
    c = _curvas(desde=date(2026, 11, 3))
    c.update(_curvas(desde=date(2025, 11, 1), hasta=date(2025, 11, 30)))
    r = seleccionar_recientes("2026-11-16", c, CORTE, FESTIVOS, max_semanas=2)
    assert len(r.fechas) == 1 and r.fechas[0].year == 2025
    assert r.fechas[0].isoformat() in FESTIVOS
    assert {"fecha": r.fechas[0].isoformat(), "motivo": "homologo"} in r.saltados


def test_sin_nada_devuelve_vacio():
    r = seleccionar_recientes("2026-11-24", {}, CORTE, FESTIVOS)
    assert r.fechas == []


def test_saltar_puente():
    # objetivo sáb 07/11, corte 06/11: 31/10 es sábado previo al lunes festivo 02/11 (puente)
    c = _curvas()
    r_off = seleccionar_recientes("2026-11-07", c, date(2026, 11, 6), FESTIVOS)
    assert r_off.fechas == _f("2026-10-31")
    r_on = seleccionar_recientes("2026-11-07", c, date(2026, 11, 6), FESTIVOS, saltar_puente=True)
    assert r_on.fechas == _f("2026-10-24")
    assert {"fecha": "2026-10-31", "motivo": "puente"} in r_on.saltados


def test_n_invalido():
    with pytest.raises(ValueError):
        seleccionar_recientes("2026-11-24", _curvas(), CORTE, FESTIVOS, n=0)


# ---------------------------------------------------------------- R2 / R3 (definición única, E4)

def test_fecha_r2_semantica_unica():
    from src.adjustments.referencia_reciente import fecha_r2, fecha_r3

    lunes, d = date(2026, 11, 23), date(2026, 11, 25)
    # sin semana de referencia: None; con anual_si_falta (perfil Jamer) d − 364
    assert fecha_r2(d, lunes, None) is None
    assert fecha_r2(d, lunes, None, anual_si_falta=True) == date(2025, 11, 26)
    # con semana de referencia: d − desfase (= semana_ref_inicio + i)
    assert fecha_r2(d, lunes, date(2025, 11, 24)) == date(2025, 11, 26)
    assert fecha_r2(d, lunes, date(2026, 10, 26)) == date(2026, 10, 28)
    # desfase 0 solo se corrige con anual_si_falta
    assert fecha_r2(d, lunes, lunes) == d
    assert fecha_r2(d, lunes, lunes, anual_si_falta=True) == date(2025, 11, 26)
    # festivo cuyo R2 no es festivo: homólogo solo si se pasan festivos (control estándar no los pasa)
    navidad, lunes_nav = date(2026, 12, 25), date(2026, 12, 21)
    ref = lunes_nav - timedelta(days=364)
    assert fecha_r2(navidad, lunes_nav, ref) == date(2025, 12, 26)
    assert fecha_r2(navidad, lunes_nav, ref, FESTIVOS) == date(2025, 12, 25)
    assert fecha_r3(date(2026, 11, 9)) == date(2025, 11, 10)
    assert fecha_r3("2026-11-09") == date(2025, 11, 10)


def test_unico_lag_anio_en_data():
    from src.adjustments import crecimiento
    from src.adjustments.data import LAG_ANIO

    assert LAG_ANIO == 364 and crecimiento.LAG_ANIO == LAG_ANIO
