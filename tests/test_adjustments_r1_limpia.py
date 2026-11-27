"""Tests de `r1_limpia` (R1 sin Semana Santa, 24/dic–6/ene ni días atípicos): selector, params y nivel."""

from datetime import date, timedelta

import pytest

from src.adjustments import referencia_reciente as rr
from src.adjustments.data import curvas_reales, dias_semana, festivos_colombia, iso
from src.adjustments.pipeline import build_context
from src.adjustments.steps import forma, nivel
from tests.ajustes_sinteticos import curvas_planas, fechas as _f, historico_sintetico, seleccionar

FESTIVOS = set(festivos_colombia(range(2024, 2028)))


def _curvas(desde=date(2025, 10, 1), hasta=date(2027, 1, 31), factores=None, excluir=()):
    """Curvas planas de 100 MW (festivos 80); `factores` {iso: k} multiplica la energía de ese día."""
    return curvas_planas(FESTIVOS, desde, hasta, factores, excluir)


def _sel(objetivo, corte, curvas=None, n=1, **kw):
    return seleccionar(objetivo, corte, curvas if curvas is not None else _curvas(), FESTIVOS, n, **kw)


# ---------------------------------------------------------------- selector

def test_salta_semana_santa():
    # mié 8/abr/2026, corte sáb 4/abr: el ancla es el Miércoles Santo 1/abr (clase normal, en Semana Santa)
    corte = date(2026, 4, 4)
    assert _sel("2026-04-08", corte).fechas == _f("2026-04-01")
    r = _sel("2026-04-08", corte, r1_limpia=True)
    assert r.fechas == _f("2026-03-25")
    assert {"fecha": "2026-04-01", "motivo": "semana_santa"} in r.saltados
    assert r.alertas == []


def test_n2_tambien_limpia_los_extra():
    corte = date(2026, 4, 4)
    assert _sel("2026-04-08", corte, n=2).fechas == _f("2026-04-01", "2026-03-25")
    assert _sel("2026-04-08", corte, n=2, r1_limpia=True).fechas == _f("2026-03-25", "2026-03-18")
    # un extra atípico también se salta
    curvas = _curvas(factores={"2026-03-25": 0.85})
    r = _sel("2026-04-15", date(2026, 4, 11), curvas, n=2, r1_limpia=True)
    assert r.fechas == _f("2026-04-08", "2026-03-18")
    motivos = {s["fecha"]: s["motivo"] for s in r.saltados}
    assert motivos["2026-04-01"] == "semana_santa" and motivos["2026-03-25"] == "atipico"


def test_salta_temporada_fin_de_anio():
    # mié 13/ene/2027, corte sáb 9/ene: 6/ene y 30/dic están en 24/dic–6/ene
    corte = date(2027, 1, 9)
    assert _sel("2027-01-13", corte).fechas == _f("2027-01-06")
    r = _sel("2027-01-13", corte, r1_limpia=True)
    assert r.fechas == _f("2026-12-23")
    assert {"fecha": "2027-01-06", "motivo": "temporada"} in r.saltados
    assert {"fecha": "2026-12-30", "motivo": "temporada"} in r.saltados


def test_objetivo_en_temporada_acepta_r1_de_temporada():
    # mié 31/dic/2025 (dentro de 24/dic–6/ene), corte sáb 27/dic: R1 = mié 24/dic (temporada, clase normal, no
    # atípico) se acepta; tampoco cuenta como "temporada" en saltados
    corte = date(2025, 12, 27)
    r = _sel("2025-12-31", corte, r1_limpia=True)
    assert r.fechas == _f("2025-12-24") == _sel("2025-12-31", corte).fechas
    assert not any(s["motivo"] == "temporada" for s in r.saltados) and r.alertas == []


def test_objetivo_normal_de_enero_salta_temporada():
    # mié 14/ene/2026 (fuera de temporada), corte sáb 3/ene: 31/dic y 24/dic son temporada → R1 = 17/dic
    corte = date(2026, 1, 3)
    assert _sel("2026-01-14", corte).fechas == _f("2025-12-31")
    r = _sel("2026-01-14", corte, r1_limpia=True)
    assert r.fechas == _f("2025-12-17")
    assert {"fecha": "2025-12-31", "motivo": "temporada"} in r.saltados
    assert {"fecha": "2025-12-24", "motivo": "temporada"} in r.saltados


def test_objetivo_en_semana_santa_acepta_r1_de_semana_santa():
    # dom 5/abr/2026 (Pascua, en Semana Santa), corte dom 29/mar: R1 = Domingo de Ramos (en Semana Santa) se acepta
    corte = date(2026, 3, 29)
    r = _sel("2026-04-05", corte, r1_limpia=True)
    assert r.fechas == _f("2026-03-29") == _sel("2026-04-05", corte).fechas
    assert not any(s["motivo"] == "semana_santa" for s in r.saltados)
    # un domingo normal después de Semana Santa sí salta el Domingo de Ramos
    r = _sel("2026-04-12", corte, r1_limpia=True)
    assert r.fechas == _f("2026-03-22")
    assert {"fecha": "2026-03-29", "motivo": "semana_santa"} in r.saltados


@pytest.mark.parametrize("factor,umbral,saltado", [
    (0.90, rr.UMBRAL_ATIPICO, True),
    (1.07, rr.UMBRAL_ATIPICO, True),
    (1.05, rr.UMBRAL_ATIPICO, False),
    (0.95, rr.UMBRAL_ATIPICO, False),
    (1.05, 0.04, True),
])
def test_atipico(factor, umbral, saltado):
    # jue 26/nov/2026, corte sáb 14/nov: ancla jue 12/nov
    curvas = _curvas(factores={"2026-11-12": factor})
    r = _sel("2026-11-26", date(2026, 11, 14), curvas, r1_limpia=True, umbral_atipico=umbral)
    assert r.fechas == (_f("2026-11-05") if saltado else _f("2026-11-12"))
    assert ({"fecha": "2026-11-12", "motivo": "atipico"} in r.saltados) is saltado


def test_atipico_requiere_tres_previos():
    # con solo 2 jueves previos con datos no se juzga atípico
    curvas = _curvas(desde=date(2026, 10, 25), factores={"2026-11-12": 0.8})
    assert _sel("2026-11-26", date(2026, 11, 14), curvas, r1_limpia=True).fechas == _f("2026-11-12")
    curvas = _curvas(desde=date(2026, 10, 18), factores={"2026-11-12": 0.8})
    assert _sel("2026-11-26", date(2026, 11, 14), curvas, r1_limpia=True).fechas == _f("2026-11-05")


def test_atipico_usa_previos_normales():
    # la mediana de los previos resiste un previo atípico
    curvas = _curvas(factores={"2026-11-05": 0.5, "2026-11-12": 1.0})
    assert _sel("2026-11-26", date(2026, 11, 14), curvas, r1_limpia=True).fechas == _f("2026-11-12")


def test_enero_mediana_sin_temporada():
    # normales 100, 24/dic–6/ene 80, 7–10/ene 90. Mié 21/ene/2026, corte 20/ene → ancla mié 14/ene (100).
    # Previos mié: 7/ene (90), 31/dic y 24/dic (temporada, no cuentan), 17/dic, 10/dic, ... (100) → mediana 100:
    # el 14/ene NO es atípico (con la temporada en la mediana saldría 85 y se saltaría un día limpio).
    factores = {}
    d = date(2025, 12, 24)
    while d <= date(2026, 1, 10):
        factores[d.isoformat()] = 0.8 if d <= date(2026, 1, 6) else 0.9
        d += timedelta(days=1)
    curvas = _curvas(factores=factores)
    r = _sel("2026-01-21", date(2026, 1, 20), curvas, r1_limpia=True)
    assert r.fechas == _f("2026-01-14")
    assert not any(s["motivo"] == "atipico" for s in r.saltados) and r.alertas == []


def test_sin_candidato_limpio_usa_el_de_siempre_con_alerta():
    corte = date(2026, 4, 4)
    r = _sel("2026-04-08", corte, max_semanas=1, r1_limpia=True)
    assert r.fechas == _f("2026-04-01")
    assert [a.codigo for a in r.alertas] == [rr.ALERTA_R1_SIN_LIMPIA]
    a = r.alertas[0]
    assert a.nivel == "info" and a.fechas == ["2026-04-08"]
    assert a.detalle["descartados"] == [{"fecha": "2026-04-01", "motivo": "semana_santa"}]
    assert a.detalle["referencia"] == "2026-04-01"


@pytest.mark.parametrize("objetivo,corte", [("2026-11-16", date(2026, 11, 14)), ("2026-05-01", date(2026, 4, 25))])
def test_objetivo_festivo_no_cambia(objetivo, corte):
    # r1_limpia no toca la selección de un objetivo festivo (el 1/may sigue tomando el Viernes Santo)
    assert _sel(objetivo, corte, r1_limpia=True).to_dict() == _sel(objetivo, corte).to_dict()


def test_objetivo_en_semana_santa_no_se_reclasifica():
    # el Miércoles Santo como objetivo: su R1 (25/mar, limpia) es la misma de siempre
    corte = date(2026, 3, 28)
    assert _sel("2026-04-01", corte, r1_limpia=True).fechas == _sel("2026-04-01", corte).fechas == _f("2026-03-25")


def test_flag_off_identico():
    curvas = _curvas(factores={"2026-11-12": 0.8})
    for objetivo, corte in [("2026-11-26", date(2026, 11, 14)), ("2026-04-08", date(2026, 4, 4)),
                            ("2027-01-13", date(2027, 1, 9)), ("2026-11-17", date(2026, 11, 14))]:
        base = _sel(objetivo, corte, curvas, n=2)
        assert _sel(objetivo, corte, curvas, n=2, r1_limpia=False, festivo_por_tipo=False).to_dict() == base.to_dict()
        assert base.festivo is None and "festivo" not in base.to_dict()


# ---------------------------------------------------------------- params

@pytest.mark.parametrize("mal", [
    {"r1_limpia": "si"}, {"r1_limpia": None}, {"festivo_por_tipo": 1},
    {"r1_umbral_atipico": 0}, {"r1_umbral_atipico": -0.1}, {"r1_umbral_atipico": 0.6},
    {"r1_umbral_atipico": "0.06"}, {"r1_umbral_atipico": None}, {"r1_umbral_atipico": True},
    {"r1_umbral_atipico": float("nan")},
])
def test_params_invalidos(mal):
    with pytest.raises(ValueError):
        nivel.validar_params({"modo": "reciente", **mal})
    with pytest.raises(ValueError):
        forma.validar_params({"fuente": "reciente_clase", **mal})


def test_params_validos_y_persistidos():
    p = nivel.validar_params({"modo": "reciente", "r1_limpia": True, "r1_umbral_atipico": 0.5})
    assert p["r1_limpia"] is True and p["r1_umbral_atipico"] == 0.5 and p["festivo_por_tipo"] is False
    assert nivel.validar_params(p) == p
    q = forma.validar_params({"fuente": "reciente_clase", "r1_limpia": True})
    assert q["r1_limpia"] is True and q["r1_umbral_atipico"] == rr.UMBRAL_ATIPICO
    # con su default y fuera de la réplica no se persisten
    assert "r1_limpia" not in forma.validar_params({})
    assert "r1_limpia" not in nivel.validar_params({})
    assert forma.validar_params({"r1_limpia": True})["r1_limpia"] is True


# ---------------------------------------------------------------- nivel (histórico sintético)

def test_nivel_reciente_usa_r1_limpia():
    # semana 21/abr/2025, emisión 17/abr → corte 16/abr = Miércoles Santo
    hist = historico_sintetico()
    semana = date(2025, 4, 21)
    ctx = build_context("Sintetica", semana, fecha_emision=semana - timedelta(days=4), historico=hist)
    prev = {iso(d): [1000.0] * 24 for d in dias_semana(semana)}
    off = nivel.step(ctx, prev, {"modo": "reciente"})
    on = nivel.step(ctx, prev, {"modo": "reciente", "r1_limpia": True})
    assert off.diagnostics["dias"]["2025-04-23"]["r1"]["fecha"] == "2025-04-16"
    dia = on.diagnostics["dias"]["2025-04-23"]
    assert dia["r1"]["fecha"] == "2025-04-09"
    assert {"fecha": "2025-04-16", "motivo": "semana_santa"} in dia["saltados"]
    curvas = curvas_reales(ctx.historico)
    e_r1 = float(curvas["2025-04-09"].sum())
    assert dia["E_obj"] == pytest.approx(e_r1 * (1 + dia["g_usado"]))
