"""Tests de `festivo_por_tipo`: tipo de festivo, factor F = E(H)/E(N_H), nivel/forma y no-regresión con flags off."""

import re
from datetime import date, timedelta

import numpy as np
import pytest

from src.adjustments import analogos, data, perfiles
from src.adjustments import referencia_reciente as rr
from src.adjustments.data import curvas_reales, dias_semana, es_semana_santa, festivos_colombia, iso, tipo_festivo
from src.adjustments.pipeline import build_context, run_pipeline
from src.adjustments.steps import forma, nivel
from tests.ajustes_sinteticos import ProveedorPlano, curvas_planas, fechas as _f, historico_sintetico, seleccionar

FESTIVOS = set(festivos_colombia(range(2024, 2028)))
HIST = historico_sintetico()


# ---------------------------------------------------------------- calendario

@pytest.mark.parametrize("fecha,tipo", [
    ("2026-05-01", "fijo"),          # viernes
    ("2026-08-07", "fijo"),          # viernes
    ("2026-12-08", "fijo"),          # martes
    ("2025-12-25", "fijo"),          # jueves
    ("2025-07-20", "fijo"),          # domingo
    ("2026-04-02", "semana_santa"),  # Jueves Santo
    ("2026-04-03", "semana_santa"),  # Viernes Santo
    ("2025-04-17", "semana_santa"),
    ("2025-04-18", "semana_santa"),
    ("2026-11-16", "lunes"),         # festivo trasladado a lunes
    ("2024-01-01", "lunes"),         # fijo que cayó lunes
    ("2024-03-25", "lunes"),         # San José trasladado a lunes, dentro de Semana Santa
    ("2027-03-22", "lunes"),         # ídem 2027
    ("2024-03-28", "semana_santa"),  # Jueves Santo 2024
    ("2026-11-17", None),            # no festivo
    ("2026-04-05", None),            # domingo de Pascua (no es festivo en la lista)
])
def test_tipo_festivo(fecha, tipo):
    assert tipo_festivo(fecha, FESTIVOS) == tipo


def test_semana_santa_rango_y_fuente_unica():
    assert es_semana_santa(date(2026, 3, 29)) and es_semana_santa(date(2026, 4, 5))
    assert not es_semana_santa(date(2026, 3, 28)) and not es_semana_santa(date(2026, 4, 6))
    assert data.domingos_semana_santa(2025) == (date(2025, 4, 13), date(2025, 4, 20))
    assert analogos.es_semana_santa is data.es_semana_santa


# ---------------------------------------------------------------- selector

def _curvas(desde=date(2024, 10, 1), hasta=date(2026, 12, 31), factores=None, excluir=(), festivos=FESTIVOS):
    """Curvas planas: normal 100 MW, festivo 80 MW, Viernes/Jueves Santo 70 MW; `factores` {iso: k}."""
    return curvas_planas(festivos, desde, hasta, factores, excluir, mw_semana_santa=70.0)


def _sel(objetivo, corte, curvas=None, n=1, festivos=FESTIVOS, **kw):
    return seleccionar(objetivo, corte, curvas if curvas is not None else _curvas(), festivos, n, **kw)


def _anios(info):
    return [x["anio"] for x in info["anios_probados"]]


def _descartes(info):
    return [x["descarte"] for x in info["anios_probados"]]


def test_primero_de_mayo_usa_normal_y_factor_del_homologo():
    corte = date(2026, 4, 25)
    curvas = _curvas(factores={"2025-05-01": 0.9})  # H = 1/may/2025 (jue): 72 MW
    # sin el flag el 1/may (vie) toma el Viernes Santo (festivo viernes más reciente)
    assert _sel("2026-05-01", corte, curvas).fechas == _f("2026-04-03")
    r = _sel("2026-05-01", corte, curvas, n=2, festivo_por_tipo=True)
    assert r.fechas == _f("2026-04-24", "2026-04-17")  # viernes normales recientes
    assert r.clase == "festivo" and r.alertas == []
    assert r.festivo["tipo_festivo"] == "fijo"
    assert r.festivo["homologo"] == "2025-05-01"
    assert r.festivo["normal_homologo"] == "2025-04-24"  # jueves anterior (el primero que se prueba)
    assert r.festivo["anios_probados"] == [{"anio": 2025, "homologo": "2025-05-01", "descarte": None}]
    assert r.factor_festivo == pytest.approx(72.0 / 100.0)
    assert r.festivo["E_homologo"] == pytest.approx(72.0 * 24)
    assert r.to_dict()["festivo"]["factor"] == pytest.approx(0.72)


def test_jueves_santo_usa_homologo_de_semana_santa():
    r = _sel("2026-04-02", date(2026, 3, 28), festivo_por_tipo=True)
    assert r.fechas == _f("2026-03-26")
    assert r.festivo["tipo_festivo"] == "semana_santa"
    assert r.festivo["homologo"] == "2025-04-17" and r.festivo["normal_homologo"] == "2025-04-10"
    assert r.factor_festivo == pytest.approx(0.70)


def test_normal_homologo_orden_antes_primero():
    # orden −7, −14, +7, +14. Sin datos el jueves anterior (−7) → −14 es el 17/abr/2025, Jueves Santo (festivo y
    # Semana Santa: no sirve) → +7
    r = _sel("2026-05-01", date(2026, 4, 25), _curvas(excluir={"2025-04-24"}), festivo_por_tipo=True)
    assert r.festivo["normal_homologo"] == "2025-05-08"
    # sin −7 ni +7 → queda +14
    r = _sel("2026-05-01", date(2026, 4, 25), _curvas(excluir={"2025-04-24", "2025-05-08"}), festivo_por_tipo=True)
    assert r.festivo["normal_homologo"] == "2025-05-15"


def test_primero_de_enero_normal_homologo_antes_y_fuera_de_temporada():
    # jue 1/ene/2026: H = mié 1/ene/2025 (laboral, como el objetivo). −7 = 25/dic/2024 (festivo) → −14 = 18/dic/2024
    # (antes que +7 = 8/ene/2025)
    corte = date(2025, 12, 27)
    r = _sel("2026-01-01", corte, festivo_por_tipo=True)
    assert r.festivo["homologo"] == "2025-01-01" and r.festivo["normal_homologo"] == "2024-12-18"
    assert r.fechas == _f("2025-12-18")  # R1 = jueves normal reciente (25/dic es festivo)
    # si el 25/dic no fuera festivo seguiría sin servir: está en 24/dic–6/ene (temporada)
    sin_navidad = FESTIVOS - {"2024-12-25"}
    r = _sel("2026-01-01", corte, _curvas(festivos=sin_navidad), festivos=sin_navidad, festivo_por_tipo=True)
    assert r.festivo["normal_homologo"] == "2024-12-18"


FESTIVOS_LARGO = set(festivos_colombia(range(2016, 2028)))


def test_homologo_del_mismo_tipo_de_dia_mas_reciente():
    # mar 8/dic/2020: 8/dic/2019 domingo y 8/dic/2018 sábado no sirven → 8/dic/2017 (viernes, laboral)
    curvas = _curvas(desde=date(2017, 9, 1), hasta=date(2020, 12, 31), festivos=FESTIVOS_LARGO)
    r = _sel("2020-12-08", date(2020, 12, 5), curvas, festivos=FESTIVOS_LARGO, festivo_por_tipo=True)
    assert r.festivo["homologo"] == "2017-12-08" and _anios(r.festivo) == [2019, 2018, 2017]
    assert _descartes(r.festivo) == [rr.DESCARTE_OTRO_TIPO_DIA, rr.DESCARTE_OTRO_TIPO_DIA, None]
    assert r.factor_festivo == pytest.approx(0.80) and r.alertas == []
    assert r.fechas == _f("2020-12-01")


def test_sin_homologo_del_mismo_tipo_de_dia_fallback_con_alerta():
    # dom 20/jul/2025: 20/jul de 2024 (sáb), 2023 (jue) y 2022 (mié) → ninguno es domingo
    curvas = _curvas(desde=date(2022, 1, 1), hasta=date(2025, 12, 31), festivos=FESTIVOS_LARGO)
    corte = date(2025, 7, 15)
    off = _sel("2025-07-20", corte, curvas, festivos=FESTIVOS_LARGO)
    on = _sel("2025-07-20", corte, curvas, festivos=FESTIVOS_LARGO, festivo_por_tipo=True)
    assert on.fechas == off.fechas and on.saltados == off.saltados and on.factor_festivo is None
    assert on.festivo["homologo"] is None and _anios(on.festivo) == [2024, 2023, 2022]
    assert [a.codigo for a in on.alertas] == [rr.ALERTA_FESTIVO_SIN_HOMOLOGO]
    det = on.alertas[0].detalle
    assert det["motivo"] == rr.MOTIVO_SIN_HOMOLOGO_MISMO_TIPO == "sin_homologo_mismo_tipo_dia"
    assert det["anios_probados"] == on.festivo["anios_probados"]
    assert _descartes(det) == [rr.DESCARTE_OTRO_TIPO_DIA] * 3


def test_festivo_entre_semana_no_toma_homologo_lunes_puente():
    # mar 8/dic/2026: 8/dic/2025 es lunes (puente) → no; 2024 domingo → no; 2023 viernes → sí
    curvas = _curvas(desde=date(2023, 10, 1), hasta=date(2026, 12, 31), festivos=FESTIVOS_LARGO)
    r = _sel("2026-12-08", date(2026, 12, 5), curvas, festivos=FESTIVOS_LARGO, festivo_por_tipo=True)
    assert r.festivo["homologo"] == "2023-12-08" and r.factor_festivo == pytest.approx(0.80)
    assert r.festivo["anios_probados"] == [
        {"anio": 2025, "homologo": "2025-12-08", "descarte": rr.DESCARTE_LUNES_PUENTE},
        {"anio": 2024, "homologo": "2024-12-08", "descarte": rr.DESCARTE_OTRO_TIPO_DIA},
        {"anio": 2023, "homologo": "2023-12-08", "descarte": None},
    ]


def test_primero_de_mayo_miercoles_no_toma_lunes():
    # mié 1/may/2024: 2023 lunes (puente), 2022 domingo, 2021 sábado → sin homólogo usable
    curvas = _curvas(desde=date(2021, 1, 1), hasta=date(2024, 12, 31), festivos=FESTIVOS_LARGO)
    r = _sel("2024-05-01", date(2024, 4, 27), curvas, festivos=FESTIVOS_LARGO, festivo_por_tipo=True)
    assert r.festivo["homologo"] is None and r.factor_festivo is None
    assert _descartes(r.festivo) == [rr.DESCARTE_LUNES_PUENTE, rr.DESCARTE_OTRO_TIPO_DIA, rr.DESCARTE_OTRO_TIPO_DIA]
    assert [a.detalle["motivo"] for a in r.alertas] == [rr.MOTIVO_SIN_HOMOLOGO_MISMO_TIPO]


def test_solo_homologo_lunes_puente_motivo_propio():
    # mar 8/dic/2026 con datos solo desde oct/2025: el único homólogo con datos es el lunes 8/dic/2025
    curvas = _curvas(desde=date(2025, 10, 1), hasta=date(2026, 12, 31), festivos=FESTIVOS_LARGO)
    r = _sel("2026-12-08", date(2026, 12, 5), curvas, festivos=FESTIVOS_LARGO, festivo_por_tipo=True)
    assert r.factor_festivo is None
    assert _descartes(r.festivo) == [rr.DESCARTE_LUNES_PUENTE, rr.DESCARTE_SIN_DATOS, rr.DESCARTE_SIN_DATOS]
    a = r.alertas[0]
    assert a.detalle["motivo"] == rr.MOTIVO_HOMOLOGO_LUNES_PUENTE
    assert rr.TEXTO_MOTIVO_SIN_HOMOLOGO[rr.MOTIVO_HOMOLOGO_LUNES_PUENTE] in a.mensaje


def test_festivo_lunes_no_busca_homologo():
    # lun 1/ene/2024 es tipo "lunes" → la regla no aplica (selección de siempre, sin homólogo)
    assert _sel("2024-01-01", date(2023, 12, 30), _curvas(desde=date(2023, 1, 1), festivos=FESTIVOS_LARGO),
                festivos=FESTIVOS_LARGO, festivo_por_tipo=True).festivo["anios_probados"] == []


def test_motivos_son_codigos_con_texto_completo():
    codigos = {v for k, v in vars(rr).items() if k.startswith("MOTIVO_")}
    assert codigos == set(rr.TEXTO_MOTIVO_SIN_HOMOLOGO)
    assert all(re.fullmatch(r"[a-z_]+", c) for c in codigos)
    assert set(rr.info_festivo("fijo")) == {"tipo_festivo", "homologo", "normal_homologo", "E_homologo",
                                           "E_normal_homologo", "factor", "anios_probados"}


def test_homologo_con_energia_cero_no_da_factor():
    # E(H) = 0 → H no sirve (su forma no sería usable); 2024 sin datos → sin factor
    r = _sel("2026-05-01", date(2026, 4, 25), _curvas(factores={"2025-05-01": 0.0}), festivo_por_tipo=True)
    assert r.factor_festivo is None and r.festivo["homologo"] is None
    assert [a.detalle["motivo"] for a in r.alertas] == [rr.MOTIVO_SIN_HOMOLOGO]


def test_festivo_lunes_no_cambia():
    corte = date(2026, 11, 14)
    off = _sel("2026-11-16", corte, n=2)
    on = _sel("2026-11-16", corte, n=2, festivo_por_tipo=True)
    assert on.fechas == off.fechas and on.saltados == off.saltados and on.alertas == off.alertas
    assert on.festivo["tipo_festivo"] == "lunes" and on.factor_festivo is None


@pytest.mark.parametrize("excluir,motivo", [
    ({"2025-05-01"}, "sin_homologo"),
    ({"2025-04-24", "2025-05-08", "2025-05-15"}, "sin_normal_homologo"),
])
def test_sin_homologo_comportamiento_actual_con_alerta(excluir, motivo):
    corte = date(2026, 4, 25)
    curvas = _curvas(excluir=excluir)
    off = _sel("2026-05-01", corte, curvas)
    on = _sel("2026-05-01", corte, curvas, festivo_por_tipo=True)
    assert on.fechas == off.fechas == _f("2026-04-03") and on.saltados == off.saltados
    assert on.factor_festivo is None
    assert [a.codigo for a in on.alertas] == [rr.ALERTA_FESTIVO_SIN_HOMOLOGO]
    assert on.alertas[0].nivel == "info" and on.alertas[0].detalle["motivo"] == motivo


def test_sin_normal_reciente_comportamiento_actual_con_alerta():
    # ningún viernes normal con datos en la ventana
    corte = date(2026, 4, 25)
    viernes = {(date(2026, 4, 24) - timedelta(days=7 * i)).isoformat() for i in range(12)} - {"2026-04-03"}
    on = _sel("2026-05-01", corte, _curvas(excluir=viernes), festivo_por_tipo=True)
    assert on.fechas == _f("2026-04-03") and on.factor_festivo is None
    assert on.festivo["homologo"] == "2025-05-01"  # el homólogo sí estaba; faltó la R1 normal
    assert [a.detalle["motivo"] for a in on.alertas] == ["sin_normal_reciente"]


def test_con_r1_limpia_el_normal_del_festivo_tambien_se_limpia():
    curvas = _curvas(factores={"2026-04-24": 0.85})
    corte = date(2026, 4, 25)
    assert _sel("2026-05-01", corte, curvas, festivo_por_tipo=True).fechas == _f("2026-04-24")
    r = _sel("2026-05-01", corte, curvas, festivo_por_tipo=True, r1_limpia=True)
    assert r.fechas == _f("2026-04-17")
    assert {"fecha": "2026-04-24", "motivo": "atipico"} in r.saltados


# ---------------------------------------------------------------- nivel y forma (histórico sintético)

SEMANA_7AGO = date(2025, 8, 4)  # jue 7/ago/2025 festivo fijo; emisión 31/jul → corte 30/jul


def _ctx(semana=SEMANA_7AGO, historico=HIST):
    return build_context("Sintetica", semana, fecha_emision=semana - timedelta(days=4), historico=historico)


def _prev(ctx, mw=1000.0):
    return {iso(d): [mw] * 24 for d in dias_semana(ctx.semana_inicio)}


def test_nivel_aplica_factor_festivo():
    ctx = _ctx()
    curvas = curvas_reales(ctx.historico)
    off = nivel.step(ctx, _prev(ctx), {"modo": "reciente"})
    on = nivel.step(ctx, _prev(ctx), {"modo": "reciente", "festivo_por_tipo": True})
    dia = on.diagnostics["dias"]["2025-08-07"]
    assert dia["r1"]["fecha"] == "2025-07-24" and dia["clase"] == "festivo"
    assert dia["tipo_festivo"] == "fijo" and dia["homologo"] == "2024-08-07"
    assert dia["normal_homologo"] == "2024-07-31"
    f_esp = curvas["2024-08-07"].sum() / curvas["2024-07-31"].sum()
    assert dia["factor_festivo"] == pytest.approx(f_esp)
    assert dia["E_obj"] == pytest.approx(curvas["2025-07-24"].sum() * (1 + dia["g_usado"]) * f_esp)
    assert dia["lag_dias"] == 14 and dia["regla"] == "reciente_factor_festivo"
    assert sum(on.curves["2025-08-07"]) == pytest.approx(dia["E"])
    # el resto de días, idéntico; sin el flag no hay claves nuevas
    for f in on.curves:
        if f != "2025-08-07":
            assert on.curves[f] == off.curves[f]
    assert all("tipo_festivo" not in v for v in off.diagnostics["dias"].values())
    # sintético: festivo ≈ 17/24 de un laboral → el nivel festivo de ambos caminos es comparable
    assert dia["E"] == pytest.approx(off.diagnostics["dias"]["2025-08-07"]["E"], rel=0.05)


def test_forma_usa_forma_del_homologo():
    ctx = _ctx()
    curvas = curvas_reales(ctx.historico)
    res = forma.step(ctx, _prev(ctx), {"fuente": "reciente_clase", "n": 1, "festivo_por_tipo": True})
    dia = res.diagnostics["dias"]["2025-08-07"]
    assert dia["fuente"] == "festivo_homologo_tipo" and dia["fechas_fuente"] == ["2024-08-07"]
    assert dia["factor_festivo"] is not None and dia["tipo_festivo"] == "fijo"
    esperado = curvas["2024-08-07"] / curvas["2024-08-07"].sum() * 24000.0
    np.testing.assert_allclose(res.curves["2025-08-07"], esperado)


def _correr(semana, params, historico=HIST):
    ctx = _ctx(semana, historico)
    return run_pipeline(ctx, ProveedorPlano(), params)


def _con_flags(params, **flags):
    out = {k: dict(v) for k, v in params.items()}
    out["nivel"].update(flags)
    out["forma"].update(flags)
    return out


def test_pipeline_perfil_jamer_con_festivo_por_tipo():
    params = _con_flags(perfiles.expandir("jamer", None), festivo_por_tipo=True)
    recs = {r.paso: r for r in _correr(SEMANA_7AGO, params)}
    dn = recs["nivel"].result.diagnostics["dias"]["2025-08-07"]
    df = recs["forma"].result.diagnostics["dias"]["2025-08-07"]
    assert df["fuente"] == "festivo_homologo_tipo" and df["delta_plano_mw"] is not None
    assert sum(recs["forma"].result.curves["2025-08-07"]) == pytest.approx(dn["E"])
    assert recs["nivel"].params["festivo_por_tipo"] is True and recs["forma"].params["festivo_por_tipo"] is True


def test_pipeline_alerta_sin_homologo_una_sola_vez():
    # 7/ago/2025 (jue): sin 2024 (mié) ni 2023 (lun) queda 2022 (domingo) → sin homólogo del mismo tipo de día
    hist = HIST[~HIST["FECHA"].isin(["2024-08-07", "2023-08-07"])].reset_index(drop=True)
    params = _con_flags(perfiles.expandir("jamer", None), festivo_por_tipo=True)
    recs = {r.paso: r for r in _correr(SEMANA_7AGO, params, hist)}
    cod = rr.ALERTA_FESTIVO_SIN_HOMOLOGO
    assert any(a.codigo == cod and "2025-08-07" in a.fechas for a in recs["nivel"].result.alerts)
    assert not any(a.codigo == cod for a in recs["forma"].result.alerts)


# ---------------------------------------------------------------- no-regresión con flags off

# Energía y pico diarios de `forma` del perfil jamer en la semana 4/ago/2025 (sintético), medidos con el código
# anterior a festivo_por_tipo / r1_limpia.
SNAPSHOT_7AGO = {
    "2025-08-04": (28315.834096, 1433.559021), "2025-08-05": (28318.12729, 1433.654571),
    "2025-08-06": (28320.420671, 1433.750128), "2025-08-07": (19867.146637, 1156.105629),
    "2025-08-08": (28325.007988, 1433.941267), "2025-08-09": (24786.389185, 1286.527417),
    "2025-08-10": (21247.197037, 1182.140586),
}


def _resultado(recs):
    return [(r.paso, r.result.curves, r.result.diagnostics, [a.to_dict() for a in r.result.alerts]) for r in recs]


_FLAGS = ("festivo_por_tipo", "r1_limpia")
# Apaga explícitamente las flags vía params del usuario (el perfil `jamer` trae festivo_por_tipo=True).
_FLAGS_OFF = {"nivel": {k: False for k in _FLAGS}, "forma": {k: False for k in _FLAGS}}


def _sin_claves(perfil):
    """Perfil expandido sin las claves de las flags (equivale al perfil anterior a festivos_r1)."""
    out = perfiles.expandir(perfil, None)
    for paso in ("nivel", "forma"):
        for k in _FLAGS:
            out[paso].pop(k, None)
    return out


@pytest.mark.parametrize("perfil", perfiles.PERFILES)
@pytest.mark.parametrize("semana", [date(2025, 8, 4), date(2025, 4, 14), date(2025, 12, 22), date(2025, 9, 8)])
def test_flags_off_identico_a_sin_claves(perfil, semana):
    sin = _resultado(_correr(semana, _sin_claves(perfil)))
    falso = _resultado(_correr(semana, perfiles.expandir(perfil, _FLAGS_OFF)))
    assert sin == falso


def test_flags_off_igual_al_snapshot():
    recs = {r.paso: r for r in _correr(SEMANA_7AGO, perfiles.expandir("jamer", _FLAGS_OFF))}
    curvas = recs["forma"].result.curves
    for f, (energia, pico) in SNAPSHOT_7AGO.items():
        assert sum(curvas[f]) == pytest.approx(energia, abs=1e-4)
        assert max(curvas[f]) == pytest.approx(pico, abs=1e-4)
