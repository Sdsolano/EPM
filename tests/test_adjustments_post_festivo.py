"""Tests de `post_festivo_factor`: F_pf = mediana de E(pf)/E(N), fuentes homólogo/recientes/combinado, nivel/forma,
alerta POST_FESTIVO_SIN_FACTOR y no-regresión con el flag apagado."""

import re
from datetime import date, timedelta

import numpy as np
import pytest

from src.adjustments import perfiles
from src.adjustments import referencia_reciente as rr
from src.adjustments.data import CLASE_POST_FESTIVO, clase_dia, curvas_reales, dias_semana, festivos_colombia, iso
from src.adjustments.pipeline import build_context, run_pipeline
from src.adjustments.steps import forma, nivel
from tests.ajustes_sinteticos import ProveedorPlano, curvas_planas, fechas as _f, historico_sintetico, seleccionar

FESTIVOS = set(festivos_colombia(range(2022, 2028)))
HIST = historico_sintetico()

# Martes 3/nov/2026: post-festivo de Todos los Santos (lunes 2/nov). Corte martes 27/oct (caso del PLAN).
OBJ = "2026-11-03"
CORTE = date(2026, 10, 27)
HOMOLOGOS = ("2025-11-04", "2024-11-05", "2023-11-07")  # día siguiente al homólogo de Todos los Santos (martes)
RECIENTES = ("2026-10-13", "2026-08-18", "2026-07-21", "2026-07-14")  # 4 martes post-festivo más recientes <= corte


def _curvas(factores=None, excluir=(), desde=date(2023, 1, 1), hasta=date(2026, 12, 31)):
    """Curvas planas (normal 100 MW, festivo 80 MW; los post-festivos son 100 salvo `factores`)."""
    return curvas_planas(FESTIVOS, desde, hasta, factores, excluir)


def _sel(objetivo=OBJ, corte=CORTE, curvas=None, n=1, **kw):
    return seleccionar(objetivo, corte, curvas if curvas is not None else _curvas(), FESTIVOS, n, **kw)


def _pf(objetivo=OBJ, corte=CORTE, curvas=None, fuente="combinado", **kw):
    return _sel(objetivo, corte, curvas, post_festivo_factor=True, post_festivo_fuente=fuente, **kw)


def _pares(r):
    return [x["post_festivo"] for x in r.post_festivo["pares"]]


# ---------------------------------------------------------------- clase, constantes y params

def test_clase_y_constantes():
    assert clase_dia(OBJ, FESTIVOS) == CLASE_POST_FESTIVO
    assert (rr.ANIOS_POST_FESTIVO, rr.RECIENTES_POST_FESTIVO, rr.MIN_RATIOS_POST_FESTIVO) == (3, 4, 3)
    assert rr.DEFAULTS_R1["post_festivo_factor"] is False and rr.DEFAULTS_R1["post_festivo_fuente"] == "combinado"
    assert set(rr.info_post_festivo("homologo")) == {"fuente", "pares", "descartados", "factor"}


def test_motivos_post_festivo_son_codigos_con_texto():
    codigos = {v for k, v in vars(rr).items() if k.startswith("MOTIVO_PF_")}
    assert codigos == set(rr.TEXTO_MOTIVO_POST_FESTIVO)
    assert all(re.fullmatch(r"[a-z_]+", c) for c in codigos)


@pytest.mark.parametrize("paso", ["nivel", "forma"])
@pytest.mark.parametrize("mal", [{"post_festivo_factor": "si"}, {"post_festivo_factor": None},
                                 {"post_festivo_fuente": "otra"}, {"post_festivo_fuente": 1}])
def test_params_invalidos(paso, mal):
    with pytest.raises(ValueError):
        if paso == "nivel":
            nivel.validar_params({"modo": "reciente", **mal})
        else:
            forma.validar_params({"fuente": "reciente_clase", **mal})


def test_params_validos_se_persisten_y_kwargs():
    p = nivel.validar_params({"modo": "reciente", "post_festivo_factor": True, "post_festivo_fuente": "homologo"})
    assert p["post_festivo_factor"] is True and p["post_festivo_fuente"] == "homologo"
    q = forma.validar_params({"fuente": "reciente_clase", "post_festivo_factor": True})
    assert q["post_festivo_factor"] is True and q["post_festivo_fuente"] == "combinado"
    k = rr.kwargs_r1(p)
    assert k["post_festivo_factor"] is True and k["post_festivo_fuente"] == "homologo"
    # con otra fuente de forma / modo alpha y valores default no se persisten
    assert "post_festivo_factor" not in forma.validar_params({"fuente": "promedio_n"})
    assert "post_festivo_factor" not in nivel.validar_params({})


# ---------------------------------------------------------------- fuentes y mediana

def test_fuente_homologo():
    curvas = _curvas(factores={"2025-11-04": 0.99, "2024-11-05": 0.95, "2023-11-07": 1.03, "2026-10-13": 0.80})
    r = _pf(curvas=curvas, fuente="homologo")
    assert r.fechas == _f("2026-10-27")  # R1 = martes normal más reciente
    assert r.clase == CLASE_POST_FESTIVO and r.alertas == []
    assert _pares(r) == list(HOMOLOGOS)
    assert [x["normal"] for x in r.post_festivo["pares"]] == ["2025-10-28", "2024-10-29", "2023-10-31"]
    assert {x["origen"] for x in r.post_festivo["pares"]} == {"homologo"}
    assert r.factor_post_festivo == pytest.approx(0.99)  # mediana de 0.99, 0.95, 1.03 (el 13/oct no cuenta)
    assert r.factor_festivo is None


def test_fuente_recientes():
    curvas = _curvas(factores={"2026-10-13": 0.94, "2026-08-18": 0.97, "2026-07-21": 1.0, "2026-07-14": 0.98,
                               "2025-11-04": 0.5})
    r = _pf(curvas=curvas, fuente="recientes")
    assert _pares(r) == list(RECIENTES)
    assert [x["normal"] for x in r.post_festivo["pares"]] == ["2026-10-06", "2026-08-11", "2026-07-07", "2026-07-07"]
    assert r.factor_post_festivo == pytest.approx(np.median([0.94, 0.97, 1.0, 0.98]))


def test_fuente_combinado_es_la_union_y_mediana():
    fac = {"2026-10-13": 0.94, "2026-08-18": 0.97, "2026-07-21": 1.0, "2026-07-14": 0.98,
           "2025-11-04": 0.99, "2024-11-05": 0.95, "2023-11-07": 1.03}
    r = _pf(curvas=_curvas(factores=fac))
    assert _pares(r) == [*RECIENTES, *HOMOLOGOS]  # del más reciente al más antiguo
    assert r.post_festivo["fuente"] == "combinado"
    assert r.factor_post_festivo == pytest.approx(np.median(list(fac.values())))
    assert [x["ratio"] for x in r.post_festivo["pares"]] == pytest.approx([fac[x] for x in _pares(r)])


def test_combinado_sin_fechas_repetidas():
    # sin datos de los martes post-festivo de 2026, los recientes llegan a 2025 e incluyen el homólogo 2025-11-04
    pf_2026 = {iso(date(2026, 1, 6) + timedelta(days=7 * k)) for k in range(43)}
    pf_2026 = {x for x in pf_2026 if clase_dia(x, FESTIVOS) == CLASE_POST_FESTIVO}
    r = _pf(curvas=_curvas(excluir=pf_2026))
    pares = _pares(r)
    assert len(pares) == len(set(pares))
    assert "2025-11-04" in pares
    origen = {x["post_festivo"]: x["origen"] for x in r.post_festivo["pares"]}
    assert origen["2025-11-04"] == "homologo+reciente"
    assert sum(1 for o in origen.values() if "reciente" in o) == rr.RECIENTES_POST_FESTIVO


# ---------------------------------------------------------------- mínimo 3 ratios, mismo día de semana, corte

def test_menos_de_tres_ratios_usa_seleccion_de_siempre_con_alerta():
    curvas = _curvas(excluir={"2024-11-05", "2023-11-07"})
    off = _sel(curvas=curvas)
    r = _pf(curvas=curvas, fuente="homologo")
    assert r.fechas == off.fechas == _f("2026-10-13")  # post-festivo reciente (comportamiento de siempre)
    assert r.factor_post_festivo is None and r.post_festivo["factor"] is None
    assert [a.codigo for a in r.alertas] == [rr.ALERTA_POST_FESTIVO_SIN_FACTOR]
    det = r.alertas[0].detalle
    assert det["motivo"] == rr.MOTIVO_PF_POCOS_RATIOS and det["n_ratios"] == 1 and det["minimo_ratios"] == 3
    assert {x["descarte"] for x in det["descartados"]} == {rr.DESCARTE_SIN_DATOS}
    assert r.alertas[0].nivel == "info" and r.alertas[0].fechas == [OBJ]


def test_mismo_dia_de_semana():
    # miércoles 9/dic/2026 (8/dic martes): los homólogos caen martes 2025, lunes 2024 y sábado 2023 → descartados
    r = _pf("2026-12-09", date(2026, 12, 5), fuente="homologo")
    assert r.post_festivo["pares"] == []
    assert [x["descarte"] for x in r.post_festivo["descartados"]] == [rr.DESCARTE_OTRO_DIA_SEMANA] * 3
    assert r.alertas[0].codigo == rr.ALERTA_POST_FESTIVO_SIN_FACTOR
    # con cualquier fuente, todos los pf usados tienen el mismo día de semana que el objetivo
    for fuente in rr.FUENTES_POST_FESTIVO:
        r = _pf(fuente=fuente)
        assert all(date.fromisoformat(x).weekday() == 1 for x in _pares(r))


def test_sin_fuga_posterior_al_corte():
    corte = date(2026, 10, 17)  # sábado: el +7 / +14 del 13/oct cae después del corte
    curvas = _curvas(excluir={"2026-10-06", "2026-09-29"},
                     factores={"2026-10-20": 5.0, "2026-10-27": 5.0, OBJ: 5.0})
    r = _pf(OBJ, corte, curvas)
    usados = [d for x in r.post_festivo["pares"] for d in (x["post_festivo"], x["normal"])]
    assert usados and all(date.fromisoformat(d) <= corte for d in usados)
    assert {"fecha": "2026-10-13", "origen": "reciente", "anio": None, "descarte": rr.DESCARTE_SIN_NORMAL} in \
        r.post_festivo["descartados"]
    assert all(f <= corte for f in r.fechas)
    assert all(x["ratio"] == pytest.approx(1.0) for x in r.post_festivo["pares"])  # nada de los días ×5


def test_r1_limpia_aplica_a_la_r1_normal():
    curvas = _curvas(factores={"2026-10-27": 1.2})  # martes normal atípico
    assert _pf(curvas=curvas).fechas == _f("2026-10-27")
    r = _pf(curvas=curvas, r1_limpia=True)
    assert r.fechas == _f("2026-10-20")
    assert {"fecha": "2026-10-27", "motivo": "atipico"} in r.saltados
    assert r.factor_post_festivo == pytest.approx(1.0)


def test_objetivo_no_post_festivo_no_cambia():
    for objetivo, corte in (("2026-11-10", CORTE), ("2026-11-02", CORTE), ("2026-12-08", date(2026, 12, 5))):
        off = _sel(objetivo, corte)
        on = _pf(objetivo, corte)
        assert on.to_dict() == off.to_dict() and on.post_festivo is None


def test_flag_off_identico_a_sin_kwargs():
    for objetivo in (OBJ, "2026-11-10", "2026-11-02"):
        base = _sel(objetivo, n=2)
        for fuente in rr.FUENTES_POST_FESTIVO:
            r = _sel(objetivo, n=2, post_festivo_factor=False, post_festivo_fuente=fuente)
            assert r.to_dict() == base.to_dict()
            assert "post_festivo" not in r.to_dict()


# ---------------------------------------------------------------- nivel y forma (histórico sintético)

SEMANA_NOV = date(2025, 11, 3)  # lun 3/nov/2025 festivo (Todos los Santos) → mar 4/nov post-festivo
PF = "2025-11-04"
_PF_ON = {"post_festivo_factor": True, "post_festivo_fuente": "combinado"}


def _ctx(semana=SEMANA_NOV, historico=HIST):
    return build_context("Sintetica", semana, fecha_emision=semana - timedelta(days=4), historico=historico)


def _prev(ctx, mw=1000.0):
    return {iso(d): [mw] * 24 for d in dias_semana(ctx.semana_inicio)}


def test_nivel_aplica_factor_post_festivo():
    ctx = _ctx()
    curvas = curvas_reales(ctx.historico)
    off = nivel.step(ctx, _prev(ctx), {"modo": "reciente"})
    on = nivel.step(ctx, _prev(ctx), {"modo": "reciente", **_PF_ON})
    dia = on.diagnostics["dias"][PF]
    assert dia["clase"] == CLASE_POST_FESTIVO and dia["r1"]["fecha"] == "2025-10-28"  # martes normal
    assert dia["regla"] == "reciente_factor_post_festivo" and dia["lag_dias"] == 7
    pares = dia["post_festivo_pares"]
    assert len(pares) >= 3 and dia["post_festivo_fuente"] == "combinado"
    ratios = [curvas[x["post_festivo"]].sum() / curvas[x["normal"]].sum() for x in pares]
    assert dia["post_festivo_ratios"] == pytest.approx(ratios)
    assert dia["factor_post_festivo"] == pytest.approx(float(np.median(ratios)))
    e_obj = curvas["2025-10-28"].sum() * (1 + dia["g_usado"]) * dia["factor_post_festivo"]
    assert dia["E_obj"] == pytest.approx(e_obj)
    assert sum(on.curves[PF]) == pytest.approx(dia["E"])
    # el resto de días, idéntico; sin el flag no hay claves nuevas
    for f in on.curves:
        if f != PF:
            assert on.curves[f] == off.curves[f]
    assert all("factor_post_festivo" not in v for v in off.diagnostics["dias"].values())


def test_forma_usa_forma_de_r1_normal():
    ctx = _ctx()
    curvas = curvas_reales(ctx.historico)
    res = forma.step(ctx, _prev(ctx), {"fuente": "reciente_clase", "n": 1, **_PF_ON})
    dia = res.diagnostics["dias"][PF]
    assert dia["fuente"] == "reciente_clase" and dia["fechas_fuente"] == ["2025-10-28"]
    assert dia["factor_post_festivo"] is not None and dia["clase"] == CLASE_POST_FESTIVO
    esperado = curvas["2025-10-28"] / curvas["2025-10-28"].sum() * 24000.0
    np.testing.assert_allclose(res.curves[PF], esperado)


def _correr(semana, params, historico=HIST):
    return run_pipeline(_ctx(semana, historico), ProveedorPlano(), params)


def _con(params, **flags):
    out = {k: dict(v) for k, v in params.items()}
    out["nivel"].update(flags)
    out["forma"].update(flags)
    return out


def test_pipeline_perfil_jamer_nivel_y_forma_consistentes():
    recs = {r.paso: r for r in _correr(SEMANA_NOV, _con(perfiles.expandir("jamer", None), **_PF_ON))}
    dn = recs["nivel"].result.diagnostics["dias"][PF]
    df = recs["forma"].result.diagnostics["dias"][PF]
    assert dn["post_festivo_pares"] == df["post_festivo_pares"]
    assert dn["factor_post_festivo"] == df["factor_post_festivo"]
    assert df["fechas_fuente"] == [dn["r1"]["fecha"]] and df["delta_plano_mw"] is not None
    assert sum(recs["forma"].result.curves[PF]) == pytest.approx(dn["E"])
    assert recs["nivel"].params["post_festivo_factor"] is True and recs["forma"].params["post_festivo_factor"] is True


def test_pipeline_alerta_sin_factor_una_sola_vez():
    hist = HIST[~HIST["FECHA"].isin(["2024-11-05", "2023-11-07"])].reset_index(drop=True)
    params = _con(perfiles.expandir("jamer", None), post_festivo_factor=True, post_festivo_fuente="homologo")
    recs = {r.paso: r for r in _correr(SEMANA_NOV, params, hist)}
    cod = rr.ALERTA_POST_FESTIVO_SIN_FACTOR
    alerta = next(a for a in recs["nivel"].result.alerts if a.codigo == cod)
    assert PF in alerta.fechas and alerta.detalle["por_fecha"][PF]["motivo"] == rr.MOTIVO_PF_POCOS_RATIOS
    assert not any(a.codigo == cod for a in recs["forma"].result.alerts)
    assert recs["nivel"].result.diagnostics["dias"][PF]["regla"] == "reciente"


# ---------------------------------------------------------------- no-regresión con el flag apagado

def _resultado(recs):
    return [(r.paso, r.result.curves, r.result.diagnostics, [a.to_dict() for a in r.result.alerts]) for r in recs]


def _sin_claves(perfil):
    out = perfiles.expandir(perfil, None)
    for paso in ("nivel", "forma"):
        for k in ("post_festivo_factor", "post_festivo_fuente"):
            out[paso].pop(k, None)
    return out


@pytest.mark.parametrize("perfil", perfiles.PERFILES)
@pytest.mark.parametrize("semana", [SEMANA_NOV, date(2025, 8, 4), date(2025, 7, 21), date(2025, 12, 22)])
def test_flag_off_identico_a_sin_claves(perfil, semana):
    sin = _resultado(_correr(semana, _sin_claves(perfil)))
    for fuente in rr.FUENTES_POST_FESTIVO:
        off = {"post_festivo_factor": False, "post_festivo_fuente": fuente}
        falso = _resultado(_correr(semana, perfiles.expandir(perfil, {"nivel": dict(off), "forma": dict(off)})))
        assert sin == falso


def test_perfiles_no_activan_el_flag():
    for perfil in perfiles.PERFILES:
        p = perfiles.params_perfil(perfil)
        assert "post_festivo_factor" not in p["nivel"] and "post_festivo_factor" not in p["forma"]
