"""Tests unitarios de los pasos del pipeline de ajustes (datos sintéticos)."""

from datetime import date, timedelta

import numpy as np
import pytest

from src.adjustments.data import dias_semana, festivos_colombia, iso, tipo_dia
from src.adjustments.pipeline import build_context
from src.adjustments.steps import base, control, forma, nivel, referencia
from tests.ajustes_sinteticos import ProveedorPlano, energia_real, forma_tipo, historico_sintetico

HIST = historico_sintetico()
FESTIVOS = set(festivos_colombia(range(2014, 2027)))
SEMANA_NORMAL = date(2025, 9, 8)  # sin festivos
SEMANA_SANTA = date(2025, 4, 14)  # jueves y viernes santo


def _ctx(semana=SEMANA_NORMAL, emision=None):
    emision = emision or semana - timedelta(days=4)  # jueves previo → corte miércoles
    return build_context("Sintetica", semana, fecha_emision=emision, historico=HIST)


def _base(ctx, mw=1000.0):
    return base.step(ctx, {}, {}, base_provider=ProveedorPlano(mw)).curves


def _hasta_nivel(ctx, alpha=0.5):
    prev = _base(ctx)
    ref = referencia.step(ctx, prev, {})
    ctx.semana_ref_inicio = date.fromisoformat(ref.diagnostics["semana_ref_inicio"])
    return prev, nivel.step(ctx, ref.curves, {"alpha": alpha})


# ---------------------------------------------------------------- base

def test_base_cubre_brecha_y_semana():
    ctx = _ctx()
    prov = ProveedorPlano()
    res = base.step(ctx, {}, {}, base_provider=prov)
    assert ctx.fecha_corte == date(2025, 9, 3)
    assert prov.llamadas == [("Sintetica", date(2025, 9, 3), 11)]
    assert res.diagnostics["dias_brecha"] == 4
    assert res.diagnostics["rango"] == ["2025-09-04", "2025-09-14"]
    assert len(res.curves) == 11 and not res.alerts


def test_base_incompleta_emite_alerta():
    ctx = _ctx()
    res = base.step(ctx, {}, {}, base_provider=ProveedorPlano(faltar=["2025-09-10"]))
    assert "2025-09-10" not in res.curves
    assert [a.codigo for a in res.alerts] == ["BASE_INCOMPLETA"]


def test_base_sin_provider_falla():
    with pytest.raises(ValueError):
        base.step(_ctx(), {}, {})


# ---------------------------------------------------------------- referencia

def test_referencia_default_364_y_no_modifica_curvas():
    ctx = _ctx()
    prev = _base(ctx)
    res = referencia.step(ctx, prev, {})
    assert res.diagnostics["semana_ref_inicio"] == "2024-09-09"
    assert res.diagnostics["metodo"] == "anio_anterior_364"
    assert res.curves == prev and res.alerts == []


def test_referencia_con_festivo_busca_homologo_mismo_dia_semana():
    ctx = _ctx(SEMANA_SANTA)
    res = referencia.step(ctx, _base(ctx), {})
    # −364 sería 2024-04-15; Semana Santa 2024 fue la semana del 2024-03-25.
    assert res.diagnostics["metodo"] == "festivo_homologo"
    assert res.diagnostics["semana_ref_inicio"] == "2024-03-25"
    assert res.diagnostics["coincidencias"] == 2
    assert res.diagnostics["homologos"] == {"2025-04-17": "2024-03-28", "2025-04-18": "2024-03-29"}


def test_referencia_forzada_por_param_se_lleva_a_lunes():
    ctx = _ctx()
    res = referencia.step(ctx, {}, {"semana_ref_inicio": "2023-09-13"})  # miércoles
    assert res.diagnostics["semana_ref_inicio"] == "2023-09-11"
    assert res.diagnostics["metodo"] == "parametro"
    assert [a.codigo for a in res.alerts] == ["REFERENCIA_AJUSTADA_A_LUNES"]


def test_referencia_sin_datos_alerta_warning():
    ctx = _ctx()
    res = referencia.step(ctx, {}, {"semana_ref_inicio": "2019-09-09"})  # antes del histórico
    alerta = next(a for a in res.alerts if a.codigo == "REFERENCIA_SIN_DATOS")
    assert alerta.nivel == "warning" and len(alerta.fechas) == 7


# ---------------------------------------------------------------- referencia: H2 (años excluidos)

HIST_LARGO = historico_sintetico(date(2015, 1, 1), date(2026, 6, 30))


def _ctx_largo(semana):
    return build_context("Sintetica", semana, fecha_emision=semana - timedelta(days=2), historico=HIST_LARGO)


def test_referencia_festivo_fijo_no_elige_2020_y_respeta_max_anios():
    # 1-may-2026 (viernes): el único año con el homólogo en viernes dentro de 6 años es 2020.
    ctx = _ctx_largo(date(2026, 4, 27))
    res = referencia.step(ctx, {}, {})
    assert res.diagnostics["semana_ref_inicio"] == "2025-04-28"  # −364, no 2020-04-27
    assert res.diagnostics["metodo"] == "anio_anterior_364"
    assert res.diagnostics["max_anios_atras"] == 4 and res.diagnostics["anios_excluidos"] == [2020]
    assert "REFERENCIA_SIN_HOMOLOGO" in [a.codigo for a in res.alerts]
    # aun ampliando la búsqueda a 6 años, 2020 sigue excluido
    res6 = referencia.step(ctx, {}, {"max_anios_atras": 6})
    assert res6.diagnostics["semana_ref_inicio"] == "2025-04-28"
    # sin exclusión y con 6 años se reproduce el comportamiento viejo (2020)
    viejo = referencia.step(ctx, {}, {"max_anios_atras": 6, "anios_excluidos": []})
    assert viejo.diagnostics["semana_ref_inicio"] == "2020-04-27"
    assert viejo.diagnostics["anios_atras"] == 6


def test_referencia_max_anios_atras_limita_busqueda():
    ctx = _ctx_largo(date(2026, 4, 27))
    res = referencia.step(ctx, {}, {"max_anios_atras": 5, "anios_excluidos": []})
    assert res.diagnostics["semana_ref_inicio"] == "2025-04-28"  # 2020 está a 6 años
    with pytest.raises(ValueError):
        referencia.step(ctx, {}, {"max_anios_atras": 0})


def test_referencia_default_364_salta_2020():
    ctx = _ctx_largo(date(2021, 9, 6))  # sin festivos; −364 = 2020-09-07
    res = referencia.step(ctx, {}, {})
    assert res.diagnostics["semana_ref_inicio"] == "2019-09-09"  # −728
    assert res.diagnostics["metodo"] == "anio_anterior_364"
    assert res.alerts == []


def test_referencia_festivo_movil_salta_2020():
    # 2021-06-07 lunes festivo (Corpus): el homólogo de 2020 también es lunes pero 2020 está excluido.
    ctx = _ctx_largo(date(2021, 6, 7))
    res = referencia.step(ctx, {}, {})
    assert res.diagnostics["metodo"] == "festivo_homologo"
    assert res.diagnostics["semana_ref_inicio"].startswith("2019-")
    assert res.diagnostics["anios_atras"] == 2


def test_referencia_forzada_en_anio_excluido_alerta_y_nivel_sin_evidencia():
    ctx = _ctx_largo(date(2021, 9, 6))
    res = referencia.step(ctx, {}, {"semana_ref_inicio": "2020-09-07"})
    assert res.diagnostics["semana_ref_inicio"] == "2020-09-07"
    alerta = next(a for a in res.alerts if a.codigo == "REFERENCIA_ANIO_EXCLUIDO")
    assert alerta.nivel == "warning"
    ctx.semana_ref_inicio = date(2020, 9, 7)
    ctx.extras["anios_excluidos"] = res.diagnostics["anios_excluidos"]
    res_nivel = nivel.step(ctx, _base(ctx), {})
    assert any(a.codigo == "SIN_EVIDENCIA_NIVEL" for a in res_nivel.alerts)
    # con anios_excluidos = [] el nivel sí usa 2020
    ctx.extras["anios_excluidos"] = []
    assert not nivel.step(ctx, _base(ctx), {}).alerts


def test_nivel_anclas_ref_no_caen_en_anio_excluido():
    # Semana objetivo enero 2022, ref enero 2021: las anclas ref (d' − lag) caen en dic-2020.
    ctx = _ctx_largo(date(2022, 1, 3))
    ctx.semana_ref_inicio = date(2021, 1, 4)
    res = nivel.step(ctx, _base(ctx), {})
    for f, dia in res.diagnostics["dias"].items():
        for clave in ("dia_ref", "ancla_ref"):
            if dia[clave]:
                assert not dia[clave].startswith("2020-"), (f, dia)
    ctx.extras["anios_excluidos"] = []
    res_sin = nivel.step(ctx, _base(ctx), {})
    assert any((d["ancla_ref"] or "").startswith("2020-") for d in res_sin.diagnostics["dias"].values())


# ---------------------------------------------------------------- nivel

def test_nivel_formula_con_alpha_y_forma_intacta():
    ctx = _ctx()
    prev, res = _hasta_nivel(ctx, alpha=0.5)
    for d in dias_semana(ctx.semana_inicio):
        f = iso(d)
        dia = res.diagnostics["dias"][f]
        e_true = energia_real(d, FESTIVOS)
        # Con crecimiento exponencial, E* = E(ancla)·E(d')/E(a') recupera el real exacto.
        assert dia["E_objetivo"] == pytest.approx(e_true, rel=1e-9)
        assert sum(res.curves[f]) == pytest.approx(0.5 * 24_000 + 0.5 * e_true, rel=1e-9)
        assert dia["alpha"] == 0.5
        # base plana → sigue plana (solo cambia el nivel)
        assert np.ptp(res.curves[f]) == pytest.approx(0.0, abs=1e-9)
    # Ancla: último mismo día de semana <= corte (miércoles 2025-09-03)
    assert res.diagnostics["dias"]["2025-09-08"]["ancla"] == "2025-09-01"
    assert res.diagnostics["dias"]["2025-09-11"]["ancla"] == "2025-08-28"
    # la brecha no se toca
    for f in ["2025-09-04", "2025-09-05", "2025-09-06", "2025-09-07"]:
        assert res.curves[f] == prev[f]


def test_nivel_sin_evidencia_alpha_cero_y_alerta():
    ctx = _ctx()
    prev = _base(ctx)
    ctx.semana_ref_inicio = None
    res = nivel.step(ctx, prev, {"alpha": 0.5})
    assert res.curves == prev
    alerta = next(a for a in res.alerts if a.codigo == "SIN_EVIDENCIA_NIVEL")
    assert alerta.nivel == "warning" and len(alerta.fechas) == 7
    assert all(d["alpha"] == 0.0 for d in res.diagnostics["dias"].values())


def test_nivel_festivo_usa_festivo_de_referencia():
    ctx = _ctx(SEMANA_SANTA)
    _, res = _hasta_nivel(ctx)
    jueves = res.diagnostics["dias"]["2025-04-17"]
    assert jueves["dia_ref"] == "2024-03-28"
    assert jueves["regla"] == "festivo_ref_misma_posicion"
    assert iso(date.fromisoformat(jueves["ancla_ref"])) not in FESTIVOS
    assert not any(a.codigo == "SIN_EVIDENCIA_NIVEL" for a in res.alerts)


def test_nivel_alpha_invalido():
    with pytest.raises(ValueError):
        nivel.step(_ctx(), {}, {"alpha": 1.5})


# ---------------------------------------------------------------- forma

@pytest.mark.parametrize("fuente", ["reciente", "referencia", "promedio_n"])
@pytest.mark.parametrize("beta", [1.0, 0.5])
def test_forma_reconcilia_energia_exacta(fuente, beta):
    ctx = _ctx()
    _, res_nivel = _hasta_nivel(ctx)
    res = forma.step(ctx, res_nivel.curves, {"fuente": fuente, "beta": beta})
    for d in dias_semana(ctx.semana_inicio):
        f = iso(d)
        assert abs(sum(res.curves[f]) - sum(res_nivel.curves[f])) <= 1e-6
        assert res.diagnostics["dias"][f]["fuente"] == fuente
        if beta == 1.0:
            esperado = forma_tipo(tipo_dia(d, FESTIVOS)) * sum(res_nivel.curves[f])
            np.testing.assert_allclose(res.curves[f], esperado, rtol=1e-9)
    # brecha intacta
    assert res.curves["2025-09-05"] == res_nivel.curves["2025-09-05"]


def test_forma_diagnostica_pico():
    ctx = _ctx()
    _, res_nivel = _hasta_nivel(ctx)
    res = forma.step(ctx, res_nivel.curves, {})
    dia = res.diagnostics["dias"]["2025-09-09"]
    assert dia["pico_despues"]["hora"] == int(np.argmax(forma_tipo("laboral"))) + 1
    assert dia["pico_antes"]["mw"] == pytest.approx(max(res_nivel.curves["2025-09-09"]))


def test_forma_festivo_usa_perfil_homologo():
    ctx = _ctx(SEMANA_SANTA)
    _, res_nivel = _hasta_nivel(ctx)
    res = forma.step(ctx, res_nivel.curves, {})
    viernes = res.diagnostics["dias"]["2025-04-18"]
    assert viernes["fuente"].startswith("festivo_")
    assert "2024-03-29" in viernes["fechas_fuente"]
    np.testing.assert_allclose(
        np.array(res.curves["2025-04-18"]) / sum(res.curves["2025-04-18"]),
        forma_tipo("festivo"), rtol=1e-9,
    )
    assert abs(sum(res.curves["2025-04-18"]) - sum(res_nivel.curves["2025-04-18"])) <= 1e-6


def test_forma_params_invalidos():
    with pytest.raises(ValueError):
        forma.step(_ctx(), {}, {"fuente": "otra"})
    with pytest.raises(ValueError):
        forma.step(_ctx(), {}, {"beta": -0.1})


# ---------------------------------------------------------------- control

def _curvas_con_tendencia_esperada(ctx):
    """Semana objetivo + brecha con la energía 'real' sintética (tendencia = la esperada)."""
    curves = {}
    d = ctx.fecha_corte + timedelta(days=1)
    while d <= ctx.semana_inicio + timedelta(days=6):
        curves[iso(d)] = (forma_tipo(tipo_dia(d, FESTIVOS)) * energia_real(d, FESTIVOS)).tolist()
        d += timedelta(days=1)
    return curves


def _energia(curves):
    return {f: float(sum(v)) for f, v in curves.items()}


def test_control_sin_cambios_tras_nivel_no_alerta_y_no_modifica():
    ctx = _ctx()
    ctx.semana_ref_inicio = date(2024, 9, 9)
    prev = _curvas_con_tendencia_esperada(ctx)
    res = control.step(ctx, prev, {}, energia_nivel=_energia(prev))
    assert res.curves == prev
    assert res.alerts == []
    assert res.diagnostics["desvio_pp"] == pytest.approx(0.0, abs=1e-12)
    # diagnóstico informativo de tendencia de referencia (curvas = real → ≈ 0)
    assert abs(res.diagnostics["diferencia_tendencia_pp"]) < 0.01


def test_control_edicion_que_sube_un_dia_10pct_alerta():
    ctx = _ctx()
    ctx.semana_ref_inicio = date(2024, 9, 9)
    nivel_curvas = _curvas_con_tendencia_esperada(ctx)
    prev = {f: list(v) for f, v in nivel_curvas.items()}
    prev["2025-09-10"] = [v * 1.10 for v in prev["2025-09-10"]]
    res = control.step(ctx, prev, {"banda_pp": 1.0}, energia_nivel=_energia(nivel_curvas))
    alerta = next(a for a in res.alerts if a.codigo == "TENDENCIA_FUERA_BANDA")
    assert alerta.nivel == "warning"
    assert alerta.fechas[0] == "2025-09-10"
    e_sem = sum(_energia(nivel_curvas)[iso(d)] for d in dias_semana(ctx.semana_inicio))
    esperado = 0.10 * sum(nivel_curvas["2025-09-10"]) / e_sem * 100
    assert res.diagnostics["desvio_pp"] == pytest.approx(esperado, rel=1e-9)
    assert res.diagnostics["dias"]["2025-09-10"]["desvio_pp"] == pytest.approx(esperado, rel=1e-9)
    # con banda amplia no alerta
    res_amplia = control.step(ctx, prev, {"banda_pp": 5.0}, energia_nivel=_energia(nivel_curvas))
    assert not any(a.codigo == "TENDENCIA_FUERA_BANDA" for a in res_amplia.alerts)


def test_control_sin_energia_nivel_no_evalua_banda():
    ctx = _ctx()
    ctx.semana_ref_inicio = date(2024, 9, 9)
    res = control.step(ctx, _curvas_con_tendencia_esperada(ctx), {})
    assert [a.codigo for a in res.alerts] == ["SIN_ENERGIA_NIVEL"]
    assert res.diagnostics["desvio_pp"] is None


def test_control_bloqueos_estructurales():
    ctx = _ctx()
    ctx.semana_ref_inicio = date(2024, 9, 9)
    prev = _curvas_con_tendencia_esperada(ctx)
    energia_nivel = _energia(prev)
    del prev["2025-09-08"]
    prev["2025-09-09"] = prev["2025-09-09"][:23]
    prev["2025-09-10"] = [float("nan")] + prev["2025-09-10"][1:]
    prev["2025-09-11"] = [-1.0] + prev["2025-09-11"][1:]
    prev["2025-09-12"] = [None] + prev["2025-09-12"][1:]
    res = control.step(ctx, prev, {}, energia_esperada=energia_nivel, energia_nivel=energia_nivel)
    bloqueos = {a.codigo: a.fechas for a in res.alerts if a.nivel == "bloqueo"}
    assert bloqueos["PERIODOS_FALTANTES"] == ["2025-09-08", "2025-09-09"]
    assert bloqueos["VALORES_NO_FINITOS"] == ["2025-09-10", "2025-09-12"]
    assert bloqueos["VALORES_NEGATIVOS"] == ["2025-09-11"]
    assert "RECONCILIACION_ROTA" not in bloqueos  # las inválidas ya están bloqueadas
    assert any(a.codigo == "SIN_CONTROL_TENDENCIA" for a in res.alerts)


def test_control_reconciliacion_rota_bloquea():
    ctx = _ctx()
    ctx.semana_ref_inicio = date(2024, 9, 9)
    prev = _curvas_con_tendencia_esperada(ctx)
    esperada = _energia(prev)
    esperada["2025-09-12"] += 1.0
    res = control.step(ctx, prev, {}, energia_esperada=esperada)
    alerta = next(a for a in res.alerts if a.codigo == "RECONCILIACION_ROTA")
    assert alerta.nivel == "bloqueo" and alerta.fechas == ["2025-09-12"]


def test_control_tendencia_referencia_usa_e_objetivo_y_marca_dias_sin_evidencia():
    ctx = _ctx()
    ctx.semana_ref_inicio = date(2024, 9, 9)
    prev = _curvas_con_tendencia_esperada(ctx)
    fechas = [iso(d) for d in dias_semana(ctx.semana_inicio)]
    nivel_dias = {f: {"E_objetivo": sum(prev[f]), "E_base": 1.0} for f in fechas}
    nivel_dias[fechas[2]] = {"E_objetivo": None, "E_base": sum(prev[fechas[2]]) * 1.10}
    res = control.step(ctx, prev, {}, energia_nivel=_energia(prev), nivel_dias=nivel_dias)
    assert res.diagnostics["dias_sin_evidencia"] == [fechas[2]]
    dia = res.diagnostics["dias"][fechas[2]]
    assert dia["fuente_esperada"] == "energia_base"
    assert dia["energia_esperada"] == pytest.approx(sum(prev[fechas[2]]) * 1.10)
    assert res.diagnostics["dias"][fechas[0]]["energia_esperada"] == pytest.approx(sum(prev[fechas[0]]))
    # el día sin evidencia arrastra la referencia: diferencia ≈ −10 % de ese día
    esperado_pp = -0.10 * sum(prev[fechas[2]]) / res.diagnostics["energia_semana_previa"] * 100
    assert res.diagnostics["diferencia_tendencia_pp"] == pytest.approx(esperado_pp, rel=1e-9)
    # es informativo: no alerta por defecto…
    assert res.alerts == []
    # …salvo que se pida banda_referencia_pp
    res_alerta = control.step(ctx, prev, {"banda_referencia_pp": 0.5},
                              energia_nivel=_energia(prev), nivel_dias=nivel_dias)
    alerta = next(a for a in res_alerta.alerts if a.codigo == "TENDENCIA_REFERENCIA_FUERA_BANDA")
    assert alerta.fechas[0] == fechas[2]


def test_control_sin_evidencia_en_ningun_dia_no_calcula_tendencia_referencia():
    ctx = _ctx()
    ctx.semana_ref_inicio = date(2024, 9, 9)
    prev = _curvas_con_tendencia_esperada(ctx)
    nivel_dias = {iso(d): {"E_objetivo": None, "E_base": 1.0} for d in dias_semana(ctx.semana_inicio)}
    res = control.step(ctx, prev, {"banda_referencia_pp": 0.1}, energia_nivel=_energia(prev),
                       nivel_dias=nivel_dias)
    assert res.alerts == []
    assert res.diagnostics["tendencia_esperada_pct"] is None
    assert res.diagnostics["tendencia_resultante_pct"] is not None


def test_control_sin_referencia_no_calcula_tendencia_referencia():
    ctx = _ctx()
    ctx.semana_ref_inicio = None
    prev = _curvas_con_tendencia_esperada(ctx)
    res = control.step(ctx, prev, {}, energia_nivel=_energia(prev))
    assert res.alerts == []
    assert res.diagnostics["tendencia_esperada_pct"] is None


# ---------------------------------------------------------------- validar_params

@pytest.mark.parametrize(
    "modulo, params",
    [
        (nivel, {"alpha": None}),
        (nivel, {"alpha": float("inf")}),
        (nivel, {"alpha": float("nan")}),
        (nivel, {"alpha": 2.0}),
        (nivel, {"alpha": -0.1}),
        (nivel, {"alpha": "0.5"}),
        (nivel, {"alpha": True}),
        (nivel, {"alfa": 0.5}),  # typo: clave desconocida
        (referencia, {"anios_excluidos": 2020}),
        (referencia, {"anios_excluidos": None}),
        (referencia, {"anios_excluidos": ["2020"]}),
        (referencia, {"anios_excluidos": [2020.5]}),
        (referencia, {"max_anios_atras": 0}),
        (referencia, {"max_anios_atras": None}),
        (referencia, {"max_anios_atras": 2.5}),
        (referencia, {"semana_ref_inicio": "no-es-fecha"}),
        (referencia, {"semana_ref_inicio": 20240909}),
        (forma, {"fuente": "desconocida"}),
        (forma, {"fuente": None}),
        (forma, {"n": 0}),
        (forma, {"n": float("inf")}),
        (forma, {"beta": 1.5}),
        (control, {"banda_pp": 0}),
        (control, {"banda_pp": None}),
        (control, {"banda_pp": float("-inf")}),
        (control, {"top_dias": 0}),
        (control, {"banda_referencia_pp": -1}),
        (base, {"lo_que_sea": 1}),
    ],
)
def test_validar_params_invalidos_lanzan_value_error(modulo, params):
    with pytest.raises(ValueError):
        modulo.validar_params(params)


def test_validar_params_completa_defaults_y_castea():
    assert nivel.validar_params({}) == {"alpha": 0.25}
    assert nivel.validar_params({"alpha": 1}) == {"alpha": 1.0}
    assert forma.validar_params({"n": 3.0}) == {"fuente": "promedio_n", "n": 3, "beta": 1.0}
    ref = referencia.validar_params({"semana_ref_inicio": date(2023, 9, 13), "anios_excluidos": [2021, 2020]})
    assert ref == {"semana_ref_inicio": "2023-09-13", "max_anios_atras": 4, "anios_excluidos": [2020, 2021]}
    assert referencia.validar_params({"anios_excluidos": []})["anios_excluidos"] == []
    assert control.validar_params(None) == {"banda_pp": 1.0, "top_dias": 3, "banda_referencia_pp": 2.0}
    assert control.validar_params({"banda_referencia_pp": None})["banda_referencia_pp"] is None
    # claves obsoletas de records viejos se descartan sin error
    assert base.validar_params({"modelo_version": "v1"}) == {}
    assert "max_anios" not in referencia.validar_params({"max_anios": 6})


def test_base_toma_modelo_version_de_ctx_extras():
    ctx = _ctx()
    ctx.extras["modelo_version"] = "xgb-2026-07"
    res = base.step(ctx, {}, {}, base_provider=ProveedorPlano())
    assert res.diagnostics["modelo_version"] == "xgb-2026-07"
    assert "modelo_version" not in base.DEFAULTS


def test_pasos_toleran_curva_previa_invalida_sin_excepcion():
    ctx = _ctx()
    prev = _base(ctx)
    prev["2025-09-10"] = [float("nan")] + prev["2025-09-10"][1:]
    prev["2025-09-11"] = prev["2025-09-11"][:23]
    ref = referencia.step(ctx, prev, {})
    ctx.semana_ref_inicio = date.fromisoformat(ref.diagnostics["semana_ref_inicio"])
    res_nivel = nivel.step(ctx, ref.curves, {})
    res_forma = forma.step(ctx, res_nivel.curves, {})
    assert res_forma.curves["2025-09-11"] == prev["2025-09-11"]
    bloqueos = {a.codigo for a in control.step(ctx, res_forma.curves, {}).alerts if a.nivel == "bloqueo"}
    assert {"VALORES_NO_FINITOS", "PERIODOS_FALTANTES"} <= bloqueos


# ---------------------------------------------------------------- params: sin OverflowError/TypeError (fix ronda 2)

@pytest.mark.parametrize(
    "modulo,params",
    [
        (nivel, {"alpha": 10**400}),
        (nivel, {"alpha": -(10**400)}),
        (forma, {"n": 10**400}),
        (forma, {"n": 61}),
        (forma, {"n": 1e300}),
        (forma, {"beta": 10**400}),
        (referencia, {"anios_excluidos": [10**400]}),
        (referencia, {"anios_excluidos": [1989]}),
        (referencia, {"anios_excluidos": [2101]}),
        (referencia, {"anios_excluidos": 2020}),
        (referencia, {"anios_excluidos": list(range(1990, 2101)) * 3}),
        (referencia, {"max_anios_atras": 10**400}),
        (referencia, {"max_anios_atras": 21}),
        (referencia, {"semana_ref_inicio": "0001-01-01"}),
        (referencia, {"semana_ref_inicio": "9999-12-27"}),
        (referencia, {"semana_ref_inicio": "1989-12-31"}),
        (referencia, {"semana_ref_inicio": "2101-01-03"}),
        (control, {"top_dias": 10**400}),
        (control, {"top_dias": 8}),
        (control, {"banda_pp": 10**400}),
        (control, {"banda_referencia_pp": 10**400}),
    ],
)
def test_validar_params_extremos_solo_value_error(modulo, params):
    with pytest.raises(ValueError):
        modulo.validar_params(params)


def test_validar_params_limites_validos():
    assert forma.validar_params({"n": 60})["n"] == 60
    assert referencia.validar_params({"max_anios_atras": 20})["max_anios_atras"] == 20
    assert referencia.validar_params({"anios_excluidos": [1990, 2100]})["anios_excluidos"] == [1990, 2100]
    assert referencia.validar_params({"semana_ref_inicio": "1990-01-01"})["semana_ref_inicio"] == "1990-01-01"
    assert control.validar_params({"top_dias": 7})["top_dias"] == 7
    assert forma.validar_params({"n": np.int64(5)})["n"] == 5


def test_referencia_forzada_posterior_a_semana_objetivo_value_error():
    ctx = _ctx()
    with pytest.raises(ValueError, match="posterior"):
        referencia.step(ctx, _base(ctx), {"semana_ref_inicio": iso(SEMANA_NORMAL + timedelta(days=7))})
    # la misma semana objetivo (no posterior) se acepta
    res = referencia.step(ctx, _base(ctx), {"semana_ref_inicio": iso(SEMANA_NORMAL)})
    assert res.diagnostics["semana_ref_inicio"] == iso(SEMANA_NORMAL)


@pytest.mark.parametrize("ref", ["1990-01-01", "1995-06-14", "2025-09-08"])
def test_valores_limite_validos_no_desbordan_en_los_pasos(ref):
    # Cualquier valor que pase validar_params debe correr sin excepción en
    # referencia → nivel → forma → control (aritmética de fechas acotada).
    ctx = _ctx()
    prev = _base(ctx)
    res_ref = referencia.step(ctx, prev, {"semana_ref_inicio": ref, "max_anios_atras": 20,
                                          "anios_excluidos": [1990, 2100]})
    ctx.semana_ref_inicio = date.fromisoformat(res_ref.diagnostics["semana_ref_inicio"])
    ctx.extras["anios_excluidos"] = res_ref.diagnostics["anios_excluidos"]
    res_nivel = nivel.step(ctx, res_ref.curves, {"alpha": 1.0})
    res_forma = forma.step(ctx, res_nivel.curves, {"fuente": "referencia", "n": 60})
    control.step(ctx, res_forma.curves, {"top_dias": 7})
    # sin semana forzada, max_anios_atras=20 recorre 20 años sin error
    referencia.step(ctx, prev, {"max_anios_atras": 20})
    referencia.step(_ctx(SEMANA_SANTA), _base(_ctx(SEMANA_SANTA)), {"max_anios_atras": 20})


@pytest.mark.parametrize("semana", [date(1, 1, 1), date(9999, 12, 27), date(1989, 12, 25)])
def test_build_context_semana_fuera_de_rango_value_error(semana):
    with pytest.raises(ValueError):
        build_context("Sintetica", semana, historico=HIST)
