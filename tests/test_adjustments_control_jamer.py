"""Alertas Jamer de `control` (saltos, crecimiento, acumulado anual): informativas y apagadas por defecto."""

from datetime import date, timedelta

import pytest

from src.adjustments.data import curvas_reales, dias_semana, energia_diaria, iso
from src.adjustments.pipeline import build_context
from src.adjustments.steps import control
from tests.ajustes_sinteticos import historico_sintetico

HIST = historico_sintetico()
SEMANA = date(2025, 9, 8)


def _ctx():
    ctx = build_context("Sintetica", SEMANA, fecha_emision=SEMANA - timedelta(days=4), historico=HIST)
    ctx.semana_ref_inicio = SEMANA - timedelta(days=364)
    return ctx


def _prev(ctx, factor=1.0):
    """Curvas de la semana objetivo = las de R2 (lag 364) × factor."""
    reales = curvas_reales(ctx.historico)
    return {
        iso(d): [float(x) * factor for x in reales[iso(d - timedelta(days=364))]]
        for d in dias_semana(ctx.semana_inicio)
    }


def _codigos(res):
    return [a.codigo for a in res.alerts]


NUEVAS = {"SALTO_ENTRE_DIAS", "CRECIMIENTO_FUERA_DE_TENDENCIA", "ACUMULADO_ANUAL_FUERA_RANGO"}


def test_defaults_apagados_no_emiten_alertas_nuevas_pero_publican_diagnosticos():
    ctx = _ctx()
    prev = _prev(ctx, 1.20)
    prev[iso(SEMANA + timedelta(days=1))] = [v + 500 for v in prev[iso(SEMANA + timedelta(days=1))]]
    res = control.step(ctx, prev, {})
    assert not NUEVAS & set(_codigos(res))
    d = res.diagnostics
    assert d["crecimiento"]["semana_pct"] > 15 and d["crecimiento"]["Y_pct"] is not None
    assert len(d["crecimiento"]["dias"]) == 7
    assert d["saltos"]["banda_mw"] is None and len(d["saltos"]["pares"]) == 6
    assert d["acumulado_anual"] is None
    assert control.DEFAULTS["banda_salto_mw"] is None
    assert control.DEFAULTS["banda_crecimiento_pp"] is None
    assert control.DEFAULTS["cupo_dias_anuales"] is None


def test_salto_igual_al_de_referencia_no_alerta_y_salto_extra_si():
    ctx = _ctx()
    prev = _prev(ctx)
    res = control.step(ctx, prev, {"banda_salto_mw": 40.0})
    assert "SALTO_ENTRE_DIAS" not in _codigos(res)
    assert all(abs(q["dif_mw"]) < 1e-6 for q in res.diagnostics["saltos"]["pares"])

    martes = iso(SEMANA + timedelta(days=1))
    prev[martes] = [v + 100.0 if 3 <= i + 1 <= 5 else v for i, v in enumerate(prev[martes])]
    res = control.step(ctx, prev, {"banda_salto_mw": 40.0})
    a = next(x for x in res.alerts if x.codigo == "SALTO_ENTRE_DIAS")
    assert a.nivel == "info"
    # sube el par lun->mar (+100) y baja mar->mié (-100): ambos fuera de banda; fechas = día de llegada
    assert a.fechas == [iso(SEMANA + timedelta(days=1)), iso(SEMANA + timedelta(days=2))]
    por_a = {q["a"]: q for q in a.detalle["pares"]}
    assert por_a[martes]["dif_mw"] == pytest.approx(100.0)
    # banda mayor: sin alerta
    assert "SALTO_ENTRE_DIAS" not in _codigos(control.step(ctx, prev, {"banda_salto_mw": 150.0}))


def test_salto_caso_jamer_86_8_vs_42_7_dif_44():
    ctx = _ctx()
    prev = _prev(ctx)
    reales = curvas_reales(ctx.historico)
    ref_lun = SEMANA - timedelta(days=364)
    base_ref = float(sum(reales[iso(ref_lun + timedelta(days=1))][i] - reales[iso(ref_lun)][i] for i in (2, 3, 4)) / 3)
    # fuerza que el martes final tenga un salto = salto_ref + 44.1 sobre el valle
    martes = iso(SEMANA + timedelta(days=1))
    prev[martes] = [v + 44.1 if 3 <= i + 1 <= 5 else v for i, v in enumerate(prev[martes])]
    res = control.step(ctx, prev, {"banda_salto_mw": 40.0})
    par = next(q for q in res.diagnostics["saltos"]["pares"] if q["a"] == martes)
    assert par["delta_final"] == pytest.approx(base_ref + 44.1)
    assert par["dif_mw"] == pytest.approx(44.1)
    assert "SALTO_ENTRE_DIAS" in _codigos(res)


def test_salto_periodos_valle_configurable_solo_p4():
    ctx = _ctx()
    prev = _prev(ctx)
    martes = iso(SEMANA + timedelta(days=1))
    prev[martes] = [v + 60.0 if i == 3 else v for i, v in enumerate(prev[martes])]  # solo P4
    # P3-P5: 60/3 = 20 MW de dif -> no supera 40; solo P4: 60 -> sí
    assert "SALTO_ENTRE_DIAS" not in _codigos(control.step(ctx, prev, {"banda_salto_mw": 40.0}))
    res = control.step(ctx, prev, {"banda_salto_mw": 40.0, "periodos_valle": [4]})
    assert "SALTO_ENTRE_DIAS" in _codigos(res)


def test_salto_sin_referencia_o_curva_invalida_no_falla():
    ctx = _ctx()
    ctx.semana_ref_inicio = None
    res = control.step(ctx, _prev(_ctx()), {"banda_salto_mw": 10.0})
    assert res.diagnostics["saltos"]["motivo"] == "sin semana de referencia"
    assert "SALTO_ENTRE_DIAS" not in _codigos(res)
    ctx = _ctx()
    prev = _prev(ctx)
    prev[iso(SEMANA + timedelta(days=2))] = [None] * 24
    res = control.step(ctx, prev, {"banda_salto_mw": 10.0})
    assert len(res.diagnostics["saltos"]["pares"]) == 4  # se omiten los 2 pares con el día inválido
    assert "PERIODOS_FALTANTES" in _codigos(res) or any(a.nivel == "bloqueo" for a in res.alerts)


def test_crecimiento_dentro_de_tendencia_no_alerta_y_fuera_si():
    ctx = _ctx()
    dentro = control.step(ctx, _prev(ctx, 1.03), {"banda_crecimiento_pp": 3.0})
    c = dentro.diagnostics["crecimiento"]
    assert c["semana_pct"] == pytest.approx(3.0, abs=1e-6)
    assert abs(c["Y_pct"] - 3.0) < 1.0 and c["fallback"] is False
    assert "CRECIMIENTO_FUERA_DE_TENDENCIA" not in _codigos(dentro)

    fuera = control.step(ctx, _prev(ctx, 1.10), {"banda_crecimiento_pp": 3.0})
    a = next(x for x in fuera.alerts if x.codigo == "CRECIMIENTO_FUERA_DE_TENDENCIA")
    assert a.nivel == "info" and len(a.fechas) == 3
    assert a.detalle["diferencia_pp"] == pytest.approx(10.0 - c["Y_pct"], abs=0.01)
    # crecer MENOS que la tendencia no alerta (solo excesos)
    assert "CRECIMIENTO_FUERA_DE_TENDENCIA" not in _codigos(
        control.step(ctx, _prev(ctx, 0.9), {"banda_crecimiento_pp": 3.0}))


def test_crecimiento_sin_pares_suficientes_no_alerta():
    ctx = _ctx()
    res = control.step(ctx, _prev(ctx, 1.3), {"banda_crecimiento_pp": 3.0, "crecimiento_min_pares": 400,
                                              "crecimiento_ventana_dias": 28})
    assert res.diagnostics["crecimiento"]["Y_pct"] is None
    assert res.diagnostics["crecimiento"]["fallback"] is True
    assert "CRECIMIENTO_FUERA_DE_TENDENCIA" not in _codigos(res)
    assert res.diagnostics["crecimiento"]["semana_pct"] == pytest.approx(30.0, abs=1e-6)


def _aprobada_con_errores(ctx, n_fuera):
    real = energia_diaria(ctx.historico)
    fechas = sorted(f for f in real if f.startswith("2025-") and f <= iso(ctx.fecha_corte))
    return {f: real[f] * (1.10 if i < n_fuera else 1.0) for i, f in enumerate(fechas)}


def test_acumulado_apagado_sin_cupo_y_con_cupo_alerta_cerca_y_excedido():
    ctx = _ctx()
    prev = _prev(ctx)
    ap = _aprobada_con_errores(ctx, 9)
    # sin cupo: no emite aunque haya datos
    res = control.step(ctx, prev, {}, aprobada_anual=ap)
    assert "ACUMULADO_ANUAL_FUERA_RANGO" not in _codigos(res) and res.diagnostics["acumulado_anual"] is None
    # cupo 10, 9 días fuera (>= 80 %): cerca
    res = control.step(ctx, prev, {"cupo_dias_anuales": 10}, aprobada_anual=ap)
    a = next(x for x in res.alerts if x.codigo == "ACUMULADO_ANUAL_FUERA_RANGO")
    assert a.nivel == "info" and a.detalle["estado"] == "cerca" and a.detalle["dias_fuera"] == 9
    assert res.diagnostics["acumulado_anual"]["estado"] == "cerca"
    # cupo 5: excedido; banda mayor al error (10 %): dentro, sin alerta
    res = control.step(ctx, prev, {"cupo_dias_anuales": 5}, aprobada_anual=ap)
    assert next(x for x in res.alerts if x.codigo == "ACUMULADO_ANUAL_FUERA_RANGO").detalle["estado"] == "excedido"
    res = control.step(ctx, prev, {"cupo_dias_anuales": 5, "banda_error_pct": 12.0}, aprobada_anual=ap)
    assert "ACUMULADO_ANUAL_FUERA_RANGO" not in _codigos(res)
    assert res.diagnostics["acumulado_anual"]["dias_fuera"] == 0


def test_acumulado_lee_de_ctx_extras_y_sin_datos_no_alerta():
    ctx = _ctx()
    prev = _prev(ctx)
    res = control.step(ctx, prev, {"cupo_dias_anuales": 3})
    assert res.diagnostics["acumulado_anual"]["estado"] == "sin_datos"
    assert "ACUMULADO_ANUAL_FUERA_RANGO" not in _codigos(res)
    ctx.extras["acumulado_aprobada"] = _aprobada_con_errores(ctx, 4)
    res = control.step(ctx, prev, {"cupo_dias_anuales": 3})
    assert "ACUMULADO_ANUAL_FUERA_RANGO" in _codigos(res)


def test_alertas_nuevas_no_bloquean_ni_alteran_curvas_ni_existentes():
    ctx = _ctx()
    prev = _prev(ctx, 1.2)
    base = control.step(ctx, prev, {})
    todo = control.step(ctx, prev, {"banda_salto_mw": 1.0, "banda_crecimiento_pp": 1.0, "cupo_dias_anuales": 1},
                        aprobada_anual=_aprobada_con_errores(ctx, 5))
    assert all(a.nivel == "info" for a in todo.alerts if a.codigo in NUEVAS)
    assert not [a for a in todo.alerts if a.nivel == "bloqueo"]
    assert todo.curves == base.curves
    # las alertas existentes siguen idénticas
    assert [a.codigo for a in todo.alerts if a.codigo not in NUEVAS] == _codigos(base)
    # TENDENCIA_FUERA_BANDA sigue funcionando
    energia = {f: sum(v) for f, v in prev.items()}
    prev2 = {**prev, iso(SEMANA): [v * 1.2 for v in prev[iso(SEMANA)]]}
    res = control.step(ctx, prev2, {}, energia_nivel=energia)
    assert "TENDENCIA_FUERA_BANDA" in _codigos(res)


@pytest.mark.parametrize("params", [
    {"banda_salto_mw": 0}, {"banda_salto_mw": "x"}, {"periodos_valle": []}, {"periodos_valle": [0]},
    {"periodos_valle": [25]}, {"periodos_valle": 4}, {"banda_crecimiento_pp": -1},
    {"cupo_dias_anuales": -1}, {"cupo_dias_anuales": 1.5}, {"banda_error_pct": 0}, {"crecimiento_ventana_dias": 3},
])
def test_validar_params_rechaza_invalidos(params):
    with pytest.raises(ValueError):
        control.validar_params(params)


def test_validar_params_normaliza():
    p = control.validar_params({"banda_salto_mw": 40, "periodos_valle": [5, 3, 4, 3], "cupo_dias_anuales": 25.0})
    assert p["banda_salto_mw"] == 40.0 and p["periodos_valle"] == [3, 4, 5] and p["cupo_dias_anuales"] == 25
    # sin pedirlas, los params validados no cambian (compatibilidad con records guardados)
    assert control.validar_params({}) == {"banda_pp": 1.0, "top_dias": 3, "banda_referencia_pp": 2.0}
    assert control.DEFAULTS_JAMER["periodos_valle"] == [3, 4, 5]


# ---------------------------------------------------------------- R2 única y lecturas únicas (E4, E6)

def _referencias(ctx, desplazamiento=364):
    """`diagnostics["referencias"]` mínimo de nivel: r2 = d − desplazamiento."""
    return {
        iso(d): {"r1": None, "r1_extra": [], "r3": None,
                 "r2": {"fecha": iso(d - timedelta(days=desplazamiento)), "curva": []}}
        for d in dias_semana(ctx.semana_inicio)
    }


def test_con_referencias_de_nivel_igual_r2_da_lo_mismo_que_el_control_estandar():
    ctx = _ctx()
    prev = _prev(ctx, 1.05)
    estandar = control.step(ctx, prev, {"banda_salto_mw": 10.0, "banda_crecimiento_pp": 1.0})
    con_refs = control.step(ctx, prev, {"banda_salto_mw": 10.0, "banda_crecimiento_pp": 1.0},
                            referencias=_referencias(ctx))
    assert con_refs.diagnostics["saltos"] == estandar.diagnostics["saltos"]
    assert con_refs.diagnostics["crecimiento"] == estandar.diagnostics["crecimiento"]
    assert [a.to_dict() for a in con_refs.alerts] == [a.to_dict() for a in estandar.alerts]


def test_con_referencias_de_nivel_usa_su_r2_no_semana_ref_mas_i():
    ctx = _ctx()
    prev = _prev(ctx, 1.0)
    energia = energia_diaria(ctx.historico)
    f = iso(SEMANA + timedelta(days=1))
    res = control.step(ctx, prev, {}, referencias=_referencias(ctx, desplazamiento=371))
    dia = res.diagnostics["crecimiento"]["dias"][f]
    assert dia["energia_ref"] == pytest.approx(energia[iso(SEMANA + timedelta(days=1) - timedelta(days=371))])
    estandar = control.step(ctx, prev, {})
    assert estandar.diagnostics["crecimiento"]["dias"][f]["energia_ref"] == pytest.approx(
        energia[iso(ctx.semana_ref_inicio + timedelta(days=1))])
    # r2 ausente en un día -> ese día no es comparable (no se inventa otra referencia)
    refs = _referencias(ctx)
    refs[f]["r2"] = None
    parcial = control.step(ctx, prev, {}, referencias=refs)
    assert f not in parcial.diagnostics["crecimiento"]["dias"]
    assert parcial.diagnostics["crecimiento"]["motivo"] == "faltan días de la semana final o de la referencia"


def test_con_referencias_no_exige_semana_ref_inicio_y_sin_ellas_si():
    ctx = _ctx()
    ctx.semana_ref_inicio = None
    prev = _prev(_ctx())
    assert control.step(ctx, prev, {}).diagnostics["saltos"]["motivo"] == "sin semana de referencia"
    res = control.step(ctx, prev, {}, referencias=_referencias(ctx))
    assert res.diagnostics["saltos"]["motivo"] is None and len(res.diagnostics["saltos"]["pares"]) == 6
    assert res.diagnostics["crecimiento"]["semana_pct"] is not None


def test_control_lee_el_historico_una_sola_vez(monkeypatch):
    ctx = _ctx()
    prev = _prev(ctx)
    llamadas = []
    original = control.curvas_reales
    monkeypatch.setattr(control, "curvas_reales", lambda h: llamadas.append(1) or original(h))
    monkeypatch.setattr(control, "energia_diaria", lambda h: llamadas.append(1) or {}, raising=False)
    control.step(ctx, prev, {"banda_salto_mw": 10.0, "banda_crecimiento_pp": 1.0, "cupo_dias_anuales": 2},
                 aprobada_anual={"2025-09-01": 1.0})
    assert len(llamadas) == 1
