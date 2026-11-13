"""Tests del pipeline de ajustes: orden, omitido/editado, rerun_from, bloqueos, serialización."""

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.adjustments.data import curvas_reales, dias_semana, energia_curvas, iso
from src.adjustments.pipeline import (
    STEP_ORDER,
    approval_blockers,
    build_context,
    final_curves,
    orden_de,
    rerun_from,
    run_pipeline,
)
from src.adjustments.types import (
    Alert,
    RunContext,
    StepRecord,
    StepResult,
    StepState,
    validar_curva,
    validar_curvas_editadas,
)
from tests.ajustes_sinteticos import ProveedorPlano, historico_sintetico

HIST = historico_sintetico()
SEMANA = date(2025, 9, 8)
FECHAS_SEMANA = [iso(d) for d in dias_semana(SEMANA)]


def _ctx(semana=SEMANA):
    return build_context("Sintetica", semana, fecha_emision=semana - timedelta(days=4), historico=HIST)


def _record(orden, estado, params=None, curves=None):
    return StepRecord(orden, STEP_ORDER[orden], StepState(estado), params or {}, StepResult(curves or {}))


# ---------------------------------------------------------------- build_context

def test_build_context_corte_y_recorte():
    ctx = _ctx()
    assert ctx.fecha_corte == date(2025, 9, 3)
    assert ctx.historico["fecha"].max() == pd.Timestamp("2025-09-03")
    assert list(ctx.historico.columns) == ["fecha", *[f"P{i}" for i in range(1, 25)], "TOTAL"]
    assert "2025-11-03" in ctx.festivos and "2019-01-01" in ctx.festivos


def test_build_context_salta_dias_incompletos_y_no_entra_en_la_semana():
    hist = HIST.copy()
    hist.loc[hist["FECHA"] == "2025-09-03", "P5"] = np.nan
    ctx = build_context("Sintetica", SEMANA, fecha_emision=date(2025, 9, 4), historico=hist)
    assert ctx.fecha_corte == date(2025, 9, 2)
    # emisión posterior al lunes objetivo: el corte nunca entra en la semana objetivo
    ctx2 = build_context("Sintetica", SEMANA, fecha_emision=date(2025, 9, 12), historico=HIST)
    assert ctx2.fecha_corte == date(2025, 9, 7)


def test_build_context_exige_lunes():
    with pytest.raises(ValueError):
        build_context("Sintetica", date(2025, 9, 9), historico=HIST)


# ---------------------------------------------------------------- run_pipeline

def test_run_pipeline_completo():
    ctx = _ctx()
    prov = ProveedorPlano()
    records = run_pipeline(ctx, prov)
    assert [r.paso for r in records] == STEP_ORDER
    assert [r.orden for r in records] == [0, 1, 2, 3, 4]
    assert all(r.estado == StepState.APLICADO for r in records)
    assert len(prov.llamadas) == 1
    assert ctx.semana_ref_inicio == date(2024, 9, 9)
    # params resueltos con defaults
    assert records[2].params["alpha"] == 0.25 and records[3].params["fuente"] == "promedio_n"
    assert records[1].params["max_anios_atras"] == 4 and records[1].params["anios_excluidos"] == [2020]
    # forma reconcilia con nivel
    e_nivel = energia_curvas(records[2].result.curves)
    for f in FECHAS_SEMANA:
        assert abs(sum(records[3].result.curves[f]) - e_nivel[f]) <= 1e-6
    assert approval_blockers(records) == []
    finales = final_curves(records)
    assert finales == records[3].result.curves
    assert len(finales) == 11  # brecha (4) + semana (7)
    for f in ["2025-09-04", "2025-09-07"]:
        assert finales[f] == records[0].result.curves[f]


def test_records_serializables_ida_y_vuelta():
    records = run_pipeline(_ctx(), ProveedorPlano())
    payload = json.loads(json.dumps([r.to_dict() for r in records]))
    vuelta = [StepRecord.from_dict(d) for d in payload]
    assert [r.to_dict() for r in vuelta] == payload
    assert vuelta[2].estado is StepState.APLICADO


def test_run_context_serializable_ida_y_vuelta():
    ctx = _ctx()
    ctx.semana_ref_inicio = date(2024, 9, 9)
    d = json.loads(json.dumps(ctx.to_dict()))
    assert "historico" not in d
    otro = RunContext.from_dict(d, historico=ctx.historico)
    assert otro.semana_inicio == ctx.semana_inicio and otro.semana_ref_inicio == date(2024, 9, 9)
    completo = RunContext.from_dict(json.loads(json.dumps(ctx.to_dict(include_historico=True))))
    assert len(completo.historico) == len(ctx.historico)


def test_paso_omitido_es_identidad():
    records = run_pipeline(_ctx(), ProveedorPlano(), overrides={2: _record(2, "omitido")})
    assert records[2].estado == StepState.OMITIDO
    assert records[2].result.curves == records[1].result.curves
    assert records[2].result.alerts == []


def test_base_no_se_puede_omitir():
    with pytest.raises(ValueError):
        run_pipeline(_ctx(), ProveedorPlano(), overrides={0: _record(0, "omitido")})


def test_referencia_omitida_deja_nivel_sin_evidencia():
    ctx = _ctx()
    records = run_pipeline(ctx, ProveedorPlano(), overrides={1: _record(1, "omitido")})
    assert ctx.semana_ref_inicio is None
    assert records[2].result.curves == records[1].result.curves
    assert any(a.codigo == "SIN_EVIDENCIA_NIVEL" for a in records[2].result.alerts)


def test_paso_editado_se_valida_y_aplica():
    editada = {"2025-09-10": [900.0] * 24}
    records = run_pipeline(_ctx(), ProveedorPlano(), overrides={3: _record(3, "editado", curves=editada)})
    assert records[3].result.curves["2025-09-10"] == [900.0] * 24
    assert records[3].result.diagnostics["fechas_editadas"] == ["2025-09-10"]
    # los demás días conservan el ajuste calculado del paso (forma aplicada)
    sin_edicion = run_pipeline(_ctx(), ProveedorPlano())
    assert records[3].result.curves["2025-09-11"] == sin_edicion[3].result.curves["2025-09-11"]
    assert records[3].result.curves["2025-09-11"] != records[2].result.curves["2025-09-11"]
    # forma editada: no se exige reconciliar con nivel
    assert approval_blockers(records) == []


@pytest.mark.parametrize(
    "curvas",
    [
        {"2025-09-10": [900.0] * 23},
        {"2025-09-10": [900.0] * 23 + [-1.0]},
        {"2025-09-10": [900.0] * 23 + [float("nan")]},
        {"2025-09-05": [900.0] * 24},  # brecha: fuera de la semana objetivo
        {},
    ],
)
def test_paso_editado_invalido_falla(curvas):
    with pytest.raises(ValueError):
        run_pipeline(_ctx(), ProveedorPlano(), overrides={2: _record(2, "editado", curves=curvas)})


def test_base_editada_superpone_dias_al_pronostico():
    editada = {"2025-09-05": [700.0] * 24}  # la base sí puede editar la brecha
    records = run_pipeline(_ctx(), ProveedorPlano(), overrides={0: _record(0, "editado", curves=editada)})
    assert records[0].result.curves["2025-09-05"] == [700.0] * 24
    assert records[0].result.curves["2025-09-06"] == [1000.0] * 24
    assert len(records[0].result.curves) == 11
    assert final_curves(records)["2025-09-05"] == [700.0] * 24


def test_control_no_admite_edicion():
    with pytest.raises(ValueError):
        run_pipeline(_ctx(), ProveedorPlano(),
                     overrides={4: _record(4, "editado", curves={"2025-09-10": [1.0] * 24})})


# ---------------------------------------------------------------- rerun_from

def test_rerun_from_recalcula_posteriores_sin_tocar_anteriores():
    ctx = _ctx()
    prov = ProveedorPlano()
    records = run_pipeline(ctx, prov)
    nuevos = rerun_from(ctx, records, 2, prov, _record(2, "aplicado", {"alpha": 1.0}))
    assert len(prov.llamadas) == 1  # la base no se recalcula
    assert nuevos[0] is records[0] and nuevos[1] is records[1]
    assert nuevos[2].params["alpha"] == 1.0
    f = "2025-09-10"
    assert sum(nuevos[2].result.curves[f]) != pytest.approx(sum(records[2].result.curves[f]))
    assert sum(nuevos[3].result.curves[f]) == pytest.approx(sum(nuevos[2].result.curves[f]), abs=1e-6)
    assert nuevos[4].result.diagnostics["energia_semana_ajustada"] != pytest.approx(
        records[4].result.diagnostics["energia_semana_ajustada"]
    )
    # mismo resultado que correr todo de cero con esos params
    desde_cero = run_pipeline(_ctx(), ProveedorPlano(), params={"nivel": {"alpha": 1.0}})
    assert final_curves(desde_cero) == final_curves(nuevos)


def test_rerun_from_referencia_forzada_propaga_a_nivel():
    ctx = _ctx()
    prov = ProveedorPlano()
    records = run_pipeline(ctx, prov)
    nuevos = rerun_from(ctx, records, 1, prov, _record(1, "aplicado", {"semana_ref_inicio": "2023-09-11"}))
    assert ctx.semana_ref_inicio == date(2023, 9, 11)
    assert nuevos[2].result.diagnostics["semana_ref_inicio"] == "2023-09-11"
    assert nuevos[4].result.diagnostics["semana_ref_inicio"] == "2023-09-11"


def test_rerun_from_toma_referencia_de_records_con_ctx_nuevo():
    prov = ProveedorPlano()
    records = run_pipeline(_ctx(), prov, params={"referencia": {"semana_ref_inicio": "2023-09-11"}})
    ctx_nuevo = _ctx()  # p.ej. reconstruido desde el Store: sin semana_ref_inicio
    nuevos = rerun_from(ctx_nuevo, records, 3, prov, _record(3, "aplicado", {"fuente": "referencia"}))
    assert ctx_nuevo.semana_ref_inicio == date(2023, 9, 11)
    assert nuevos[3].result.diagnostics["dias"]["2025-09-10"]["fechas_fuente"] == ["2023-09-13"]


def test_rerun_from_conserva_edicion_posterior():
    ctx = _ctx()
    prov = ProveedorPlano()
    editada = {"2025-09-10": [900.0] * 24}
    records = run_pipeline(ctx, prov, overrides={3: _record(3, "editado", curves=editada)})
    nuevos = rerun_from(ctx, records, 2, prov, _record(2, "aplicado", {"alpha": 0.0}))
    assert nuevos[3].estado == StepState.EDITADO
    assert nuevos[3].result.curves["2025-09-10"] == [900.0] * 24
    # días no editados: forma recalculada sobre el nuevo nivel (reconcilia con él)
    assert sum(nuevos[3].result.curves["2025-09-11"]) == pytest.approx(
        sum(nuevos[2].result.curves["2025-09-11"]), abs=1e-6)
    assert nuevos[3].result.curves["2025-09-11"] != records[3].result.curves["2025-09-11"]


def test_rerun_from_valida_paso():
    ctx = _ctx()
    records = run_pipeline(ctx, ProveedorPlano())
    with pytest.raises(ValueError):
        rerun_from(ctx, records, 2, ProveedorPlano(), _record(3, "aplicado"))


# ---------------------------------------------------------------- H1: tendencia esperada desde E*

def _proveedor_real(festivos):
    """Base = energía/forma 'real' sintética (pronóstico perfecto)."""
    from tests.ajustes_sinteticos import energia_real, forma_tipo
    from src.adjustments.data import tipo_dia

    def prov(ucp, end_date, n_days):
        out = {}
        for i in range(1, n_days + 1):
            d = end_date + timedelta(days=i)
            out[iso(d)] = (forma_tipo(tipo_dia(d, festivos)) * energia_real(d, festivos)).tolist()
        return out

    return prov


def test_control_tendencia_ref_con_festivo_objetivo_sin_festivo_no_alerta():
    # 2025-05-12 no tiene festivos; su semana −364 (2024-05-13) arranca con lunes festivo.
    semana = date(2025, 5, 12)
    ctx = _ctx(semana)
    festivos = set(ctx.festivos)
    records = run_pipeline(ctx, _proveedor_real(festivos))
    assert ctx.semana_ref_inicio == date(2024, 5, 13) and "2024-05-13" in festivos
    ctrl = records[4].result
    assert not any(a.codigo.startswith("TENDENCIA_") for a in ctrl.alerts)
    assert abs(ctrl.diagnostics["diferencia_tendencia_pp"]) < 0.01
    assert ctrl.diagnostics["dias_sin_evidencia"] == []
    # la tendencia esperada sale de E*(d) del paso nivel (pasado por el pipeline)
    e_obj = sum(records[2].result.diagnostics["dias"][f]["E_objetivo"] for f in [iso(d) for d in dias_semana(semana)])
    assert ctrl.diagnostics["energia_semana_esperada"] == pytest.approx(e_obj, rel=1e-12)
    # la fórmula vieja (semana ref vs su semana anterior, en bruto) habría alertado
    from src.adjustments.data import energia_diaria
    reales = energia_diaria(ctx.historico)
    ref = dias_semana(ctx.semana_ref_inicio)
    t_vieja = sum(reales[iso(d)] for d in ref) / sum(reales[iso(d - timedelta(days=7))] for d in ref) - 1
    assert abs(t_vieja * 100 - ctrl.diagnostics["tendencia_resultante_pct"]) > 1.0


def test_control_con_nivel_omitido_recalcula_e_objetivo():
    semana = date(2025, 5, 12)
    ctx = _ctx(semana)
    records = run_pipeline(ctx, _proveedor_real(set(ctx.festivos)), overrides={2: _record(2, "omitido")})
    ctrl = records[4].result
    assert ctrl.diagnostics["tendencia_esperada_pct"] is not None
    assert abs(ctrl.diagnostics["diferencia_tendencia_pp"]) < 0.01
    # enclavamiento contra la energía del paso base (nivel omitido)
    assert ctrl.diagnostics["origen_energia_nivel"] == "base"
    assert ctrl.diagnostics["desvio_pp"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------- curvas_calculadas (R1-C)

def test_editado_guarda_curva_calculada_de_los_dias_editados():
    editada = {"2025-09-10": [900.0] * 24}
    records = run_pipeline(_ctx(), ProveedorPlano(), overrides={3: _record(3, "editado", curves=editada)})
    sin_edicion = run_pipeline(_ctx(), ProveedorPlano())
    res = records[3].result
    assert res.curves["2025-09-10"] == [900.0] * 24
    assert list(res.curvas_calculadas) == ["2025-09-10"]
    assert res.curvas_calculadas["2025-09-10"] == sin_edicion[3].result.curves["2025-09-10"]
    # aplicados/omitidos no traen curvas calculadas; ida y vuelta por dict
    assert sin_edicion[3].result.curvas_calculadas == {}
    vuelta = StepRecord.from_dict(json.loads(json.dumps(records[3].to_dict())))
    assert vuelta.result.curvas_calculadas == res.curvas_calculadas
    # from_dict tolera la ausencia del campo (records viejos)
    d = records[3].to_dict()
    del d["result"]["curvas_calculadas"]
    assert StepRecord.from_dict(d).result.curvas_calculadas == {}


def test_base_editada_guarda_calculada_y_energia_refleja_edicion():
    editada = {"2025-09-10": [700.0] * 24}
    records = run_pipeline(_ctx(), ProveedorPlano(), overrides={0: _record(0, "editado", curves=editada)})
    res = records[0].result
    assert res.curvas_calculadas == {"2025-09-10": [1000.0] * 24}
    assert res.diagnostics["energia"]["2025-09-10"] == pytest.approx(700.0 * 24)
    assert res.diagnostics["energia"]["2025-09-11"] == pytest.approx(1000.0 * 24)


def test_rerun_from_recalcula_curvas_calculadas_de_la_edicion():
    ctx = _ctx()
    prov = ProveedorPlano()
    records = run_pipeline(ctx, prov, overrides={3: _record(3, "editado", curves={"2025-09-10": [900.0] * 24})})
    nuevos = rerun_from(ctx, records, "nivel", prov, _record(2, "aplicado", {"alpha": 1.0}))
    assert nuevos[3].result.curves["2025-09-10"] == [900.0] * 24
    assert sum(nuevos[3].result.curvas_calculadas["2025-09-10"]) == pytest.approx(
        sum(nuevos[2].result.curves["2025-09-10"]), abs=1e-6)


# ---------------------------------------------------------------- control: enclavamiento

def test_control_sin_ediciones_no_alerta_y_desvio_cero():
    records = run_pipeline(_ctx(), ProveedorPlano())
    ctrl = records[4].result
    assert ctrl.diagnostics["origen_energia_nivel"] == "nivel"
    assert ctrl.diagnostics["desvio_pp"] == pytest.approx(0.0, abs=1e-9)
    assert not any(a.codigo == "TENDENCIA_FUERA_BANDA" for a in ctrl.alerts)


@pytest.mark.parametrize("orden", [2, 3])
def test_control_edicion_que_sube_un_dia_10pct_alerta(orden):
    sin_edicion = run_pipeline(_ctx(), ProveedorPlano())
    f = "2025-09-10"
    editada = {f: [v * 1.10 for v in sin_edicion[orden].result.curves[f]]}
    records = run_pipeline(_ctx(), ProveedorPlano(), overrides={orden: _record(orden, "editado", curves=editada)})
    ctrl = records[4].result
    alerta = next(a for a in ctrl.alerts if a.codigo == "TENDENCIA_FUERA_BANDA")
    assert alerta.fechas[0] == f
    e_nivel = sum(sum(sin_edicion[2].result.curves[x]) for x in FECHAS_SEMANA)
    esperado = 0.10 * sum(sin_edicion[2].result.curves[f]) / e_nivel * 100
    assert ctrl.diagnostics["desvio_pp"] == pytest.approx(esperado, rel=1e-6)
    assert approval_blockers(records) == []  # es warning, no bloqueo


# ---------------------------------------------------------------- curvas no finitas

def test_base_con_nan_bloquea_sin_excepcion_y_es_json_safe():
    class ProveedorConNan(ProveedorPlano):
        def __call__(self, ucp, end_date, n_days):
            out = super().__call__(ucp, end_date, n_days)
            out["2025-09-10"][3] = float("nan")
            out["2025-09-11"][5] = float("inf")
            return out

    ctx = _ctx()
    prov = ProveedorConNan()
    records = run_pipeline(ctx, prov)
    codigos = {a.codigo: a.fechas for a in approval_blockers(records)}
    assert codigos["VALORES_NO_FINITOS"] == ["2025-09-10", "2025-09-11"]
    payload = json.dumps([r.to_dict() for r in records], allow_nan=False)
    vuelta = [StepRecord.from_dict(d) for d in json.loads(payload)]
    assert vuelta[0].result.curves["2025-09-10"][3] is None
    assert vuelta[0].result.curves["2025-09-11"][5] is None
    # recalcular desde records cargados (con None) tampoco lanza y sigue bloqueando
    nuevos = rerun_from(ctx, vuelta, 2, prov, _record(2, "aplicado", {"alpha": 0.5}))
    assert "VALORES_NO_FINITOS" in {a.codigo for a in approval_blockers(nuevos)}
    json.dumps([r.to_dict() for r in nuevos], allow_nan=False)


def test_validar_curvas_editadas():
    permitidas = FECHAS_SEMANA
    validar_curvas_editadas({"2025-09-10": [1.0] * 24}, permitidas)
    for curvas in [
        {},
        {"2025-09-05": [1.0] * 24},
        {"2025-09-10": [1.0] * 23},
        {"2025-09-10": [None] + [1.0] * 23},
        {"2025-09-10": ["x"] + [1.0] * 23},
        {"2025-09-10": [float("inf")] + [1.0] * 23},
        {"2025-09-10": [-1.0] + [1.0] * 23},
        {"2025-09-10": None},
        {"2025-09-10": 5},
    ]:
        with pytest.raises(ValueError):
            validar_curvas_editadas(curvas, permitidas)
    assert validar_curva([1.0] * 24) is None
    assert validar_curva(None) == "PERIODOS_FALTANTES"
    assert validar_curva([[1.0] * 24]) == "PERIODOS_FALTANTES"


# ---------------------------------------------------------------- params inválidos

@pytest.mark.parametrize(
    "params",
    [
        {"nivel": {"alpha": None}},
        {"nivel": {"alpha": float("inf")}},
        {"nivel": {"alpha": 2.0}},
        {"referencia": {"anios_excluidos": 2020}},
        {"forma": {"fuente": "desconocida"}},
        {"control": {"banda_pp": None}},
        {"paso_x": {}},
    ],
)
def test_params_invalidos_value_error_en_run_y_rerun(params):
    with pytest.raises(ValueError):
        run_pipeline(_ctx(), ProveedorPlano(), params=params)
    paso, p = next(iter(params.items()))
    if paso in STEP_ORDER:
        ctx = _ctx()
        records = run_pipeline(ctx, ProveedorPlano())
        with pytest.raises(ValueError):
            rerun_from(ctx, records, paso, ProveedorPlano(), _record(orden_de(paso), "aplicado", p))


# ---------------------------------------------------------------- identificación por nombre

def test_orden_de():
    assert [orden_de(p) for p in STEP_ORDER] == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        orden_de("dias_especiales")


def test_rerun_from_acepta_nombre_de_paso():
    ctx = _ctx()
    records = run_pipeline(ctx, ProveedorPlano())
    por_nombre = rerun_from(ctx, records, "nivel", ProveedorPlano(), _record(2, "aplicado", {"alpha": 1.0}))
    por_orden = rerun_from(_ctx(), records, 2, ProveedorPlano(), _record(2, "aplicado", {"alpha": 1.0}))
    assert final_curves(por_nombre) == final_curves(por_orden)
    with pytest.raises(ValueError):
        rerun_from(ctx, records, "forma", ProveedorPlano(), _record(2, "aplicado"))


def _desalinear(records, como):
    rs = [StepRecord.from_dict(r.to_dict()) for r in records]
    if como == "nombres_cruzados":
        rs[2].paso, rs[3].paso = rs[3].paso, rs[2].paso
    elif como == "falta_paso":
        rs = rs[:4]
    elif como == "paso_repetido":
        rs[3] = StepRecord.from_dict(rs[2].to_dict())
    elif como == "paso_desconocido":
        rs[3].paso = "dias_especiales"
    elif como == "orden_incoherente":
        rs[2].orden, rs[3].orden = 3, 2
    return rs


@pytest.mark.parametrize(
    "como", ["nombres_cruzados", "falta_paso", "paso_repetido", "paso_desconocido", "orden_incoherente"]
)
def test_rerun_from_records_desalineados_value_error(como):
    ctx = _ctx()
    records = _desalinear(run_pipeline(ctx, ProveedorPlano()), como)
    with pytest.raises(ValueError):
        rerun_from(ctx, records, "control", ProveedorPlano(), _record(4, "aplicado"))


def test_run_pipeline_override_con_paso_que_no_coincide_value_error():
    with pytest.raises(ValueError):
        run_pipeline(_ctx(), ProveedorPlano(), overrides={2: _record(3, "omitido")})
    # por nombre también vale
    records = run_pipeline(_ctx(), ProveedorPlano(), overrides={"nivel": _record(2, "omitido")})
    assert records[2].estado == StepState.OMITIDO


# ---------------------------------------------------------------- bloqueos

def test_bloqueo_por_dia_faltante_incluso_con_control_omitido():
    prov = ProveedorPlano(faltar=["2025-09-12"])
    records = run_pipeline(_ctx(), prov)
    codigos = {a.codigo for a in approval_blockers(records)}
    assert "PERIODOS_FALTANTES" in codigos

    records_sin_control = run_pipeline(_ctx(), prov, overrides={4: _record(4, "omitido")})
    bloqueos = approval_blockers(records_sin_control)
    assert [a.codigo for a in bloqueos] == ["PERIODOS_FALTANTES"]
    assert bloqueos[0].fechas == ["2025-09-12"]


def test_approval_blockers_sin_records():
    assert [a.codigo for a in approval_blockers([])] == ["SIN_PASOS"]


def test_alert_ida_y_vuelta():
    a = Alert("warning", "X", "m", ["2025-01-01"], {"v": np.float64(1.5)})
    assert Alert.from_dict(json.loads(json.dumps(a.to_dict()))) == Alert("warning", "X", "m", ["2025-01-01"], {"v": 1.5})


# ---------------------------------------------------------------- humo con CSV real

CSV_ANTIOQUIA = Path(__file__).resolve().parents[1] / "data" / "raw" / "Antioquia" / "datos.csv"


@pytest.mark.skipif(not CSV_ANTIOQUIA.exists(), reason="sin data/raw/Antioquia/datos.csv")
def test_humo_csv_real_antioquia():
    """
    Semana objetivo pedida: 2026-11-02 (lunes festivo). Si el CSV no llega hasta
    ahí (hoy termina en 2026-07-04), se usa la última semana con datos del lunes
    al sábado: 2026-06-29, que también arranca con festivo (San Pedro y San Pablo).
    """
    from src.adjustments.data import cargar_historico

    hist = cargar_historico("Antioquia")
    ultimo = hist["fecha"].max().date()
    semana = date(2026, 11, 2)
    if ultimo < semana - timedelta(days=2):  # sin datos hasta esa fecha → última semana disponible
        semana = (ultimo - timedelta(days=ultimo.weekday()))
    ctx = build_context("Antioquia", semana, fecha_emision=semana - timedelta(days=2), historico=hist)
    reales = curvas_reales(ctx.historico)

    def copia_semana_previa(ucp, end_date, n_days):
        out = {}
        for i in range(1, n_days + 1):
            d = end_date + timedelta(days=i)
            src = d - timedelta(days=7)
            while iso(src) not in reales:
                src -= timedelta(days=7)
            out[iso(d)] = list(reales[iso(src)])
        return out

    records = run_pipeline(ctx, copia_semana_previa)
    assert [r.paso for r in records] == STEP_ORDER
    assert approval_blockers(records) == []
    finales = final_curves(records)
    for d in dias_semana(semana):
        assert len(finales[iso(d)]) == 24
    festivos_semana = records[1].result.diagnostics["festivos_objetivo"]
    for f in festivos_semana:
        assert records[3].result.diagnostics["dias"][f]["fuente"].startswith("festivo_")
    json.dumps([r.to_dict() for r in records])


# ---------------------------------------------------------------- control: coherencia con la referencia (fix ronda 2)

SEMANA_TENDENCIA = date(2025, 5, 12)


def _proveedor_escalado(festivos, factor, desde):
    """Base 'perfecta' con los días >= `desde` multiplicados por `factor`."""
    real = _proveedor_real(festivos)

    def prov(ucp, end_date, n_days):
        out = real(ucp, end_date, n_days)
        return {f: ([v * factor for v in c] if f >= iso(desde) else c) for f, c in out.items()}

    return prov


def _alerta(records, codigo):
    return next((a for a in records[4].result.alerts if a.codigo == codigo), None)


def test_control_tendencia_referencia_default_info_con_alpha_cero():
    ctx = _ctx(SEMANA_TENDENCIA)
    prov = _proveedor_escalado(set(ctx.festivos), 1.10, SEMANA_TENDENCIA)
    records = run_pipeline(ctx, prov, params={"nivel": {"alpha": 0.0}})
    ctrl = records[4].result
    alerta = _alerta(records, "TENDENCIA_REFERENCIA_FUERA_BANDA")
    assert alerta is not None and alerta.nivel == "info"
    assert ctrl.diagnostics["banda_referencia_pp"] == 2.0
    assert ctrl.diagnostics["diferencia_tendencia_pp"] > 2.0
    assert alerta.detalle["diferencia_tendencia_pp"] == pytest.approx(ctrl.diagnostics["diferencia_tendencia_pp"])
    # el enclavamiento no lo ve (no hubo ediciones después de nivel)
    assert _alerta(records, "TENDENCIA_FUERA_BANDA") is None
    assert ctrl.diagnostics["desvio_pp"] == pytest.approx(0.0, abs=1e-9)
    assert approval_blockers(records) == []


def test_control_tendencia_referencia_con_nivel_omitido():
    ctx = _ctx(SEMANA_TENDENCIA)
    prov = _proveedor_escalado(set(ctx.festivos), 0.90, SEMANA_TENDENCIA)
    records = run_pipeline(ctx, prov, overrides={2: _record(2, "omitido")})
    alerta = _alerta(records, "TENDENCIA_REFERENCIA_FUERA_BANDA")
    assert alerta is not None and alerta.nivel == "info"
    assert records[4].result.diagnostics["diferencia_tendencia_pp"] < -2.0
    assert records[4].result.diagnostics["origen_energia_nivel"] == "base"
    assert _alerta(records, "TENDENCIA_FUERA_BANDA") is None


def test_control_tendencia_referencia_con_base_editada():
    ctx = _ctx(SEMANA_TENDENCIA)
    festivos = set(ctx.festivos)
    sin_edicion = run_pipeline(_ctx(SEMANA_TENDENCIA), _proveedor_real(festivos))
    assert _alerta(sin_edicion, "TENDENCIA_REFERENCIA_FUERA_BANDA") is None
    fechas = [iso(d) for d in dias_semana(SEMANA_TENDENCIA)]
    editada = {f: [v * 1.10 for v in sin_edicion[0].result.curves[f]] for f in fechas}
    records = run_pipeline(ctx, _proveedor_real(festivos),
                           overrides={0: _record(0, "editado", curves=editada)})
    alerta = _alerta(records, "TENDENCIA_REFERENCIA_FUERA_BANDA")
    assert alerta is not None and alerta.nivel == "info"
    # α=0.25 devuelve un cuarto hacia E*: queda ≈ +7.5 pp sobre la referencia
    assert records[4].result.diagnostics["diferencia_tendencia_pp"] > 2.0
    assert _alerta(records, "TENDENCIA_FUERA_BANDA") is None


def test_control_tendencia_referencia_null_la_desactiva():
    ctx = _ctx(SEMANA_TENDENCIA)
    prov = _proveedor_escalado(set(ctx.festivos), 1.10, SEMANA_TENDENCIA)
    records = run_pipeline(ctx, prov, params={"nivel": {"alpha": 0.0},
                                              "control": {"banda_referencia_pp": None}})
    assert _alerta(records, "TENDENCIA_REFERENCIA_FUERA_BANDA") is None
    assert records[4].result.diagnostics["diferencia_tendencia_pp"] > 2.0  # el diagnóstico sigue


# ---------------------------------------------------------------- brecha inválida (fix ronda 2)

def test_brecha_invalida_warning_no_bloquea():
    class ProveedorBrechaRota(ProveedorPlano):
        def __call__(self, ucp, end_date, n_days):
            out = super().__call__(ucp, end_date, n_days)
            out["2025-09-05"][0] = float("nan")
            del out["2025-09-06"]
            return out

    ctx = _ctx()
    records = run_pipeline(ctx, ProveedorBrechaRota())
    alerta = _alerta(records, "BRECHA_INVALIDA")
    assert alerta is not None and alerta.nivel == "warning"
    assert alerta.fechas == ["2025-09-05", "2025-09-06"]
    assert alerta.detalle["motivos"] == {"2025-09-05": "VALORES_NO_FINITOS", "2025-09-06": "PERIODOS_FALTANTES"}
    assert records[4].result.diagnostics["dias_brecha"] == ["2025-09-04", "2025-09-05", "2025-09-06", "2025-09-07"]
    # approval_blockers solo cubre la semana objetivo
    assert approval_blockers(records) == []
    json.dumps([r.to_dict() for r in records], allow_nan=False)


def test_brecha_valida_sin_alerta():
    records = run_pipeline(_ctx(), ProveedorPlano())
    assert _alerta(records, "BRECHA_INVALIDA") is None


def test_referencia_forzada_posterior_value_error_en_run_pipeline():
    with pytest.raises(ValueError):
        run_pipeline(_ctx(), ProveedorPlano(),
                     params={"referencia": {"semana_ref_inicio": iso(SEMANA + timedelta(days=14))}})
