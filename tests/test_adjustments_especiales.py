"""Tests del paso `especiales`: detección, sugerencias, aplicación y migración de 5 pasos."""

from dataclasses import replace
from datetime import date, timedelta

import pytest

from src.adjustments import analogos as buscador
from src.adjustments.data import PERIODOS, dias_semana, iso
from src.adjustments.pipeline import (
    STEP_ORDER,
    approval_blockers,
    build_context,
    final_curves,
    rerun_from,
    run_pipeline,
)
from src.adjustments.steps import especiales
from src.adjustments.types import StepRecord, StepResult, StepState
from tests.ajustes_sinteticos import ProveedorPlano, historico_sintetico

HIST = historico_sintetico()
SEMANA_SANTA = date(2025, 4, 7)  # contiene el Domingo de Ramos 2025-04-13


def _ctx(semana, hist=HIST):
    return build_context(
        "Sintetica", semana, fecha_emision=semana - timedelta(days=4), historico=hist
    )


def _record(orden, estado, params=None, curves=None):
    return StepRecord(orden, STEP_ORDER[orden], StepState(estado), params or {}, StepResult(curves or {}))


def _prev_semana(ctx):
    return {iso(d): [1000.0] * 24 for d in dias_semana(ctx.semana_inicio)}


def _hist_semana_santa(factor=1.5):
    """Histórico sintético con las Semanas Santas 2021-2024 multiplicadas por `factor`."""
    hist = HIST.copy()
    fechas = []
    d = date(2021, 1, 1)
    while d <= date(2024, 12, 31):
        if buscador.es_semana_santa(d):
            fechas.append(d.isoformat())
        d += timedelta(days=1)
    hist.loc[hist["FECHA"].isin(fechas), PERIODOS] *= factor
    return hist


# ---------------------------------------------------------------- params

def test_validar_params_defaults_y_validos():
    assert especiales.validar_params(None) == {"aplicar": False, "alpha": 0.25, "ancla": "actual"}
    assert especiales.validar_params({"aplicar": True}) == {
        "aplicar": True, "alpha": 0.25, "ancla": "actual"}
    assert especiales.validar_params({"ancla": "analogos"}) == {
        "aplicar": False, "alpha": 0.25, "ancla": "analogos"}


@pytest.mark.parametrize(
    "params",
    [
        {"alpha": -0.1},
        {"alpha": 1.1},
        {"alpha": None},
        {"alpha": "0.5"},
        {"ancla": "otra"},
        {"aplicar": 1},
        {"aplicar": "true"},
        {"clave_desconocida": 1},
    ],
)
def test_validar_params_invalidos(params):
    with pytest.raises(ValueError):
        especiales.validar_params(params)


def test_run_pipeline_params_invalidos_especiales():
    with pytest.raises(ValueError):
        run_pipeline(_ctx(SEMANA_SANTA), ProveedorPlano(), params={"especiales": {"alpha": 2.0}})


# ---------------------------------------------------------------- detección

@pytest.mark.parametrize(
    "semana, motivo, fecha",
    [
        (SEMANA_SANTA, "semana_santa", "2025-04-13"),
        (date(2024, 12, 23), "24_dic", "2024-12-24"),
        (date(2024, 12, 30), "31_dic", "2024-12-31"),
        (date(2025, 11, 17), "post_festivo", "2025-11-18"),
    ],
)
def test_deteccion_de_los_cuatro_motivos(semana, motivo, fecha):
    ctx = _ctx(semana)
    result = especiales.step(ctx, _prev_semana(ctx), {})
    dias = result.diagnostics["dias"]
    assert dias[fecha]["motivo"] == motivo
    assert fecha in result.diagnostics["dias_especiales"]
    assert motivo in result.diagnostics["motivos"]


def test_post_festivo_no_marca_el_festivo_ni_el_domingo():
    ctx = _ctx(date(2025, 11, 17))
    result = especiales.step(ctx, _prev_semana(ctx), {})
    dias = result.diagnostics["dias"]
    assert "2025-11-17" not in dias  # lunes festivo
    assert dias["2025-11-18"]["motivo"] == "post_festivo"
    desc = buscador.descriptor_dia(date(2025, 11, 18), set(ctx.festivos))
    assert desc["relacion_festivo"] == "despues"
    filtros = especiales._filtros_dia("post_festivo", {}, desc)
    assert filtros["relacion_festivo"] == "despues"
    assert filtros["distancia_festivo"] == 1


def _ctx_con_festivos(semana, extra):
    ctx = _ctx(semana)
    ctx.festivos = sorted(set(ctx.festivos) | set(extra))
    return ctx


def test_post_festivo_viernes_tras_jueves_festivo_usa_puente():
    ctx = _ctx_con_festivos(date(2025, 6, 16), {"2025-06-19"})
    desc = buscador.descriptor_dia(date(2025, 6, 20), set(ctx.festivos))
    assert desc["relacion_festivo"] == "puente"
    result = especiales.step(ctx, _prev_semana(ctx), {})
    assert result.diagnostics["dias"]["2025-06-20"]["motivo"] == "post_festivo"
    filtros = especiales._filtros_dia("post_festivo", {}, desc)
    assert filtros["relacion_festivo"] == "puente"
    assert "distancia_festivo" not in filtros


def test_post_festivo_sabado_tras_viernes_festivo_usa_puente():
    ctx = _ctx_con_festivos(date(2025, 6, 16), {"2025-06-20"})
    desc = buscador.descriptor_dia(date(2025, 6, 21), set(ctx.festivos))
    assert desc["relacion_festivo"] == "puente"
    result = especiales.step(ctx, _prev_semana(ctx), {})
    assert result.diagnostics["dias"]["2025-06-21"]["motivo"] == "post_festivo"
    filtros = especiales._filtros_dia("post_festivo", {}, desc)
    assert filtros["relacion_festivo"] == "puente"
    assert "distancia_festivo" not in filtros


# ---------------------------------------------------------------- default sin efecto

def test_default_no_cambia_curvas_y_sugiere_dias():
    ctx = _ctx(SEMANA_SANTA)
    records = run_pipeline(ctx, ProveedorPlano())
    sin_especiales = run_pipeline(
        _ctx(SEMANA_SANTA), ProveedorPlano(), overrides={"especiales": _record(4, "omitido")}
    )
    assert [r.paso for r in records] == STEP_ORDER
    assert final_curves(records) == final_curves(sin_especiales)
    assert final_curves(records) == records[3].result.curves  # identidad sobre forma

    esp = records[4]
    assert esp.paso == "especiales"
    assert esp.params == {"aplicar": False, "alpha": 0.25, "ancla": "actual"}
    dia = esp.result.diagnostics["dias"]["2025-04-13"]
    assert dia["motivo"] == "semana_santa"
    assert dia["aplicado"] is False
    assert dia["E_final"] == pytest.approx(dia["E_previa"])
    assert esp.result.diagnostics["dias_ajustados"] == []
    alerta = next(a for a in esp.result.alerts if a.codigo == "ESPECIALES_SUGERIDOS")
    assert alerta.nivel == "info" and "2025-04-13" in alerta.fechas


# ---------------------------------------------------------------- aplicar

def test_aplicar_solo_cambia_los_dias_especiales():
    hist = _hist_semana_santa()
    base = run_pipeline(_ctx(SEMANA_SANTA, hist), ProveedorPlano())
    aplicado = run_pipeline(
        _ctx(SEMANA_SANTA, hist), ProveedorPlano(),
        params={"especiales": {"aplicar": True, "ancla": "analogos"}},
    )
    esp = aplicado[4]
    assert esp.result.diagnostics["dias_ajustados"] == ["2025-04-13"]

    for d in dias_semana(SEMANA_SANTA):
        f = iso(d)
        if f == "2025-04-13":
            assert esp.result.curves[f] != base[4].result.curves[f]
        else:
            assert esp.result.curves[f] == base[4].result.curves[f]

    dia = esp.result.diagnostics["dias"]["2025-04-13"]
    assert dia["aplicado"] is True
    assert dia["E_objetivo"] > dia["E_previa"]
    assert dia["E_final"] == pytest.approx(
        (1 - dia["alpha"]) * dia["E_previa"] + dia["alpha"] * dia["E_objetivo"])
    assert dia["formas_fuente"]
    assert esp.result.diagnostics["alpha"] == 0.25


def test_aplicar_con_analogos_del_analista():
    hist = _hist_semana_santa()
    mapa = {"2025-04-13": ["2024-03-24", "2023-04-02"]}
    records = run_pipeline(
        _ctx(SEMANA_SANTA, hist), ProveedorPlano(),
        params={"especiales": {"aplicar": True, "ancla": "analogos", "analogos": mapa}},
    )
    dia = records[4].result.diagnostics["dias"]["2025-04-13"]
    assert dia["seleccion"] == "analista"
    assert dia["fechas_fuente"] == ["2024-03-24", "2023-04-02"]
    assert dia["aplicado"] is True


def test_aplicar_no_genera_reconciliacion_rota():
    records = run_pipeline(
        _ctx(SEMANA_SANTA, _hist_semana_santa()), ProveedorPlano(),
        params={"especiales": {"aplicar": True}},
    )
    assert records[4].result.diagnostics["dias_ajustados"] == ["2025-04-13"]
    assert not any(a.codigo == "RECONCILIACION_ROTA" for a in records[5].result.alerts)
    assert approval_blockers(records) == []


def test_aplicar_con_ancla_actual_conserva_la_forma_previa():
    ctx = _ctx(SEMANA_SANTA)
    ctx.semana_ref_inicio = date(2024, 4, 8)
    prev = _prev_semana(ctx)
    result = especiales.step(ctx, prev, {"aplicar": True, "ancla": "actual"})
    f = "2025-04-13"
    dia = result.diagnostics["dias"][f]
    assert dia["aplicado"] is True
    ratio = dia["E_final"] / dia["E_previa"]
    assert result.curves[f] == pytest.approx([1000.0 * ratio] * 24)


def test_alpha_cero_no_cambia_curvas_ni_forma():
    hist = _hist_semana_santa()
    records = run_pipeline(
        _ctx(SEMANA_SANTA, hist), ProveedorPlano(),
        params={"especiales": {"aplicar": True, "alpha": 0.0, "ancla": "analogos"}},
    )
    assert records[4].result.curves == records[3].result.curves
    dia = records[4].result.diagnostics["dias"]["2025-04-13"]
    assert dia["aplicado"] is False
    assert dia["alpha_efectivo"] == 0.0
    assert dia["E_final"] == pytest.approx(dia["E_previa"])
    assert records[4].result.diagnostics["dias_ajustados"] == []


@pytest.mark.parametrize("orden", [2, 3])
def test_especiales_preserva_edicion_manual_previa(orden):
    editada = {"2025-04-13": [1234.0] * 24}
    records = run_pipeline(
        _ctx(SEMANA_SANTA, _hist_semana_santa()), ProveedorPlano(),
        params={"especiales": {"aplicar": True, "ancla": "analogos"}},
        overrides={orden: _record(orden, "editado", curves=editada)},
    )
    esp = records[4]
    f = "2025-04-13"
    assert esp.result.curves[f] == records[3].result.curves[f]
    assert esp.result.diagnostics["dias_preservados"] == [f]
    assert esp.result.diagnostics["dias_ajustados"] == []
    assert esp.result.diagnostics["dias"][f]["regla"] == "edicion_preservada"
    assert any(a.codigo == "ESPECIALES_EDICION_PRESERVADA" for a in esp.result.alerts)


def test_respeta_limite_de_filtros_del_analista():
    records = run_pipeline(
        _ctx(SEMANA_SANTA, _hist_semana_santa()), ProveedorPlano(),
        params={"especiales": {"aplicar": True, "ancla": "analogos",
                               "filtros_analogos": {"limite": 2}}},
    )
    dia = records[4].result.diagnostics["dias"]["2025-04-13"]
    assert records[4].result.diagnostics["filtros_analogos"]["limite"] == 2
    assert 0 < len(dia["analogos"]) <= 2


# ---------------------------------------------------------------- elecciones IA

def test_elecciones_ia_solo_diagnostico_sin_tocar_curvas():
    ctx = _ctx(date(2025, 9, 8))
    ctx.extras["eventos"] = {
        "2025-09-10": "Elecciones regionales y votación",
        "2025-09-11": "Concierto masivo",
    }
    base = run_pipeline(_ctx(date(2025, 9, 8)), ProveedorPlano())
    records = run_pipeline(ctx, ProveedorPlano(), params={"especiales": {"aplicar": True}})
    esp = records[4]
    assert esp.result.diagnostics["elecciones_ia"] == ["2025-09-10"]
    assert esp.result.diagnostics["dias"]["2025-09-10"]["motivo"] == "elecciones_ia"
    assert "2025-09-10" not in esp.result.diagnostics["dias_ajustados"]
    assert esp.result.curves == base[4].result.curves
    alerta = next(a for a in esp.result.alerts if a.codigo == "ELECCIONES_IA")
    assert alerta.fechas == ["2025-09-10"]
    assert alerta.detalle["verificado"] is False
    assert alerta.detalle["eventos"]["2025-09-10"] == "Elecciones regionales y votación"


def test_sin_eventos_ia_no_hay_elecciones():
    result = especiales.step(_ctx(date(2025, 9, 8)), _prev_semana(_ctx(date(2025, 9, 8))), {})
    assert result.diagnostics["elecciones_ia"] == []
    assert not any(a.codigo == "ELECCIONES_IA" for a in result.alerts)


# ---------------------------------------------------------------- overrides por nombre

def test_overrides_y_params_por_nombre_especiales():
    records = run_pipeline(
        _ctx(SEMANA_SANTA), ProveedorPlano(),
        params={"especiales": {"aplicar": True, "alpha": 0.7}},
    )
    assert records[4].paso == "especiales"
    assert records[4].params == {"aplicar": True, "alpha": 0.7, "ancla": "actual"}
    assert records[4].result.diagnostics["alpha"] == 0.7

    omitido = run_pipeline(
        _ctx(SEMANA_SANTA), ProveedorPlano(), overrides={"especiales": _record(4, "omitido")}
    )
    assert omitido[4].estado is StepState.OMITIDO
    assert omitido[4].result.diagnostics == {"omitido": True}


# ---------------------------------------------------------------- migración de 5 pasos

def _records_5_pasos(semana=SEMANA_SANTA):
    records = run_pipeline(
        _ctx(semana), ProveedorPlano(), overrides={"especiales": _record(4, "omitido")}
    )
    return [
        records[0],
        records[1],
        records[2],
        records[3],
        replace(records[5], orden=4),
    ]


def test_migracion_5_pasos_desde_control_conserva_el_record_migrado():
    semana = SEMANA_SANTA
    viejos = _records_5_pasos(semana)
    assert [r.paso for r in viejos] == ["base", "referencia", "nivel", "forma", "control"]
    nuevos = rerun_from(_ctx(semana), viejos, "control", ProveedorPlano(), _record(5, "aplicado"))
    assert [r.paso for r in nuevos] == STEP_ORDER
    assert [r.orden for r in nuevos] == [0, 1, 2, 3, 4, 5]
    esp = nuevos[4]
    assert esp.estado is StepState.OMITIDO
    assert esp.result.diagnostics == {"omitido": True, "migrado": True}
    assert esp.result.curves == nuevos[3].result.curves


def test_migracion_5_pasos_desde_nivel_devuelve_seis():
    semana = SEMANA_SANTA
    viejos = _records_5_pasos(semana)
    nuevos = rerun_from(
        _ctx(semana), viejos, "nivel", ProveedorPlano(), _record(2, "aplicado", {"alpha": 0.25})
    )
    assert [r.paso for r in nuevos] == STEP_ORDER
    assert nuevos[4].paso == "especiales"
    assert nuevos[4].estado is StepState.OMITIDO
    assert nuevos[5].paso == "control" and nuevos[5].orden == 5
