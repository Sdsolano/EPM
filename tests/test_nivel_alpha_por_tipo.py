"""Tests del param opcional `alpha_por_tipo` del paso nivel (datos sintéticos, sin red)."""

import json
from datetime import date, timedelta

import pytest

from src.adjustments.data import dias_semana, festivos_colombia, iso, tipo_dia
from src.adjustments.pipeline import build_context, run_pipeline
from src.adjustments.steps import base, nivel, referencia
from src.adjustments.types import StepRecord
from tests.ajustes_sinteticos import ProveedorPlano, historico_sintetico

HIST = historico_sintetico()
FESTIVOS = set(festivos_colombia(range(2014, 2027)))
SEMANA_NORMAL = date(2025, 9, 8)  # sin festivos
SEMANA_SANTA = date(2025, 4, 14)  # jueves y viernes santo festivos


def _ctx(semana):
    return build_context("Sintetica", semana, fecha_emision=semana - timedelta(days=4), historico=HIST)


def _correr_nivel(semana, params):
    ctx = _ctx(semana)
    prev = base.step(ctx, {}, {}, base_provider=ProveedorPlano(1000.0)).curves
    ref = referencia.step(ctx, prev, {})
    ctx.semana_ref_inicio = date.fromisoformat(ref.diagnostics["semana_ref_inicio"])
    return ctx, ref.curves, nivel.step(ctx, ref.curves, params)


@pytest.mark.parametrize("semana", [SEMANA_NORMAL, SEMANA_SANTA])
def test_sin_alpha_por_tipo_identico(semana):
    _, _, sin = _correr_nivel(semana, {"alpha": 0.4})
    _, _, con_none = _correr_nivel(semana, {"alpha": 0.4, "alpha_por_tipo": None})
    assert con_none.curves == sin.curves
    assert con_none.diagnostics == sin.diagnostics
    assert "alpha_por_tipo" not in sin.diagnostics
    assert all("tipo_dia" not in d for d in sin.diagnostics["dias"].values())


def test_validar_params_sin_clave_no_cambia_normalizados():
    assert nivel.validar_params(None) == {"alpha": 0.25}
    assert nivel.validar_params({"alpha": 0.3}) == {"alpha": 0.3}
    assert nivel.validar_params({"alpha_por_tipo": None}) == {"alpha": 0.25}
    assert "alpha_por_tipo" not in nivel.DEFAULTS


@pytest.mark.parametrize("semana", [SEMANA_NORMAL, SEMANA_SANTA])
def test_todas_las_claves_iguales_a_alpha_identico(semana):
    _, _, sin = _correr_nivel(semana, {"alpha": 0.4})
    todas = {t: 0.4 for t in ("laboral", "sabado", "domingo", "festivo")}
    _, _, con = _correr_nivel(semana, {"alpha": 0.4, "alpha_por_tipo": todas})
    assert con.curves == sin.curves
    for f, d in con.diagnostics["dias"].items():
        assert d["alpha"] == sin.diagnostics["dias"][f]["alpha"]
        assert d["E_final"] == sin.diagnostics["dias"][f]["E_final"]
    assert con.diagnostics["alpha_por_tipo"] == todas


def test_clave_parcial_festivo_cambia_solo_festivos():
    _, prev, sin = _correr_nivel(SEMANA_SANTA, {"alpha": 0.4})
    ctx, _, con = _correr_nivel(SEMANA_SANTA, {"alpha": 0.4, "alpha_por_tipo": {"festivo": 1.0}})
    festivos_semana = []
    for d in dias_semana(ctx.semana_inicio):
        f = iso(d)
        dia = con.diagnostics["dias"][f]
        assert dia["tipo_dia"] == tipo_dia(d, ctx.festivos)
        if dia["tipo_dia"] == "festivo":
            festivos_semana.append(f)
            assert dia["alpha"] == 1.0
            assert dia["E_final"] == pytest.approx(dia["E_objetivo"], rel=1e-12)
            assert con.curves[f] != sin.curves[f]
        else:
            assert dia["alpha"] == sin.diagnostics["dias"][f]["alpha"]
            assert con.curves[f] == sin.curves[f]
    assert festivos_semana == ["2025-04-17", "2025-04-18"]
    # la brecha no se toca
    for f in set(prev) - {iso(d) for d in dias_semana(ctx.semana_inicio)}:
        assert con.curves[f] == prev[f]


def test_sin_evidencia_sigue_alpha_cero_aunque_haya_alpha_por_tipo():
    ctx = _ctx(SEMANA_NORMAL)
    prev = base.step(ctx, {}, {}, base_provider=ProveedorPlano(1000.0)).curves
    ctx.semana_ref_inicio = None
    res = nivel.step(ctx, prev, {"alpha": 0.5, "alpha_por_tipo": {"laboral": 1.0, "sabado": 1.0, "domingo": 1.0}})
    assert res.curves == prev
    assert all(d["alpha"] == 0.0 for d in res.diagnostics["dias"].values())


@pytest.mark.parametrize(
    "valor",
    [
        {"lunes": 0.5},
        {"Laboral": 0.5},
        {"festivo": 1.5},
        {"festivo": -0.1},
        {"festivo": float("nan")},
        {"festivo": float("inf")},
        {"festivo": None},
        {"festivo": True},
        {"festivo": "0.5"},
        [0.5],
        0.5,
        "festivo",
    ],
)
def test_validacion_alpha_por_tipo(valor):
    with pytest.raises(ValueError):
        nivel.validar_params({"alpha_por_tipo": valor})
    with pytest.raises(ValueError):
        nivel.step(_ctx(SEMANA_NORMAL), {}, {"alpha_por_tipo": valor})


def test_validacion_normaliza_a_float_y_conserva_otras_validaciones():
    p = nivel.validar_params({"alpha": 0.3, "alpha_por_tipo": {"sabado": 1, "festivo": 0}})
    assert p == {"alpha": 0.3, "alpha_por_tipo": {"sabado": 1.0, "festivo": 0.0}}
    assert all(isinstance(v, float) for v in p["alpha_por_tipo"].values())
    with pytest.raises(ValueError):
        nivel.validar_params({"alpha": 1.5, "alpha_por_tipo": {"festivo": 0.5}})
    with pytest.raises(ValueError):
        nivel.validar_params({"alfa": 0.3})


def test_params_persisten_json_ida_y_vuelta_en_pipeline():
    ctx = _ctx(SEMANA_SANTA)
    params = {"nivel": {"alpha": 0.4, "alpha_por_tipo": {"festivo": 0.9}}}
    records = run_pipeline(ctx, ProveedorPlano(), params=params)
    rec_nivel = next(r for r in records if r.paso == "nivel")
    assert rec_nivel.params == {"alpha": 0.4, "alpha_por_tipo": {"festivo": 0.9}}
    payload = json.loads(json.dumps([r.to_dict() for r in records], allow_nan=False))
    vuelta = [StepRecord.from_dict(d) for d in payload]
    assert [r.to_dict() for r in vuelta] == payload
    # los params persistidos revalidan igual
    p_vuelta = next(r for r in vuelta if r.paso == "nivel").params
    assert nivel.validar_params(p_vuelta) == rec_nivel.params
