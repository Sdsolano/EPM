"""Tests de `src/adjustments/aprendizaje.py` y `Store.list_runs` (solo lectura).

No requieren red ni base de datos: usan StepRecord sintéticos y InMemoryStore.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pytest

from src.adjustments.aprendizaje import _energia, a_dict, resumir
from src.adjustments.store import InMemoryStore
from src.adjustments.types import StepRecord, StepResult, StepState

LUNES = date(2026, 11, 16)
MARTES = date(2026, 11, 17)
DIAS = [LUNES.isoformat(), MARTES.isoformat()]


def _plana(valor: float) -> list:
    return [float(valor)] * 24


def _rec_editado(paso: str, calculadas: dict, editadas: dict, orden: int = 0) -> StepRecord:
    return StepRecord(
        orden=orden, paso=paso, estado=StepState.EDITADO, params={},
        result=StepResult(
            curves=editadas, curvas_calculadas=calculadas,
            diagnostics={"fechas_editadas": sorted(editadas)},
        ),
    )


def _rec_sin_calculada(paso: str, fechas: list, orden: int = 0) -> StepRecord:
    return StepRecord(
        orden=orden, paso=paso, estado=StepState.EDITADO, params={},
        result=StepResult(
            curves={f: _plana(100.0) for f in fechas}, curvas_calculadas={},
            diagnostics={"fechas_editadas": sorted(fechas)},
        ),
    )


def _run(run_id: str, ucp: str, records: list, festivos=None) -> dict:
    return {"meta": {"id": run_id, "ucp": ucp, "extras": {"festivos": festivos or []}},
            "records": records}


# ---------------------------------------------------------------------------
# resumir
# ---------------------------------------------------------------------------


def test_energia_robusta_a_ndarray_y_vacios():
    assert _energia(np.array([1.0, 2.0, 3.0])) == pytest.approx(6.0)
    assert _energia(np.array([])) is None
    assert _energia(np.array([np.nan])) is None
    assert _energia([]) is None
    assert _energia(None) is None
    assert _energia([1.0, 2.0]) == pytest.approx(3.0)

def test_edicion_nivel_y_forma_medias_signos_y_agrupacion():
    run = _run("r1", "UCP1", [
        _rec_editado("nivel", {DIAS[0]: _plana(100.0)}, {DIAS[0]: _plana(110.0)}, orden=2),
        _rec_editado("forma", {DIAS[1]: _plana(100.0)}, {DIAS[1]: _plana(90.0)}, orden=3),
    ])
    res = resumir([run])

    grupos = {(g["paso"], g["tipo_dia"]): g for g in res["grupos"]}
    assert set(grupos) == {("nivel", "laboral"), ("forma", "laboral")}

    nivel = grupos[("nivel", "laboral")]
    assert nivel["ucp"] == "UCP1"
    assert nivel["n_dias"] == 1 and nivel["n_corridas"] == 1 and nivel["n_casos"] == 1
    assert nivel["delta_pct_medio"] == pytest.approx(10.0)  # porcentaje: 0.10 → 10.0
    assert nivel["delta_pct_mediano"] == pytest.approx(10.0)
    assert nivel["pct_positivo"] == pytest.approx(100.0)
    assert nivel["p50_abs"] == pytest.approx(10.0)
    assert nivel["p90_abs"] == pytest.approx(10.0)
    assert nivel["fechas"] == [DIAS[0]]
    assert nivel["recurrente"] is False  # min_casos=3 por defecto y solo 1 día

    forma = grupos[("forma", "laboral")]
    assert forma["delta_pct_medio"] == pytest.approx(-10.0)
    assert forma["delta_pct_mediano"] == pytest.approx(-10.0)
    assert forma["pct_positivo"] == pytest.approx(0.0)
    assert forma["fechas"] == [DIAS[1]]

    glob = res["resumen_global"]
    assert glob["n_casos"] == 2 and glob["n_dias"] == 2 and glob["n_corridas"] == 1
    assert glob["delta_pct_medio"] == pytest.approx(0.0)
    assert glob["delta_pct_mediano"] == pytest.approx(0.0)
    assert glob["pct_positivo"] == pytest.approx(50.0)
    assert glob["p50_abs"] == pytest.approx(10.0)
    assert glob["recurrente"] is False
    assert res["omitidos_sin_calculada"] == 0
    assert res["min_casos"] == 3

    assert set(res["por_dia_semana"]) == {"lunes", "martes"}
    assert res["por_dia_semana"]["lunes"]["delta_pct_medio"] == pytest.approx(10.0)
    assert res["por_dia_semana"]["lunes"]["recurrente"] is False
    assert res["por_dia_semana"]["martes"]["delta_pct_medio"] == pytest.approx(-10.0)


def test_tipo_de_dia_festivo_desde_extras():
    run = _run("r1", "UCP1", [
        _rec_editado("nivel", {DIAS[0]: _plana(100.0)}, {DIAS[0]: _plana(105.0)}),
    ], festivos=[DIAS[0]])
    res = resumir([run])
    assert [g["tipo_dia"] for g in res["grupos"]] == ["festivo"]


def test_registros_sin_curvas_calculadas_se_omiten_sin_lanzar():
    viejo = _rec_sin_calculada("nivel", [DIAS[0], DIAS[1]])
    aplicado = StepRecord(orden=1, paso="forma", estado=StepState.APLICADO, params={},
                          result=StepResult(curves={DIAS[0]: _plana(100.0)}))
    res = resumir([_run("r_viejo", "UCP1", [viejo, aplicado])])

    assert res["grupos"] == []
    assert res["resumen_global"]["n_casos"] == 0
    assert res["resumen_global"]["delta_pct_medio"] is None
    assert res["resumen_global"]["pct_positivo"] is None
    assert res["omitidos_sin_calculada"] == 2  # un día omitido por cada fecha editada sin calculada


def test_record_editado_sin_fechas_ni_calculada_no_rompe():
    rec = StepRecord(orden=0, paso="nivel", estado=StepState.EDITADO, params={},
                     result=StepResult(curves={DIAS[0]: _plana(100.0)}))
    res = resumir([_run("r", "UCP1", [rec])])
    assert res["grupos"] == [] and res["omitidos_sin_calculada"] == 0


def test_min_casos_marca_recurrente_y_cuenta_corridas():
    run1 = _run("r1", "UCP1", [
        _rec_editado("nivel", {DIAS[0]: _plana(100.0)}, {DIAS[0]: _plana(110.0)}),
    ])
    run2 = _run("r2", "UCP1", [
        _rec_editado("nivel", {DIAS[0]: _plana(100.0), DIAS[1]: _plana(100.0)},
                     {DIAS[0]: _plana(120.0), DIAS[1]: _plana(80.0)}),
    ])
    res = resumir([run1, run2], min_casos=2)
    grupo = res["grupos"][0]

    assert grupo["n_casos"] == 3
    assert grupo["n_dias"] == 2 and grupo["n_corridas"] == 2
    assert grupo["delta_pct_medio"] == pytest.approx((0.1 + 0.2 - 0.2) / 3 * 100, abs=1e-4)
    assert grupo["delta_pct_mediano"] == pytest.approx(10.0)
    assert grupo["pct_positivo"] == pytest.approx(2 / 3 * 100, abs=1e-4)
    assert grupo["p50_abs"] == pytest.approx(20.0)
    assert grupo["recurrente"] is True  # 2 días >= min_casos
    assert len(grupo["fechas"]) == 2

    assert resumir([run1, run2], min_casos=3)["grupos"][0]["recurrente"] is False

    # por_dia_semana usa el mismo umbral: aquí 1 día por día de semana
    por_dia = resumir([run1, run2], min_casos=2)["por_dia_semana"]
    assert por_dia["lunes"]["recurrente"] is False and por_dia["martes"]["recurrente"] is False


def test_min_casos_default_es_3():
    dos = [_run("r1", "UCP1", [
        _rec_editado("nivel", {DIAS[0]: _plana(100.0)}, {DIAS[0]: _plana(110.0)}),
    ]), _run("r2", "UCP1", [
        _rec_editado("nivel", {DIAS[1]: _plana(100.0)}, {DIAS[1]: _plana(110.0)}),
    ])]
    res = resumir(dos)
    assert res["min_casos"] == 3
    assert res["grupos"][0]["n_dias"] == 2 and res["grupos"][0]["recurrente"] is False

    tercer_dia = (LUNES + timedelta(days=7)).isoformat()
    tres = dos + [_run("r3", "UCP1", [
        _rec_editado("nivel", {tercer_dia: _plana(100.0)}, {tercer_dia: _plana(110.0)}),
    ])]
    res = resumir(tres)
    assert res["grupos"][0]["n_dias"] == 3 and res["grupos"][0]["recurrente"] is True


def test_fechas_top_10_por_delta_absoluto():
    dias = [(date(2026, 1, 5) + timedelta(days=7 * k)).isoformat() for k in range(12)]  # 12 lunes
    editadas = {f: _plana(100.0 + k) for k, f in enumerate(dias)}
    calculadas = {f: _plana(100.0) for f in dias}
    res = resumir([_run("r", "UCP1", [_rec_editado("nivel", calculadas, editadas)])])

    fechas = res["grupos"][0]["fechas"]
    assert len(fechas) == 10
    assert fechas[0] == dias[-1]  # +12% es el mayor |delta|
    assert fechas[-1] == dias[2]  # los dos menores (+1%, +2%) quedan fuera


def test_a_dict_es_json_estricto():
    run = _run("r", "UCP1", [
        _rec_editado("nivel", {DIAS[0]: _plana(100.0)}, {DIAS[0]: _plana(110.0)}),
    ])
    payload = a_dict(resumir([run]))
    texto = json.dumps(payload, allow_nan=False)
    assert '"grupos"' in texto and '"resumen_global"' in texto


# ---------------------------------------------------------------------------
# InMemoryStore.list_runs
# ---------------------------------------------------------------------------

def _meta(ucp: str, semana: date) -> dict:
    return {
        "ucp": ucp, "semana_inicio": semana, "fecha_corte": semana - date.resolution,
        "fecha_emision": semana, "modelo_version": "m1", "params_iniciales": {},
        "extras": {}, "creado_por": "t",
    }


def _records_list_runs() -> list:
    return [StepRecord(orden=0, paso="base", estado=StepState.APLICADO, params={},
                       result=StepResult(curves={LUNES.isoformat(): _plana(1.0)}))]


def test_inmemory_list_runs_orden_filtros_y_limite():
    store = InMemoryStore()
    s1, s2, s3 = date(2026, 11, 2), date(2026, 11, 9), date(2026, 11, 16)
    r1 = store.create_run(_meta("UCP1", s1), _records_list_runs())
    r3 = store.create_run(_meta("UCP1", s3), _records_list_runs())
    r2 = store.create_run(_meta("UCP1", s2), _records_list_runs())
    otra = store.create_run(_meta("UCP2", s2), _records_list_runs())

    todas = store.list_runs(None, None, None, 200)
    assert todas[0]["meta"]["id"] == r3  # más reciente primero
    assert {run["meta"]["id"] for run in todas} == {r1, r2, r3, otra}
    assert set(todas[0]) == {"meta", "records"}
    assert todas[0]["records"][0].paso == "base"

    assert [run["meta"]["id"] for run in store.list_runs("UCP1", None, None, 200)] == [r3, r2, r1]
    assert [run["meta"]["id"] for run in store.list_runs("UCP2", None, None, 200)] == [otra]

    rango = store.list_runs("UCP1", s1, s2, 200)
    assert [run["meta"]["id"] for run in rango] == [r2, r1]  # extremos inclusivos

    assert [run["meta"]["id"] for run in store.list_runs(None, None, None, 1)] == [r3]
    assert store.list_runs("NO_EXISTE", None, None, 200) == []
    # el store acota el límite a 1..500 (no acepta 1e9 ni valores < 1)
    assert len(store.list_runs(None, None, None, 10 ** 9)) == 4
    assert [run["meta"]["id"] for run in store.list_runs(None, None, None, 0)] == [r3]


def test_inmemory_list_runs_no_expone_eventos_ni_estado_mutable():
    store = InMemoryStore()
    rid = store.create_run(_meta("UCP1", LUNES), _records_list_runs())
    run = store.list_runs("UCP1", None, None, 10)[0]
    assert "eventos" not in run
    run["meta"]["ucp"] = "MUTADO"
    assert store.get_run(rid)["meta"]["ucp"] == "UCP1"
