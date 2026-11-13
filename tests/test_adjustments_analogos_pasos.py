"""Tests de la integración de días análogos (Fase 2b) en los pasos forma, nivel y referencia."""

from datetime import date, timedelta

import numpy as np
import pytest

from src.adjustments import analogos as buscador
from src.adjustments.data import PERIODOS, dias_semana, energia_curvas, energia_diaria, festivos_colombia, iso
from src.adjustments.pipeline import approval_blockers, build_context, rerun_from, run_pipeline
from src.adjustments.steps import _analogos, base, forma, nivel, referencia
from src.adjustments.types import StepRecord, StepResult, StepState
from tests.ajustes_sinteticos import ProveedorPlano, historico_sintetico

# Fixtures del router (pytest las toma del namespace del módulo).
from tests.test_adjustments_router import (  # noqa: F401
    _crear,
    _steps,
    client,
    env,
    raw_dir,
    store,
)

FESTIVOS = set(festivos_colombia(range(2019, 2027)))
FACTOR_POST = 0.9  # energía relativa del día laboral siguiente a un festivo


def _forma_post() -> np.ndarray:
    """Forma distinta (pico de madrugada) para los días laborales después de un festivo."""
    h = np.arange(24)
    v = 1.0 + 0.5 * np.exp(-((h - 4) ** 2) / 6)
    return v / v.sum()


def _es_post(d: date) -> bool:
    return d.weekday() < 5 and iso(d) not in FESTIVOS and iso(d - timedelta(days=1)) in FESTIVOS


def _hist_con_post():
    """Histórico sintético donde los días laborales post-festivo tienen otra forma y −10 % de energía."""
    df = historico_sintetico()
    forma_post = _forma_post()
    for i, f in enumerate(df["FECHA"]):
        d = date.fromisoformat(f)
        if _es_post(d):
            energia = float(df.loc[i, PERIODOS].sum()) * FACTOR_POST
            df.loc[i, PERIODOS] = forma_post * energia
    df["TOTAL"] = df[PERIODOS].sum(axis=1)
    return df


HIST = _hist_con_post()
SEMANA = date(2025, 8, 18)  # lunes festivo (Asunción) → martes 19 post-festivo
LUNES, MARTES, MIERCOLES = "2025-08-18", "2025-08-19", "2025-08-20"


def _ctx(semana=SEMANA):
    return build_context("Sintetica", semana, fecha_emision=semana - timedelta(days=4), historico=HIST)


def _hasta_nivel(ctx, params_nivel=None):
    prev = base.step(ctx, {}, {}, base_provider=ProveedorPlano()).curves
    ref = referencia.step(ctx, prev, {})
    ctx.semana_ref_inicio = date.fromisoformat(ref.diagnostics["semana_ref_inicio"])
    ctx.extras["anios_excluidos"] = ref.diagnostics["anios_excluidos"]
    return ref, nivel.step(ctx, ref.curves, params_nivel or {})


def _forma(v):
    v = np.asarray(v, dtype=float)
    return v / v.sum()


# ---------------------------------------------------------------- forma

def test_forma_analogos_automatica_capta_post_festivo_y_conserva_energia():
    ctx = _ctx()
    assert ctx.fecha_corte == date(2025, 8, 13)
    _, res_nivel = _hasta_nivel(ctx)
    prev = res_nivel.curves
    default = forma.step(ctx, prev, {})
    res = forma.step(ctx, prev, {"fuente": "analogos"})

    for f in [iso(d) for d in dias_semana(SEMANA)]:
        assert abs(sum(res.curves[f]) - sum(prev[f])) <= 1e-6
    dia = res.diagnostics["dias"][MARTES]
    assert dia["fuente"] == "analogos" and dia["seleccion"] == "automatica"
    assert len(dia["fechas_fuente"]) == 4  # limite = n
    assert all(_es_post(date.fromisoformat(x)) and x <= iso(ctx.fecha_corte) for x in dia["fechas_fuente"])
    assert all(date.fromisoformat(x).weekday() == 1 for x in dia["fechas_fuente"])
    assert sum(dia["pesos"]) == pytest.approx(1.0, abs=1e-5)
    # capta la forma post-festivo; el promedio_n (martes normales) no
    assert np.allclose(_forma(res.curves[MARTES]), _forma_post(), atol=1e-9)
    assert not np.allclose(_forma(default.curves[MARTES]), _forma_post(), atol=1e-3)
    # festivo sin entrada en el mapa: se conserva el perfil homólogo (igual que el default)
    assert res.curves[LUNES] == default.curves[LUNES]
    assert res.diagnostics["dias"][LUNES]["seleccion"] == "homologo_festivo"
    assert res.diagnostics["dias"][LUNES]["fuente"].startswith("festivo_")
    assert res.diagnostics["filtros_analogos"]["limite"] == 4
    assert 2020 in res.diagnostics["filtros_analogos"]["anios_excluidos"]


def test_forma_analogos_con_mapa_pesos_por_score_y_no_observadas():
    ctx = _ctx()
    _, res_nivel = _hasta_nivel(ctx)
    prev = res_nivel.curves
    elegidas = ["2024-08-20", "2025-08-12"]  # un martes post-festivo y un martes normal
    mapa = {MARTES: elegidas + ["2018-05-08"], LUNES: ["2025-08-10"]}  # 2018: fuera del histórico
    res = forma.step(ctx, prev, {"fuente": "analogos", "analogos": mapa})

    dia = res.diagnostics["dias"][MARTES]
    assert dia["seleccion"] == "analista" and dia["fechas_fuente"] == elegidas
    puntuados = buscador.puntuar_fechas(ctx.historico, ctx.festivos, MARTES, elegidas)
    scores = np.array([s for _, s, _ in puntuados])
    assert dia["pesos"] == pytest.approx(list(scores / scores.sum()), abs=1e-6)
    esperada = sum(w * _forma(c) for w, (_, _, c) in zip(scores / scores.sum(), puntuados))
    assert np.allclose(_forma(res.curves[MARTES]), esperada, atol=1e-9)
    assert abs(sum(res.curves[MARTES]) - sum(prev[MARTES])) <= 1e-6
    # festivo CON entrada en el mapa: usa la elección del analista
    assert res.diagnostics["dias"][LUNES]["fechas_fuente"] == ["2025-08-10"]
    assert np.allclose(_forma(res.curves[LUNES]), _forma(ctx.historico.set_index("fecha").loc["2025-08-10", PERIODOS]))
    alerta = next(a for a in res.alerts if a.codigo == "ANALOGOS_NO_OBSERVADOS")
    assert alerta.nivel == "info" and alerta.detalle == {MARTES: ["2018-05-08"]}


def test_forma_analogos_sin_candidatos_cae_a_promedio_n():
    ctx = _ctx()
    _, res_nivel = _hasta_nivel(ctx)
    prev = res_nivel.curves
    default = forma.step(ctx, prev, {})
    res = forma.step(ctx, prev, {"fuente": "analogos", "filtros_analogos": {"anio_max": 1995}})
    alerta = next(a for a in res.alerts if a.codigo == "FORMA_FUENTE_ALTERNA")
    no_festivos = [iso(d) for d in dias_semana(SEMANA)][1:]
    assert sorted(alerta.fechas) == no_festivos
    assert all(alerta.detalle[f] == "sin_analogos" for f in no_festivos)
    for f in no_festivos:
        assert res.curves[f] == default.curves[f]
        assert res.diagnostics["dias"][f]["fuente"] == "promedio_n"


def test_forma_y_nivel_params_nuevos_no_cambian_nada_con_otra_fuente_o_ancla():
    ctx = _ctx()
    ref, res_nivel = _hasta_nivel(ctx)
    extra = {"analogos": {MARTES: ["2024-08-20"]}, "filtros_analogos": {"meses": [8]}}
    for fuente in ("promedio_n", "reciente", "referencia"):
        a = forma.step(ctx, res_nivel.curves, {"fuente": fuente})
        b = forma.step(ctx, res_nivel.curves, {"fuente": fuente, **extra})
        assert a.curves == b.curves and a.diagnostics == b.diagnostics and a.alerts == b.alerts
    a = nivel.step(ctx, ref.curves, {})
    b = nivel.step(ctx, ref.curves, {"ancla": "actual", **extra})
    assert a.curves == b.curves and a.diagnostics == b.diagnostics and a.alerts == b.alerts


def test_validar_params_analogos_normaliza_es_idempotente_y_omite_defaults():
    assert forma.validar_params({}) == {"fuente": "promedio_n", "n": 4, "beta": 1.0}
    assert nivel.validar_params({"ancla": "actual", "analogos": None}) == {"alpha": 0.25}
    p = forma.validar_params({
        "fuente": "analogos",
        "analogos": {date(2025, 8, 19): ["2024-08-20", date(2024, 8, 20), "2023-08-22T00:00"]},
        "filtros_analogos": {"meses": [8, 7, 8]},
    })
    assert p["analogos"] == {"2025-08-19": ["2024-08-20", "2023-08-22"]}
    assert p["filtros_analogos"]["meses"] == [7, 8] and p["filtros_analogos"]["limite"] == 20
    assert forma.validar_params(p) == p and forma.validar_params(forma.validar_params(p)) == p
    q = nivel.validar_params({"ancla": "analogos", "alpha": 0.5})
    assert q == {"alpha": 0.5, "ancla": "analogos"} and nivel.validar_params(q) == q


def test_defaults_sin_rastros_de_analogos_y_claves_desconocidas_rechazadas():
    ctx = _ctx()
    _, res_nivel = _hasta_nivel(ctx)
    res = forma.step(ctx, res_nivel.curves, {})
    assert set(res.diagnostics) == {"fuente", "n", "beta", "dias"}
    assert all(not {"seleccion", "pesos"} & set(d) for d in res.diagnostics["dias"].values())
    assert not any(a.codigo == "ANALOGOS_NO_OBSERVADOS" for a in res.alerts)
    assert set(res_nivel.diagnostics) == {"alpha", "semana_ref_inicio", "desfase_dias",
                                          "anios_excluidos", "dias"}
    for modulo in (forma, nivel):
        with pytest.raises(ValueError):
            modulo.validar_params({"otra": 1})


@pytest.mark.parametrize("modulo, params", [
    (forma, {"analogos": []}),
    (forma, {"analogos": "2025-08-19"}),
    (forma, {"analogos": {"2025-08-19": "2024-08-20"}}),
    (forma, {"analogos": {"2025-08-19": []}}),
    (forma, {"analogos": {"no-fecha": ["2024-08-20"]}}),
    (forma, {"analogos": {"2025-08-19": ["2024-13-01"]}}),
    (forma, {"analogos": {"2025-08-19": [20240820]}}),
    (forma, {"analogos": {"2025-08-19": ["1989-12-31"]}}),
    (forma, {"analogos": {"2025-08-19": ["2024-08-20"] * 30 + [f"2023-01-{i:02d}" for i in range(1, 32)]}}),
    (forma, {"analogos": {f"2025-08-{i:02d}": ["2024-08-20"] for i in range(10, 18)}}),
    (forma, {"filtros_analogos": {"desconocido": 1}}),
    (forma, {"filtros_analogos": "auto"}),
    (forma, {"fuente": "analogo"}),
    (nivel, {"ancla": "otra"}),
    (nivel, {"ancla": None}),
    (nivel, {"filtros_analogos": {"limite": 0}}),
    (nivel, {"analogos": {"2025-08-19": [None]}}),
])
def test_validar_params_analogos_invalidos(modulo, params):
    with pytest.raises(ValueError):
        modulo.validar_params(params)


@pytest.mark.parametrize("mapa, error", [
    ({"2025-08-25": ["2024-08-20"]}, "semana objetivo"),
    ({MARTES: ["2025-08-14"]}, "posterior a la fecha de corte"),
    ({MARTES: [MARTES]}, "posterior a la fecha de corte"),
])
@pytest.mark.parametrize("paso", ["forma", "nivel"])
def test_mapa_validado_contra_el_contexto(mapa, error, paso):
    ctx = _ctx()
    ref, res_nivel = _hasta_nivel(ctx)
    with pytest.raises(ValueError, match=error):
        if paso == "forma":
            forma.step(ctx, res_nivel.curves, {"fuente": "analogos", "analogos": mapa})
        else:
            nivel.step(ctx, ref.curves, {"ancla": "analogos", "analogos": mapa})


def test_mapa_rechaza_el_propio_dia_aunque_sea_anterior_al_corte():
    ctx = _ctx()
    ctx.fecha_corte = date(2025, 8, 30)  # contexto artificial: corte posterior a la semana
    with pytest.raises(ValueError, match="propio análogo"):
        _analogos.validar_mapa_ctx("forma", {MARTES: [MARTES]}, ctx)


# ---------------------------------------------------------------- nivel

def test_nivel_ancla_analogos_ratio_esperado_y_sin_fuga():
    ctx = _ctx()
    ref, actual = _hasta_nivel(ctx, {"alpha": 1.0})
    res = nivel.step(ctx, ref.curves, {"alpha": 1.0, "ancla": "analogos"})
    energia = energia_diaria(ctx.historico)
    corte = iso(ctx.fecha_corte)

    dia = res.diagnostics["dias"][MARTES]
    assert dia["regla"] == "analogos" and dia["seleccion"] == "automatica"
    assert dia["ancla"] == "2025-08-12" and dia["lag_dias"] == 7
    assert 1 <= len(dia["analogos"]) <= _analogos.LIMITE_AUTO_NIVEL
    ratio = 0.0
    for a in dia["analogos"]:
        assert a["fecha"] <= corte and a["ancla"] <= corte and a["ancla"] < a["fecha"]
        # el ancla del análogo está en el mismo régimen que la del objetivo (martes normal, no post-festivo)
        assert date.fromisoformat(a["ancla"]).weekday() == 1 and not _es_post(date.fromisoformat(a["ancla"]))
        assert a["ratio"] == pytest.approx(energia[a["fecha"]] / energia[a["ancla"]])
        assert a["ancla"] not in ctx.festivos and date.fromisoformat(a["ancla"]).year != 2020
        ratio += a["peso"] * a["ratio"]
    assert sum(a["peso"] for a in dia["analogos"]) == pytest.approx(1.0, abs=1e-5)
    assert dia["ratio"] == pytest.approx(ratio, rel=1e-5)
    assert dia["E_objetivo"] == pytest.approx(energia["2025-08-12"] * dia["ratio"])
    # el caso construido: post-festivo = 0.9 × un martes normal (± crecimiento de una semana)
    assert dia["ratio"] == pytest.approx(FACTOR_POST, rel=2e-3)
    assert sum(res.curves[MARTES]) == pytest.approx(dia["E_objetivo"])
    # un miércoles normal tiene ratio ≈ 1 (crecimiento de una semana)
    assert res.diagnostics["dias"][MIERCOLES]["ratio"] == pytest.approx(1.0, abs=2e-3)
    assert res.diagnostics["ancla"] == "analogos"
    assert not any(a.codigo == "NIVEL_ANCLA_ALTERNA" for a in res.alerts)


def test_nivel_ancla_analogos_con_mapa():
    ctx = _ctx()
    ref, _ = _hasta_nivel(ctx)
    res = nivel.step(ctx, ref.curves, {"ancla": "analogos", "analogos": {MARTES: ["2024-08-20", "2018-01-02"]}})
    dia = res.diagnostics["dias"][MARTES]
    assert dia["seleccion"] == "analista"
    assert [a["fecha"] for a in dia["analogos"]] == ["2024-08-20"]
    assert dia["analogos"][0]["ancla"] == "2024-08-13" and dia["analogos"][0]["peso"] == 1.0
    assert any(a.codigo == "ANALOGOS_NO_OBSERVADOS" for a in res.alerts)


def test_nivel_ancla_analogos_sin_ratios_usa_evidencia_actual_con_alerta():
    ctx = _ctx()
    ref, actual = _hasta_nivel(ctx)
    res = nivel.step(ctx, ref.curves, {"ancla": "analogos", "filtros_analogos": {"anio_max": 1995}})
    alerta = next(a for a in res.alerts if a.codigo == "NIVEL_ANCLA_ALTERNA")
    assert alerta.nivel == "info" and alerta.fechas == [iso(d) for d in dias_semana(SEMANA)]
    assert res.curves == actual.curves
    assert res.diagnostics["dias"] == actual.diagnostics["dias"]


def test_nivel_ancla_actual_intacta_y_evidencias_no_cambia():
    ctx = _ctx()
    ref, actual = _hasta_nivel(ctx)
    antes = nivel.evidencias(ctx)
    nivel.step(ctx, ref.curves, {"ancla": "analogos"})
    assert nivel.evidencias(ctx) == antes
    assert "ancla" not in actual.diagnostics
    assert all("analogos" not in d for d in actual.diagnostics["dias"].values())


# ---------------------------------------------------------------- referencia

def test_referencia_semanas_sugeridas_forma_y_orden():
    ctx = _ctx()
    prev = base.step(ctx, {}, {}, base_provider=ProveedorPlano()).curves
    res = referencia.step(ctx, prev, {})
    sug = res.diagnostics["semanas_sugeridas"]
    assert 1 <= len(sug) <= _analogos.TOP_SEMANAS
    scores = [s["score"] for s in sug]
    assert scores == sorted(scores, reverse=True)
    semana_obj = {iso(d) for d in dias_semana(SEMANA)}
    for s in sug:
        lunes = date.fromisoformat(s["semana_inicio"])
        assert lunes.weekday() == 0 and lunes + timedelta(days=6) <= ctx.fecha_corte
        assert lunes.year != 2020 and (lunes + timedelta(days=6)).year != 2020
        assert set(s["dias"]) <= semana_obj
        for obj, analoga in s["dias"].items():
            a = date.fromisoformat(analoga)
            assert lunes <= a <= lunes + timedelta(days=6)
            assert a.weekday() == date.fromisoformat(obj).weekday()
        assert s["score"] <= len(s["dias"]) + 1e-9
    # la mejor sugerencia para una semana con lunes festivo es otra semana con lunes festivo
    assert iso(date.fromisoformat(sug[0]["semana_inicio"])) in ctx.festivos
    # no cambia la semana elegida
    assert res.diagnostics["semana_ref_inicio"] == "2024-08-19"


def test_referencia_semanas_sugeridas_respeta_anios_excluidos():
    ctx = _ctx()
    prev = base.step(ctx, {}, {}, base_provider=ProveedorPlano()).curves
    res = referencia.step(ctx, prev, {"anios_excluidos": [2024, 2020]})
    assert all("2024" not in s["semana_inicio"] for s in res.diagnostics["semanas_sugeridas"])


# ---------------------------------------------------------------- pipeline + PATCH

PARAMS_FORMA = {"fuente": "analogos", "analogos": {MARTES: ["2024-08-20", "2023-08-22"]}}


def test_pipeline_completo_con_analogos_y_rerun_equivalente_al_patch():
    ctx = _ctx()
    params = {"nivel": {"ancla": "analogos"}, "forma": PARAMS_FORMA}
    records = run_pipeline(ctx, ProveedorPlano(), params=params)
    assert records[2].params == {"alpha": 0.25, "ancla": "analogos"}
    assert records[3].params == {"fuente": "analogos", "n": 4, "beta": 1.0,
                                 "analogos": {MARTES: ["2024-08-20", "2023-08-22"]}}
    assert records[3].result.diagnostics["dias"][MARTES]["fechas_fuente"] == ["2024-08-20", "2023-08-22"]
    e_nivel = energia_curvas(records[2].result.curves)
    for f in [iso(d) for d in dias_semana(SEMANA)]:
        assert abs(sum(records[3].result.curves[f]) - e_nivel[f]) <= 1e-6
    assert approval_blockers(records) == []
    control = records[4].result.diagnostics
    assert control  # el control corre con E_objetivo de nivel por análogos

    # rerun_from cambiando solo forma (equivalente al PATCH) == corrida directa
    ctx2 = _ctx()
    previos = run_pipeline(ctx2, ProveedorPlano(), params={"nivel": {"ancla": "analogos"}})
    nuevo = StepRecord(3, "forma", StepState.APLICADO, PARAMS_FORMA, StepResult({}))
    rehecho = rerun_from(ctx2, previos, "forma", ProveedorPlano(), nuevo)
    assert [r.to_dict() for r in rehecho] == [r.to_dict() for r in records]


def test_patch_forma_a_analogos_via_router(client):
    creada = _crear(client)
    rid = creada["id"]
    martes = "2026-11-17"  # tras el lunes festivo 2026-11-16
    mapa = {martes: ["2025-11-18", date(2026, 11, 10).isoformat(), "2025-11-18"]}
    r = client.patch(f"/adjustments/{rid}/steps/forma",
                     json={"estado": "aplicado", "version_esperada": 1,
                           "params": {"fuente": "analogos", "analogos": mapa}})
    assert r.status_code == 200, r.text
    forma_out = _steps(r.json())["forma"]
    assert forma_out["params"]["fuente"] == "analogos"
    assert forma_out["params"]["analogos"] == {martes: ["2025-11-18", "2026-11-10"]}
    dia = forma_out["diagnostics"]["dias"][martes]
    assert dia["fechas_fuente"] == ["2025-11-18", "2026-11-10"] and dia["seleccion"] == "analista"
    guardada = client.get(f"/adjustments/{rid}").json()
    assert _steps(guardada)["forma"]["params"] == forma_out["params"]

    # fecha posterior al corte → 422 y la corrida no cambia
    r = client.patch(f"/adjustments/{rid}/steps/forma",
                     json={"estado": "aplicado", "version_esperada": 2,
                           "params": {"fuente": "analogos", "analogos": {martes: ["2026-11-14"]}}})
    assert r.status_code == 422 and "corte" in r.json()["detail"]
    r = client.patch(f"/adjustments/{rid}/steps/nivel",
                     json={"estado": "aplicado", "version_esperada": 2, "params": {"ancla": "otra"}})
    assert r.status_code == 422
