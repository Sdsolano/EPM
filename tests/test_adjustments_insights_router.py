"""Router /adjustments/insights/*: forma de cada respuesta (CONTRATO_B), errores y degradación sin historial."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.adjustments import insights as I
from src.adjustments import router as router_mod
from src.adjustments import router_insights
from tests.test_adjustments_router import (  # noqa: F401 — fixtures reexportados
    BODY,
    CORTE,
    SEMANA,
    UCP,
    _base_externa,
    _crear as _crear_base,
    _curva,
    _semana_iso,
    _steps,
    client,
    client_sin_store,
    env,
    raw_dir,
    store,
)

N_PERIODOS = 24


@pytest.fixture(autouse=True)
def _sin_openai(monkeypatch):
    async def vacio(ucp, lunes, **kwargs):
        return {}

    monkeypatch.setattr(router_mod.eventos, "eventos_de_semana", vacio)


def _escribir_emitidos(directorio, desde: date, hasta: date, sesgo_sabado: float = -0.03) -> None:
    """Historial emitido sintético: el real del histórico de prueba con el sábado `sesgo_sabado` corto."""
    filas = []
    lunes = desde
    while lunes <= hasta:
        for i in range(7):
            d = lunes + timedelta(days=i)
            err = sesgo_sabado if d.weekday() == 5 else (0.002 if (d.toordinal() % 2) else -0.002)
            filas.append({"fecha": d.isoformat(), "semana_inicio": lunes.isoformat(),
                          **{f"P{k + 1}": round(v * (1 + err), 3) for k, v in enumerate(_curva(d))}})
        lunes += timedelta(days=7)
    pd.DataFrame(filas).to_csv(directorio / f"{UCP}_k1.csv", index=False)


@pytest.fixture
def emitidos(tmp_path, monkeypatch):
    d = tmp_path / "precision"
    d.mkdir()
    _escribir_emitidos(d, date(2026, 3, 2), date(2026, 11, 9))
    monkeypatch.setattr(I, "PRECISION_DIR", d)
    return d


@pytest.fixture
def api(env, store):
    """TestClient con /adjustments y /adjustments/insights sobre el mismo InMemoryStore."""
    app = FastAPI()
    app.include_router(router_mod.router)
    app.include_router(router_insights.router)
    app.dependency_overrides[router_mod.get_store_dep] = lambda: store
    return TestClient(app)


@pytest.fixture
def api_sin_store(env):
    app = FastAPI()
    app.include_router(router_insights.router)
    app.dependency_overrides[router_mod.get_store_dep] = lambda: None
    return TestClient(app)


def _crear(api, semana: date = SEMANA, corte: date = CORTE):
    body = {"ucp": UCP, "semana_inicio": semana.isoformat(), "usuario": "analista",
            "base_curvas": _base_externa(corte, semana, 1.0)}
    r = api.post("/adjustments/run", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def _get(api, ruta, **params):
    return api.get(f"/adjustments/insights/{ruta}", params=params)


# ---------------------------------------------------------------------------
# resumen
# ---------------------------------------------------------------------------

def test_resumen_forma_del_contrato(api, emitidos):
    run = _crear(api)
    r = _get(api, "resumen", run_id=run["id"])
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["run_id"] == run["id"] and j["ucp"] == UCP and j["nombre_ucp"] == UCP
    assert j["semana_inicio"] == SEMANA.isoformat() and j["semana_fin"] == (SEMANA + timedelta(days=6)).isoformat()
    assert j["estado"] == "borrador" and j["version"] == run["version"]
    assert j["titulo"].startswith("Semana del 16 al 22 nov: demanda prevista ") and "GWh" in j["titulo"]
    assert set(j["confianza"]) == {"nivel", "texto", "wape_reciente_pct", "semanas_evaluadas"}
    assert j["confianza"]["nivel"] in ("alta", "media") and j["confianza"]["semanas_evaluadas"] == 8
    k = j["kpis"]
    assert {"energia_semana_mwh", "energia_semana_anterior_mwh", "variacion_vs_anterior_pct", "pico_mw",
            "pico_fecha", "pico_hora", "error_reciente_pct", "error_reciente_tendencia", "alertas",
            "recomendaciones"} <= set(k)
    assert {"total", "warning", "info"} <= set(k["alertas"]) and {"total", "alta"} == set(k["recomendaciones"])
    assert k["energia_semana_mwh"] == pytest.approx(
        sum(sum(v) for v in run["final_curves"].values() if True) - sum(
            sum(v) for f, v in run["final_curves"].items() if f not in _semana_iso()), rel=1e-4)
    assert len(j["dias"]) == 7
    assert set(j["dias"][0]) == {"fecha", "dia", "tipo_dia", "energia_mwh", "pico_mw", "pico_hora", "ajuste_pct",
                                 "estado", "n_recomendaciones"}
    assert j["dias"][0]["dia"] == "lunes" and j["dias"][5]["tipo_dia"] == "sabado"
    assert all(d["estado"] in ("ok", "revisar", "critico") for d in j["dias"])
    assert j["siguiente_paso"]["accion"] in ("revisar_recomendaciones", "revisar_alertas", "aprobar", "ya_aprobada")
    for i in j["insights"]:
        assert set(i) == {"id", "nivel", "titulo", "texto", "fechas"}
        assert not any(w in (i["titulo"] + i["texto"]) for w in ("α", "ratio", "ancla"))


def test_resumen_errores(api, api_sin_store):
    assert _get(api, "resumen", run_id="no-existe").status_code == 404
    assert api.get("/adjustments/insights/resumen").status_code == 422
    assert _get(api_sin_store, "resumen", run_id="x").status_code == 503


def test_resumen_sin_historial_de_pronosticos_no_falla(api, tmp_path, monkeypatch):
    vacio = tmp_path / "vacio"
    vacio.mkdir()
    monkeypatch.setattr(I, "PRECISION_DIR", vacio)
    run = _crear(api)
    j = _get(api, "resumen", run_id=run["id"]).json()
    assert j["confianza"]["nivel"] == "sin_datos" and j["kpis"]["error_reciente_pct"] is None
    assert _get(api, "recomendaciones", run_id=run["id"]).json()["recomendaciones"] is not None
    assert _get(api, "series", run_id=run["id"]).status_code == 200


def test_run_sin_historial_real_degrada(api, env, emitidos):
    run = _crear(api)

    def sin_datos(ucp):
        raise FileNotFoundError("sin datos")

    router_mod.set_history_loader(sin_datos)  # el fixture env restaura el loader al terminar
    for ruta in ("resumen", "recomendaciones", "series", "banda"):
        r = _get(api, ruta, run_id=run["id"])
        assert r.status_code == 200, (ruta, r.text)
    s = _get(api, "series", run_id=run["id"]).json()
    assert all(v is None for v in s["series"]["real"]) and s["series"]["banda_inferior"] is None
    assert all(v is not None for v in s["series"]["ajustado"])
    assert _get(api, "resumen", run_id=run["id"]).json()["confianza"]["nivel"] == "sin_datos"


# ---------------------------------------------------------------------------
# precisión
# ---------------------------------------------------------------------------

def test_precision_forma_del_contrato(api, emitidos):
    r = _get(api, "precision", ucp=UCP, semanas=6)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["ucp"] == UCP and j["hasta"] == "2026-11-02" and j["desde"] == "2026-09-28"
    assert set(j["resumen"]) == {"semanas", "wape_pct", "mape_energia_semana_pct", "sesgo_energia_pct",
                                 "mejor_semana", "peor_semana", "tendencia"}
    assert j["resumen"]["semanas"] == 6 and len(j["semanas"]) == 6
    s = j["semanas"][-1]
    assert s["semana_inicio"] == "2026-11-02" and s["fuente"] == "modelo_historico" and s["dias_con_datos"] == 7
    assert set(s["dias"][0]) == {"fecha", "dia", "tipo_dia", "wape_pct", "error_energia_pct", "energia_pronosticada_mwh",
                                 "energia_real_mwh", "pico_pronosticado_mw", "pico_real_mw", "hora_pico_pronosticada",
                                 "hora_pico_real"}
    assert {t["tipo_dia"] for t in j["por_tipo_dia"]} <= {"laboral", "sabado", "domingo", "festivo"}
    assert len(j["por_hora"]["sesgo_pct"]) == N_PERIODOS and len(j["por_hora"]["wape_pct"]) == N_PERIODOS
    assert [t for t in j["por_tipo_dia"] if t["tipo_dia"] == "sabado"][0]["sesgo_energia_pct"] < -2.5


def test_precision_hasta_semanas_y_errores(api, emitidos):
    j = _get(api, "precision", ucp=UCP, semanas=3, hasta="2026-06-01").json()
    assert j["hasta"] == "2026-06-01" and len(j["semanas"]) == 3
    assert _get(api, "precision", ucp=UCP, hasta="2026-06-02").status_code == 422  # no es lunes
    assert _get(api, "precision", ucp=UCP, semanas=0).status_code == 422
    assert _get(api, "precision", ucp=UCP, semanas=53).status_code == 422
    assert _get(api, "precision", ucp="Desconocida").status_code == 404
    assert _get(api, "precision", ucp="../x").status_code == 422
    assert api.get("/adjustments/insights/precision").status_code == 422


def test_precision_sin_historial_emitido_no_es_500(api, tmp_path, monkeypatch):
    vacio = tmp_path / "vacio"
    vacio.mkdir()
    monkeypatch.setattr(I, "PRECISION_DIR", vacio)
    j = _get(api, "precision", ucp=UCP).json()
    assert j["semanas"] == [] and j["por_tipo_dia"] == [] and j["resumen"]["tendencia"] == "sin_datos"
    assert j["desde"] is None and j["hasta"] is None


def test_precision_usa_corridas_aprobadas(api, emitidos, store):
    semana_pasada = date(2026, 11, 2)  # lunes con real completo en el histórico sintético (hasta 2026-11-13)
    run = _crear(api, semana=semana_pasada, corte=semana_pasada - timedelta(days=1))
    assert api.post(f"/adjustments/{run['id']}/approve", json={"usuario": "jefe", "version_esperada": run["version"]}
                    ).status_code == 200
    j = _get(api, "precision", ucp=UCP, semanas=4, hasta=semana_pasada.isoformat()).json()
    por = {s["semana_inicio"]: s["fuente"] for s in j["semanas"]}
    assert por[semana_pasada.isoformat()] == "corrida_aprobada"
    assert por[(semana_pasada - timedelta(days=7)).isoformat()] == "modelo_historico"


# ---------------------------------------------------------------------------
# recomendaciones
# ---------------------------------------------------------------------------

def test_recomendaciones_y_aplicar_con_un_clic(api, emitidos):
    run = _crear(api)
    r = _get(api, "recomendaciones", run_id=run["id"])
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["run_id"] == run["id"] and j["version"] == run["version"] and j["estado"] == "borrador"
    sab = (SEMANA + timedelta(days=5)).isoformat()
    rec = next(x for x in j["recomendaciones"] if x["id"] == f"sesgo-{sab}")
    assert {"id", "fecha", "dia", "tipo_dia", "prioridad", "categoria", "titulo", "explicacion", "evidencia",
            "impacto_esperado", "aplicable", "ajuste"} == set(rec)
    assert rec["dia"] == "sábado" and rec["categoria"] in ("nivel", "forma") and rec["aplicable"] is True
    assert rec["evidencia"]["n"] >= 6 and rec["evidencia"]["fuente"] == "historial"
    aj = rec["ajuste"]
    assert set(aj) == {"resumen", "delta_energia_pct", "delta_energia_mwh", "curva_sugerida", "patch"}
    assert len(aj["curva_sugerida"]) == 24 and aj["delta_energia_pct"] > 2.0
    patch = aj["patch"]
    assert patch["paso"] in ("forma", "especiales") and patch["body"]["estado"] == "editado"

    # 1 clic: PATCH con el body del patch + version_esperada
    body = {**patch["body"], "version_esperada": run["version"], "usuario": "analista"}
    p = api.patch(f"/adjustments/{run['id']}/steps/{patch['paso']}", json=body)
    assert p.status_code == 200, p.text
    nuevo = p.json()
    assert nuevo["version"] == run["version"] + 1
    assert nuevo["final_curves"][sab] == pytest.approx(aj["curva_sugerida"], rel=1e-9)
    # el resto de la semana no cambia
    for f in _semana_iso():
        if f != sab:
            assert nuevo["final_curves"][f] == pytest.approx(run["final_curves"][f], rel=1e-9)

    j2 = _get(api, "recomendaciones", run_id=run["id"]).json()
    assert j2["version"] == nuevo["version"]
    assert f"sesgo-{sab}" in j2["ya_resueltas"]
    assert all(x["id"] != f"sesgo-{sab}" for x in j2["recomendaciones"])


def test_recomendaciones_de_corrida_aprobada_no_son_aplicables(api, emitidos):
    run = _crear(api)
    assert api.post(f"/adjustments/{run['id']}/approve", json={"usuario": "jefe", "version_esperada": run["version"]}
                    ).status_code == 200
    j = _get(api, "recomendaciones", run_id=run["id"]).json()
    assert j["estado"] == "aprobada" and j["recomendaciones"]
    assert all(r["aplicable"] is False for r in j["recomendaciones"])
    res = _get(api, "resumen", run_id=run["id"]).json()
    assert res["estado"] == "aprobada" and res["siguiente_paso"]["accion"] == "ya_aprobada"


def test_recomendaciones_errores(api, api_sin_store):
    assert _get(api, "recomendaciones", run_id="no-existe").status_code == 404
    assert api.get("/adjustments/insights/recomendaciones").status_code == 422
    assert _get(api_sin_store, "recomendaciones", run_id="x").status_code == 503


# ---------------------------------------------------------------------------
# banda
# ---------------------------------------------------------------------------

def test_banda_por_ucp(api, emitidos):
    r = _get(api, "banda", ucp=UCP)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["ucp"] == UCP and j["nivel_confianza"] == 0.8 and j["metodo"] == "cuantiles_empiricos_error_relativo"
    assert j["n_semanas_historial"] == optionally(j) and "dias" not in j
    assert set(j["por_tipo_dia"]) == {"laboral", "sabado", "domingo", "festivo"}
    sab = j["por_tipo_dia"]["sabado"]
    assert sab["n_dias"] > 20 and len(sab["p10_pct"]) == 24
    assert all(a <= b <= c for a, b, c in zip(sab["p10_pct"], sab["p50_pct"], sab["p90_pct"]))
    assert sab["p50_pct"][0] == pytest.approx(3.09, abs=0.05)  # real/pronóstico − 1 con el sábado 3 % corto


def optionally(j):
    return j["n_semanas_historial"]


def test_banda_con_run_y_filtros(api, emitidos):
    run = _crear(api)
    j = _get(api, "banda", run_id=run["id"]).json()
    assert j["ucp"] == UCP and set(j["dias"]) <= set(_semana_iso()) and len(j["dias"]) == 7
    assert j["dias"][SEMANA.isoformat()]["tipo_dia"] == "festivo"  # 16 nov 2026: Todos los Santos (trasladado)
    d = j["dias"][(SEMANA + timedelta(days=1)).isoformat()]
    assert d["tipo_dia"] == "laboral" and len(d["inferior"]) == 24 and len(d["superior"]) == 24
    assert all(a < b for a, b in zip(d["inferior"], d["superior"]))
    assert _get(api, "banda", run_id=run["id"], ucp=UCP).status_code == 200
    solo = _get(api, "banda", ucp=UCP, tipo_dia="sabado", fecha_inicio="2026-03-02", fecha_fin="2026-03-31").json()
    assert list(solo["por_tipo_dia"]) == ["sabado"] and solo["por_tipo_dia"]["sabado"]["n_dias"] == 4


def test_banda_errores_y_sin_historial(api, tmp_path, monkeypatch):
    assert api.get("/adjustments/insights/banda").status_code == 422
    assert _get(api, "banda", ucp="Desconocida").status_code == 404
    assert _get(api, "banda", ucp=UCP, tipo_dia="otro").status_code == 422
    assert _get(api, "banda", ucp=UCP, fecha_inicio="2026-05-01", fecha_fin="2026-04-01").status_code == 422
    assert _get(api, "banda", run_id="no-existe").status_code == 404
    vacio = tmp_path / "vacio"
    vacio.mkdir()
    monkeypatch.setattr(I, "PRECISION_DIR", vacio)
    j = _get(api, "banda", ucp=UCP).json()
    assert j["n_semanas_historial"] == 0
    assert all(t["n_dias"] == 0 and t["p10_pct"] is None for t in j["por_tipo_dia"].values())
    run = _crear(api)
    assert _get(api, "banda", run_id=run["id"]).json()["dias"] == {}


# ---------------------------------------------------------------------------
# series
# ---------------------------------------------------------------------------

def test_series_semana_futura(api, emitidos):
    run = _crear(api)
    r = _get(api, "series", run_id=run["id"])
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["run_id"] == run["id"] and j["unidad"] == "MW" and j["puntos"] == 168 and j["semana_inicio"] == SEMANA.isoformat()
    assert j["fechas"] == _semana_iso() and len(j["etiquetas_eje"]) == 7
    assert j["etiquetas_eje"][1] == {"i": 24, "texto": "mar 17"}
    s = j["series"]
    for k in ("base", "ajustado", "real", "banda_inferior", "banda_superior"):
        assert len(s[k]) == 168
    assert all(v is None for v in s["real"])  # semana futura
    assert all(v is not None for v in s["base"]) and all(v is not None for v in s["ajustado"])
    assert all(isinstance(v, float) for v in s["banda_inferior"] + s["banda_superior"])
    ref = _steps(run)["referencia"]["diagnostics"]["semana_ref_inicio"]
    assert s["referencia_semana_inicio"] == ref and len(s["referencia"]) == 168
    assert set(j["energia_diaria"]) == {"fechas", "base", "ajustado", "referencia", "real"}
    assert all(len(j["energia_diaria"][k]) == 7 for k in ("base", "ajustado", "referencia", "real"))
    assert [d["tipo_dia"] for d in j["dias"]][5:] == ["sabado", "domingo"] and not any(d["editado"] for d in j["dias"])


def test_series_semana_pasada_trae_real_y_marca_editados(api, emitidos):
    pasada = date(2026, 11, 2)
    run = _crear(api, semana=pasada, corte=pasada - timedelta(days=1))
    f = (pasada + timedelta(days=2)).isoformat()
    curva = [v * 1.02 for v in run["final_curves"][f]]
    p = api.patch(f"/adjustments/{run['id']}/steps/forma", json={
        "estado": "editado", "curvas": {f: curva}, "version_esperada": run["version"], "usuario": "a"})
    assert p.status_code == 200, p.text
    j = _get(api, "series", run_id=run["id"]).json()
    assert all(v is not None for v in j["series"]["real"])
    assert j["series"]["real"][:24] == pytest.approx(_curva(pasada), abs=0.01)
    assert [d["editado"] for d in j["dias"]] == [False, False, True, False, False, False, False]
    assert j["energia_diaria"]["real"][0] == pytest.approx(sum(_curva(pasada)), abs=0.5)


def test_series_errores(api, api_sin_store):
    assert _get(api, "series", run_id="no-existe").status_code == 404
    assert api.get("/adjustments/insights/series").status_code == 422
    assert _get(api_sin_store, "series", run_id="x").status_code == 503


# ---------------------------------------------------------------------------
# auth y registro
# ---------------------------------------------------------------------------

def test_token_obligatorio_si_esta_configurado(api, emitidos, monkeypatch):
    monkeypatch.setenv(router_mod.ENV_SERVICE_TOKEN, "secreto")
    for ruta, params in (("resumen", {"run_id": "x"}), ("precision", {"ucp": UCP}), ("recomendaciones", {"run_id": "x"}),
                         ("banda", {"ucp": UCP}), ("series", {"run_id": "x"})):
        assert _get(api, ruta, **params).status_code == 401
    r = api.get("/adjustments/insights/precision", params={"ucp": UCP}, headers={router_mod.HEADER_SERVICE_TOKEN: "secreto"})
    assert r.status_code == 200


def test_rutas_registradas_en_la_api_principal():
    from src.api import main

    rutas = set(main.app.openapi()["paths"])
    for ruta in ("resumen", "precision", "recomendaciones", "banda", "series"):
        assert f"/adjustments/insights/{ruta}" in rutas
