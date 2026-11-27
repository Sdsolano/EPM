"""Tests de POST/GET /adjustments/aprendizaje/error (P1 "Lo que aprendió"): solo lectura, sin store."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.adjustments import data as data_mod
from src.adjustments import router as router_mod
from src.adjustments.store import InMemoryStore

LUNES0 = date(2026, 8, 3)
URL = "/adjustments/aprendizaje/error"


def _entrada(semanas: int = 6, factor: float = 1.01):
    emis, reales = [], {}
    for w in range(semanas):
        lunes = LUNES0 + timedelta(days=7 * w)
        curvas = {}
        for k in range(7):
            d = (lunes + timedelta(days=k)).isoformat()
            real = [100.0 + h for h in range(24)]
            reales[d] = real
            curvas[d] = [x * factor for x in real]
        emis.append({"semana_inicio": lunes.isoformat(), "version": 1, "curvas": curvas})
    return emis, reales


def _body(**kw):
    emis, reales = _entrada()
    return {"ucp": "Antioquia", "emisiones": emis, "reales": reales, "festivos": [], **kw}


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def client(monkeypatch, store):
    monkeypatch.delenv(router_mod.ENV_SERVICE_TOKEN, raising=False)
    app = FastAPI()
    app.include_router(router_mod.router)
    app.dependency_overrides[router_mod.get_store_dep] = lambda: store
    return TestClient(app)


def test_post_ok(client):
    r = client.post(URL, json=_body(semanas=4))
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["estado"] == "ok" and d["ucp"] == "Antioquia"
    assert d["sesgo_global"]["pct"] == pytest.approx(1.0, abs=0.05)
    assert d["ultima_semana"]


def test_post_sin_datos(client):
    r = client.post(URL, json={"ucp": "Antioquia", "emisiones": [], "reales": {}})
    assert r.status_code == 200
    assert r.json()["estado"] == "sin_datos"


@pytest.mark.parametrize("mut", [
    lambda b: b.update(ucp="  "),
    lambda b: b.update(semanas=2),
    lambda b: b["emisiones"][0].update(semana_inicio="2026-08-04"),  # no es lunes
    lambda b: b["reales"].update({"2026-08-03": [1.0] * 23}),
    lambda b: b["reales"].update({"2026-08-03": [-1.0] * 24}),
    lambda b: b["reales"].update({"no-fecha": [1.0] * 24}),
])
def test_post_422(client, mut):
    b = _body()
    mut(b)
    r = client.post(URL, json=b)
    assert r.status_code == 422, r.text
    assert r.json()["detail"]


def test_post_no_toca_store_ni_curvas(client, store):
    b = _body()
    antes = json.dumps(b, sort_keys=True)
    assert client.post(URL, json=b).status_code == 200
    assert json.dumps(b, sort_keys=True) == antes
    assert store.list_runs(None, None, None, 50) == []


def test_post_sin_store(monkeypatch):
    monkeypatch.delenv(router_mod.ENV_SERVICE_TOKEN, raising=False)
    app = FastAPI()
    app.include_router(router_mod.router)
    app.dependency_overrides[router_mod.get_store_dep] = lambda: None
    assert TestClient(app).post(URL, json=_body()).status_code == 200


def test_ruta_no_cae_en_run_id(client):
    # GET sin ucp valida debe ser 422 de nuestra ruta, no un 404 de /{run_id}
    assert client.get(URL).status_code == 422


def test_token_respetado(monkeypatch, store):
    monkeypatch.setenv(router_mod.ENV_SERVICE_TOKEN, "secreto")
    app = FastAPI()
    app.include_router(router_mod.router)
    app.dependency_overrides[router_mod.get_store_dep] = lambda: store
    c = TestClient(app)
    assert c.post(URL, json=_body()).status_code == 401
    ok = c.post(URL, json=_body(), headers={router_mod.HEADER_SERVICE_TOKEN: "secreto"})
    assert ok.status_code == 200


@pytest.fixture
def raiz_demo(tmp_path, monkeypatch):
    (tmp_path / "data" / "raw").mkdir(parents=True)
    monkeypatch.setattr(data_mod, "RAW_DATA_DIR", tmp_path / "data" / "raw")
    (tmp_path / "retro").mkdir()
    return tmp_path


def test_get_lee_snapshot(client, raiz_demo):
    emis, reales = _entrada()
    (raiz_demo / "retro" / "Antioquia.json").write_text(
        json.dumps({"emisiones": emis, "reales": reales, "festivos": []}), encoding="utf-8")
    r = client.get(URL, params={"ucp": "Antioquia", "semanas": 4})
    assert r.status_code == 200, r.text
    assert r.json()["estado"] == "ok"
    assert r.json()["sesgo_global"]["pct"] == pytest.approx(1.0, abs=0.05)


def test_get_sin_snapshot_es_sin_datos_y_ucp_invalida_422(client, raiz_demo):
    r = client.get(URL, params={"ucp": "Atlantico"})
    assert r.status_code == 200
    d = r.json()
    assert d["estado"] == "sin_datos" and d["ucp"] == "Atlantico" and "Atlantico" in d["motivo"]
    assert client.get(URL, params={"ucp": "../x"}).status_code == 422
    assert client.get(URL, params={"ucp": "Antioquia", "semanas": 99}).status_code == 422
    (raiz_demo / "retro" / "Roto.json").write_text("{no json", encoding="utf-8")
    assert client.get(URL, params={"ucp": "Roto"}).status_code == 422


def test_get_snapshot_sintetico_con_sesgo_conocido_festivo_y_domingo_negativos(client, raiz_demo):
    """Fixture generado aquí (sin datos reales): 8 semanas, reportado = real * 1.01 salvo
    el domingo 2026-09-27 (0.95) y el festivo lunes 2026-09-21 (0.96) de la última semana."""
    lunes0 = date(2026, 8, 3)
    festivo = "2026-09-21"
    domingo = "2026-09-27"
    emis, reales = [], {}
    for w in range(8):
        lunes = lunes0 + timedelta(days=7 * w)
        curvas = {}
        for k in range(7):
            d = (lunes + timedelta(days=k)).isoformat()
            real = [100.0 + h for h in range(24)]
            reales[d] = real
            f = 0.95 if d == domingo else 0.96 if d == festivo else 1.01
            curvas[d] = [x * f for x in real]
        emis.append({"semana_inicio": lunes.isoformat(), "version": 1, "curvas": curvas})
    (raiz_demo / "retro" / "Sintetica.json").write_text(
        json.dumps({"emisiones": emis, "reales": reales, "festivos": [festivo]}), encoding="utf-8")
    r = client.get(URL, params={"ucp": "Sintetica", "semanas": 8})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["estado"] == "ok" and out["base"]["hasta"] == "2026-09-27"
    ult = {x["fecha"]: x for x in out["ultima_semana"]}
    assert ult[domingo]["tipo_dia"] == "domingo" and ult[domingo]["error_pct"] == pytest.approx(-5.0, abs=0.1)
    assert ult[festivo]["tipo_dia"] == "festivo" and ult[festivo]["error_pct"] == pytest.approx(-4.0, abs=0.1)
    # sesgo conocido: 54 dias a +1 %, domingo -5 % y festivo -4 % (winsorizado a +-5 no cambia nada)
    esperado = (54 * 1.0 + (-5.0) + (-4.0)) / 56
    assert out["sesgo_global"]["n"] == 56
    assert out["sesgo_global"]["pct"] == pytest.approx(esperado, abs=0.05)
    assert out["ventanas"][2]["sesgo_pct"] is None  # 12 semanas > 8 pedidas: no disponible
