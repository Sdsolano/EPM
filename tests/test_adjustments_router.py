"""Tests del router /adjustments y del store (InMemory + Postgres temporal opcional).

No importa src/api/main.py (pesado): monta un FastAPI mínimo con solo el router, un
InMemoryStore y fakes inyectados para el modelo base, el refresco de datos, el histórico
y los festivos.

Postgres: si `AJUSTES_DATABASE_URL` está definido, los tests marcados crean una base temporal
(aplican sql/001 + sql/002) y la borran al terminar. Si no hay permiso CREATE DATABASE, usan la
base indicada con una UCP única y borran sus filas al terminar.
"""

from __future__ import annotations

import importlib.util
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.adjustments import data as data_mod
from src.adjustments import pipeline as pl
from src.adjustments import router as router_mod
from src.adjustments.data import festivos_colombia
from src.adjustments.store import (
    ApprovalBlockedError,
    InMemoryStore,
    PostgresStore,
    RunApprovedError,
    RunNotFoundError,
    VersionConflictError,
)
from src.adjustments.types import Alert, StepRecord, StepResult, StepState

UCP = "Antioquia"
SEMANA = date(2026, 11, 16)  # lunes
CORTE = date(2026, 11, 13)  # último dato real del histórico sintético (viernes)

# forma horaria típica (pico nocturno) y factor por día de semana
_FORMA = np.array([0.8, 0.75, 0.72, 0.7, 0.72, 0.8, 0.9, 1.0, 1.05, 1.08, 1.1, 1.12,
                   1.1, 1.08, 1.07, 1.06, 1.05, 1.1, 1.25, 1.3, 1.22, 1.1, 0.98, 0.88])
_DOW = [1.0, 1.01, 1.01, 1.02, 1.0, 0.9, 0.8]


def _curva(d: date, escala: float = 1.0) -> list:
    tendencia = 1.0 + 0.00005 * (d - date(2023, 1, 1)).days
    return [float(x) for x in 400.0 * _FORMA * _DOW[d.weekday()] * tendencia * escala]


def _filas(desde: date, hasta: date) -> pd.DataFrame:
    rows = []
    for ts in pd.date_range(desde, hasta, freq="D"):
        vals = _curva(ts.date())
        rows.append({"fecha": ts, **{f"P{i + 1}": v for i, v in enumerate(vals)}})
    df = pd.DataFrame(rows)
    df["TOTAL"] = df[[f"P{i}" for i in range(1, 25)]].sum(axis=1)
    return df


HIST = _filas(date(2023, 1, 1), CORTE)
FESTIVOS = festivos_colombia(range(2016, 2028))


class FakeEnv:
    """Fakes del modelo base, refresco, histórico, festivos y verificación de modelo."""

    def __init__(self) -> None:
        self.modelo_entrenado = True
        self.hist = HIST.copy()
        self.calls = []            # llamadas al modelo: (ucp, end_date, n_days)
        self.log = []              # orden: "refresh", "history", "model"
        self.dias_nuevos = 0       # el próximo refresco agrega N días reales al histórico
        self.modelo_version = "fake_model_v1"
        self.model_error = None
        self.refresh_error = None
        self.festivos_error = False
        self.quitar_dia = None     # fecha ISO que el modelo no devuelve

    async def provider(self, ucp, end_date, n_days):
        self.log.append("model")
        self.calls.append((ucp, end_date, n_days))
        if self.model_error is not None:
            raise self.model_error
        curves = {
            (end_date + timedelta(days=k)).isoformat(): _curva(end_date + timedelta(days=k), 1.03)
            for k in range(1, n_days + 1)
        }
        if self.quitar_dia:
            curves.pop(self.quitar_dia, None)
        return curves, {"modelo_version": self.modelo_version}

    def refresher(self, ucp):  # sync: el router lo corre en threadpool
        self.log.append("refresh")
        if self.refresh_error is not None:
            raise self.refresh_error
        if self.dias_nuevos:
            ultimo = self.hist["fecha"].max().date()
            nuevo = _filas(ultimo + timedelta(days=1), ultimo + timedelta(days=self.dias_nuevos))
            self.hist = pd.concat([self.hist, nuevo], ignore_index=True)
            self.dias_nuevos = 0

    def model_checker(self, ucp):
        self.log.append("model_check")
        return self.modelo_entrenado

    def history(self, ucp):
        self.log.append("history")
        return self.hist.copy()

    def festivos(self, ucp, inicio, fin):
        if self.festivos_error:
            raise ConnectionError("API de festivos caída")
        return [f for f in FESTIVOS if inicio.isoformat() <= f <= fin.isoformat()]


def _registrar_ucp(ucp: str) -> None:
    """Crea data/raw/{ucp}/datos.csv (vacío) en el RAW_DATA_DIR vigente: la UCP pasa la validación."""
    carpeta = Path(data_mod.RAW_DATA_DIR) / ucp
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "datos.csv").write_text("")


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    """RAW_DATA_DIR temporal con la UCP de prueba (el histórico real lo da el fake)."""
    raiz = tmp_path / "data" / "raw"
    raiz.mkdir(parents=True)
    monkeypatch.setattr(data_mod, "RAW_DATA_DIR", raiz)
    _registrar_ucp(UCP)
    return raiz


@pytest.fixture
def env(raw_dir, monkeypatch):
    monkeypatch.delenv(router_mod.ENV_SERVICE_TOKEN, raising=False)
    e = FakeEnv()
    prev = (
        router_mod.set_base_provider(e.provider),
        router_mod.set_data_refresher(e.refresher),
        router_mod.set_history_loader(e.history),
        router_mod.set_festivos_loader(e.festivos),
        router_mod.set_model_checker(e.model_checker),
    )
    yield e
    router_mod.set_base_provider(prev[0])
    router_mod.set_data_refresher(prev[1])
    router_mod.set_history_loader(prev[2])
    router_mod.set_festivos_loader(prev[3])
    router_mod.set_model_checker(prev[4])


def _client(store):
    app = FastAPI()
    app.include_router(router_mod.router)
    app.dependency_overrides[router_mod.get_store_dep] = lambda: store
    return TestClient(app)


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def client(env, store):
    return _client(store)


@pytest.fixture
def client_sin_store(env):
    return _client(None)


BODY = {"ucp": UCP, "semana_inicio": SEMANA.isoformat()}


def _json_nan(payload):
    """Cuerpo JSON con NaN/Infinity literales (httpx los rechaza con json=)."""
    return {"content": json.dumps(payload), "headers": {"Content-Type": "application/json"}}


def _semana_iso(lunes: date = SEMANA):
    return [(lunes + timedelta(days=k)).isoformat() for k in range(7)]


def _crear(client, **extra):
    r = client.post("/adjustments/run", json={**BODY, "usuario": "analista", **extra})
    assert r.status_code == 201, r.text
    return r.json()


def _steps(data):
    return {s["paso"]: s for s in data["steps"]}


def _base_externa(corte: date, lunes: date, escala: float = 1.02):
    domingo = lunes + timedelta(days=6)
    n = (domingo - corte).days
    return {(corte + timedelta(days=k)).isoformat(): _curva(corte + timedelta(days=k), escala)
            for k in range(1, n + 1)}


# ---------------------------------------------------------------------------
# preview / run / get
# ---------------------------------------------------------------------------

def test_preview_funciona_sin_store(client_sin_store, env):
    r = client_sin_store.post("/adjustments/preview", json=BODY)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["id"] is None and data["persistido"] is False and data["version"] is None
    assert [s["paso"] for s in data["steps"]] == pl.STEP_ORDER
    assert data["meta"]["fecha_corte"] == CORTE.isoformat()
    assert data["meta"]["modelo_version"] == "fake_model_v1"
    assert _steps(data)["base"]["diagnostics"]["modelo_version"] == "fake_model_v1"
    assert data["meta"]["extras"]["fuente_base"] == "modelo"
    assert data["meta"]["extras"]["fuente_festivos"] == "api"
    assert "festivos" not in data["meta"]["extras"]
    # refresco ANTES de leer el histórico; modelo con end_date = último dato real
    assert env.log == ["model_check", "refresh", "history", "model"]  # modelo verificado antes de refrescar
    assert env.calls == [(UCP, CORTE, 9)]
    for fecha in _semana_iso():
        assert len(data["final_curves"][fecha]) == 24


def test_refresco_antes_de_build_context_usa_corte_nuevo(client, env):
    env.dias_nuevos = 2  # el refresco trae sábado y domingo reales
    creada = _crear(client)
    nuevo_corte = CORTE + timedelta(days=2)
    assert creada["meta"]["fecha_corte"] == nuevo_corte.isoformat()
    assert creada["meta"]["extras"]["ultimo_dato_real"] == nuevo_corte.isoformat()
    assert env.calls == [(UCP, nuevo_corte, 7)]
    assert sorted(_steps(creada)["base"]["curves"]) == _semana_iso()


def test_semana_pasada_con_modelo_422(client, env):
    for lunes in (date(2026, 11, 9), date(2026, 11, 2)):
        r = client.post("/adjustments/preview", json={"ucp": UCP, "semana_inicio": lunes.isoformat()})
        assert r.status_code == 422, r.text
        assert "datos reales" in r.json()["detail"]
    assert env.calls == []


def test_fecha_emision_solo_con_base_curvas(client, env):
    r = client.post("/adjustments/preview", json={**BODY, "fecha_emision": "2026-11-14"})
    assert r.status_code == 422
    r = client.post("/adjustments/preview", json={**BODY, "modelo_version": "x"})
    assert r.status_code == 422
    assert env.calls == [] and "refresh" not in env.log


def test_endpoints_persistentes_503_sin_store(client_sin_store):
    assert client_sin_store.post("/adjustments/run", json=BODY).status_code == 503
    rid = "00000000-0000-0000-0000-000000000000"
    assert client_sin_store.get(f"/adjustments/{rid}").status_code == 503
    assert client_sin_store.get(f"/adjustments/{rid}/versiones").status_code == 503
    assert client_sin_store.get("/adjustments/aprobada",
                                params={"ucp": UCP, "semana_inicio": SEMANA.isoformat()}).status_code == 503
    assert client_sin_store.patch(f"/adjustments/{rid}/steps/nivel",
                                  json={"estado": "omitido", "version_esperada": 1}).status_code == 503
    assert client_sin_store.post(f"/adjustments/{rid}/approve",
                                 json={"usuario": "x", "version_esperada": 1}).status_code == 503


def test_run_y_get(client):
    creada = _crear(client)
    assert creada["persistido"] is True and creada["id"]
    assert creada["version"] == 1 and creada["meta"]["version"] == 1
    assert creada["meta"]["estado"] == "borrador"
    assert creada["meta"]["creado_por"] == "analista"
    assert [e["accion"] for e in creada["eventos"]] == ["crear"]
    assert creada["eventos"][0]["version"] == 1
    assert creada["meta"]["params_iniciales"] == {} and "params" not in creada["meta"]

    r = client.get(f"/adjustments/{creada['id']}")
    assert r.status_code == 200
    leida = r.json()
    assert leida["steps"] == creada["steps"]
    assert leida["final_curves"] == creada["final_curves"]


def test_404_corrida_inexistente(client):
    rid = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/adjustments/{rid}").status_code == 404
    assert client.get("/adjustments/no-es-uuid").status_code == 404
    assert client.get(f"/adjustments/{rid}/versiones").status_code == 404
    assert client.get(f"/adjustments/{rid}/versiones/1").status_code == 404
    assert client.patch(f"/adjustments/{rid}/steps/nivel",
                        json={"estado": "omitido", "version_esperada": 1}).status_code == 404
    assert client.post(f"/adjustments/{rid}/approve", json={"usuario": "x", "version_esperada": 1}).status_code == 404


def test_validaciones_request(client):
    assert client.post("/adjustments/preview", json={**BODY, "semana_inicio": "2026-11-17"}).status_code == 422
    assert client.post("/adjustments/preview", json={**BODY, "params": {"inexistente": {}}}).status_code == 422
    assert client.post("/adjustments/preview", json={**BODY, "semana_inicio": "9999-12-27"}).status_code == 422
    assert client.post("/adjustments/preview", json={**BODY, "base_ref": "x" * 201}).status_code == 422


@pytest.mark.parametrize("semana", ["0001-01-01", "1989-12-25", "9999-12-27", "2101-01-03"])
def test_semana_fuera_de_rango_422_sin_tocar_nada(client, env, semana):
    assert date.fromisoformat(semana).weekday() == 0  # lunes: el 422 es por el rango, no por el día
    router_mod.set_data_refresher(RefresherQueFalla())
    for ruta in ("/adjustments/preview", "/adjustments/run"):
        r = client.post(ruta, json={**BODY, "semana_inicio": semana})
        assert r.status_code == 422, r.text
        assert "rango" in r.text
    r = client.post("/adjustments/preview", json={**BODY, "semana_inicio": "2026-11-16", "fecha_emision": "0001-01-01",
                                                  "base_curvas": _base_externa(CORTE, SEMANA)})
    assert r.status_code == 422
    assert env.log == []  # ni histórico, ni festivos, ni modelo


def test_overflow_en_params_422(client):
    r = client.post("/adjustments/preview", json={**BODY, "params": {"nivel": {"alpha": 10 ** 400}}})
    assert r.status_code == 422, r.text
    creada = _crear(client)
    r = client.patch(f"/adjustments/{creada['id']}/steps/nivel",
                     json={"estado": "aplicado", "version_esperada": 1, "params": {"alpha": 10 ** 400}})
    assert r.status_code == 422, r.text


def test_base_ref_y_params_iniciales(client):
    creada = _crear(client, base_ref="sphere:pronostico/123", params={"nivel": {"alpha": 0.5}})
    assert creada["meta"]["extras"]["base_ref"] == "sphere:pronostico/123"
    assert creada["meta"]["params_iniciales"] == {"nivel": {"alpha": 0.5}}
    assert creada["eventos"][0]["detalle"]["base_ref"] == "sphere:pronostico/123"
    # tras un PATCH, params_iniciales sigue siendo el request original; el vigente está en el paso
    r = client.patch(f"/adjustments/{creada['id']}/steps/nivel",
                     json={"estado": "aplicado", "version_esperada": 1, "params": {"alpha": 0.75}})
    data = r.json()
    assert data["meta"]["params_iniciales"] == {"nivel": {"alpha": 0.5}}
    assert _steps(data)["nivel"]["params"]["alpha"] == 0.75
    sin_ref = client.post("/adjustments/preview", json=BODY).json()
    assert "base_ref" not in sin_ref["meta"]["extras"]


# ---------------------------------------------------------------------------
# validación de UCP, modelo entrenado y auth
# ---------------------------------------------------------------------------

class RefresherQueFalla:
    def __call__(self, ucp):
        raise AssertionError(f"no debía refrescarse la UCP {ucp!r}")


@pytest.mark.parametrize("ucp,codigo", [("../x", 422), ("a/b", 422), ("Foo", 404), ("", 422),
                                        ("x" * 65, 422), ("..", 422), ("Antioquia ", 422)])
def test_ucp_invalida_no_toca_disco_ni_red(client, env, raw_dir, ucp, codigo):
    router_mod.set_data_refresher(RefresherQueFalla())

    def provider_que_falla(*_a):
        raise AssertionError("no debía llamarse al modelo")

    router_mod.set_base_provider(provider_que_falla)
    antes = sorted(p.relative_to(raw_dir.parent.parent) for p in raw_dir.parent.parent.rglob("*"))
    locks_antes = set(router_mod._ucp_locks)
    for ruta in ("/adjustments/preview", "/adjustments/run"):
        r = client.post(ruta, json={**BODY, "ucp": ucp})
        assert r.status_code == codigo, (ruta, r.text)
    despues = sorted(p.relative_to(raw_dir.parent.parent) for p in raw_dir.parent.parent.rglob("*"))
    assert antes == despues
    assert set(router_mod._ucp_locks) == locks_antes
    assert env.log == []  # ni histórico, ni verificación de modelo
    assert client.get("/adjustments/aprobada",
                      params={"ucp": ucp, "semana_inicio": SEMANA.isoformat()}).status_code in (404, 422)


def test_ucp_symlink_fuera_de_raw_404(client, env, raw_dir, tmp_path):
    afuera = tmp_path / "afuera"
    afuera.mkdir()
    (afuera / "datos.csv").write_text("")
    (raw_dir / "Escape").symlink_to(afuera)
    router_mod.set_data_refresher(RefresherQueFalla())
    assert client.post("/adjustments/preview", json={**BODY, "ucp": "Escape"}).status_code == 404


def test_ucp_sin_modelo_409_sin_refrescar(client, env):
    env.modelo_entrenado = False
    router_mod.set_data_refresher(RefresherQueFalla())
    r = client.post("/adjustments/run", json={**BODY, "usuario": "analista"})
    assert r.status_code == 409 and "/retrain" in r.json()["detail"]
    assert env.calls == [] and env.log == ["model_check"]
    # con base_curvas no se usa el modelo: no se verifica
    curvas = _base_externa(CORTE, SEMANA)
    r = client.post("/adjustments/preview", json={**BODY, "fecha_emision": "2026-11-14", "base_curvas": curvas})
    assert r.status_code == 200, r.text


def test_reaplicar_base_sin_modelo_409(client, env):
    creada = _crear(client)
    env.modelo_entrenado = False
    n = len(env.calls)
    r = client.patch(f"/adjustments/{creada['id']}/steps/base", json={"estado": "aplicado", "version_esperada": 1})
    assert r.status_code == 409 and len(env.calls) == n


def test_auth_con_token(client, env, monkeypatch):
    monkeypatch.setenv(router_mod.ENV_SERVICE_TOKEN, "s3cr3t-largo")
    assert client.post("/adjustments/preview", json=BODY).status_code == 401
    assert client.post("/adjustments/preview", json=BODY,
                       headers={"X-Ajustes-Token": "otro"}).status_code == 401
    assert client.get("/adjustments/00000000-0000-0000-0000-000000000000").status_code == 401
    assert client.post("/adjustments/preview", json={"ucp": "../x"}).status_code == 401  # auth antes que validación
    ok = {"X-Ajustes-Token": "s3cr3t-largo"}
    assert client.post("/adjustments/preview", json=BODY, headers=ok).status_code == 200
    r = client.post("/adjustments/run", json={**BODY, "usuario": "a"}, headers=ok)
    assert r.status_code == 201
    assert client.get(f"/adjustments/{r.json()['id']}", headers=ok).status_code == 200
    assert client.get(f"/adjustments/{r.json()['id']}").status_code == 401


def test_auth_sin_token_permite_y_avisa_una_vez(client, env, monkeypatch, caplog):
    monkeypatch.setattr(router_mod, "_aviso_sin_token", False)
    with caplog.at_level("WARNING", logger=router_mod.logger.name):
        assert client.post("/adjustments/preview", json=BODY).status_code == 200
        assert client.post("/adjustments/preview", json=BODY,
                           headers={"X-Ajustes-Token": "cualquiera"}).status_code == 200
    avisos = [r for r in caplog.records if router_mod.ENV_SERVICE_TOKEN in r.getMessage()]
    assert len(avisos) == 1


@pytest.mark.parametrize("params", [
    {"nivel": {"alpha": 5}},
    {"nivel": {"alpha": "mucho"}},
    {"nivel": {"alpha": None}},
    {"forma": {"fuente": "inventada"}},
    {"control": {"banda_pp": -1}},
])
def test_params_invalidos_422(client, params):
    r = client.post("/adjustments/preview", json={**BODY, "params": params})
    assert r.status_code == 422, r.text


def test_error_del_modelo_502(client, env):
    env.model_error = HTTPException(status_code=500, detail="Error en pipeline de datos: boom\nTraceback...")
    r = client.post("/adjustments/run", json=BODY)
    assert r.status_code == 502, r.text
    detail = r.json()["detail"]
    assert "pronóstico base" in detail["mensaje"] and "Traceback" not in detail["causa"]

    env.model_error = RuntimeError("modelo roto")
    assert client.post("/adjustments/preview", json=BODY).status_code == 502


def test_error_del_refresco_502(client, env):
    env.refresh_error = ConnectionError("XM no responde")
    r = client.post("/adjustments/preview", json=BODY)
    assert r.status_code == 502 and env.calls == []


def test_festivos_caen_a_libreria_con_alerta(client, env):
    env.festivos_error = True
    creada = _crear(client)
    assert creada["meta"]["extras"]["fuente_festivos"] == "libreria"
    alertas = _steps(creada)["base"]["alerts"]
    assert any(a["codigo"] == "FESTIVOS_FUENTE_ALTERNA" and a["nivel"] == "info" for a in alertas)
    # la alerta se conserva al recalcular desde la base
    r = client.patch(f"/adjustments/{creada['id']}/steps/base", json={"estado": "aplicado", "version_esperada": 1})
    assert r.status_code == 200, r.text
    assert any(a["codigo"] == "FESTIVOS_FUENTE_ALTERNA" for a in _steps(r.json())["base"]["alerts"])


# ---------------------------------------------------------------------------
# base_curvas (pronóstico externo)
# ---------------------------------------------------------------------------

def test_base_curvas_no_llama_modelo_ni_refresco(client, env):
    lunes = date(2026, 11, 9)  # semana ya observada: válida con base externa
    emision = date(2026, 11, 7)  # sábado → corte viernes 2026-11-06
    corte = date(2026, 11, 6)
    body = {"ucp": UCP, "semana_inicio": lunes.isoformat(), "fecha_emision": emision.isoformat(),
            "base_curvas": _base_externa(corte, lunes), "modelo_version": "emitido_v7", "usuario": "backend"}
    r = client.post("/adjustments/run", json=body)
    assert r.status_code == 201, r.text
    data = r.json()
    assert env.calls == [] and "refresh" not in env.log
    assert data["meta"]["fecha_corte"] == corte.isoformat()
    assert data["meta"]["modelo_version"] == "emitido_v7"
    assert data["meta"]["extras"]["fuente_base"] == "externa"
    base = _steps(data)["base"]
    assert base["curves"][lunes.isoformat()] == pytest.approx(body["base_curvas"][lunes.isoformat()])
    assert base["diagnostics"]["modelo_version"] == "emitido_v7"

    # re-aplicar una base externa no es posible (no hay modelo que la regenere)
    r = client.patch(f"/adjustments/{data['id']}/steps/base", json={"estado": "aplicado", "version_esperada": 1})
    assert r.status_code == 422


def test_base_curvas_incompleta_422(client, env):
    curvas = _base_externa(CORTE, SEMANA)
    curvas.pop(SEMANA.isoformat())
    ext = {**BODY, "fecha_emision": "2026-11-14"}
    r = client.post("/adjustments/preview", json={**ext, "base_curvas": curvas})
    assert r.status_code == 422 and r.json()["detail"]["faltan"] == [SEMANA.isoformat()]
    curvas = _base_externa(CORTE, SEMANA)
    curvas[SEMANA.isoformat()] = [1.0] * 23
    assert client.post("/adjustments/preview", json={**ext, "base_curvas": curvas}).status_code == 422
    assert env.calls == []


def _nan_base_flow(client):
    curvas = _base_externa(CORTE, SEMANA)
    martes = (SEMANA + timedelta(days=1)).isoformat()
    curvas[martes][5] = float("nan")
    r = client.post("/adjustments/run", **_json_nan({**BODY, "fecha_emision": "2026-11-14", "base_curvas": curvas}))
    assert r.status_code == 201, r.text
    creada = r.json()
    assert creada["bloqueos"], "NaN en la base debe bloquear la aprobación"
    assert creada["final_curves"][martes][5] is None or creada["bloqueos"]
    r = client.post(f"/adjustments/{creada['id']}/approve", json={"usuario": "jefe", "version_esperada": 1})
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["bloqueos"]
    leida = client.get(f"/adjustments/{creada['id']}").json()
    assert leida["meta"]["estado"] == "borrador" and leida["eventos"][-1]["accion"] == "aprobacion_rechazada"


def test_nan_en_base_201_con_bloqueo_y_approve_422(client):
    _nan_base_flow(client)


def test_nan_en_base_del_modelo(client, env):
    original = env.provider

    async def con_nan(ucp, end_date, n_days):
        curves, info = await original(ucp, end_date, n_days)
        curves[SEMANA.isoformat()][0] = float("inf")
        return curves, info

    router_mod.set_base_provider(con_nan)
    creada = _crear(client)
    assert creada["bloqueos"]


# ---------------------------------------------------------------------------
# PATCH (pasos por nombre + versión)
# ---------------------------------------------------------------------------

def test_patch_omitir_paso_recalcula_posteriores(client, env):
    creada = _crear(client)
    llamadas_base = len(env.calls)
    r = client.patch(f"/adjustments/{creada['id']}/steps/nivel",
                     json={"estado": "omitido", "version_esperada": 1, "motivo": "probar sin nivel",
                           "usuario": "analista"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["version"] == 2
    steps = _steps(data)
    assert steps["nivel"]["estado"] == "omitido"
    assert steps["nivel"]["curves"] == steps["referencia"]["curves"]  # omitido = identidad
    for fecha in _semana_iso():
        assert sum(steps["forma"]["curves"][fecha]) == pytest.approx(sum(steps["nivel"]["curves"][fecha]), rel=1e-6)
    assert len(env.calls) == llamadas_base  # no se volvió a llamar al modelo
    ev = data["eventos"][-1]
    assert ev["accion"] == "omitir_paso" and ev["paso"] == "nivel" and ev["version"] == 2
    assert ev["usuario"] == "analista" and ev["motivo"] == "probar sin nivel"
    assert client.get(f"/adjustments/{creada['id']}").json()["steps"] == data["steps"]


def test_patch_editar_curvas_conserva_calculada(client):
    creada = _crear(client)
    calculado = _steps(creada)["nivel"]["curves"]
    fecha = SEMANA.isoformat()
    editada = [500.0] * 24
    r = client.patch(f"/adjustments/{creada['id']}/steps/nivel",
                     json={"estado": "editado", "version_esperada": 1, "curvas": {fecha: editada},
                           "motivo": "ajuste manual"})
    assert r.status_code == 200, r.text
    nivel = _steps(r.json())["nivel"]
    assert nivel["estado"] == "editado"
    assert nivel["curves"][fecha] == editada
    assert nivel["curvas_calculadas"][fecha] == pytest.approx(calculado[fecha])  # automático vs analista
    otro = (SEMANA + timedelta(days=1)).isoformat()
    assert nivel["curves"][otro] == pytest.approx(calculado[otro])  # no editado = calculado
    assert nivel["diagnostics"]["fechas_editadas"] == [fecha]

    # una segunda edición acumula los días ya editados
    r = client.patch(f"/adjustments/{creada['id']}/steps/nivel",
                     json={"estado": "editado", "version_esperada": 2, "curvas": {otro: [400.0] * 24}})
    assert r.status_code == 200, r.text
    steps = _steps(r.json())
    assert steps["nivel"]["curves"][fecha] == editada and steps["nivel"]["curves"][otro] == [400.0] * 24
    assert steps["nivel"]["diagnostics"]["fechas_editadas"] == sorted([fecha, otro])
    assert steps["nivel"]["curvas_calculadas"][otro] == pytest.approx(calculado[otro])
    assert sum(steps["forma"]["curves"][fecha]) == pytest.approx(500.0 * 24, rel=1e-6)


def _reeditar_base_flow(client, ucp=UCP):
    """Editar base día X y luego día Y: curvas_calculadas[X] sigue siendo la salida del modelo."""
    creada = _crear(client, ucp=ucp)
    rid = creada["id"]
    modelo = _steps(creada)["base"]["curves"]
    x, y = SEMANA.isoformat(), (SEMANA + timedelta(days=2)).isoformat()
    r = client.patch(f"/adjustments/{rid}/steps/base",
                     json={"estado": "editado", "version_esperada": 1, "curvas": {x: [450.0] * 24}})
    assert r.status_code == 200, r.text
    r = client.patch(f"/adjustments/{rid}/steps/base",
                     json={"estado": "editado", "version_esperada": 2, "curvas": {y: [420.0] * 24}})
    assert r.status_code == 200, r.text
    base = _steps(r.json())["base"]
    assert base["curves"][x] == [450.0] * 24 and base["curves"][y] == [420.0] * 24
    assert base["curvas_calculadas"][x] == pytest.approx(modelo[x])
    assert base["curvas_calculadas"][y] == pytest.approx(modelo[y])
    assert base["diagnostics"]["fechas_editadas"] == sorted([x, y])
    leida = _steps(client.get(f"/adjustments/{rid}").json())["base"]
    assert leida["curvas_calculadas"][x] == pytest.approx(modelo[x])
    return rid


def test_reeditar_base_conserva_salida_del_modelo(client):
    _reeditar_base_flow(client)


@pytest.mark.parametrize("body", [
    {"estado": "editado"},                                            # sin curvas
    {"estado": "editado", "curvas": {"2026-11-16": [1.0] * 23}},      # 23 periodos
    {"estado": "editado", "curvas": {"2026-11-16": [-1.0] * 24}},     # negativos
    {"estado": "editado", "curvas": {"2030-01-01": [1.0] * 24}},      # fuera del horizonte
    {"estado": "editado", "curvas": {"2026-11-14": [1.0] * 24}},      # brecha (solo editable en base)
    {"estado": "editado", "curvas": {"16/11/2026": [1.0] * 24}},      # fecha inválida
    {"estado": "omitido", "curvas": {"2026-11-16": [1.0] * 24}},      # curvas sin editado
    {"estado": "pausado"},                                            # estado inválido
    {"estado": "aplicado", "params": {"alpha": 7}},                   # params inválidos
])
def test_patch_invalido_422(client, body):
    creada = _crear(client)
    r = client.patch(f"/adjustments/{creada['id']}/steps/nivel", json={"version_esperada": 1, **body})
    assert r.status_code == 422, r.text
    assert client.get(f"/adjustments/{creada['id']}").json()["version"] == 1


def test_patch_nan_en_edicion_422(client):
    creada = _crear(client)
    curva = [100.0] * 24
    curva[3] = float("nan")
    r = client.patch(f"/adjustments/{creada['id']}/steps/forma",
                     **_json_nan({"estado": "editado", "version_esperada": 1, "curvas": {SEMANA.isoformat(): curva}}))
    assert r.status_code == 422, r.text


def test_patch_paso_invalido_y_omitir_base(client):
    creada = _crear(client)
    rid = creada["id"]
    for paso in ("9", "2", "inexistente"):
        assert client.patch(f"/adjustments/{rid}/steps/{paso}",
                            json={"estado": "omitido", "version_esperada": 1}).status_code == 422
    assert client.patch(f"/adjustments/{rid}/steps/base",
                        json={"estado": "omitido", "version_esperada": 1}).status_code == 422
    assert client.patch(f"/adjustments/{rid}/steps/control",
                        json={"estado": "editado", "version_esperada": 1,
                              "curvas": {SEMANA.isoformat(): [1.0] * 24}}).status_code == 422
    # version_esperada es obligatoria
    assert client.patch(f"/adjustments/{rid}/steps/nivel", json={"estado": "omitido"}).status_code == 422


def test_patch_version_vieja_409(client):
    creada = _crear(client)
    rid = creada["id"]
    r = client.patch(f"/adjustments/{rid}/steps/nivel", json={"estado": "omitido", "version_esperada": 1})
    assert r.status_code == 200
    r = client.patch(f"/adjustments/{rid}/steps/forma", json={"estado": "omitido", "version_esperada": 1})
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["version_actual"] == 2
    leida = client.get(f"/adjustments/{rid}").json()
    assert leida["version"] == 2 and _steps(leida)["forma"]["estado"] == "aplicado"


class _StoreConCarrera(InMemoryStore):
    """Simula otra escritura entre la lectura del router y su replace_records."""

    def __init__(self) -> None:
        super().__init__()
        self.interferir = False

    def replace_records(self, run_id, records, evento, version_esperada, meta_updates=None):
        if self.interferir:
            self.interferir = False
            actual = self.get_run(run_id)
            super().replace_records(run_id, actual["records"], {"accion": "editar_paso", "usuario": "otro"},
                                    version_esperada)
        return super().replace_records(run_id, records, evento, version_esperada, meta_updates)


def test_patch_concurrente_409(env):
    store = _StoreConCarrera()
    client = _client(store)
    creada = _crear(client)
    store.interferir = True
    r = client.patch(f"/adjustments/{creada['id']}/steps/nivel", json={"estado": "omitido", "version_esperada": 1})
    assert r.status_code == 409, r.text
    leida = client.get(f"/adjustments/{creada['id']}").json()
    assert leida["version"] == 2 and _steps(leida)["nivel"]["estado"] == "aplicado"  # ganó la otra escritura


def test_patch_reaplicar_base_actualiza_modelo_version(client, env):
    creada = _crear(client)
    n = len(env.calls)
    env.modelo_version = "fake_model_v2"
    r = client.patch(f"/adjustments/{creada['id']}/steps/base", json={"estado": "aplicado", "version_esperada": 1})
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(env.calls) == n + 1 and env.calls[-1] == (UCP, CORTE, 9)
    assert data["meta"]["modelo_version"] == "fake_model_v2"
    assert _steps(data)["base"]["diagnostics"]["modelo_version"] == "fake_model_v2"
    ev = data["eventos"][-1]
    assert ev["detalle"]["modelo_version"] == "fake_model_v2"
    assert ev["detalle"]["modelo_version_anterior"] == "fake_model_v1"


def test_patch_reaplicar_base_con_datos_nuevos_422(client, env):
    creada = _crear(client)
    env.dias_nuevos = 1
    n = len(env.calls)
    r = client.patch(f"/adjustments/{creada['id']}/steps/base", json={"estado": "aplicado", "version_esperada": 1})
    assert r.status_code == 422, r.text
    assert len(env.calls) == n  # nunca se llama al modelo con un corte pasado


# ---------------------------------------------------------------------------
# approve
# ---------------------------------------------------------------------------

def test_approve_sin_bloqueos_y_409_despues(client):
    creada = _crear(client)
    assert creada["bloqueos"] == []
    rid = creada["id"]
    r = client.post(f"/adjustments/{rid}/approve", json={"usuario": "jefe", "motivo": "ok", "version_esperada": 1})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["meta"]["estado"] == "aprobada" and data["version"] == 2
    assert data["meta"]["approved_by"] == "jefe"
    ev = data["eventos"][-1]
    assert ev["accion"] == "aprobar" and ev["detalle"]["excepciones_justificadas"] == []
    assert "warnings_vigentes" in ev["detalle"]

    # congelada
    assert client.patch(f"/adjustments/{rid}/steps/nivel",
                        json={"estado": "omitido", "version_esperada": 2}).status_code == 409
    assert client.post(f"/adjustments/{rid}/approve", json={"usuario": "jefe", "version_esperada": 2}).status_code == 409


def _warnings(data):
    return sorted({a["codigo"] for s in data["steps"] for a in s["alerts"] if a["nivel"] == "warning"})


def test_approve_excepciones_justificadas(client):
    creada = _crear(client)
    rid = creada["id"]
    # edición fuerte del nivel → warning del enclavamiento de control
    fecha = SEMANA.isoformat()
    editada = [x * 1.5 for x in _steps(creada)["nivel"]["curves"][fecha]]
    r = client.patch(f"/adjustments/{rid}/steps/nivel",
                     json={"estado": "editado", "version_esperada": 1, "curvas": {fecha: editada}})
    assert r.status_code == 200, r.text
    vigentes = _warnings(r.json())
    assert vigentes, "la edición debería producir al menos un warning"

    # código que no está vigente → 422 y no se aprueba
    r = client.post(f"/adjustments/{rid}/approve",
                    json={"usuario": "jefe", "version_esperada": 2,
                          "excepciones_justificadas": [vigentes[0], "NO_EXISTE"]})
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["codigos_invalidos"] == ["NO_EXISTE"]
    assert r.json()["detail"]["warnings_vigentes"] == vigentes
    assert client.get(f"/adjustments/{rid}").json()["meta"]["estado"] == "borrador"

    # justificar uno (no es obligatorio justificar todos)
    r = client.post(f"/adjustments/{rid}/approve",
                    json={"usuario": "jefe", "version_esperada": 2,
                          "excepciones_justificadas": [vigentes[0], vigentes[0]]})
    assert r.status_code == 200, r.text
    det = r.json()["eventos"][-1]["detalle"]
    assert det["excepciones_justificadas"] == [vigentes[0]]
    assert sorted({w["codigo"] for w in det["warnings_vigentes"]}) == vigentes
    assert det["warnings_sin_justificar"] == vigentes[1:]


def test_approve_version_vieja_409(client):
    creada = _crear(client)
    rid = creada["id"]
    assert client.patch(f"/adjustments/{rid}/steps/nivel",
                        json={"estado": "omitido", "version_esperada": 1}).status_code == 200
    r = client.post(f"/adjustments/{rid}/approve", json={"usuario": "jefe", "version_esperada": 1})
    assert r.status_code == 409 and r.json()["detail"]["version_actual"] == 2
    assert client.post(f"/adjustments/{rid}/approve", json={"usuario": "jefe"}).status_code == 422  # obligatoria


def _approve_reemplaza_flow(client):
    params = {"ucp": UCP, "semana_inicio": SEMANA.isoformat()}
    assert client.get("/adjustments/aprobada", params=params).status_code == 404
    a = _crear(client)
    assert client.post(f"/adjustments/{a['id']}/approve", json={"usuario": "jefe", "version_esperada": 1}).status_code == 200
    vigente = client.get("/adjustments/aprobada", params=params)
    assert vigente.status_code == 200 and vigente.json()["id"] == a["id"]

    b = _crear(client, params={"nivel": {"alpha": 0.5}})
    r = client.post(f"/adjustments/{b['id']}/approve", json={"usuario": "jefe", "version_esperada": 1})
    assert r.status_code == 200, r.text
    vigente = client.get("/adjustments/aprobada", params=params).json()
    assert vigente["id"] == b["id"] and vigente["meta"]["estado"] == "aprobada"
    vieja = client.get(f"/adjustments/{a['id']}").json()
    assert vieja["meta"]["estado"] == "reemplazada" and vieja["meta"]["reemplazada_por"] == b["id"]
    assert vieja["eventos"][-1]["accion"] == "reemplazar"
    assert vieja["eventos"][-1]["version"] == vieja["version"]  # misma versión en InMemory y Postgres
    # la reemplazada sigue congelada
    assert client.patch(f"/adjustments/{a['id']}/steps/nivel",
                        json={"estado": "omitido", "version_esperada": vieja["version"]}).status_code == 409
    otra = {"ucp": UCP, "semana_inicio": "2026-11-23"}
    assert client.get("/adjustments/aprobada", params=otra).status_code == 404
    return a, b


def test_approve_reemplaza_aprobada_anterior(client):
    _approve_reemplaza_flow(client)


def test_approve_con_bloqueos_422(client, monkeypatch):
    creada = _crear(client)
    bloqueo = Alert(nivel="bloqueo", codigo="PERIODOS_FALTANTES", mensaje="faltan periodos",
                    fechas=[SEMANA.isoformat()])
    monkeypatch.setattr(pl, "approval_blockers", lambda records: [bloqueo])
    r = client.post(f"/adjustments/{creada['id']}/approve", json={"usuario": "jefe", "version_esperada": 1})
    assert r.status_code == 422
    assert r.json()["detail"]["bloqueos"][0]["codigo"] == "PERIODOS_FALTANTES"
    monkeypatch.undo()
    leida = client.get(f"/adjustments/{creada['id']}").json()
    assert leida["meta"]["estado"] == "borrador" and leida["version"] == 1
    assert leida["eventos"][-1]["accion"] == "aprobacion_rechazada"


def test_approve_bloqueo_real_por_base_incompleta(client, env):
    """Base del modelo sin un día de la semana objetivo: bloqueo de aprobación."""
    env.quitar_dia = (SEMANA + timedelta(days=2)).isoformat()
    creada = _crear(client)
    assert creada["bloqueos"], "una base incompleta debería bloquear la aprobación"
    r = client.post(f"/adjustments/{creada['id']}/approve", json={"usuario": "jefe", "version_esperada": 1})
    assert r.status_code == 422 and r.json()["detail"]["bloqueos"]


def test_historial_de_versiones(client):
    creada = _crear(client)
    rid = creada["id"]
    client.patch(f"/adjustments/{rid}/steps/nivel", json={"estado": "omitido", "version_esperada": 1,
                                                          "motivo": "sin nivel", "usuario": "ana"})
    client.post(f"/adjustments/{rid}/approve", json={"usuario": "jefe", "version_esperada": 2})
    versiones = client.get(f"/adjustments/{rid}/versiones").json()
    assert [v["version"] for v in versiones] == [1, 2, 3]
    assert versiones[1]["creado_por"] == "ana" and versiones[1]["motivo"] == "sin nivel"
    v1 = client.get(f"/adjustments/{rid}/versiones/1").json()
    assert v1["steps"] == creada["steps"]
    v2 = client.get(f"/adjustments/{rid}/versiones/2").json()
    assert _steps(v2)["nivel"]["estado"] == "omitido"
    assert client.get(f"/adjustments/{rid}/versiones/9").status_code == 404


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def _records_fake():
    curva = {SEMANA.isoformat(): [1.0] * 24}
    return [
        StepRecord(orden=0, paso="base", estado=StepState.APLICADO, params={},
                   result=StepResult(curves=curva, diagnostics={"n": 1})),
        StepRecord(orden=1, paso="referencia", estado=StepState.OMITIDO, params={"x": 1},
                   result=StepResult(curves=curva, alerts=[Alert("warning", "X", "m", [SEMANA.isoformat()])])),
    ]


def _meta(ucp=UCP):
    return {"ucp": ucp, "semana_inicio": SEMANA, "fecha_corte": CORTE, "fecha_emision": CORTE + timedelta(days=1),
            "modelo_version": "m1", "params_iniciales": {"nivel": {"alpha": 0.5}}, "extras": {"fuente_base": "modelo"},
            "creado_por": "t"}


def _ejercitar_store(store, ucp=UCP):
    rid = store.create_run(_meta(ucp), _records_fake(), {"usuario": "t", "accion": "crear", "detalle": {"a": 1}})
    run = store.get_run(rid)
    assert run["meta"]["estado"] == "borrador" and run["meta"]["version"] == 1
    assert run["meta"]["semana_inicio"] == SEMANA.isoformat()
    assert run["meta"]["params_iniciales"] == {"nivel": {"alpha": 0.5}}
    assert [(e["accion"], e["version"], e["detalle"]) for e in run["eventos"]] == [("crear", 1, {"a": 1})]
    with pytest.raises(ValueError):  # evento inválido: no se crea nada
        store.create_run(_meta(ucp), _records_fake(), {"usuario": "t"})
    assert run["meta"]["extras"] == {"fuente_base": "modelo"}
    assert [r.to_dict() for r in run["records"]] == [r.to_dict() for r in _records_fake()]

    nuevos = _records_fake()[:1]
    with pytest.raises(VersionConflictError) as exc:
        store.replace_records(rid, nuevos, {"accion": "editar_paso"}, version_esperada=7)
    assert exc.value.actual == 1
    v = store.replace_records(rid, nuevos, {"usuario": "u", "accion": "editar_paso", "paso": "referencia",
                                            "motivo": "m"}, version_esperada=1, meta_updates={"modelo_version": "m2"})
    assert v == 2
    run = store.get_run(rid)
    assert len(run["records"]) == 1 and run["meta"]["version"] == 2 and run["meta"]["modelo_version"] == "m2"
    ev = run["eventos"][-1]
    assert ev["accion"] == "editar_paso" and ev["paso"] == "referencia" and ev["version"] == 2
    with pytest.raises(ValueError):
        store.replace_records(rid, nuevos, {"accion": "x"}, version_esperada=2, meta_updates={"ucp": "otra"})

    # NaN/inf no rompen la persistencia (se guardan como null)
    con_nan = _records_fake()
    con_nan[0].result.curves[SEMANA.isoformat()][0] = float("nan")
    assert store.replace_records(rid, con_nan, {"accion": "editar_paso"}, version_esperada=2) == 3
    assert store.get_run(rid)["records"][0].result.curves[SEMANA.isoformat()][0] is None

    # approve re-evalúa bloqueos dentro del lock
    bloqueo = Alert("bloqueo", "PERIODOS_FALTANTES", "x")
    with pytest.raises(ApprovalBlockedError):
        store.approve(rid, "jefe", None, version_esperada=3, check=lambda recs: [bloqueo])
    with pytest.raises(VersionConflictError):
        store.approve(rid, "jefe", None, version_esperada=2)
    vistos = []
    assert store.approve(rid, "jefe", "ok", version_esperada=3,
                         check=lambda recs: vistos.append(len(recs)) or [], detalle={"k": 1}) == 4
    assert vistos == [2]
    run = store.get_run(rid)
    assert run["meta"]["estado"] == "aprobada" and run["meta"]["approved_by"] == "jefe"
    assert run["meta"]["approved_at"] and run["eventos"][-1]["detalle"] == {"k": 1}
    assert store.get_approved(ucp, SEMANA)["meta"]["id"] == rid
    with pytest.raises(RunApprovedError):
        store.replace_records(rid, nuevos, {"accion": "editar_paso"}, version_esperada=4)
    with pytest.raises(RunApprovedError):
        store.approve(rid, "jefe", None, version_esperada=4)

    # otra aprobada para la misma (ucp, semana) reemplaza a la anterior
    rid2 = store.create_run(_meta(ucp), _records_fake())
    store.approve(rid2, "jefe", None, version_esperada=1)
    viejo_run = store.get_run(rid)
    viejo = viejo_run["meta"]
    assert viejo["estado"] == "reemplazada" and viejo["reemplazada_por"] == rid2
    assert viejo_run["eventos"][-1]["accion"] == "reemplazar" and viejo_run["eventos"][-1]["version"] == 4
    assert store.get_approved(ucp, SEMANA.isoformat())["meta"]["id"] == rid2
    assert store.get_approved(ucp, SEMANA + timedelta(days=7)) is None

    versiones = store.list_versions(rid)
    assert [x["version"] for x in versiones] == [1, 2, 3, 4]
    assert versiones[0]["motivo"] == "crear"
    v1 = store.get_version(rid, 1)
    assert [r.to_dict() for r in v1["records"]] == [r.to_dict() for r in _records_fake()]
    assert len(store.get_version(rid, 2)["records"]) == 1
    assert store.get_version(rid, 99) is None

    falso = "00000000-0000-0000-0000-000000000000"
    assert store.get_run(falso) is None and store.get_run("no-es-uuid") is None
    assert store.list_versions(falso) is None
    with pytest.raises(RunNotFoundError):
        store.replace_records(falso, nuevos, {"accion": "x"}, version_esperada=1)
    with pytest.raises(RunNotFoundError):
        store.approve(falso, "u", None, version_esperada=1)
    return rid, rid2


def test_inmemory_store():
    _ejercitar_store(InMemoryStore())


# ---------------------------------------------------------------------------
# Postgres temporal (solo con AJUSTES_DATABASE_URL)
# ---------------------------------------------------------------------------

def _apply_schema():
    script = Path(__file__).resolve().parents[1] / "scripts" / "apply_adjustments_schema.py"
    spec = importlib.util.spec_from_file_location("apply_adjustments_schema", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.apply_schema


@pytest.fixture(scope="module")
def pg():
    """(dsn, ucp) sobre una base temporal que se borra al final (o filas propias si no hay CREATEDB)."""
    dsn = os.getenv("AJUSTES_DATABASE_URL")
    if not dsn:
        pytest.skip("AJUSTES_DATABASE_URL no definido")
    import psycopg2
    from psycopg2.extensions import make_dsn, parse_dsn

    nombre = f"epm_ajustes_test_{uuid.uuid4().hex[:10]}"
    admin = psycopg2.connect(dsn)
    admin.autocommit = True
    try:
        with admin.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{nombre}"')
        temporal = True
        test_dsn = make_dsn(dsn, dbname=nombre)
    except psycopg2.Error:
        temporal = False
        test_dsn = dsn
    ucp = UCP if temporal else f"TEST_{uuid.uuid4().hex[:8]}"
    try:
        apply_schema = _apply_schema()
        apply_schema(test_dsn)
        apply_schema(test_dsn)  # idempotente
        yield test_dsn, ucp
    finally:
        if temporal:
            with admin.cursor() as cur:
                cur.execute(f'DROP DATABASE IF EXISTS "{nombre}" WITH (FORCE)')
        else:
            conn = psycopg2.connect(test_dsn)
            with conn, conn.cursor() as cur:
                cur.execute("DELETE FROM ajustes.corrida WHERE ucp = %s", (ucp,))
            conn.close()
        admin.close()
    assert parse_dsn(test_dsn)


def test_postgres_store(pg):
    dsn, ucp = pg
    store = PostgresStore(dsn)
    _ejercitar_store(store, ucp)

    # replace_records es atómico: si falla a mitad, no se pierden los pasos previos ni sube la versión
    rid = store.create_run(_meta(ucp), _records_fake())
    malo = _records_fake() + [_records_fake()[0]]  # paso base duplicado -> viola PK (corrida_id, paso)
    with pytest.raises(Exception):
        store.replace_records(rid, malo, {"accion": "editar_paso"}, version_esperada=1)
    run = store.get_run(rid)
    assert len(run["records"]) == 2 and run["eventos"] == [] and run["meta"]["version"] == 1

    # list_runs: dos queries (corridas + pasos por ANY), sin eventos y con el límite acotado
    corridas = store.list_runs(ucp, None, None, 10 ** 9)
    assert corridas and all(set(c) == {"meta", "records"} and c["records"] for c in corridas)
    semanas = [c["meta"]["semana_inicio"] for c in corridas]
    assert semanas == sorted(semanas, reverse=True)


def test_postgres_aprobaciones_concurrentes(pg):
    dsn, ucp = pg
    store = PostgresStore(dsn)
    semana = SEMANA + timedelta(days=70)
    meta = {**_meta(ucp), "semana_inicio": semana}
    ids = [store.create_run(meta, _records_fake()) for _ in range(4)]
    barrera = threading.Barrier(len(ids))

    def aprobar(rid):
        barrera.wait()
        return store.approve(rid, "jefe", None, version_esperada=1)

    with ThreadPoolExecutor(len(ids)) as ex:
        list(ex.map(aprobar, ids))
    estados = [store.get_run(r)["meta"]["estado"] for r in ids]
    assert estados.count("aprobada") == 1 and estados.count("reemplazada") == 3
    assert store.get_approved(ucp, semana)["meta"]["estado"] == "aprobada"

    # dos PATCH concurrentes con la misma versión: uno gana, el otro 409
    rid = store.create_run({**meta, "semana_inicio": semana + timedelta(days=7)}, _records_fake())
    barrera2 = threading.Barrier(2)

    def editar(_):
        barrera2.wait()
        try:
            return store.replace_records(rid, _records_fake(), {"accion": "editar_paso"}, version_esperada=1)
        except VersionConflictError:
            return "409"

    with ThreadPoolExecutor(2) as ex:
        resultados = sorted(map(str, ex.map(editar, range(2))))
    assert resultados == ["2", "409"]


def test_postgres_flujo_router(pg, env):
    dsn, ucp = pg
    client = _client(PostgresStore(dsn))
    _registrar_ucp(ucp)
    body = {"ucp": ucp, "semana_inicio": SEMANA.isoformat()}

    r = client.post("/adjustments/run", json={**body, "usuario": "analista"})
    assert r.status_code == 201, r.text
    creada = r.json()
    rid = creada["id"]
    r = client.patch(f"/adjustments/{rid}/steps/nivel",
                     json={"estado": "editado", "version_esperada": 1,
                           "curvas": {SEMANA.isoformat(): [500.0] * 24}})
    assert r.status_code == 200, r.text
    nivel = _steps(r.json())["nivel"]
    assert nivel["curvas_calculadas"][SEMANA.isoformat()] == pytest.approx(
        _steps(creada)["nivel"]["curves"][SEMANA.isoformat()])
    assert client.patch(f"/adjustments/{rid}/steps/forma",
                        json={"estado": "omitido", "version_esperada": 1}).status_code == 409
    r = client.post(f"/adjustments/{rid}/approve", json={"usuario": "jefe", "version_esperada": 2})
    assert r.status_code == 200, r.text
    assert client.get("/adjustments/aprobada",
                      params={"ucp": ucp, "semana_inicio": SEMANA.isoformat()}).json()["id"] == rid
    versiones = client.get(f"/adjustments/{rid}/versiones").json()
    assert [v["version"] for v in versiones] == [1, 2, 3]

    # segunda corrida aprobada reemplaza a la primera
    r = client.post("/adjustments/run", json={**body, "usuario": "analista"})
    rid2 = r.json()["id"]
    assert client.post(f"/adjustments/{rid2}/approve", json={"usuario": "jefe", "version_esperada": 1}).status_code == 200
    vieja = client.get(f"/adjustments/{rid}").json()["meta"]
    assert vieja["estado"] == "reemplazada" and vieja["reemplazada_por"] == rid2

    # NaN en base externa: 201 con bloqueo y approve 422 (no rompe jsonb)
    curvas = _base_externa(CORTE, SEMANA)
    martes = (SEMANA + timedelta(days=1)).isoformat()
    curvas[martes][5] = float("nan")
    r = client.post("/adjustments/run", **_json_nan({**body, "fecha_emision": "2026-11-14", "base_curvas": curvas}))
    assert r.status_code == 201, r.text
    assert r.json()["bloqueos"]
    r = client.post(f"/adjustments/{r.json()['id']}/approve", json={"usuario": "jefe", "version_esperada": 1})
    assert r.status_code == 422


def test_postgres_router_reeditar_base(pg, env):
    dsn, ucp = pg
    _registrar_ucp(ucp)
    client = _client(PostgresStore(dsn))
    _reeditar_base_flow(client, ucp)
