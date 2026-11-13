"""Integración de la fase 4 en el router: /automatico, eventos en control y /aprendizaje."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta

import pytest

from src.adjustments import router as router_mod
from tests.test_adjustments_router import (  # noqa: F401 — fixtures reexportados
    BODY,
    SEMANA,
    UCP,
    _client,
    _crear,
    _steps,
    client,
    client_sin_store,
    env,
    raw_dir,
    store,
)


def _semana_iso(lunes=SEMANA):
    return [(lunes + timedelta(days=k)).isoformat() for k in range(7)]


@pytest.fixture(autouse=True)
def _sin_openai(monkeypatch):
    """Por defecto ningún test de este módulo consulta OpenAI (los tests lo sobreescriben)."""

    async def vacio(ucp, lunes, **kwargs):
        return {}

    monkeypatch.setattr(router_mod.eventos, "eventos_de_semana", vacio)
    return vacio


# ---------------------------------------------------------------------------
# GET /adjustments/automatico
# ---------------------------------------------------------------------------

def test_automatico_200_con_ucp_real(client):
    r = client.get("/adjustments/automatico", params={"ucp": UCP})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ucp"] == UCP
    assert set(data["tipos"]) == {"laboral", "sabado", "domingo", "festivo"}
    assert data["tipos"]["laboral"]["forma"] == "automatico"
    assert data["fuente"].endswith("backtest_modelo_20261113.md")
    # Contrato de 3 estados para el front: `segmentos` + `recomendaciones`.
    assert data["segmentos"]["domingo"]["nivel"] == "no_aplicar"
    assert data["segmentos"]["laboral"]["forma"] == "automatico"
    assert {"tipo_dia": "domingo", "paso": "nivel", "recomendacion": "no_aplicar"} in data[
        "recomendaciones"
    ]


def test_automatico_404_ucp_inexistente(client):
    assert client.get("/adjustments/automatico", params={"ucp": "NoExiste"}).status_code == 404


def test_automatico_422_ucp_invalida(client):
    assert client.get("/adjustments/automatico", params={"ucp": "../x"}).status_code == 422


def test_automatico_401_con_token(client, monkeypatch):
    monkeypatch.setenv(router_mod.ENV_SERVICE_TOKEN, "s3cr3t-largo")
    assert client.get("/adjustments/automatico", params={"ucp": UCP}).status_code == 401
    ok = {"X-Ajustes-Token": "s3cr3t-largo"}
    assert client.get("/adjustments/automatico", params={"ucp": UCP}, headers=ok).status_code == 200


def test_automatico_no_requiere_store_ni_lo_captura_run_id(client_sin_store):
    # /{run_id} devolvería 503 sin store; /automatico responde 200 (ruta registrada antes)
    assert client_sin_store.get("/adjustments/automatico", params={"ucp": UCP}).status_code == 200


# ---------------------------------------------------------------------------
# Eventos (F06) en el paso control
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("excepcion", [RuntimeError("openai caído"), asyncio.TimeoutError()])
def test_eventos_fallo_no_rompe_la_corrida(client, monkeypatch, excepcion):
    async def falla(ucp, lunes, **kwargs):
        raise excepcion

    monkeypatch.setattr(router_mod.eventos, "eventos_de_semana", falla)
    r = client.post("/adjustments/preview", json=BODY)
    assert r.status_code == 200, r.text
    assert not any(a["codigo"] == "EVENTOS_SEMANA" for a in _steps(r.json())["control"]["alerts"])


def test_eventos_alerta_en_control_sin_exponer_dict(client, monkeypatch):
    eventos = {"2026-11-17": "Partido de la selección", "2026-11-19": "Festival"}
    llamadas = []

    async def falso(ucp, lunes, **kwargs):
        llamadas.append((ucp, lunes))
        return eventos

    monkeypatch.setattr(router_mod.eventos, "eventos_de_semana", falso)
    r = client.post("/adjustments/preview", json=BODY)
    assert r.status_code == 200, r.text
    data = r.json()
    alerta = next(a for a in _steps(data)["control"]["alerts"] if a["codigo"] == "EVENTOS_SEMANA")
    assert alerta["detalle"]["eventos"] == eventos
    assert alerta["detalle"]["fuente"] == "openai" and alerta["detalle"]["verificado"] is False
    assert "eventos" not in data["meta"]["extras"]
    assert llamadas == [(UCP, SEMANA)]


def test_eventos_se_reusan_en_patch(client, monkeypatch):
    llamadas = []

    async def falso(ucp, lunes, **kwargs):
        llamadas.append(lunes)
        return {"2026-11-18": "Concierto"}

    monkeypatch.setattr(router_mod.eventos, "eventos_de_semana", falso)
    creada = _crear(client)
    assert llamadas == [SEMANA]
    r = client.patch(f"/adjustments/{creada['id']}/steps/nivel",
                     json={"estado": "omitido", "version_esperada": 1})
    assert r.status_code == 200, r.text
    assert llamadas == [SEMANA]  # se reusó meta.extras.eventos, no se volvió a consultar
    assert any(a["codigo"] == "EVENTOS_SEMANA" for a in _steps(r.json())["control"]["alerts"])


def test_patch_corrida_vieja_sin_eventos_no_consulta(client, store, monkeypatch):
    llamadas = []

    async def falso(ucp, lunes, **kwargs):
        llamadas.append(lunes)
        return {"2026-11-18": "Concierto"}

    monkeypatch.setattr(router_mod.eventos, "eventos_de_semana", falso)
    creada = _crear(client)
    assert llamadas == [SEMANA]

    # Simula una corrida vieja: sin `meta.extras.eventos`.
    store._runs[creada["id"]]["meta"]["extras"].pop("eventos", None)
    r = client.patch(f"/adjustments/{creada['id']}/steps/nivel",
                     json={"estado": "omitido", "version_esperada": 1})
    assert r.status_code == 200, r.text
    assert llamadas == [SEMANA]  # no se consultó OpenAI al recalcular
    assert not any(a["codigo"] == "EVENTOS_SEMANA" for a in _steps(r.json())["control"]["alerts"])


def test_eventos_timeout_corto():
    assert router_mod.EVENTOS_TIMEOUT == 5


# ---------------------------------------------------------------------------
# aplicar_automatico (R2): solo diagnósticos
# ---------------------------------------------------------------------------

def test_aplicar_automatico_diagnostics_sin_cambiar_curvas(client):
    base = client.post("/adjustments/preview", json=BODY).json()
    auto = client.post("/adjustments/preview", json={**BODY, "aplicar_automatico": True}).json()
    assert auto["final_curves"] == base["final_curves"]
    assert "automatico" not in _steps(base)["nivel"]["diagnostics"]
    assert "automatico" not in _steps(auto)["especiales"]["diagnostics"]

    for paso in ("nivel", "forma"):
        diag = _steps(auto)[paso]["diagnostics"]["automatico"]
        assert set(diag) == set(_semana_iso())
        for fecha in _semana_iso():
            assert diag[fecha]["tipo_dia"] in {"laboral", "sabado", "domingo", "festivo"}
            assert diag[fecha]["estado"] in {"automatico", "sugerido"}
            assert diag[fecha]["recomendacion"] in {"no_aplicar", "automatico", "sugerido"}
            assert diag[fecha]["paso"] == paso  # contrato PLAN_4C
    # Antioquia: forma automática en días laborales (el lunes de esta semana es festivo)
    martes = (SEMANA + timedelta(days=1)).isoformat()
    forma = _steps(auto)["forma"]["diagnostics"]["automatico"]
    assert forma[martes] == {"tipo_dia": "laboral", "estado": "automatico", "paso": "forma",
                             "recomendacion": "automatico"}
    assert forma[SEMANA.isoformat()] == {"tipo_dia": "festivo", "estado": "sugerido",
                                         "paso": "forma", "recomendacion": "sugerido"}
    # Antioquia/domingo/nivel está marcado "no aplicar" en el §8.
    domingo = (SEMANA + timedelta(days=6)).isoformat()
    nivel = _steps(auto)["nivel"]["diagnostics"]["automatico"]
    assert nivel[domingo] == {"tipo_dia": "domingo", "estado": "sugerido", "paso": "nivel",
                              "recomendacion": "no_aplicar"}


def test_modos_paso_en_meta_siempre(client):
    # R2 es una clasificación de revisión para el front: viaja aunque el flag esté apagado.
    data = client.post("/adjustments/preview", json=BODY).json()
    assert data["meta"]["extras"]["modos_paso"] == {"nivel": "sugerido", "forma": "sugerido"}

    auto = client.post("/adjustments/preview", json={**BODY, "aplicar_automatico": True}).json()
    assert auto["meta"]["extras"]["modos_paso"] == data["meta"]["extras"]["modos_paso"]


def test_modos_paso_se_conserva_en_patch(client):
    creada = _crear(client)
    assert creada["meta"]["extras"]["modos_paso"]["forma"] == "sugerido"
    r = client.patch(f"/adjustments/{creada['id']}/steps/forma",
                     json={"estado": "omitido", "version_esperada": 1})
    assert r.status_code == 200, r.text
    assert r.json()["meta"]["extras"]["modos_paso"] == creada["meta"]["extras"]["modos_paso"]


def test_modos_paso_se_recalcula_en_corrida_vieja(client, store):
    creada = _crear(client)
    store._runs[creada["id"]]["meta"]["extras"].pop("modos_paso", None)  # corrida previa al contrato
    data = client.get(f"/adjustments/{creada['id']}").json()
    assert data["meta"]["extras"]["modos_paso"] == {"nivel": "sugerido", "forma": "sugerido"}


def test_segmentos_paso_en_meta_siempre(client):
    # R2 de 3 estados para el front: viaja aunque el flag esté apagado.
    data = client.post("/adjustments/preview", json=BODY).json()
    seg = data["meta"]["extras"]["segmentos_paso"]
    assert list(seg) == ["laboral", "sabado", "domingo", "festivo"]
    assert seg["laboral"]["forma"] == "automatico"
    assert seg["domingo"]["nivel"] == "no_aplicar"
    assert seg["sabado"]["nivel"] == "sugerido"

    auto = client.post("/adjustments/preview", json={**BODY, "aplicar_automatico": True}).json()
    assert auto["meta"]["extras"]["segmentos_paso"] == seg


def test_segmentos_paso_se_conserva_en_patch(client):
    creada = _crear(client)
    r = client.patch(f"/adjustments/{creada['id']}/steps/forma",
                     json={"estado": "omitido", "version_esperada": 1})
    assert r.status_code == 200, r.text
    assert r.json()["meta"]["extras"]["segmentos_paso"] == creada["meta"]["extras"]["segmentos_paso"]


def test_segmentos_paso_se_recalcula_en_corrida_vieja(client, store):
    creada = _crear(client)
    store._runs[creada["id"]]["meta"]["extras"].pop("segmentos_paso", None)
    data = client.get(f"/adjustments/{creada['id']}").json()
    assert data["meta"]["extras"]["segmentos_paso"]["laboral"]["forma"] == "automatico"
    assert data["meta"]["extras"]["segmentos_paso"]["domingo"]["nivel"] == "no_aplicar"


def test_corrida_vieja_de_5_pasos_se_migra_al_leer_y_en_patch(client, store):
    creada = _crear(client)
    run = store._runs[creada["id"]]
    recs = run["records"]
    run["records"] = [recs[0], recs[1], recs[2], recs[3], replace(recs[5], orden=4)]

    data = client.get(f"/adjustments/{creada['id']}").json()
    assert [s["paso"] for s in data["steps"]] == ["base", "referencia", "nivel", "forma",
                                                 "especiales", "control"]
    assert data["steps"][4]["estado"] == "omitido"

    # PATCH de `especiales` funciona sin otro PATCH previo que migre la corrida.
    r = client.patch(f"/adjustments/{creada['id']}/steps/especiales",
                     json={"estado": "omitido", "version_esperada": 1})
    assert r.status_code == 200, r.text
    assert [s["paso"] for s in r.json()["steps"]] == ["base", "referencia", "nivel", "forma",
                                                      "especiales", "control"]


def test_aplicar_automatico_se_conserva_en_patch(client):
    creada = _crear(client, aplicar_automatico=True)
    r = client.patch(f"/adjustments/{creada['id']}/steps/forma",
                     json={"estado": "omitido", "version_esperada": 1})
    assert r.status_code == 200, r.text
    assert "automatico" in _steps(r.json())["nivel"]["diagnostics"]


# ---------------------------------------------------------------------------
# GET /adjustments/aprendizaje
# ---------------------------------------------------------------------------

def _editar_nivel(client, creada, fecha, valores):
    r = client.patch(f"/adjustments/{creada['id']}/steps/nivel",
                     json={"estado": "editado", "version_esperada": 1, "curvas": {fecha: valores}})
    assert r.status_code == 200, r.text
    return r.json()


def test_aprendizaje_200_con_dos_corridas_y_una_edicion(client):
    primera = _crear(client)
    _crear(client)
    fecha = SEMANA.isoformat()
    _editar_nivel(client, primera, fecha, [500.0] * 24)

    r = client.get("/adjustments/aprendizaje", params={"ucp": UCP})
    assert r.status_code == 200, r.text
    data = r.json()
    assert set(data) >= {"grupos", "resumen_global", "por_dia_semana", "omitidos_sin_calculada", "min_casos"}
    assert data["min_casos"] == 3  # default
    grupo = next(g for g in data["grupos"] if g["paso"] == "nivel" and g["ucp"] == UCP)
    assert grupo["n_dias"] == 1 and grupo["n_corridas"] == 1
    assert grupo["delta_pct_medio"] > 0 and grupo["pct_positivo"] == 100.0  # porcentaje ×100
    assert grupo["recurrente"] is False  # 1 día < min_casos=3
    assert fecha in grupo["fechas"]
    assert data["resumen_global"]["n_casos"] == 1

    con_umbral = client.get("/adjustments/aprendizaje", params={"ucp": UCP, "min_casos": 1})
    assert con_umbral.status_code == 200, con_umbral.text
    assert con_umbral.json()["grupos"][0]["recurrente"] is True


def test_aprendizaje_metricas_en_porcentaje(client):
    primera = _crear(client)
    _crear(client)
    fecha = SEMANA.isoformat()
    calculada = _steps(primera)["nivel"]["curves"][fecha]
    editada = [v * 1.1 for v in calculada]
    _editar_nivel(client, primera, fecha, editada)

    data = client.get("/adjustments/aprendizaje", params={"ucp": UCP, "min_casos": 1}).json()
    grupo = next(g for g in data["grupos"] if g["paso"] == "nivel")
    assert grupo["delta_pct_medio"] == pytest.approx(10.0, abs=1e-3)
    assert grupo["p50_abs"] == pytest.approx(10.0, abs=1e-3)
    assert grupo["pct_positivo"] == pytest.approx(100.0)
    assert data["por_dia_semana"]["lunes"]["delta_pct_medio"] == pytest.approx(10.0, abs=1e-3)


def test_aprendizaje_filtros_de_fechas(client):
    _crear(client)
    r = client.get("/adjustments/aprendizaje", params={
        "ucp": UCP,
        "desde": (SEMANA + timedelta(days=7)).isoformat(),
        "hasta": (SEMANA + timedelta(days=14)).isoformat(),
    })
    assert r.status_code == 200, r.text
    assert r.json()["grupos"] == [] and r.json()["resumen_global"]["n_dias"] == 0


def test_aprendizaje_503_sin_store(client_sin_store):
    assert client_sin_store.get("/adjustments/aprendizaje",
                                params={"ucp": UCP}).status_code == 503


class _StoreSinListRuns:
    """Store mínimo sin `list_runs` (contrato viejo): el endpoint responde 501."""

    def get_run(self, run_id):  # pragma: no cover - no debería usarse
        return None


def test_aprendizaje_501_store_sin_list_runs(env):
    client = _client(_StoreSinListRuns())
    r = client.get("/adjustments/aprendizaje", params={"ucp": UCP})
    assert r.status_code == 501, r.text


def test_aprendizaje_422_desde_mayor_que_hasta(client):
    r = client.get("/adjustments/aprendizaje", params={
        "desde": (SEMANA + timedelta(days=7)).isoformat(),
        "hasta": SEMANA.isoformat(),
    })
    assert r.status_code == 422, r.text


def test_aprendizaje_404_ucp_inexistente(client):
    assert client.get("/adjustments/aprendizaje", params={"ucp": "NoExiste"}).status_code == 404
