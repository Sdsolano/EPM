"""Comparador ASR vs mesa vs real (`src/adjustments/comparar_asr.py`, router y script) con datos sintéticos."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.adjustments import asr, data as data_mod, pipeline
from src.adjustments import comparar_asr as cmp
from src.adjustments import router as router_mod
from src.adjustments import router_asr
from tests.ajustes_sinteticos import ProveedorPlano, forma_tipo, historico_sintetico

ROOT = Path(__file__).resolve().parent.parent
UCP = "Sintetica"
LUNES = date(2025, 5, 19)  # semana completa observada en el histórico sintético
CORTE = LUNES - timedelta(days=2)


def _senda_plana_tipo(fecha: str):
    """Senda falsa: la forma laboral sintética (el validador recorta poco)."""
    return forma_tipo("laboral"), 7, "normal"


@pytest.fixture(scope="module")
def historico():
    return data_mod.normalizar_historico(historico_sintetico(date(2022, 1, 1), date(2025, 6, 30)), ucp=UCP)


@pytest.fixture(scope="module")
def festivos():
    return data_mod.festivos_colombia(range(2021, 2027))


@pytest.fixture(scope="module")
def ctx_factory(historico, festivos):
    def make():
        hist = historico[historico["fecha"] <= pd.Timestamp(CORTE)]
        return pipeline.build_context(UCP, LUNES, fecha_emision=CORTE + timedelta(days=1),
                                      historico=hist, festivos=festivos)
    return make


@pytest.fixture(scope="module")
def reales(historico):
    return data_mod.curvas_reales(historico)


def test_metricas_dia():
    real = np.arange(1, 25, dtype=float)
    m = cmp.metricas_dia(2 * real, real)
    assert m["err_forma_pct"] == pytest.approx(0.0)
    assert m["wape_pct"] == pytest.approx(100.0)
    assert m["err_energia_pct"] == pytest.approx(100.0)
    assert m["hora_pico_pred"] == m["hora_pico_real"] == 24 and m["dif_hora_pico_h"] == 0
    sin_real = cmp.metricas_dia(real)
    assert "wape_pct" not in sin_real and sin_real["energia_pred"] == pytest.approx(300.0)


def test_dif_hora_pico_circular():
    r = np.zeros(24); r[23] = 10  # pico en la hora 24
    p = np.zeros(24); p[0] = 10   # pico en la hora 1
    assert cmp.metricas_dia(p + 1, r + 1)["dif_hora_pico_h"] == 1


def test_comparar_semana_series_y_mesa(ctx_factory, reales):
    prov = ProveedorPlano(1000.0)
    res = cmp.comparar_semana(ctx_factory(), prov, reales=reales, escenarios=("anual", "recientes"),
                              senda_fn=_senda_plana_tipo, descomponer=True)
    fechas = res["fechas"]
    assert len(fechas) == 7 and fechas[0] == LUNES.isoformat()
    assert res["fecha_corte"] == CORTE.isoformat()
    for s in ("base", "mesa", "real", "asr_anual", "asr_anual_sin_validador", "asr_recientes",
              "asr_anual_solo_nivel", "asr_anual_solo_forma"):
        assert s in res["series"], s
    assert all(res["series"]["base"][f] == [1000.0] * 24 for f in fechas)
    # la mesa es exactamente /preview con la misma base (pipeline con defaults)
    mesa = pipeline.final_curves(pipeline.run_pipeline(ctx_factory(), ProveedorPlano(1000.0)))
    for f in fechas:
        assert np.allclose(res["series"]["mesa"][f], mesa[f])
    # sin validador el ASR es h1 + (h2 − h3), independiente de la base (salvo redondeos)
    hist = data_mod.curvas_reales(ctx_factory().historico)
    for f in fechas:
        h = [hist[x.isoformat()] for x in asr.fechas_escenario("anual", date.fromisoformat(f), CORTE)]
        assert np.max(np.abs(np.asarray(res["series"]["asr_anual_sin_validador"][f]) - (h[0] + h[1] - h[2]))) < 0.61
    # descomposición: solo_nivel tiene la energía del ASR y la forma de la base
    f = fechas[2]
    sn = np.asarray(res["series"]["asr_anual_solo_nivel"][f])
    assert sn.sum() == pytest.approx(np.sum(res["series"]["asr_anual"][f]))
    assert np.allclose(sn / sn.sum(), 1 / 24)
    sf = np.asarray(res["series"]["asr_anual_solo_forma"][f])
    assert sf.sum() == pytest.approx(24_000.0)
    # métricas: una fila por serie (no real) y día, con error contra el real
    m = pd.DataFrame(res["metricas"])
    assert set(m["serie"]) == set(res["series"]) - {"real"}
    assert m["wape_pct"].notna().all()
    fila = m[(m["serie"] == "base") & (m["fecha"] == f)].iloc[0]
    assert fila["wape_pct"] == pytest.approx(100 * np.abs(1000.0 - reales[f]).sum() / reales[f].sum())
    sem = {r["serie"]: r for r in res["semana"]}
    assert "err_energia_semana_pct" in sem["mesa"]
    assert res["asr_detalle"]["asr_anual"][f]["metodo"] == "normal"


def test_comparar_semana_seleccion_explicita_y_analista(ctx_factory):
    ctx = ctx_factory()
    hist = data_mod.curvas_reales(ctx.historico)
    f = LUNES.isoformat()
    fechas_sel = ["2025-05-12", "2024-05-20", "2024-05-13"]
    sel = {f: [asr.Historico(x, hist[x].tolist()) for x in fechas_sel]}
    analista = {f: [5.0] * 24}
    res = cmp.comparar_semana(ctx, ProveedorPlano(), seleccion=sel, validador=(False,), analista=analista)
    assert set(res["series"]["asr_sin_validador"]) == {f}
    assert len(res["omitidos"]["asr_sin_validador"]) == 6  # los otros días no tienen selección
    assert res["series"]["analista"] == analista
    assert "real" not in res["series"]
    assert res["asr_detalle"]["asr_sin_validador"][f]["h_fechas"] == fechas_sel


def test_dias_sin_historicos_quedan_omitidos(historico, festivos):
    # histórico que empieza hace menos de un año: el escenario anual no tiene h2/h3
    hist = historico[(historico["fecha"] >= pd.Timestamp("2024-09-01")) & (historico["fecha"] <= pd.Timestamp(CORTE))]
    ctx = pipeline.build_context(UCP, LUNES, fecha_emision=CORTE + timedelta(days=1), historico=hist,
                                 festivos=festivos)
    res = cmp.comparar_semana(ctx, ProveedorPlano(), escenarios=("anual", "recientes"), validador=(False,))
    assert res["series"]["asr_anual_sin_validador"] == {}
    assert len(res["omitidos"]["asr_anual_sin_validador"]) == 7
    assert len(res["series"]["asr_recientes_sin_validador"]) == 7


def test_sin_modelo_la_serie_con_validador_queda_omitida(ctx_factory):
    class SinEntrenar:
        class normal_disaggregator:  # noqa: N801
            is_fitted = False

    res = cmp.comparar_semana(ctx_factory(), ProveedorPlano(), motor=SinEntrenar())
    assert res["series"]["asr_anual"] == {}
    assert len(res["omitidos"]["asr_anual"]) == 7
    assert len(res["series"]["asr_anual_sin_validador"]) == 7


def test_base_incompleta_falla(ctx_factory):
    with pytest.raises(ValueError):
        cmp.comparar_semana(ctx_factory(), ProveedorPlano(faltar=[LUNES.isoformat()]), validador=(False,))


def test_proveedor_ingenuo_igual_al_del_backtest(ctx_factory):
    spec = importlib.util.spec_from_file_location("backtest_adjustments", ROOT / "scripts" / "backtest_adjustments.py")
    bt = sys.modules.get("backtest_adjustments")
    if bt is None:
        bt = importlib.util.module_from_spec(spec)
        sys.modules["backtest_adjustments"] = bt
        spec.loader.exec_module(bt)
    ctx = ctx_factory()
    a = cmp.proveedor_ingenuo(ctx.historico)(UCP, ctx.fecha_corte, 8)
    b = bt.proveedor_ingenuo(ctx.historico)(UCP, ctx.fecha_corte, 8)
    assert a == b and len(a) == 8


# ---------------------------------------------------------------------------
# Endpoint POST /adjustments/comparar-asr
# ---------------------------------------------------------------------------

@pytest.fixture
def cliente(tmp_path, monkeypatch, historico, festivos):
    raiz = tmp_path / "data" / "raw"
    (raiz / UCP).mkdir(parents=True)
    (raiz / UCP / "datos.csv").write_text("")
    monkeypatch.setattr(data_mod, "RAW_DATA_DIR", raiz)
    monkeypatch.delenv(router_mod.ENV_SERVICE_TOKEN, raising=False)
    prev_h = router_mod.set_history_loader(lambda ucp: historico)
    prev_f = router_mod.set_festivos_loader(lambda ucp, i, f: [x for x in festivos if i.isoformat() <= x <= f.isoformat()])

    class Motor:
        class normal_disaggregator:  # noqa: N801
            is_fitted = True
            cluster_by_dayofweek = {i: 0 for i in range(7)}
            cluster_profiles = pd.DataFrame([forma_tipo("laboral")])

        class special_disaggregator:  # noqa: N801
            is_fitted = False

    monkeypatch.setattr(asr, "cargar_motor", lambda ucp, models_root=None: Motor())
    app = FastAPI()
    app.include_router(router_asr.router)
    yield TestClient(app)
    router_mod.set_history_loader(prev_h)
    router_mod.set_festivos_loader(prev_f)


def _base_semana(escala: float = 1.0):
    return {(CORTE + timedelta(days=i)).isoformat(): [1000.0 * escala] * 24 for i in range(1, 9)}


def test_endpoint_con_base_curvas(cliente, reales):
    body = {"ucp": UCP, "semana_inicio": LUNES.isoformat(), "fecha_emision": (CORTE + timedelta(days=1)).isoformat(),
            "base_curvas": _base_semana(), "escenarios": ["anual"]}
    r = cliente.post("/adjustments/comparar-asr", json=body)
    assert r.status_code == 200, r.text
    js = r.json()
    assert js["fecha_corte"] == CORTE.isoformat()
    assert {"base", "mesa", "asr_anual", "asr_anual_sin_validador", "real"} <= set(js["series"])
    assert js["series"]["base"][LUNES.isoformat()] == [1000.0] * 24
    assert np.allclose(js["series"]["real"][LUNES.isoformat()], reales[LUNES.isoformat()])
    assert len(js["series"]["asr_anual"]) == 7


def test_endpoint_ingenuo_y_corte_por_defecto(cliente):
    r = cliente.post("/adjustments/comparar-asr", json={"ucp": UCP, "semana_inicio": LUNES.isoformat()})
    assert r.status_code == 200, r.text
    assert r.json()["fecha_corte"] == CORTE.isoformat()  # sábado previo, como el backtest


def test_endpoint_seleccion(cliente):
    f = LUNES.isoformat()
    body = {"ucp": UCP, "semana_inicio": f, "seleccion": {f: ["2025-05-12", "2024-05-20", "2024-05-13"]}}
    r = cliente.post("/adjustments/comparar-asr", json=body)
    assert r.status_code == 200, r.text
    assert list(r.json()["series"]["asr"]) == [f]
    # histórico posterior al corte → 422 (sería fuga)
    body["seleccion"] = {f: ["2025-05-18", "2024-05-20", "2024-05-13"]}
    assert cliente.post("/adjustments/comparar-asr", json=body).status_code == 422
    body["seleccion"] = {f: ["2025-05-12", "2024-05-20"]}
    assert cliente.post("/adjustments/comparar-asr", json=body).status_code == 422
    body["seleccion"] = {"2025-06-30": ["2025-05-12", "2024-05-20", "2024-05-13"]}
    assert cliente.post("/adjustments/comparar-asr", json=body).status_code == 422


@pytest.mark.parametrize("cambio, codigo", [
    ({"base": "modelo"}, 422),
    ({"escenarios": ["otro"]}, 422),
    ({"semana_inicio": "2025-05-20"}, 422),
    ({"ucp": "NoExiste"}, 404),
    ({"base_curvas": {"2025-05-19": [1.0] * 24}}, 422),
])
def test_endpoint_errores(cliente, cambio, codigo):
    body = {"ucp": UCP, "semana_inicio": LUNES.isoformat(), **cambio}
    assert cliente.post("/adjustments/comparar-asr", json=body).status_code == codigo


def test_endpoint_exige_token(cliente, monkeypatch):
    monkeypatch.setenv(router_mod.ENV_SERVICE_TOKEN, "secreto")
    body = {"ucp": UCP, "semana_inicio": LUNES.isoformat()}
    assert cliente.post("/adjustments/comparar-asr", json=body).status_code == 401
    r = cliente.post("/adjustments/comparar-asr", json=body, headers={router_mod.HEADER_SERVICE_TOKEN: "secreto"})
    assert r.status_code == 200


def test_endpoint_registrado_en_la_api(monkeypatch):
    api_main = pytest.importorskip("src.api.main")
    monkeypatch.delenv(router_mod.ENV_SERVICE_TOKEN, raising=False)  # load_dotenv pudo definirlo
    # body inválido: 422 de validación (no 404) ⇒ la ruta está montada en la API
    r = TestClient(api_main.app).post("/adjustments/comparar-asr", json={"ucp": UCP})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("comparar_asr_script", ROOT / "scripts" / "comparar_asr.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["comparar_asr_script"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_script_historico_y_reporte(script, historico, festivos, tmp_path):
    semanas = [LUNES - timedelta(days=7 * k) for k in (2, 1, 0)]
    df = script.correr_historico(UCP, historico, semanas, lambda ucp, h: cmp.proveedor_ingenuo(h), "ingenuo",
                                 senda_fn=_senda_plana_tipo, festivos=festivos)
    assert set(df["semana_inicio"]) == {s.isoformat() for s in semanas}
    assert {"base", "mesa", "asr_anual", "asr_anual_clic_asc", "asr_recientes_sin_validador",
            "asr_anual_solo_forma"} <= set(df["serie"])
    md = tmp_path / "r.md"
    md.write_text("viejo\n## 6. Diagnóstico\ntexto manual\n", encoding="utf-8")
    script.escribir_reporte(df, md, {"fecha": "X", "csv_nombre": "x.csv", "bases": ["ingenuo"], "semanas": 3,
                                     "secciones_manuales": script.bt.secciones_manuales(md)})
    texto = md.read_text(encoding="utf-8")
    assert "## 2. Error por serie" in texto and "## 6. Diagnóstico\ntexto manual" in texto
    assert "viejo" not in texto
    r = script.resumen(df[df["serie"] == "base"], ["serie"])
    assert r["n_dias"].iloc[0] == 21


def test_script_parsear_seleccion_y_csv(script, tmp_path):
    sel = script.parsear_seleccion(["2026-11-16=2026-11-09, 2025-11-17,2025-11-10"])
    assert sel == {"2026-11-16": ["2026-11-09", "2025-11-17", "2025-11-10"]}
    with pytest.raises(ValueError):
        script.parsear_seleccion(["2026-11-16=2026-11-09,2025-11-17"])
    largo = tmp_path / "largo.csv"
    pd.DataFrame({"fecha": ["2026-11-16"] * 24, "periodo": [f"P{i}" for i in range(1, 25)],
                  "valor": list(range(24))}).to_csv(largo, index=False)
    assert script.leer_curvas_csv(largo) == {"2026-11-16": [float(i) for i in range(24)]}
    ancho = tmp_path / "ancho.csv"
    pd.DataFrame([{"fecha": "2026-11-16", **{f"P{i}": float(i) for i in range(1, 25)}}]).to_csv(ancho, index=False)
    assert script.leer_curvas_csv(ancho) == {"2026-11-16": [float(i) for i in range(1, 25)]}


def test_motor_que_falla_al_cargar_deja_la_serie_omitida(ctx_factory, monkeypatch):
    llamadas = []

    def falla(ucp, models_root=None):
        llamadas.append(ucp)
        raise RuntimeError("pickle corrupto")

    monkeypatch.setattr(asr, "cargar_motor", falla)
    res = cmp.comparar_semana(ctx_factory(), ProveedorPlano())
    assert res["series"]["asr_anual"] == {} and len(res["omitidos"]["asr_anual"]) == 7
    assert "pickle corrupto" in next(iter(res["omitidos"]["asr_anual"].values()))
    assert len(llamadas) == 1  # el fallo no se reintenta día por día
    assert len(res["series"]["asr_anual_sin_validador"]) == 7


def test_advertencia_si_el_historico_esta_desactualizado(historico, festivos):
    hist = historico[historico["fecha"] <= pd.Timestamp(CORTE - timedelta(days=30))]
    ctx = pipeline.build_context(UCP, LUNES, fecha_emision=CORTE + timedelta(days=1), historico=hist,
                                 festivos=festivos)
    res = cmp.comparar_semana(ctx, ProveedorPlano(), validador=(False,))
    assert len(res["advertencias"]) == 1
    assert cmp.comparar_semana(ctx_semana(historico, festivos), ProveedorPlano(),
                               validador=(False,))["advertencias"] == []


def ctx_semana(historico, festivos):
    hist = historico[historico["fecha"] <= pd.Timestamp(CORTE)]
    return pipeline.build_context(UCP, LUNES, fecha_emision=CORTE + timedelta(days=1), historico=hist,
                                  festivos=festivos)


def test_endpoint_curvas_invalidas_422(cliente):
    base = {"ucp": UCP, "semana_inicio": LUNES.isoformat()}
    f = LUNES.isoformat()
    assert cliente.post("/adjustments/comparar-asr",
                        json={**base, "analista_curvas": {f: [1.0] * 23}}).status_code == 422
    malas = _base_semana()
    malas[f] = [1.0] * 23 + [float("inf")]
    r = cliente.post("/adjustments/comparar-asr", content=json.dumps({**base, "base_curvas": malas}),
                     headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_script_reporte_con_escenarios_reducidos(script, historico, festivos, tmp_path):
    df = script.correr_historico(UCP, historico, [LUNES], lambda ucp, h: cmp.proveedor_ingenuo(h), "ingenuo",
                                 senda_fn=_senda_plana_tipo, festivos=festivos, escenarios=["anual"])
    assert script.series_principales(df) == ["base", "mesa", "asr_anual"]
    md = tmp_path / "r.md"
    script.escribir_reporte(df, md, {"fecha": "X", "csv_nombre": "x.csv", "bases": ["ingenuo"], "semanas": 1})
    texto = md.read_text(encoding="utf-8")
    seccion2 = texto.split("## 2.")[1].split("## 3.")[0]
    assert "| ingenuo | Sintetica | asr_anual |" in seccion2


def test_metricas_dia_real_en_cero_no_divide():
    m = cmp.metricas_dia([1.0] * 24, [0.0] * 24)
    assert m["wape_pct"] is None and m["err_energia_pct"] is None and m["err_pico_pct"] is None


def test_seleccion_fuera_de_la_semana_es_error(ctx_factory):
    ctx = ctx_factory()
    hist = data_mod.curvas_reales(ctx.historico)
    sel = {"2025-06-30": [asr.Historico(x, hist[x].tolist()) for x in ("2025-05-12", "2024-05-20", "2024-05-13")]}
    with pytest.raises(ValueError):
        cmp.comparar_semana(ctx, ProveedorPlano(), seleccion=sel, validador=(False,))


def test_validar_curvas():
    cmp.validar_curvas({"2025-05-19": [1.0] * 24}, "x")
    with pytest.raises(ValueError):
        cmp.validar_curvas({"2025-05-19": [1.0] * 23}, "x")
    with pytest.raises(ValueError):
        cmp.validar_curvas({"2025-05-19": [1.0] * 23 + [float("nan")]}, "x")
