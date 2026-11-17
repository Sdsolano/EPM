"""Tests del buscador de días análogos (R3): módulo `src/adjustments/analogos.py` y `POST /adjustments/analogos`.

Datos sintéticos (sin CSV reales ni red): histórico de `tests.ajustes_sinteticos`, festivos de la librería.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.adjustments import analogos as an
from src.adjustments import data as data_mod
from src.adjustments import router as router_mod
from src.adjustments.data import PERIODOS, festivos_colombia, normalizar_historico
from tests.ajustes_sinteticos import historico_sintetico

FESTIVOS = festivos_colombia(range(2016, 2028))
HIST = normalizar_historico(historico_sintetico(date(2019, 1, 1), date(2025, 12, 31)))
CORTE = date(2025, 12, 31)


def _d(s: str) -> date:
    return date.fromisoformat(s)


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------

def test_descriptor_martes_despues_de_lunes_festivo():
    # 2025-03-24 lunes festivo (San José, observado)
    d = an.descriptor_dia(_d("2025-03-25"), FESTIVOS)
    assert d["dia_semana"] == 1 and d["mes"] == 3 and d["anio"] == 2025
    assert d["festivo"] is False
    assert d["relacion_festivo"] == "despues" and d["distancia_festivo"] == 1
    assert d["festivo_relacionado"] == "2025-03-24"
    assert d["tipo_festivo"] == "dia_de_san_jose"
    assert d["clima"] is None


@pytest.mark.parametrize("dia", ["2025-03-22", "2025-03-23"])
def test_descriptor_fin_de_semana_antes_de_lunes_festivo_es_puente(dia):
    d = an.descriptor_dia(_d(dia), FESTIVOS)
    assert d["relacion_festivo"] == "puente"
    assert d["distancia_festivo"] is None
    assert d["festivo_relacionado"] == "2025-03-24"


def test_descriptor_sandwich_es_puente():
    # 2025-05-01 jueves festivo; viernes 2 entre festivo y sábado
    d = an.descriptor_dia(_d("2025-05-02"), FESTIVOS)
    assert d["relacion_festivo"] == "puente" and d["festivo_relacionado"] == "2025-05-01"
    assert d["tipo_festivo"] == "dia_del_trabajo"


def test_descriptor_jueves_santo():
    d = an.descriptor_dia(_d("2025-04-17"), FESTIVOS)
    assert d["festivo"] is True and d["relacion_festivo"] == "festivo"
    assert d["semana_santa"] is True
    assert d["tipo_festivo"] == "semana_santa"
    assert d["temporada_escolar"] == "vacaciones"
    assert "Jueves Santo" in d["nombre_festivo"]
    # miércoles santo: antes de festivo, en Semana Santa
    m = an.descriptor_dia(_d("2025-04-16"), FESTIVOS)
    assert (m["relacion_festivo"], m["distancia_festivo"], m["semana_santa"]) == ("antes", 1, True)
    # domingo de Ramos sí, sábado anterior no
    assert an.descriptor_dia(_d("2025-04-13"), FESTIVOS)["semana_santa"] is True
    assert an.descriptor_dia(_d("2025-04-12"), FESTIVOS)["semana_santa"] is False


@pytest.mark.parametrize("dia,esperado", [
    ("2025-01-10", "vacaciones"), ("2025-01-27", "clases"), ("2025-04-14", "vacaciones"),
    ("2025-06-20", "vacaciones"), ("2025-07-11", "clases"), ("2025-08-20", "clases"),
    ("2025-10-15", "vacaciones"),  # semana del lunes festivo de octubre (2025-10-13)
    ("2025-10-22", "clases"), ("2025-12-01", "vacaciones"),
])
def test_descriptor_temporada_escolar(dia, esperado):
    assert an.descriptor_dia(_d(dia), FESTIVOS)["temporada_escolar"] == esperado


def test_descriptor_empate_gana_despues_y_ninguna():
    festivos = {"2025-02-03", "2025-02-07"}  # lunes y viernes
    d = an.descriptor_dia(_d("2025-02-05"), festivos)  # miércoles a 2 días de ambos
    assert (d["relacion_festivo"], d["distancia_festivo"], d["festivo_relacionado"]) == ("despues", 2, "2025-02-03")
    n = an.descriptor_dia(_d("2025-08-27"), FESTIVOS)
    assert n["relacion_festivo"] == "ninguna" and n["distancia_festivo"] is None
    assert n["festivo_relacionado"] is None and n["tipo_festivo"] is None


def test_descriptor_festivo_ucp_sin_nombre():
    d = an.descriptor_dia(_d("2025-02-12"), ["2025-02-12"])
    assert d["festivo"] is True and d["tipo_festivo"] == "festivo_ucp"


def test_descriptor_con_clima():
    clima = {"2025-03-25": {"temp_media": 24.0, "lluvia_mm": 2.0, "humedad_media": 70.0}}
    assert an.descriptor_dia(_d("2025-03-25"), FESTIVOS, clima)["clima"]["temp_media"] == 24.0


# ---------------------------------------------------------------------------
# Validación de filtros
# ---------------------------------------------------------------------------

def test_validar_filtros_defaults():
    f = an.validar_filtros(None)
    assert set(f) == set(an.FILTROS_DEFAULT)
    assert f["anios_excluidos"] == [2020] and f["limite"] == 20 and f["dia_semana"] is None
    assert an.validar_filtros({"anios_excluidos": None, "limite": None})["anios_excluidos"] == [2020]
    assert an.validar_filtros({"anios_excluidos": []})["anios_excluidos"] == []
    assert an.validar_filtros({"dia_semana": 3})["dia_semana"] == [3]
    assert an.validar_filtros({"dia_semana": [4, 1, 1]})["dia_semana"] == [1, 4]
    assert an.validar_filtros({"tipo_festivo": "Día de San José (observado)"})["tipo_festivo"] == "dia_de_san_jose"
    assert an.validar_filtros({"tipo_festivo": "Viernes Santo"})["tipo_festivo"] == "semana_santa"
    c = an.validar_filtros({"clima": {"temp_tolerancia": 1.5, "lluvia": True}})["clima"]
    assert c == {"temp_tolerancia": 1.5, "lluvia": True, "temp_media": None}


@pytest.mark.parametrize("filtros", [
    "martes", [1], {"desconocido": 1}, {"dia_semana": 7}, {"dia_semana": True}, {"dia_semana": "martes"},
    {"dia_semana": []}, {"meses": [13]}, {"meses": []}, {"meses": 5}, {"temporada_dias": 0},
    {"temporada_dias": 184}, {"relacion_festivo": "cerca"}, {"distancia_festivo": 0}, {"distancia_festivo": 8},
    {"tipo_festivo": ""}, {"tipo_festivo": 5}, {"tipo_festivo": "x" * 500}, {"semana_santa": "si"},
    {"temporada_escolar": "verano"}, {"clima": 3}, {"clima": {"foo": 1}}, {"clima": {"temp_tolerancia": 0}},
    {"clima": {"temp_tolerancia": float("nan")}}, {"clima": {"lluvia": 1}}, {"clima": {"temp_media": 500}},
    {"anio_min": 1800}, {"anio_min": 2024, "anio_max": 2023}, {"anios_excluidos": 2020},
    {"anios_excluidos": ["2020"]}, {"limite": 0}, {"limite": 367}, {"limite": 2.5}, {"limite": 10 ** 400},
])
def test_validar_filtros_invalidos(filtros):
    with pytest.raises(ValueError):
        an.validar_filtros(filtros)


# ---------------------------------------------------------------------------
# Búsqueda
# ---------------------------------------------------------------------------

FILTROS_MARTES_MAYO = {"dia_semana": "auto", "meses": [5], "relacion_festivo": "auto", "distancia_festivo": "auto"}


def test_busqueda_martes_de_mayo_despues_de_festivo():
    # 2026-05-19: martes después del lunes festivo 2026-05-18 (Ascensión, observado)
    r = an.buscar_analogos(HIST, FESTIVOS, "2026-05-19", CORTE.isoformat(), FILTROS_MARTES_MAYO)
    obj = r["descriptor_objetivo"]
    assert (obj["dia_semana"], obj["mes"], obj["relacion_festivo"], obj["distancia_festivo"]) == (1, 5, "despues", 1)
    assert set(r) == {"fecha", "fecha_corte", "descriptor_objetivo", "filtros_aplicados", "candidatos", "resumen"}
    cands = r["candidatos"]
    assert len(cands) >= 4  # 2021-05-18, 2022-05-31, 2023-05-02, 2023-05-23, 2024-05-14
    fechas = {c["fecha"] for c in cands}
    assert {"2021-05-18", "2023-05-02", "2023-05-23", "2024-05-14"} <= fechas
    for c in cands:
        d = _d(c["fecha"])
        assert d.weekday() == 1 and d.month == 5
        assert c["descriptor"]["relacion_festivo"] == "despues" and c["descriptor"]["distancia_festivo"] == 1
        assert "martes" in c["razones"] and "mayo" in c["razones"]
        assert any(x.startswith("día después de festivo (") for x in c["razones"])
        assert c["observado"] is True
    fa = r["filtros_aplicados"]
    assert fa["dia_semana"] == [1] and fa["relacion_festivo"] == "despues" and fa["distancia_festivo"] == 1


def test_orden_por_score_y_fecha_y_consistencia_con_puntuar():
    r = an.buscar_analogos(HIST, FESTIVOS, "2025-06-10", "2025-06-01", {"limite": 60})
    cands = r["candidatos"]
    assert len(cands) == 60
    claves = [(-c["score"], -_d(c["fecha"]).toordinal()) for c in cands]
    assert claves == sorted(claves)
    assert all(0.0 <= c["score"] <= 1.0 for c in cands)
    obj = dict(r["descriptor_objetivo"], fecha=r["fecha"])
    for c in cands:
        score, comp, razones = an.puntuar(obj, dict(c["descriptor"], fecha=c["fecha"]))
        assert score == c["score"] and razones == c["razones"]
        assert sum(comp.values()) == pytest.approx(c["score"], abs=1e-3)
        assert set(comp) == set(c["componentes"])
    # el mejor análogo de un martes cualquiera es un martes
    assert _d(cands[0]["fecha"]).weekday() == 1


def test_solo_fechas_hasta_corte_y_nunca_la_propia():
    r = an.buscar_analogos(HIST, FESTIVOS, "2024-05-14", "2024-06-30", {"dia_semana": "auto", "limite": 366})
    fechas = [c["fecha"] for c in r["candidatos"]]
    assert "2024-05-14" not in fechas
    assert max(fechas) <= "2024-06-30"
    assert "2024-06-25" in fechas  # posterior al objetivo pero <= corte
    assert r["filtros_aplicados"]["fecha_corte_efectiva"] == "2024-06-30"


def test_corte_posterior_al_ultimo_dato_y_ausente():
    r = an.buscar_analogos(HIST, FESTIVOS, "2026-05-19", "2026-05-16", {"limite": 5})
    assert r["fecha_corte"] == "2026-05-16"
    assert r["filtros_aplicados"]["fecha_corte_efectiva"] == "2025-12-31"
    assert any("posterior al último dato real" in n for n in r["filtros_aplicados"]["notas"])
    r2 = an.buscar_analogos(HIST, FESTIVOS, "2026-05-19", None, {"limite": 5})
    assert r2["fecha_corte"] == "2025-12-31" and r2["filtros_aplicados"]["notas"] == []


def test_anios_excluidos_default_y_vacio():
    filtros = {"dia_semana": "auto", "meses": [5], "limite": 366}
    r = an.buscar_analogos(HIST, FESTIVOS, "2025-05-20", "2025-05-19", filtros)
    assert r["filtros_aplicados"]["anios_excluidos"] == [2020]
    assert not any(c["fecha"].startswith("2020") for c in r["candidatos"])
    r2 = an.buscar_analogos(HIST, FESTIVOS, "2025-05-20", "2025-05-19", {**filtros, "anios_excluidos": []})
    assert any(c["fecha"].startswith("2020") for c in r2["candidatos"])
    r3 = an.buscar_analogos(HIST, FESTIVOS, "2025-05-20", "2025-05-19",
                            {**filtros, "anio_min": 2022, "anio_max": 2023})
    assert {c["fecha"][:4] for c in r3["candidatos"]} == {"2022", "2023"}


def test_dias_faltantes_marcados_no_observados():
    hist = HIST[HIST["fecha"] != pd.Timestamp("2024-05-21")].copy()
    hist.loc[hist["fecha"] == pd.Timestamp("2024-05-28"), "P5"] = np.nan
    r = an.buscar_analogos(hist, FESTIVOS, "2025-05-27", "2025-05-26",
                           {"dia_semana": "auto", "meses": [5], "limite": 50})
    por_fecha = {c["fecha"]: c for c in r["candidatos"]}
    for f in ("2024-05-21", "2024-05-28"):
        c = por_fecha[f]
        assert c["observado"] is False
        assert c["energia"] is None and c["pico"] is None and c["curva"] is None and c["curva_normalizada"] is None
    res = r["resumen"]
    assert res["n_candidatos"] == len(r["candidatos"])
    assert res["n_observados"] == len(r["candidatos"]) - 2
    sel = an.seleccionar(hist, FESTIVOS, "2025-05-27", "2025-05-26", {"dia_semana": "auto", "meses": [5]})
    assert "2024-05-21" not in {f for f, _, _ in sel} and "2024-05-28" not in {f for f, _, _ in sel}


def test_curvas_pico_y_resumen_ponderado():
    r = an.buscar_analogos(HIST, FESTIVOS, "2025-05-27", "2025-05-26", {"dia_semana": "auto", "limite": 8})
    cands = r["candidatos"]
    for c in cands:
        assert len(c["curva"]) == 24 and len(c["curva_normalizada"]) == 24
        assert sum(c["curva_normalizada"]) == pytest.approx(1.0)
        assert c["energia"] == pytest.approx(sum(c["curva"]))
        assert c["pico"]["mw"] == pytest.approx(max(c["curva"]))
        assert c["pico"]["hora"] == int(np.argmax(c["curva"])) + 1
        assert c["clima"] is None
    w = np.array([c["score"] for c in cands])
    formas = np.array([c["curva_normalizada"] for c in cands])
    energias = np.array([c["energia"] for c in cands])
    res = r["resumen"]
    assert res["ponderado_por_score"] is True and res["n_candidatos"] == 8
    assert res["curva_promedio_normalizada"] == pytest.approx((w @ formas / w.sum()).tolist())
    assert sum(res["curva_promedio_normalizada"]) == pytest.approx(1.0)
    assert res["energia_promedio"] == pytest.approx(float(w @ energias / w.sum()))
    json.dumps(r, allow_nan=False)  # JSON-safe


def test_resumen_sin_candidatos():
    r = an.buscar_analogos(HIST, FESTIVOS, "2025-05-27", "2025-05-26", {"anio_min": 2100})
    assert r["candidatos"] == []
    assert r["resumen"] == {"n_candidatos": 0, "n_observados": 0, "curva_promedio_normalizada": None,
                            "energia_promedio": None, "ponderado_por_score": True}


def test_filtro_semana_santa_y_tipo_festivo_auto():
    r = an.buscar_analogos(HIST, FESTIVOS, "2025-04-17", "2025-04-16",
                           {"relacion_festivo": "auto", "tipo_festivo": "auto", "limite": 50})
    assert r["candidatos"]
    for c in r["candidatos"]:
        assert c["descriptor"]["tipo_festivo"] == "semana_santa" and c["descriptor"]["festivo"] is True
        assert "Semana Santa" in c["razones"]
    r2 = an.buscar_analogos(HIST, FESTIVOS, "2025-08-27", "2025-08-26", {"tipo_festivo": "auto", "limite": 3})
    assert any("tipo_festivo auto" in n for n in r2["filtros_aplicados"]["notas"])
    r3 = an.buscar_analogos(HIST, FESTIVOS, "2025-08-27", "2025-08-26",
                            {"semana_santa": True, "temporada_escolar": "vacaciones", "limite": 366})
    assert r3["candidatos"] and all(c["descriptor"]["semana_santa"] for c in r3["candidatos"])


def test_temporada_dias_circular():
    r = an.buscar_analogos(HIST, FESTIVOS, "2025-01-03", "2025-01-02", {"temporada_dias": 5, "limite": 366})
    meses = {_d(c["fecha"]).month for c in r["candidatos"]}
    assert meses == {12, 1}


def test_fecha_invalida_lanza_value_error():
    with pytest.raises(ValueError):
        an.buscar_analogos(HIST, FESTIVOS, "1800-01-01")
    with pytest.raises(ValueError):
        an.buscar_analogos(HIST, FESTIVOS, "2025-05-27", filtros={"limite": 0})


# ---------------------------------------------------------------------------
# Clima
# ---------------------------------------------------------------------------

def _clima_sintetico():
    clima = {}
    d = date(2024, 1, 1)
    while d <= CORTE:
        k = d.toordinal()
        clima[d.isoformat()] = {"temp_media": 22.0 + (k % 7), "lluvia_mm": float(k % 3), "humedad_media": 70.0}
        d += timedelta(days=1)
    return clima


def test_clima_diario_tal_cual():
    filas = [{"fecha": "2025-01-01", "periodo": p, "p_t": 20.0 + p % 2, "p_h": 80.0, "p_v": 1.0, "p_i": 0.5}
             for p in range(1, 25)]
    filas += [{"fecha": "2025-01-01", "periodo": 1, "p_t": 23.0, "p_h": 80.0, "p_v": 1.0, "p_i": 0.5}]  # repetido
    filas += [{"fecha": "2025-01-02", "periodo": p, "p_t": 25.0, "p_h": 60.0, "p_v": 1.0, "p_i": 800.0}
              for p in range(1, 25)]
    filas += [{"fecha": "2025-01-03", "periodo": p, "p_t": 25.0 if p > 6 else float("nan"), "p_h": 60.0,
               "p_v": 1.0, "p_i": 0.0} for p in range(1, 25)]  # 18 periodos finitos → sin clima
    c = an.clima_diario(pd.DataFrame(filas))
    assert set(c) == {"2025-01-01", "2025-01-02"}
    assert c["2025-01-01"]["lluvia_mm"] == pytest.approx(12.0)
    assert c["2025-01-01"]["temp_media"] == pytest.approx((22.0 + 11 * 21.0 + 12 * 20.0) / 24)  # P1 promediado
    assert c["2025-01-02"]["lluvia_mm"] == pytest.approx(800.0 * 24)  # valores raros: tal cual
    assert an.clima_diario(pd.DataFrame()) == {}


def test_cargar_clima(tmp_path):
    (tmp_path / "X").mkdir()
    assert an.cargar_clima("X", tmp_path) == {}
    filas = [{"fecha": "2025-01-01", "periodo": p, "p_t": 20.0, "p_h": 80.0, "p_v": 1.0, "p_i": 0.0}
             for p in range(1, 25)]
    pd.DataFrame(filas).to_csv(tmp_path / "X" / "clima_new.csv", index=False)
    assert an.cargar_clima("X", tmp_path)["2025-01-01"]["temp_media"] == pytest.approx(20.0)
    (tmp_path / "otra").mkdir()
    assert an.cargar_clima("../X", tmp_path / "otra") == {}  # fuera de la raíz


def test_filtro_clima_temperatura_y_lluvia():
    clima = _clima_sintetico()
    objetivo = "2025-06-10"
    t_obj = clima[objetivo]["temp_media"]
    filtros = {"clima": {"temp_tolerancia": 1.0, "lluvia": True}, "limite": 366}
    r = an.buscar_analogos(HIST, FESTIVOS, objetivo, "2025-06-09", filtros, clima=clima)
    assert r["candidatos"]
    assert r["descriptor_objetivo"]["clima"]["temp_media"] == t_obj
    assert r["filtros_aplicados"]["clima"]["temp_media"] == t_obj
    for c in r["candidatos"]:
        assert c["fecha"] >= "2024-01-01"  # sin clima → descartado
        assert abs(c["clima"]["temp_media"] - t_obj) <= 1.0
        assert c["clima"]["lluvia_mm"] >= an.LLUVIA_UMBRAL_MM
        assert "clima" in c["componentes"] and "con lluvia" in c["razones"]
    r2 = an.buscar_analogos(HIST, FESTIVOS, objetivo, "2025-06-09",
                            {"clima": {"lluvia": False}, "limite": 366}, clima=clima)
    assert r2["candidatos"] and all(c["clima"]["lluvia_mm"] < an.LLUVIA_UMBRAL_MM for c in r2["candidatos"])


def test_objetivo_sin_clima_nota_y_temp_pronosticada():
    clima = _clima_sintetico()
    filtros = {"clima": {"temp_tolerancia": 0.5}, "limite": 30}
    # objetivo futuro sin clima observado: no se filtra por temperatura y se anota
    r = an.buscar_analogos(HIST, FESTIVOS, "2026-05-19", None, filtros, clima=clima)
    assert any("no se aplica el filtro de temperatura" in n for n in r["filtros_aplicados"]["notas"])
    assert r["filtros_aplicados"]["clima"]["temp_media"] is None
    assert any(c["clima"] is None for c in r["candidatos"])  # no se descartan días sin clima
    # con temperatura pronosticada sí filtra
    r2 = an.buscar_analogos(HIST, FESTIVOS, "2026-05-19", None,
                            {"clima": {"temp_tolerancia": 0.5, "temp_media": 25.0}, "limite": 366}, clima=clima)
    assert r2["candidatos"] and all(abs(c["clima"]["temp_media"] - 25.0) <= 0.5 for c in r2["candidatos"])


# ---------------------------------------------------------------------------
# Helpers para los pasos
# ---------------------------------------------------------------------------

def test_seleccionar():
    filtros = {"dia_semana": "auto", "limite": 5}
    sel = an.seleccionar(HIST, FESTIVOS, "2025-05-27", "2025-05-26", filtros)
    assert len(sel) == 5
    r = an.buscar_analogos(HIST, FESTIVOS, "2025-05-27", "2025-05-26", filtros)
    assert [(f, s) for f, s, _ in sel] == [(c["fecha"], c["score"]) for c in r["candidatos"]]
    for (f, _, curva), c in zip(sel, r["candidatos"]):
        assert isinstance(curva, np.ndarray) and curva.shape == (24,)
        np.testing.assert_allclose(curva, c["curva"])
    # sin filtros: anios_excluidos por defecto
    sel2 = an.seleccionar(HIST, FESTIVOS, "2021-01-05", "2021-01-04", {"limite": 366})
    assert sel2 and not any(f.startswith("2020") for f, _, _ in sel2)


def test_seleccionar_con_historico_de_contexto():
    """ctx.historico: solo filas válidas <= corte (columnas fecha datetime64, P1..P24, TOTAL)."""
    ctx_hist = data_mod.recortar(HIST, date(2025, 5, 26))
    sel = an.seleccionar(ctx_hist, FESTIVOS, "2025-05-27", "2025-05-26", {"limite": 3})
    assert len(sel) == 3 and all(f <= "2025-05-26" for f, _, _ in sel)


def test_puntuar_fechas():
    hist = HIST[HIST["fecha"] != pd.Timestamp("2024-05-28")]
    out = an.puntuar_fechas(hist, FESTIVOS, "2025-05-27", ["2024-05-21", "2024-05-28", "2018-03-01"])
    assert [f for f, _, _ in out] == ["2024-05-21", "2024-05-28", "2018-03-01"]
    assert out[0][2].shape == (24,) and out[1][2] is None and out[2][2] is None
    assert out[0][1] > out[2][1]
    r = an.buscar_analogos(HIST, FESTIVOS, "2025-05-27", "2025-05-26", {"limite": 366})
    esperado = {c["fecha"]: c["score"] for c in r["candidatos"]}
    assert out[0][1] == esperado["2024-05-21"]
    with pytest.raises(ValueError):
        an.puntuar_fechas(hist, FESTIVOS, "2025-05-27", ["no-es-fecha"])


def test_rendimiento_busqueda_cacheada():
    an.seleccionar(HIST, FESTIVOS, "2025-05-27", "2025-05-26")  # construye el índice
    t0 = time.perf_counter()
    for _ in range(10):
        an.buscar_analogos(HIST, FESTIVOS, "2025-05-27", "2025-05-26", {"limite": 20})
    assert (time.perf_counter() - t0) / 10 < 0.5  # objetivo < 0.1 s; margen amplio para CI lento


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

UCP = "Antioquia"
CLAVES_CANDIDATO = {"fecha", "score", "razones", "observado", "energia", "pico", "curva", "curva_normalizada",
                    "clima", "descriptor", "componentes"}


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    raiz = tmp_path / "data" / "raw"
    (raiz / UCP).mkdir(parents=True)
    (raiz / UCP / "datos.csv").write_text("")
    monkeypatch.setattr(data_mod, "RAW_DATA_DIR", raiz)
    return raiz


@pytest.fixture
def client(raw_dir, monkeypatch):
    monkeypatch.delenv(router_mod.ENV_SERVICE_TOKEN, raising=False)
    llamadas = {"clima": 0}

    def clima(ucp):
        llamadas["clima"] += 1
        return _clima_sintetico()

    prev = (
        router_mod.set_history_loader(lambda ucp: HIST.copy()),
        router_mod.set_festivos_loader(lambda ucp, i, f: [x for x in FESTIVOS if i.isoformat() <= x <= f.isoformat()]),
        router_mod.set_clima_loader(clima),
    )
    app = FastAPI()
    app.include_router(router_mod.router)
    app.dependency_overrides[router_mod.get_store_dep] = lambda: None  # funciona sin store
    try:
        c = TestClient(app)
        c.llamadas = llamadas
        yield c
    finally:
        router_mod.set_history_loader(prev[0])
        router_mod.set_festivos_loader(prev[1])
        router_mod.set_clima_loader(prev[2])


BODY = {"ucp": UCP, "fecha": "2026-05-19", "fecha_corte": "2025-12-31", "filtros": FILTROS_MARTES_MAYO}


def test_endpoint_200_contrato(client):
    r = client.post("/adjustments/analogos", json=BODY)
    assert r.status_code == 200, r.text
    j = r.json()
    assert set(j) == {"ucp", "fecha", "fecha_corte", "descriptor_objetivo", "filtros_aplicados", "candidatos",
                      "resumen"}
    assert j["ucp"] == UCP and j["fecha"] == "2026-05-19" and j["fecha_corte"] == "2025-12-31"
    assert {"dia_semana", "mes", "festivo", "relacion_festivo", "distancia_festivo", "semana_santa",
            "temporada_escolar"} <= set(j["descriptor_objetivo"])
    assert j["candidatos"] and all(set(c) == CLAVES_CANDIDATO for c in j["candidatos"])
    c0 = j["candidatos"][0]
    assert set(c0["pico"]) == {"mw", "hora"}
    assert c0["clima"] is None or set(c0["clima"]) == {"temp_media", "lluvia_mm"}
    assert {"n_candidatos", "curva_promedio_normalizada", "energia_promedio"} <= set(j["resumen"])
    assert client.llamadas["clima"] == 1


def test_endpoint_sin_filtros_y_sin_corte(client):
    r = client.post("/adjustments/analogos", json={"ucp": UCP, "fecha": "2025-06-10"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["fecha_corte"] == "2025-12-31" and len(j["candidatos"]) == 20
    assert "2025-06-10" not in {c["fecha"] for c in j["candidatos"]}


def test_endpoint_clima_falla_busca_sin_clima(client):
    def falla(ucp):
        raise OSError("archivo corrupto")

    prev = router_mod.set_clima_loader(falla)
    try:
        r = client.post("/adjustments/analogos", json=BODY)
    finally:
        router_mod.set_clima_loader(prev)
    assert r.status_code == 200, r.text
    assert any("clima no disponible" in n for n in r.json()["filtros_aplicados"]["notas"])


@pytest.mark.parametrize("cuerpo", [
    {**BODY, "filtros": {"desconocido": 1}},
    {**BODY, "filtros": {"dia_semana": 9}},
    {**BODY, "filtros": {"limite": 0}},
    {**BODY, "filtros": {"clima": {"temp_tolerancia": -1}}},
    {**BODY, "ucp": "../etc"},
    {**BODY, "ucp": ""},
    {**BODY, "fecha": "no-fecha"},
    {**BODY, "fecha": "1800-01-01"},
    {**BODY, "fecha_corte": "2200-01-01"},
    {"ucp": UCP},
])
def test_endpoint_422(client, cuerpo):
    r = client.post("/adjustments/analogos", json=cuerpo)
    assert r.status_code == 422, r.text


def test_endpoint_422_filtros_antes_de_leer_datos(raw_dir, monkeypatch):
    monkeypatch.delenv(router_mod.ENV_SERVICE_TOKEN, raising=False)

    def no_leer(ucp):
        raise AssertionError("no debe leer el histórico con filtros inválidos")

    prev = router_mod.set_history_loader(no_leer)
    try:
        app = FastAPI()
        app.include_router(router_mod.router)
        r = TestClient(app).post("/adjustments/analogos", json={**BODY, "filtros": {"meses": [0]}})
    finally:
        router_mod.set_history_loader(prev)
    assert r.status_code == 422


def test_endpoint_404_ucp_inexistente(client):
    r = client.post("/adjustments/analogos", json={**BODY, "ucp": "NoExiste"})
    assert r.status_code == 404


def test_endpoint_401_con_token(client, monkeypatch):
    monkeypatch.setenv(router_mod.ENV_SERVICE_TOKEN, "secreto")
    assert client.post("/adjustments/analogos", json=BODY).status_code == 401
    assert client.post("/adjustments/analogos", json=BODY,
                       headers={router_mod.HEADER_SERVICE_TOKEN: "otro"}).status_code == 401
    r = client.post("/adjustments/analogos", json=BODY, headers={router_mod.HEADER_SERVICE_TOKEN: "secreto"})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Datos reales (opcional; se salta si no están los CSV)
# ---------------------------------------------------------------------------

_RAW_REAL = Path(__file__).resolve().parents[1] / "data" / "raw" / "Antioquia" / "datos.csv"


@pytest.mark.skipif(not _RAW_REAL.is_file(), reason="sin data/raw/Antioquia/datos.csv")
def test_real_antioquia_martes_despues_de_festivo():
    hist = data_mod.cargar_historico("Antioquia", _RAW_REAL.parents[1])
    fest = festivos_colombia(range(int(hist["fecha"].dt.year.min()), 2028))
    r = an.buscar_analogos(hist, fest, "2026-05-19", "2026-05-16", FILTROS_MARTES_MAYO)
    assert r["candidatos"]
    for c in r["candidatos"]:
        assert _d(c["fecha"]).weekday() == 1 and c["fecha"] <= "2026-05-16"
        assert c["descriptor"]["relacion_festivo"] == "despues"
    json.dumps(r, allow_nan=False)
