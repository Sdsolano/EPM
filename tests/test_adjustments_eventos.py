"""Tests de `src/adjustments/eventos.py` y de las alertas de eventos/clima en `control`.

Sin red ni API_KEY real: el cliente/callable de consulta es inyectable y la
cache vive en `tmp_path`. Los tests del router (integración) van en otro archivo.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.adjustments import analogos, eventos
from src.adjustments.data import dias_semana, iso
from src.adjustments.pipeline import build_context
from src.adjustments.steps import control
from tests.ajustes_sinteticos import ProveedorPlano, historico_sintetico

HIST = historico_sintetico()
SEMANA = date(2025, 9, 8)  # sin festivos
SEMANA_SANTA = date(2025, 4, 14)  # jueves y viernes santo


def _run(coro):
    return asyncio.run(coro)


def _ctx(semana: date = SEMANA):
    return build_context("Sintetica", semana, fecha_emision=semana - timedelta(days=4), historico=HIST)


def _curvas(ctx):
    return ProveedorPlano(1000.0)("Sintetica", ctx.fecha_corte, 11)


def _energia(curves):
    return {f: float(sum(v)) for f, v in curves.items()}


# ---------------------------------------------------------------------------
# eventos: cache, fallos, saneo y semana
# ---------------------------------------------------------------------------

def test_obtener_eventos_reusa_cache(tmp_path):
    llamadas = []

    def consultar(ucp, inicio, fin):
        llamadas.append((ucp, inicio, fin))
        return {"2025-09-10": "Concierto"}

    r1 = _run(eventos.obtener_eventos("X", "2025-09-08", "2025-09-14",
                                      consultar=consultar, cache_dir=tmp_path))
    r2 = _run(eventos.obtener_eventos("X", "2025-09-08", "2025-09-14",
                                      consultar=consultar, cache_dir=tmp_path))
    assert r1 == r2 == {"2025-09-10": "Concierto"}
    assert len(llamadas) == 1

    archivos = list(tmp_path.glob("*.json"))
    assert len(archivos) == 1
    data = json.loads(archivos[0].read_text(encoding="utf-8"))
    assert data["ucp"] == "X"
    assert data["inicio"] == "2025-09-08" and data["fin"] == "2025-09-14"
    assert data["eventos"] == {"2025-09-10": "Concierto"}
    assert data["generado"]

    # cache vieja (TTL vencido) → vuelve a consultar
    archivos[0].write_text(json.dumps({**data, "generado": "2000-01-01T00:00:00+00:00"}),
                           encoding="utf-8")
    _run(eventos.obtener_eventos("X", "2025-09-08", "2025-09-14",
                                 consultar=consultar, cache_dir=tmp_path))
    assert len(llamadas) == 2


def test_rangos_distintos_no_comparten_cache(tmp_path):
    llamadas = []

    def consultar(ucp, inicio, fin):
        llamadas.append((inicio, fin))
        return {inicio: f"Inicio {inicio}", fin: f"Fin {fin}"}

    corto = _run(eventos.obtener_eventos("X", "2025-09-08", "2025-09-14",
                                         consultar=consultar, cache_dir=tmp_path))
    largo = _run(eventos.obtener_eventos("X", "2025-09-08", "2025-09-21",
                                         consultar=consultar, cache_dir=tmp_path))
    assert len(llamadas) == 2
    assert corto == {"2025-09-08": "Inicio 2025-09-08", "2025-09-14": "Fin 2025-09-14"}
    assert largo == {"2025-09-08": "Inicio 2025-09-08", "2025-09-21": "Fin 2025-09-21"}
    assert len(list(tmp_path.glob("*.json"))) == 2

    # repetir el rango corto reusa SU cache (no la del rango largo)
    assert _run(eventos.obtener_eventos("X", "2025-09-08", "2025-09-14",
                                        consultar=consultar, cache_dir=tmp_path)) == corto
    assert len(llamadas) == 2


def test_obtener_eventos_fallo_devuelve_vacio(tmp_path):
    def consultar(*args, **kwargs):
        raise RuntimeError("sin red")

    assert _run(eventos.obtener_eventos("X", "2025-09-08", "2025-09-14",
                                        consultar=consultar, cache_dir=tmp_path)) == {}
    assert list(tmp_path.glob("*.json")) == []


def test_obtener_eventos_fallo_devuelve_cache_vieja(tmp_path):
    _run(eventos.obtener_eventos("X", "2025-09-08", "2025-09-14",
                                 consultar=lambda *a, **k: {"2025-09-10": "Concierto"},
                                 cache_dir=tmp_path))

    def consultar(*args, **kwargs):
        raise RuntimeError("sin red")

    r = _run(eventos.obtener_eventos("X", "2025-09-08", "2025-09-14",
                                     consultar=consultar, cache_dir=tmp_path, ttl_dias=0))
    assert r == {"2025-09-10": "Concierto"}


def test_sin_api_key_no_cachea_y_no_borra_la_previa(monkeypatch, tmp_path):
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(eventos.EventosNoDisponibles):
        eventos.consultar_openai("X", "2025-09-08", "2025-09-14")

    bueno = _run(eventos.obtener_eventos("X", "2025-09-08", "2025-09-14",
                                         consultar=lambda *a, **k: {"2025-09-10": "Concierto"},
                                         cache_dir=tmp_path))
    assert bueno == {"2025-09-10": "Concierto"}
    archivo = next(tmp_path.glob("*.json"))
    antes = json.loads(archivo.read_text(encoding="utf-8"))

    # Sin API_KEY el fallo no debe sobrescribir la cache buena (se fuerza TTL vencido).
    archivo.write_text(json.dumps({**antes, "generado": "2000-01-01T00:00:00+00:00"}),
                       encoding="utf-8")
    r = _run(eventos.obtener_eventos("X", "2025-09-08", "2025-09-14", cache_dir=tmp_path))
    assert r == {"2025-09-10": "Concierto"}
    despues = json.loads(archivo.read_text(encoding="utf-8"))
    assert despues["eventos"] == antes["eventos"]
    assert despues["generado"] == "2000-01-01T00:00:00+00:00"  # no se reescribió


def test_sin_api_key_sin_cache_previa_no_escribe(tmp_path, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    assert _run(eventos.obtener_eventos("X", "2025-09-08", "2025-09-14",
                                        cache_dir=tmp_path)) == {}
    assert list(tmp_path.glob("*.json")) == []


def test_respuesta_vacia_se_cachea_con_ttl_corto(tmp_path):
    llamadas = []

    def consultar(*args, **kwargs):
        llamadas.append(args)
        return {}

    r1 = _run(eventos.obtener_eventos("X", "2025-09-08", "2025-09-14",
                                      consultar=consultar, cache_dir=tmp_path))
    assert r1 == {} and len(llamadas) == 1
    archivos = list(tmp_path.glob("*.json"))
    assert len(archivos) == 1

    # El vacío cacheado se reusa: no se vuelve a consultar.
    assert _run(eventos.obtener_eventos("X", "2025-09-08", "2025-09-14",
                                        consultar=consultar, cache_dir=tmp_path)) == {}
    assert len(llamadas) == 1

    # Pasado el TTL corto de vacíos se reintenta aunque `ttl_dias` siga vigente.
    data = json.loads(archivos[0].read_text(encoding="utf-8"))
    data["generado"] = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    archivos[0].write_text(json.dumps(data), encoding="utf-8")
    _run(eventos.obtener_eventos("X", "2025-09-08", "2025-09-14",
                                 consultar=consultar, cache_dir=tmp_path))
    assert len(llamadas) == 2


def test_fallo_al_guardar_cache_devuelve_eventos(tmp_path):
    bloqueado = tmp_path / "no_es_directorio"
    bloqueado.write_text("x", encoding="utf-8")

    def consultar(*args, **kwargs):
        return {"2025-09-10": "Concierto"}

    r = _run(eventos.obtener_eventos("X", "2025-09-08", "2025-09-14",
                                     consultar=consultar, cache_dir=bloqueado))
    assert r == {"2025-09-10": "Concierto"}


def test_env_cache_dir_vacio_usa_default(monkeypatch):
    monkeypatch.setenv(eventos.ENV_CACHE_DIR, "")
    assert eventos._dir_cache() == eventos.CACHE_DIR_DEFAULT


def test_escribir_cache_usa_temporal_unico(tmp_path, monkeypatch):
    """El temporal incluye pid + uuid para que dos escrituras concurrentes no colisionen."""
    origen = {}
    real = os.replace

    def espia(src, dst):
        origen["src"] = str(src)
        return real(src, dst)

    monkeypatch.setattr(eventos.os, "replace", espia)

    ruta = tmp_path / "X_2025-09-08_2025-09-14.json"
    eventos._escribir_cache(ruta, "X", date(2025, 9, 8), date(2025, 9, 14), {"2025-09-08": "e"})

    assert ruta.is_file()
    nombre = Path(origen["src"]).name
    assert nombre.endswith(".tmp")
    assert str(os.getpid()) in nombre
    assert nombre != ruta.with_suffix(".tmp").name
    assert list(tmp_path.glob("*.tmp")) == []


def test_obtener_eventos_sanea_claves_valores_y_rango(tmp_path):
    def consultar(*args, **kwargs):
        return {
            "2025-09-10": "  Evento  ",
            "2025-09-11": "x" * 200,
            "2025-09-12": 123,
            "2025-9-9": "clave inválida",
            "2025-09-13": "",
            "2025-09-01": "antes del rango",
            "2025-09-15": "después del rango",
            "2030-01-01": "fuera del rango",
        }

    r = _run(eventos.obtener_eventos("X", "2025-09-08", "2025-09-14",
                                     consultar=consultar, cache_dir=tmp_path))
    assert set(r) == {"2025-09-10", "2025-09-11"}
    assert r["2025-09-10"] == "Evento"
    assert len(r["2025-09-11"]) == eventos.MAX_LEN_EVENTO


def test_eventos_de_semana_pide_lunes_a_domingo(monkeypatch):
    capturado = {}

    async def falso(ucp, inicio, fin, **kwargs):
        capturado.update(ucp=ucp, inicio=inicio, fin=fin, kwargs=kwargs)
        return {"2025-09-10": "Evento"}

    monkeypatch.setattr(eventos, "obtener_eventos", falso)
    r = _run(eventos.eventos_de_semana("X", date(2025, 9, 8)))
    assert r == {"2025-09-10": "Evento"}
    assert capturado["ucp"] == "X"
    assert capturado["inicio"] == date(2025, 9, 8)
    assert capturado["fin"] == date(2025, 9, 14)


# ---------------------------------------------------------------------------
# control: alertas informativas (solo lectura)
# ---------------------------------------------------------------------------

def test_control_con_eventos_alerta_y_curvas_identicas():
    ctx = _ctx()
    prev = _curvas(ctx)
    ctx.extras["eventos"] = {"2025-09-10": "Concierto"}

    res = control.step(ctx, prev, {"banda_referencia_pp": None}, energia_nivel=_energia(prev))

    assert res.curves == prev
    alerta = next(a for a in res.alerts if a.codigo == "EVENTOS_SEMANA")
    assert alerta.nivel == "info"
    assert "sugeridos por IA" in alerta.mensaje and "no verificados" in alerta.mensaje
    assert alerta.fechas == ["2025-09-10"]
    assert alerta.detalle == {
        "eventos": {"2025-09-10": "Concierto"},
        "fuente": "openai",
        "verificado": False,
    }


def test_control_sin_eventos_no_alerta():
    ctx = _ctx()
    prev = _curvas(ctx)
    res = control.step(ctx, prev, {"banda_referencia_pp": None}, energia_nivel=_energia(prev))
    assert res.curves == prev
    assert res.alerts == []


def test_control_eventos_fuera_de_semana_no_salen():
    ctx = _ctx()
    prev = _curvas(ctx)
    ctx.extras["eventos"] = {
        "2025-09-01": "Antes de la semana",
        "2025-09-10": "Dentro de la semana",
        "2025-09-20": "Después de la semana",
    }

    res = control.step(ctx, prev, {"banda_referencia_pp": None}, energia_nivel=_energia(prev))
    alerta = next(a for a in res.alerts if a.codigo == "EVENTOS_SEMANA")
    assert alerta.fechas == ["2025-09-10"]
    assert alerta.detalle["eventos"] == {"2025-09-10": "Dentro de la semana"}


def test_control_eventos_solo_fuera_de_semana_no_alerta():
    ctx = _ctx()
    prev = _curvas(ctx)
    ctx.extras["eventos"] = {"2025-09-20": "Después de la semana"}

    res = control.step(ctx, prev, {"banda_referencia_pp": None}, energia_nivel=_energia(prev))
    assert not any(a.codigo == "EVENTOS_SEMANA" for a in res.alerts)


def test_control_festivos_de_la_semana_alerta():
    ctx = _ctx(SEMANA_SANTA)
    prev = _curvas(ctx)
    res = control.step(ctx, prev, {"banda_referencia_pp": None}, energia_nivel=_energia(prev))

    alerta = next(a for a in res.alerts if a.codigo == "FESTIVO_EN_SEMANA")
    assert alerta.nivel == "info"
    assert set(alerta.fechas) == {"2025-04-17", "2025-04-18"}
    assert alerta.detalle["festivos"] == alerta.fechas


def _clima_semana(fechas, **cambios):
    base = date(2024, 1, 1)
    clima = {}
    for i in range(120):
        f = iso(base + timedelta(days=i))
        clima[f] = {"temp_media": 20.0, "lluvia_mm": 0.0, "humedad_media": 70.0}
    for f in fechas:
        clima[f] = {"temp_media": 20.0, "lluvia_mm": 0.0, "humedad_media": 70.0}
    clima.update(cambios)
    return clima


def test_control_clima_observado_info(monkeypatch):
    ctx = _ctx()
    prev = _curvas(ctx)
    fechas = [iso(d) for d in dias_semana(ctx.semana_inicio)]
    monkeypatch.setattr(analogos, "cargar_clima", lambda ucp, *a, **k: _clima_semana(fechas))

    res = control.step(ctx, prev, {"banda_referencia_pp": None}, energia_nivel=_energia(prev))
    alerta = next(a for a in res.alerts if a.codigo == "CLIMA_OBSERVADO")
    assert alerta.nivel == "info"
    assert alerta.detalle["temp_media"] == 20.0
    assert alerta.detalle["lluvia_mm"] == 0.0


def test_control_clima_extremo_es_warning(monkeypatch):
    ctx = _ctx()
    prev = _curvas(ctx)
    fechas = [iso(d) for d in dias_semana(ctx.semana_inicio)]
    clima = _clima_semana(fechas, **{fechas[2]: {"temp_media": 20.0, "lluvia_mm": 50.0,
                                                "humedad_media": 70.0}})
    monkeypatch.setattr(analogos, "cargar_clima", lambda ucp, *a, **k: clima)

    res = control.step(ctx, prev, {"banda_referencia_pp": None}, energia_nivel=_energia(prev))
    alerta = next(a for a in res.alerts if a.codigo == "CLIMA_OBSERVADO")
    assert alerta.nivel == "warning"
    assert alerta.detalle["lluvia_max_diaria_mm"] == 50.0
    assert res.curves == prev


def _escribir_clima(ruta, fechas):
    filas = [
        {"fecha": f, "periodo": p, "p_t": 20.0, "p_h": 70.0, "p_i": 0.0}
        for f in fechas
        for p in range(1, 25)
    ]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(ruta, index=False)


def test_cargar_clima_cachea_por_mtime_y_tamano(tmp_path, monkeypatch):
    fechas = ["2025-09-08", "2025-09-09"]
    ruta = tmp_path / "X" / "clima_new.csv"
    _escribir_clima(ruta, fechas)

    llamadas = []
    real = analogos.pd.read_csv

    def contar(*args, **kwargs):
        llamadas.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(analogos.pd, "read_csv", contar)

    c1 = analogos.cargar_clima("X", tmp_path)
    c2 = analogos.cargar_clima("X", tmp_path)
    assert set(c1) == set(fechas)
    assert c1 is c2
    assert len(llamadas) == 1

    # Mismo tamaño, mtime distinto: la cache se invalida y se relee.
    st = ruta.stat()
    os.utime(ruta, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    c3 = analogos.cargar_clima("X", tmp_path)
    assert c3 == c1
    assert len(llamadas) == 2


def test_cargar_clima_cache_lru_acotada_sin_clear_global(tmp_path, monkeypatch):
    for ucp in ("A", "B", "C", "D", "E"):
        _escribir_clima(tmp_path / ucp / "clima_new.csv", ["2025-09-08"])

    llamadas = []
    real = analogos.pd.read_csv

    def contar(*args, **kwargs):
        llamadas.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(analogos.pd, "read_csv", contar)

    for ucp in ("A", "B", "C", "D", "E"):
        analogos.cargar_clima(ucp, tmp_path)
    assert len(llamadas) == 5

    # E (la más reciente) sigue cacheada; A fue la víctima del LRU de 4.
    assert analogos.cargar_clima("E", tmp_path)
    assert len(llamadas) == 5
    assert analogos.cargar_clima("A", tmp_path)
    assert len(llamadas) == 6
    assert len(analogos._CLIMA_CACHE) == analogos.MAX_CLIMA_CACHE


def test_control_clima_no_se_rompe_con_cache(tmp_path, monkeypatch):
    ctx = _ctx()
    prev = _curvas(ctx)
    fechas = [iso(d) for d in dias_semana(ctx.semana_inicio)]
    _escribir_clima(tmp_path / ctx.ucp / "clima_new.csv", fechas)
    monkeypatch.setattr(analogos.data, "RAW_DATA_DIR", tmp_path)

    for _ in range(2):
        res = control.step(ctx, prev, {"banda_referencia_pp": None}, energia_nivel=_energia(prev))
        alerta = next(a for a in res.alerts if a.codigo == "CLIMA_OBSERVADO")
        assert alerta.nivel == "info"
        assert alerta.detalle["temp_media"] == 20.0
        assert res.curves == prev
