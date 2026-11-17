"""Tests del proveedor `modelo` aislado (scripts/backtest_modelo.py).

Rápidos y sin red: hash, festivos offline, recorte, redirección de rutas, bloqueo de
red, no-op de `full_update_csv` y restauración al salir. El entrenamiento real
(~40 s) solo corre con `EPM_TEST_MODELO_REAL=1`.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _cargar_script():
    spec = importlib.util.spec_from_file_location("backtest_modelo", ROOT / "scripts" / "backtest_modelo.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["backtest_modelo"] = mod
    spec.loader.exec_module(mod)
    return mod


bm = _cargar_script()

COLS = ["UCP", "VARIABLE", "FECHA", "Clasificador interno", "TIPO DIA"] + [f"P{i}" for i in range(1, 25)] + ["TOTAL"]


def _escribir_repo_falso(raiz: Path, ucp: str = "Prueba", dias: int = 20) -> Path:
    filas = []
    for i in range(dias):
        f = date(2026, 1, 1) + timedelta(days=i)
        vals = [round(100 + h + i * 0.1, 3) for h in range(24)]
        filas.append([ucp, "Demanda_Real", f.isoformat(), "NORMAL", "Sã¡bado"] + vals + [round(sum(vals), 3)])
    d = raiz / "data" / "raw" / ucp
    d.mkdir(parents=True)
    pd.DataFrame(filas, columns=COLS).to_csv(d / "datos.csv", index=False, encoding="utf-8")
    (d / "clima_new.csv").write_text("fecha,periodo,p_t,p_h,p_v,p_i\n2026-01-01,1,25.0,60.0,7.4,3\n", encoding="utf-8")
    (raiz / "config").mkdir()
    (raiz / "config" / "festivos.json").write_text("{}", encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# Piezas puras
# ---------------------------------------------------------------------------

def test_hash_arbol_detecta_cambios(tmp_path):
    (tmp_path / "models" / "x").mkdir(parents=True)
    (tmp_path / "models" / "x" / "a.bin").write_bytes(b"abc")
    (tmp_path / "models" / "__pycache__").mkdir()
    (tmp_path / "models" / "__pycache__" / "z.pyc").write_bytes(b"ignorado")
    h1 = bm.hash_arbol(tmp_path, ["models", "no_existe"])
    assert list(h1) == ["models/x/a.bin"]
    (tmp_path / "models" / "x" / "a.bin").write_bytes(b"abd")
    (tmp_path / "models" / "b.bin").write_bytes(b"1")
    h2 = bm.hash_arbol(tmp_path, ["models"])
    dif = bm.diferencias_hash(h1, h2)
    assert dif == {"nuevos": ["models/b.bin"], "borrados": [], "modificados": ["models/x/a.bin"]}
    assert bm.diferencias_hash(h2, h2) == {"nuevos": [], "borrados": [], "modificados": []}


def test_festivos_offline_colombia():
    f = bm.festivos_offline("2026-01-01", "2026-01-31", "Antioquia")
    assert f == ["2026-01-01", "2026-01-12"]  # Año nuevo y Reyes (trasladado a lunes)
    cruzado = bm.festivos_offline("2025-12-20", "2026-01-02")
    assert cruzado == ["2025-12-25", "2026-01-01"]
    assert bm.festivos_offline("2026-02-01", "2026-01-01") == []


def test_recortar_datos_sin_fuga_y_texto_intacto(tmp_path):
    d = _escribir_repo_falso(tmp_path)
    filas, ultima = bm.recortar_datos(d / "datos.csv", tmp_path / "rec.csv", date(2026, 1, 10))
    assert (filas, ultima) == (10, date(2026, 1, 10))
    rec = pd.read_csv(tmp_path / "rec.csv", dtype=str, encoding="utf-8")
    assert rec["FECHA"].max() == "2026-01-10"
    orig = pd.read_csv(d / "datos.csv", dtype=str, encoding="utf-8")
    pd.testing.assert_frame_equal(rec, orig.iloc[:10])  # mismas celdas (texto), solo filas <= corte
    assert bm.recortar_datos(d / "datos.csv", tmp_path / "v.csv", date(2025, 1, 1)) == (0, None)


def test_a_curvas_valida_fechas():
    corte = date(2026, 3, 7)
    fechas = pd.date_range("2026-03-08", periods=3, freq="D")
    preds = pd.DataFrame({"fecha": fechas, **{f"P{i}": [float(i)] * 3 for i in range(1, 25)}})
    c = bm.SandboxModelo._a_curvas(preds, corte, 3)
    assert list(c) == ["2026-03-08", "2026-03-09", "2026-03-10"] and c["2026-03-08"][23] == 24.0
    with pytest.raises(ValueError, match="fechas predichas"):
        bm.SandboxModelo._a_curvas(preds, corte - timedelta(days=1), 3)
    with pytest.raises(ValueError, match="fechas predichas"):
        bm.SandboxModelo._a_curvas(preds, corte, 4)


# ---------------------------------------------------------------------------
# Sandbox (sin entrenar)
# ---------------------------------------------------------------------------

def test_sandbox_aisla_rutas_red_festivos_y_restaura(tmp_path):
    import requests

    import src.api.main as api_main
    import src.config.settings as settings
    import src.pipeline.orchestrator as orq
    import src.prediction.hourly.disaggregation_engine as eng
    from src.adjustments import data as adj_data
    from src.prediction.festivos_api import FestivosAPIClient

    _escribir_repo_falso(tmp_path / "repo")
    cwd0 = os.getcwd()
    get0, connect0, update0 = requests.get, socket.socket.connect, api_main.full_update_csv
    models0, proc0, raw0 = settings.MODELS_DIR, orq.PROCESSED_DATA_DIR, adj_data.RAW_DATA_DIR

    with bm.SandboxModelo(["Prueba"], raiz_repo=tmp_path / "repo") as sb:
        d = sb.dir
        assert Path(os.getcwd()).resolve() == d
        # copias de datos y config, nada del repo real
        assert (d / "data/raw/Prueba/datos.csv").exists() and (d / "config/festivos.json").exists()
        # rutas absolutas redirigidas al sandbox
        for ruta in (settings.MODELS_DIR, settings.LOGS_DIR, orq.PROCESSED_DATA_DIR, orq.FEATURES_DATA_DIR,
                     eng.MODELS_DIR, adj_data.RAW_DATA_DIR):
            assert bm._dentro_de(Path(ruta), d), ruta
        # red bloqueada (requests y socket), loopback permitido
        with pytest.raises(bm.RedBloqueadaError):
            requests.get("https://example.com")
        with pytest.raises(bm.RedBloqueadaError):
            socket.create_connection(("203.0.113.1", 80), timeout=1)
        s = socket.socket()
        try:
            with pytest.raises(bm.RedBloqueadaError):
                s.connect(("203.0.113.1", 80))
        finally:
            s.close()
        # festivos offline en todas las rutas del flujo
        assert FestivosAPIClient().get_festivos("2026-01-01", "2026-01-31", "Antioquia") == ["2026-01-01", "2026-01-12"]
        assert FestivosAPIClient().get_festivos_set("2026-01-01", "2026-01-31", "X") == {"2026-01-01", "2026-01-12"}
        # actualización de datos desactivada
        assert api_main.full_update_csv("Prueba") is None
        assert api_main.full_update_csv is not update0
        # recorte al corte
        assert sb.preparar_corte("Prueba", date(2026, 1, 5)) == date(2026, 1, 5)
        rec = pd.read_csv(d / "data/raw/Prueba/datos.csv")
        assert rec["FECHA"].max() == "2026-01-05" and len(rec) == 5
        with pytest.raises(ValueError):
            sb.preparar_corte("Prueba", date(2025, 1, 1))
        # predecir sin modelos entrenados / con corte anterior al de los modelos
        with pytest.raises(RuntimeError):
            sb.predecir("Prueba", date(2026, 1, 5))
        sb.modelo_corte["Prueba"] = date(2026, 1, 10)
        with pytest.raises(ValueError, match="fuga"):
            sb.predecir("Prueba", date(2026, 1, 5))

    # todo restaurado y el sandbox temporal borrado
    assert os.getcwd() == cwd0
    assert requests.get is get0 and socket.socket.connect is connect0 and api_main.full_update_csv is update0
    assert settings.MODELS_DIR == models0 and orq.PROCESSED_DATA_DIR == proc0 and adj_data.RAW_DATA_DIR == raw0
    assert not d.exists()


# ---------------------------------------------------------------------------
# Entrenamiento real (opt-in)
# ---------------------------------------------------------------------------

REAL = os.environ.get("EPM_TEST_MODELO_REAL") == "1"


@pytest.mark.skipif(not REAL, reason="entrenamiento real: exportar EPM_TEST_MODELO_REAL=1 (~1 min)")
def test_entreno_real_sin_fuga_y_repo_intacto():
    ucp, corte = "Atlantico", date(2026, 3, 7)
    antes = bm.hash_arbol(ROOT)
    with bm.SandboxModelo([ucp]) as sb:
        c1 = sb.entrenar(ucp, corte, 8)
        assert list(c1)[0] == (corte + timedelta(days=1)).isoformat() and len(c1) == 8
        # no-fuga: alterar los datos posteriores al corte no cambia nada
        completo = sb.dir / "data" / "raw" / ucp / "datos.csv.completo"
        df = pd.read_csv(completo)
        fut = pd.to_datetime(df["FECHA"]) > pd.Timestamp(corte)
        cols = [f"P{i}" for i in range(1, 25)] + ["TOTAL"]
        df.loc[fut, cols] = df.loc[fut, cols] * 3
        df.to_csv(completo, index=False)
        c2 = sb.entrenar(ucp, corte, 8)
        for f in c1:
            assert np.allclose(c1[f], c2[f], atol=1e-6), f
        c3 = sb.predecir(ucp, corte + timedelta(days=7), 8)
        assert list(c3)[0] == (corte + timedelta(days=8)).isoformat()
    assert bm.hash_arbol(ROOT) == antes
