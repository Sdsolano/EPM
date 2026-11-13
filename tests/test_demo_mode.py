"""Tests del modo demo (src/api/demo.py, migraciones al arrancar, snapshot). Sin red ni OpenAI.

Postgres: solo si AJUSTES_DATABASE_URL apunta a un servidor local donde se pueda crear una base temporal;
si no, los tests de migraciones se saltan.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.adjustments import router as router_mod
from src.api import demo

ROOT = Path(__file__).resolve().parent.parent


def _basic(clave: str, usuario: str = "x") -> dict:
    return {"Authorization": "Basic " + base64.b64encode(f"{usuario}:{clave}".encode()).decode()}


@pytest.fixture
def entorno(monkeypatch):
    for v in ("DEMO_PASSWORD", "DEMO_STATIC_DIR", "DEMO_REFRESH_DATA", "DEMO_SKIP_MIGRATIONS",
              "DATABASE_URL", "AJUSTES_DATABASE_URL", "DEMO_ALLOW_NOAUTH"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("DEMO_ALLOW_NOAUTH", "1")
    anterior = router_mod._data_refresher
    yield monkeypatch
    router_mod._data_refresher = anterior


def _app_base() -> FastAPI:
    app = FastAPI()

    @app.get("/")
    def raiz():
        return {"api": "json"}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/adjustments/ping")
    def ping():
        return {"pong": True}

    @app.post("/predict")
    def predict():
        return {"ok": True}

    return app


def _static(tmp_path: Path) -> Path:
    d = tmp_path / "dist"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text("<html>SPA</html>")
    (d / "assets" / "app.js").write_text("console.log(1)")
    (tmp_path / "secreto.txt").write_text("no")
    return d


def test_demo_activo_por_variable(entorno):
    assert demo.demo_activo() is False
    entorno.setenv("DEMO_MODE", "1")
    assert demo.demo_activo() is True
    entorno.setenv("DEMO_MODE", "0")
    assert demo.demo_activo() is False


def test_sin_password_falla_cerrado(entorno):
    entorno.delenv("DEMO_ALLOW_NOAUTH")
    with pytest.raises(RuntimeError):
        demo.install_demo(_app_base())


def test_sin_password_no_hay_auth(entorno):
    app = _app_base()
    demo.install_demo(app)
    c = TestClient(app)
    assert c.get("/adjustments/ping").status_code == 200
    assert router_mod._data_refresher is None  # solo lectura por defecto


def test_auth_basic_y_cabecera(entorno):
    entorno.setenv("DEMO_PASSWORD", "clave-demo")
    app = _app_base()
    demo.install_demo(app)
    c = TestClient(app)
    assert c.get("/health").status_code == 200  # exento
    r = c.get("/adjustments/ping")
    assert r.status_code == 401 and "www-authenticate" not in r.headers
    assert c.get("/adjustments/ping", headers=_basic("mala")).status_code == 401
    assert c.get("/adjustments/ping", headers={"Authorization": "Basic !!!"}).status_code == 401
    assert c.get("/adjustments/ping", headers=_basic("clave-demo", "cualquiera")).status_code == 200
    assert c.post("/predict", headers={"X-Demo-Password": "clave-demo"}).status_code == 200
    assert c.post("/predict").status_code == 401


def test_dsn_desde_database_url(entorno):
    entorno.setenv("DATABASE_URL", "postgresql://u:p@h:5432/db")
    assert demo.resolver_dsn() == "postgresql://u:p@h:5432/db"
    import os
    assert os.environ["AJUSTES_DATABASE_URL"] == "postgresql://u:p@h:5432/db"
    entorno.setenv("AJUSTES_DATABASE_URL", "postgresql://otro/db")
    assert demo.resolver_dsn() == "postgresql://otro/db"


def test_refresh_opt_in(entorno):
    entorno.setenv("DEMO_REFRESH_DATA", "1")
    sentinel = lambda ucp: None  # noqa: E731
    router_mod.set_data_refresher(sentinel)
    demo.install_demo(_app_base())
    assert router_mod._data_refresher is sentinel


def test_estaticos_y_fallback_spa(entorno, tmp_path):
    entorno.setenv("DEMO_STATIC_DIR", str(_static(tmp_path)))
    app = _app_base()
    demo.install_demo(app)
    c = TestClient(app)
    html = {"Accept": "text/html"}
    assert "SPA" in c.get("/", headers=html).text            # "/" ya no es el JSON de la API
    assert "SPA" in c.get("/mesa/ajustes", headers=html).text  # ruta del front -> index.html
    assert c.get("/assets/app.js").text == "console.log(1)"
    assert c.get("/assets/falta.js", headers=html).status_code == 404  # con extensión: 404 real
    assert c.get("/ruta-api-inexistente", headers={"Accept": "application/json"}).status_code == 404
    assert c.get("/adjustments/ping").json() == {"pong": True}  # la API sigue intacta
    assert c.get("/health").json() == {"status": "ok"}
    assert c.post("/predict").status_code == 200
    assert c.get("/%2e%2e/secreto.txt", headers=html).text != "no"  # sin path traversal
    assert c.get("/..%2fsecreto.txt").text != "no"


def test_sin_estaticos_no_rompe(entorno, tmp_path):
    entorno.setenv("DEMO_STATIC_DIR", str(tmp_path / "no-existe"))
    app = _app_base()
    demo.install_demo(app)
    assert TestClient(app).get("/").json() == {"api": "json"}


def test_main_sin_demo_mode_no_cambia_nada(entorno):
    """Importar la API sin DEMO_MODE no instala auth ni estáticos."""
    from src.api import main

    assert not any(getattr(r, "path", "") == "/{ruta:path}" for r in main.app.router.routes)
    assert main.app.user_middleware == []


def test_migraciones_archivos_presentes():
    from src.adjustments.migrations import SQL_DIR

    nombres = [p.name for p in sorted(SQL_DIR.glob("*.sql"))]
    assert nombres[:2] == ["001_init.sql", "002_versionado.sql"]


@pytest.fixture
def base_temporal():
    import os

    dsn = os.getenv("TEST_DATABASE_URL") or os.getenv("AJUSTES_DATABASE_URL")
    if not dsn:
        pytest.skip("sin AJUSTES_DATABASE_URL/TEST_DATABASE_URL")
    try:
        import psycopg2
        from psycopg2 import sql
        from psycopg2.extensions import make_dsn, parse_dsn

        admin = psycopg2.connect(dsn)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Postgres no disponible: {e}")
    admin.autocommit = True
    nombre = f"demo_test_{uuid.uuid4().hex[:10]}"
    try:
        with admin.cursor() as cur:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(nombre)))
    except Exception as e:  # noqa: BLE001
        admin.close()
        pytest.skip(f"sin permiso CREATE DATABASE: {e}")
    try:
        yield make_dsn(dsn, dbname=nombre)
    finally:
        with admin.cursor() as cur:
            cur.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(nombre)))
        admin.close()


def test_migraciones_idempotentes(base_temporal):
    import psycopg2

    from src.adjustments.migrations import aplicar_migraciones

    assert aplicar_migraciones(base_temporal, intentos=1) == ["001_init.sql", "002_versionado.sql"]
    assert aplicar_migraciones(base_temporal, intentos=1) == ["001_init.sql", "002_versionado.sql"]  # 2a vez: ok
    with psycopg2.connect(base_temporal) as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass('ajustes.corrida'), to_regclass('ajustes.corrida_version')")
        assert all(cur.fetchone())


def test_arranque_aplica_migraciones_y_store(base_temporal, entorno):
    import psycopg2

    entorno.setenv("DATABASE_URL", base_temporal)
    app = _app_base()
    demo.install_demo(app)
    with TestClient(app):  # dispara el evento startup
        pass
    with psycopg2.connect(base_temporal) as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass('ajustes.corrida_evento')")
        assert cur.fetchone()[0]
    from src.adjustments.store import get_store

    assert get_store() is not None


def test_migraciones_sin_db_falla_con_mensaje():
    from src.adjustments.migrations import aplicar_migraciones

    with pytest.raises(RuntimeError, match="No se pudo conectar"):
        aplicar_migraciones("postgresql://nadie:x@127.0.0.1:1/db", intentos=1, espera_s=0)


def _cargar_snapshot():
    spec = importlib.util.spec_from_file_location("demo_snapshot", ROOT / "scripts" / "demo_snapshot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_snapshot_manifiesto_y_verificacion(tmp_path):
    snap = _cargar_snapshot()
    src = tmp_path / "repo"
    for ucp in snap.UCPS:
        (src / f"data/raw/{ucp}").mkdir(parents=True)
        (src / f"data/raw/{ucp}/datos.csv").write_text("a,b\n1,2\n")
        (src / f"models/{ucp}/registry").mkdir(parents=True)
        (src / f"models/{ucp}/registry/champion_model.joblib").write_bytes(b"m")
    (src / "reports/ajustes").mkdir(parents=True)
    (src / "reports/ajustes/curvas_modelo_k1_Antioquia.csv").write_text("x\n")
    dest = tmp_path / "out"
    mf = snap.construir(dest, src_root=src)
    rutas = {e["ruta"] for e in mf["archivos"]}
    assert "reports/ajustes/curvas_modelo_k1_Antioquia.csv" in rutas
    assert "data/raw/Atlantico/datos.csv" in rutas
    assert any("curvas_modelo_k1_Atlantico" in a for a in mf["avisos"])
    assert snap.verificar(dest) == []
    (dest / "data/raw/Atlantico/datos.csv").write_text("alterado")
    assert snap.verificar(dest) == ["hash distinto data/raw/Atlantico/datos.csv"]
    assert json.loads((dest / "MANIFEST.json").read_text())["total_bytes"] > 0


def test_api_demo_y_estaticos_publicos(entorno, tmp_path):
    (tmp_path / "index.html").write_text("<html>app</html>")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "a.js").write_text("x")
    entorno.setenv("DEMO_PASSWORD", "k")
    entorno.setenv("DEMO_STATIC_DIR", str(tmp_path))
    app = _app_base()
    demo.install_demo(app)
    c = TestClient(app)
    assert c.get("/").status_code == 200  # público
    assert c.get("/assets/a.js").status_code == 200
    r = c.get("/api/demo/ucps")
    assert r.status_code == 401 and "www-authenticate" not in r.headers
    u = c.get("/api/demo/ucps", headers=_basic("k", "demo"))
    assert u.status_code == 200 and isinstance(u.json(), list)
    assert all({"ucp", "nombre"} <= set(x) for x in u.json())
    if u.json():
        ucp = u.json()[0]["ucp"]
        rr = c.get(f"/api/demo/reales?ucp={ucp}&desde=2026-06-01&hasta=2026-06-03", headers=_basic("k"))
        assert rr.status_code == 200 and all(len(v) == 24 for v in rr.json().values())
    assert c.get("/api/demo/reales?ucp=Nada&desde=2026-06-01&hasta=2026-06-03", headers=_basic("k")).status_code == 404
    assert c.get("/api/demo/reales?ucp=Antioquia&desde=mal&hasta=x", headers=_basic("k")).status_code == 422


def test_demo_desactiva_actualizacion_de_datos_del_modelo(entorno):
    import sys
    import types

    fake = types.ModuleType("src.api.main")
    llamadas = []
    fake.full_update_csv = lambda ucp: llamadas.append(ucp)
    fake.full_update_csv_diario = lambda ucp: llamadas.append(ucp)
    entorno.setitem(sys.modules, "src.api.main", fake)
    demo._desactivar_actualizacion_modelo()
    fake.full_update_csv("Antioquia")
    fake.full_update_csv_diario("Atlantico")
    assert llamadas == []  # no-op: no hay red ni escritura
