"""Modo demo (DEMO_MODE=1): entorno interno autocontenido para Railway. Nada de esto actúa sin la variable.

Variables:
- DEMO_MODE            1/true/yes activa todo lo de este módulo (main.py llama a `install_demo`).
- DATABASE_URL         Postgres de Railway; se copia a AJUSTES_DATABASE_URL si esta no está definida.
- AJUSTES_DATABASE_URL (preferente si existe) DSN que usa el módulo de ajustes.
- DEMO_PASSWORD        contraseña compartida. Si no está definida NO hay auth. Nunca va en el código.
- DEMO_STATIC_DIR      carpeta del build del front (default `static_demo`); con fallback SPA a index.html.
- DEMO_REFRESH_DATA    1 para permitir que el módulo refresque datos desde la red (default: solo lectura).
- DEMO_SKIP_MIGRATIONS 1 para no aplicar migraciones al arrancar.
- API_KEY              clave de OpenAI (nombre que ya usa el código; alertas de eventos 4D).
- PORT                 lo lee main.py en el arranque directo (`python -m src.api.main`).

Auth: HTTP Basic (usuario cualquiera, clave = DEMO_PASSWORD). El navegador pregunta una vez y reenvía la
credencial en todas las peticiones del mismo origen (fetch incluido), sin login propio en el front.
También se acepta la cabecera `X-Demo-Password` para scripts. /health queda exento (healthcheck).
Es protección de entorno interno, no un sistema de usuarios.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response

logger = logging.getLogger(__name__)

ENV_MODE = "DEMO_MODE"
ENV_PASSWORD = "DEMO_PASSWORD"
ENV_STATIC = "DEMO_STATIC_DIR"
ENV_REFRESH = "DEMO_REFRESH_DATA"
ENV_SKIP_MIG = "DEMO_SKIP_MIGRATIONS"
DEFAULT_STATIC = "static_demo"
RUTAS_EXENTAS = {"/health"}
_VERDADERO = {"1", "true", "yes", "on"}


def _activo(nombre: str) -> bool:
    return os.getenv(nombre, "").strip().lower() in _VERDADERO


def demo_activo() -> bool:
    return _activo(ENV_MODE)


def resolver_dsn() -> Optional[str]:
    """Garantiza AJUSTES_DATABASE_URL (usa DATABASE_URL de Railway si hace falta) y la devuelve."""
    dsn = os.getenv("AJUSTES_DATABASE_URL") or os.getenv("DATABASE_URL")
    if dsn and not os.getenv("AJUSTES_DATABASE_URL"):
        os.environ["AJUSTES_DATABASE_URL"] = dsn
    return dsn or None


def password_valida(request: Request, esperado: str) -> bool:
    candidatos = []
    cab = request.headers.get("authorization", "")
    if cab[:6].lower() == "basic ":
        try:
            decodificado = base64.b64decode(cab[6:].strip()).decode("utf-8")
            candidatos.append(decodificado.partition(":")[2])
        except Exception:  # noqa: BLE001 — cabecera mal formada = no autenticado
            pass
    cab_dir = request.headers.get("x-demo-password")
    if cab_dir:
        candidatos.append(cab_dir)
    e = esperado.encode("utf-8")
    return any(hmac.compare_digest(c.encode("utf-8"), e) for c in candidatos)


def _es_estatico_publico(path: str, method: str, static_dir: Optional[Path]) -> bool:
    """`GET /` y archivos reales del build del front son públicos (la pantalla de acceso debe poder cargar)."""
    if method not in ("GET", "HEAD") or static_dir is None:
        return False
    if path == "/":
        return True
    raiz = static_dir.resolve()
    candidato = (raiz / path.lstrip("/")).resolve()
    return candidato.is_file() and raiz in candidato.parents


def _instalar_auth(app: FastAPI, static_dir: Optional[Path] = None) -> None:
    @app.middleware("http")
    async def _auth_demo(request: Request, call_next):
        esperado = os.getenv(ENV_PASSWORD)  # se lee por petición: nunca queda en código ni en estado
        if (not esperado or request.url.path in RUTAS_EXENTAS
                or _es_estatico_publico(request.url.path, request.method, static_dir)
                or password_valida(request, esperado)):
            return await call_next(request)
        # Sin WWW-Authenticate a propósito: así el navegador no abre su diálogo nativo (el front tiene su acceso).
        return JSONResponse({"detail": "Autenticación requerida"}, status_code=401)


def _nombre_ucp(ucp: str) -> str:
    return {"Atlantico": "Atlántico"}.get(ucp, ucp)


def _instalar_api_demo(app: FastAPI) -> None:
    """Rutas propias del front demo (JSON plano). Registrar ANTES del catch-all de estáticos."""
    from datetime import date

    from fastapi import HTTPException

    from src.adjustments import data as ajdata

    @app.get("/api/demo/ucps", include_in_schema=False)
    async def _ucps():
        raiz = Path(ajdata.RAW_DATA_DIR)
        modelos = Path(__file__).resolve().parents[2] / "models"
        salida = []
        if raiz.is_dir():
            for d in sorted(raiz.iterdir()):
                if d.is_dir() and (d / "datos.csv").is_file() and (modelos / d.name).is_dir():
                    salida.append({"ucp": d.name, "nombre": _nombre_ucp(d.name)})
        return salida

    @app.get("/api/demo/reales", include_in_schema=False)
    async def _reales(ucp: str, desde: str, hasta: str):
        if not ucp.isalnum():
            raise HTTPException(422, "ucp inválida")
        try:
            d, h = date.fromisoformat(desde), date.fromisoformat(hasta)
        except ValueError:
            raise HTTPException(422, "fechas inválidas (YYYY-MM-DD)")
        try:
            hist = await asyncio.get_running_loop().run_in_executor(None, ajdata.cargar_historico, ucp)
        except FileNotFoundError:
            raise HTTPException(404, "UCP sin histórico")
        curvas = ajdata.curvas_reales(hist)
        return {f: [float(x) for x in v] for f, v in curvas.items() if d.isoformat() <= f <= h.isoformat()}


def _instalar_estaticos(app: FastAPI, static_dir: Path) -> None:
    raiz = static_dir.resolve()
    index = raiz / "index.html"
    # "/" pasa a ser la SPA (el JSON informativo de main.py deja de ser la raíz en modo demo)
    app.router.routes[:] = [r for r in app.router.routes
                            if not (getattr(r, "path", None) == "/" and "GET" in (getattr(r, "methods", None) or ()))]

    @app.get("/{ruta:path}", include_in_schema=False)
    async def _spa(ruta: str, request: Request):
        # Registrada al final: solo recibe lo que ninguna ruta de la API atendió.
        if ruta:
            candidato = (raiz / ruta).resolve()
            if candidato.is_file() and raiz in candidato.parents:
                return FileResponse(candidato)
        ultimo = ruta.rsplit("/", 1)[-1]
        quiere_html = "text/html" in request.headers.get("accept", "")
        if (not ruta or ("." not in ultimo and quiere_html)) and index.is_file():
            return FileResponse(index, headers={"Cache-Control": "no-cache"})
        return JSONResponse({"detail": "Not Found"}, status_code=404)


def _desactivar_actualizacion_modelo() -> None:
    """`run_predict_flow` (pronóstico del modelo) llama a `full_update_csv` (API externa + escribe data/raw);
    en la demo es de solo lectura y offline respecto a datos: se reemplaza por un no-op en el módulo main."""
    import sys

    main_mod = sys.modules.get("src.api.main")
    for nombre in ("full_update_csv", "full_update_csv_diario"):
        if main_mod is not None and hasattr(main_mod, nombre):
            setattr(main_mod, nombre, lambda ucp, *a, **k: logger.info("[demo] actualización de datos omitida (%s)", ucp))


def install_demo(app: FastAPI) -> None:
    """Activa el modo demo sobre `app`. Llamar DESPUÉS de incluir todos los routers."""
    from src.adjustments.router import set_data_refresher

    dsn = resolver_dsn()
    if not dsn:
        logger.warning("[demo] sin DATABASE_URL/AJUSTES_DATABASE_URL: /adjustments responderá 503")
    static_dir = Path(os.getenv(ENV_STATIC) or DEFAULT_STATIC)
    if not os.getenv(ENV_PASSWORD):
        if not _activo("DEMO_ALLOW_NOAUTH"):
            raise RuntimeError("DEMO_MODE exige DEMO_PASSWORD (o DEMO_ALLOW_NOAUTH=1 para desarrollo local sin auth)")
        logger.warning("[demo] DEMO_PASSWORD no definido y DEMO_ALLOW_NOAUTH=1: entorno SIN autenticación")
    else:
        _instalar_auth(app, static_dir)
    if not _activo(ENV_REFRESH):
        set_data_refresher(None)  # solo lectura: sin red ni escritura sobre data/raw
        _desactivar_actualizacion_modelo()
    _instalar_api_demo(app)

    if (static_dir / "index.html").is_file():
        _instalar_estaticos(app, static_dir)
    else:
        logger.warning("[demo] no hay index.html en %s: no se sirve el front", static_dir)

    @app.on_event("startup")
    async def _migrar() -> None:
        if not dsn or _activo(ENV_SKIP_MIG):
            return
        from src.adjustments.migrations import aplicar_migraciones

        try:
            await asyncio.get_running_loop().run_in_executor(None, aplicar_migraciones, dsn)
        except Exception:  # noqa: BLE001 — el servicio arranca igual; /adjustments fallará con detalle
            logger.exception("[demo] fallaron las migraciones de ajustes")

    logger.info("[demo] modo demo activo (auth=%s, front=%s)", bool(os.getenv(ENV_PASSWORD)),
                (static_dir / "index.html").is_file())
