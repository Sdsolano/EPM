"""
Proveedor `modelo` aislado para el backtest de ajustes (Fase 2a, parte A).

Corre el flujo real de pronóstico (`src/api/main.py`: pipeline de features →
`train_model_if_needed` → `train_hourly_disaggregation_if_needed` →
`ForecastPipeline.predict_next_n_days`) dentro de un SANDBOX temporal, sin red y
sin escribir nada del repo:

- Directorio temporal con copia de `data/raw/{ucp}/{datos,clima_new}.csv` y
  `config/`; modelos propios en `<sandbox>/models/{ucp}/` (nunca se leen ni se
  escriben los del repo).
- `os.chdir` al sandbox mientras dura el contexto (main.py usa rutas relativas).
- Constantes de ruta absolutas ya importadas (`src.config.settings.*`,
  `orchestrator.PROCESSED_DATA_DIR`, `MODELS_DIR` de los módulos hourly,
  `monitoring.LOGS_DIR`, `adjustments.data.RAW_DATA_DIR`, ...) redirigidas al
  sandbox por monkeypatch y restauradas al salir.
- `full_update_csv` / `full_update_csv_diario` → no-op.
- Festivos offline: `FestivosAPIClient.get_festivos` devuelve el calendario
  oficial de Colombia de la librería `holidays` (`adjustments.data.festivos_colombia`).
- Red bloqueada: `requests` y `socket.connect` a hosts no locales lanzan
  `RedBloqueadaError` (el loopback se permite: asyncio en Windows lo usa).
- Sin fuga: antes de cada llamada se reescribe el `datos.csv` del sandbox con las
  filas `FECHA <= corte` (además se pasa `end_date=corte`). El clima se copia
  completo: el "pronóstico" de clima de los días futuros es el clima OBSERVADO
  (limitación declarada; `clima_new.csv` solo cubre desde 2025-12-01, antes de eso
  el flujo usa promedios históricos / ceros, igual que producción).

Uso como módulo (lo usa el backtest):

    with SandboxModelo(["Antioquia", "Atlantico"]) as sb:
        sb.entrenar("Antioquia", date(2026, 3, 7))          # modelos con datos <= corte
        curvas = sb.predecir("Antioquia", date(2026, 3, 14), 8)  # {fecha_iso: [P1..P24]}
        prov = sb.proveedor("Antioquia", date(2026, 3, 7))   # pipeline.BaseProvider

CLI (medición de tiempos + verificación de hashes):

    python scripts/backtest_modelo.py --ucp Antioquia Atlantico --corte 2026-03-07 --n-days 8
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import time
from contextlib import ExitStack
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adjustments import data, pipeline  # noqa: E402
from src.adjustments.types import Curves  # noqa: E402

logger = logging.getLogger("backtest_modelo")

UCPS_DEFAULT = ("Antioquia", "Atlantico")
ARCHIVOS_RAW = ("datos.csv", "clima_new.csv")
DIRS_HASH_DEFAULT = ("data/raw", "models", "data/features", "data/processed")
# Nombres de constantes de ruta que se redirigen en cualquier módulo ya importado.
NOMBRES_RUTA = (
    "BASE_DIR", "DATA_DIR", "LOGS_DIR", "MODELS_DIR", "CONFIG_DIR", "REPORTS_DIR",
    "RAW_DATA_DIR", "PROCESSED_DATA_DIR", "FEATURES_DATA_DIR",
)
# Prefijos de módulos del proyecto (algunos se importan además sin `src.` por los sys.path.append).
PREFIJOS_MODULOS = ("src.", "pipeline.", "prediction.", "models.", "config.", "utils.")
HOSTS_LOCALES = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}


class RedBloqueadaError(RuntimeError):
    """Intento de acceso a red dentro del sandbox."""


# ---------------------------------------------------------------------------
# Hash de árboles de archivos
# ---------------------------------------------------------------------------

def hash_arbol(raiz: Path | str, dirs: Sequence[str] = DIRS_HASH_DEFAULT) -> Dict[str, str]:
    """sha256 de cada archivo bajo `raiz/<dir>` (clave = ruta relativa con '/').

    Se ignoran `__pycache__`. Un directorio inexistente no aporta claves.
    """
    raiz = Path(raiz)
    out: Dict[str, str] = {}
    for d in dirs:
        base = raiz / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for bloque in iter(lambda: f.read(1 << 20), b""):
                    h.update(bloque)
            out[p.relative_to(raiz).as_posix()] = h.hexdigest()
    return out


def diferencias_hash(antes: Dict[str, str], despues: Dict[str, str]) -> Dict[str, List[str]]:
    """Archivos nuevos, borrados y modificados entre dos `hash_arbol`."""
    return {
        "nuevos": sorted(set(despues) - set(antes)),
        "borrados": sorted(set(antes) - set(despues)),
        "modificados": sorted(k for k in set(antes) & set(despues) if antes[k] != despues[k]),
    }


# ---------------------------------------------------------------------------
# Piezas del sandbox (también usadas por los tests)
# ---------------------------------------------------------------------------

def festivos_offline(start_date: str, end_date: str, ucp: str = "") -> List[str]:
    """Festivos oficiales de Colombia (librería `holidays`) en [start_date, end_date]."""
    ini = pd.Timestamp(start_date).date()
    fin = pd.Timestamp(end_date).date()
    if fin < ini:
        return []
    return [f for f in data.festivos_colombia(range(ini.year, fin.year + 1)) if ini.isoformat() <= f <= fin.isoformat()]


def recortar_datos(origen: Path, destino: Path, corte: date) -> Tuple[int, Optional[date]]:
    """Copia `origen` (datos.csv) a `destino` dejando solo filas con FECHA <= corte.

    Se filtra sobre el texto ISO de la columna FECHA (sin reformatear el resto del
    archivo). Devuelve (filas escritas, última fecha).
    """
    df = pd.read_csv(origen, dtype=str, keep_default_na=False, encoding="utf-8")
    fechas = pd.to_datetime(df["FECHA"], errors="coerce")
    mascara = fechas.notna() & (fechas <= pd.Timestamp(corte))
    df[mascara].to_csv(destino, index=False, encoding="utf-8")
    ultima = fechas[mascara].max()
    return int(mascara.sum()), (ultima.date() if pd.notna(ultima) else None)


def _es_host_local(direccion) -> bool:
    if isinstance(direccion, tuple) and direccion:
        host = str(direccion[0])
    else:  # AF_UNIX u otros
        return True
    return host in HOSTS_LOCALES or host.startswith("127.")


def _modulos_proyecto():
    for nombre, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if nombre == "src" or nombre.startswith(PREFIJOS_MODULOS) or nombre in ("config", "pipeline", "prediction"):
            yield nombre, mod


def _cargar_modulos_flujo():
    """Importa todo lo que toca el flujo de pronóstico (antes de redirigir rutas)."""
    import src.api.main as api_main  # noqa: F401  (importa pipeline, trainer, forecaster, hourly, router)
    for nombre in (
        "src.pipeline.monitoring", "src.prediction.festivos_api", "src.prediction.hourly.calendar_utils",
        "src.prediction.hourly.disaggregation_engine", "src.prediction.hourly.hourly_disaggregator",
        "src.prediction.hourly.special_days", "src.pipeline.update_csv",
    ):
        importlib.import_module(nombre)
    return api_main


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

class SandboxModelo:
    """Contexto aislado para entrenar/predecir con el flujo real del modelo.

    Args:
        ucps: UCP a preparar (se copian sus `data/raw/{ucp}/` del repo).
        raiz_repo: raíz del repo de donde se copian datos y `config/` (solo lectura).
        directorio: carpeta del sandbox; si es None se crea un `tempfile.mkdtemp`
            (y se borra al salir salvo `conservar=True`).
        usar_flujo_api: si True, cada entrenamiento/predicción llama
            `run_predict_flow` (fidelidad total con la API, incluida su predicción
            "de prueba" de 30 días que solo calcula `should_retrain`); si False
            (default) se llaman las mismas funciones del flujo con los mismos
            argumentos pero sin esa predicción de prueba (mismas curvas, más rápido).
        silenciar: baja el nivel de logging/prints del flujo durante las llamadas.
    """

    def __init__(
        self,
        ucps: Iterable[str] = UCPS_DEFAULT,
        raiz_repo: Path | str = ROOT,
        directorio: Path | str | None = None,
        conservar: bool = False,
        usar_flujo_api: bool = False,
        silenciar: bool = True,
    ):
        self.ucps = list(ucps)
        self.raiz_repo = Path(raiz_repo).resolve()
        self._dir_pedido = Path(directorio) if directorio else None
        self.conservar = conservar
        self.usar_flujo_api = usar_flujo_api
        self.silenciar = silenciar
        self.dir: Optional[Path] = None
        self._pila: Optional[ExitStack] = None
        self._api = None
        self.modelo_corte: Dict[str, date] = {}  # ucp -> corte de los modelos entrenados en el sandbox
        self.tiempos: List[Dict] = []  # registro de tiempos (operación, ucp, corte, segundos)

    # -- ciclo de vida -------------------------------------------------------
    def __enter__(self) -> "SandboxModelo":
        api_main = _cargar_modulos_flujo()
        self._api = api_main
        if self._dir_pedido is not None:
            self.dir = self._dir_pedido.resolve()
            self.dir.mkdir(parents=True, exist_ok=True)
        else:
            self.dir = Path(tempfile.mkdtemp(prefix="epm_sandbox_modelo_")).resolve()
        self._preparar_arbol()

        pila = ExitStack()
        self._pila = pila
        try:
            pila.callback(self._limpiar_dir)
            pila.enter_context(_cambiar_dir(self.dir))
            self._redirigir_rutas(pila)
            self._parchear_red(pila)
            self._parchear_festivos(pila)
            self._parchear_actualizacion(pila)
            self._aislar_file_handlers(pila)
        except BaseException:
            pila.close()
            raise
        return self

    def __exit__(self, *exc) -> None:
        if self._pila is not None:
            self._pila.close()
            self._pila = None

    def _limpiar_dir(self) -> None:
        if self.dir is not None and not self.conservar and self._dir_pedido is None:
            shutil.rmtree(self.dir, ignore_errors=True)

    def _preparar_arbol(self) -> None:
        d = self.dir
        for sub in ("data/raw", "data/features", "data/processed", "data/predictions", "models", "logs", "reports"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        cfg = self.raiz_repo / "config"
        if cfg.exists():
            shutil.copytree(cfg, d / "config", dirs_exist_ok=True)
        else:
            (d / "config").mkdir(exist_ok=True)
        for ucp in self.ucps:
            (d / "data" / "raw" / ucp).mkdir(parents=True, exist_ok=True)
            (d / "data" / "features" / ucp).mkdir(parents=True, exist_ok=True)
            (d / "models" / ucp).mkdir(parents=True, exist_ok=True)
            for nombre in ARCHIVOS_RAW:
                src_f = self.raiz_repo / "data" / "raw" / ucp / nombre
                if not src_f.exists():
                    raise FileNotFoundError(f"No existe {src_f}")
                # copia íntegra de referencia (solo lectura); `datos.csv` se recorta en cada llamada
                shutil.copy2(src_f, d / "data" / "raw" / ucp / f"{nombre}.completo")
                shutil.copy2(src_f, d / "data" / "raw" / ucp / nombre)

    # -- parches -------------------------------------------------------------
    def _mapear(self, valor: Path) -> Optional[Path]:
        """Ruta absoluta bajo la raíz del código (o de `raiz_repo`) → misma ruta relativa en el sandbox."""
        for raiz in (ROOT, self.raiz_repo):
            try:
                rel = Path(valor).resolve().relative_to(raiz)
            except (ValueError, OSError):
                continue
            return self.dir / rel
        return None

    def _redirigir_rutas(self, pila: ExitStack) -> None:
        """Toda constante de ruta del proyecto que apunte al repo → mismo sitio en el sandbox."""
        cambios: List[Tuple[object, str, object]] = []
        for _nombre, mod in _modulos_proyecto():
            for attr in NOMBRES_RUTA:
                if not hasattr(mod, attr):
                    continue
                valor = getattr(mod, attr)
                if not isinstance(valor, (str, Path)):
                    continue
                if not Path(valor).is_absolute():
                    continue  # relativas ya caen en el sandbox por el chdir
                nuevo = self._mapear(Path(valor))
                if nuevo is None:
                    continue
                nuevo_val = str(nuevo) if isinstance(valor, str) else nuevo
                cambios.append((mod, attr, valor))
                setattr(mod, attr, nuevo_val)

        def restaurar():
            for mod, attr, valor in reversed(cambios):
                setattr(mod, attr, valor)

        pila.callback(restaurar)
        self.rutas_redirigidas = [(getattr(m, "__name__", "?"), a) for m, a, _ in cambios]

    def _parchear_red(self, pila: ExitStack) -> None:
        import requests

        def bloqueado(*args, **kwargs):
            raise RedBloqueadaError(f"Acceso a red bloqueado en el sandbox: {args[:2]!r}")

        originales = []
        for obj, attr in (
            (requests, "get"), (requests, "post"), (requests, "put"), (requests, "delete"),
            (requests, "request"), (requests.Session, "request"),
        ):
            originales.append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, bloqueado)

        conectar_orig = socket.socket.connect
        conectar_ex_orig = socket.socket.connect_ex
        crear_orig = socket.create_connection

        def connect(sock, direccion):
            if not _es_host_local(direccion):
                raise RedBloqueadaError(f"Conexión de red bloqueada en el sandbox: {direccion!r}")
            return conectar_orig(sock, direccion)

        def connect_ex(sock, direccion):
            if not _es_host_local(direccion):
                raise RedBloqueadaError(f"Conexión de red bloqueada en el sandbox: {direccion!r}")
            return conectar_ex_orig(sock, direccion)

        def create_connection(direccion, *a, **k):
            if not _es_host_local(direccion):
                raise RedBloqueadaError(f"Conexión de red bloqueada en el sandbox: {direccion!r}")
            return crear_orig(direccion, *a, **k)

        socket.socket.connect = connect
        socket.socket.connect_ex = connect_ex
        socket.create_connection = create_connection

        def restaurar():
            for obj, attr, val in reversed(originales):
                setattr(obj, attr, val)
            socket.socket.connect = conectar_orig
            socket.socket.connect_ex = conectar_ex_orig
            socket.create_connection = crear_orig

        pila.callback(restaurar)

    def _parchear_festivos(self, pila: ExitStack) -> None:
        clases = []
        for nombre, mod in _modulos_proyecto():
            cls = getattr(mod, "FestivosAPIClient", None)
            if isinstance(cls, type) and cls not in [c for c, _ in clases]:
                clases.append((cls, cls.__dict__.get("get_festivos")))

        def get_festivos(self_cli, start_date, end_date, ucp):
            return festivos_offline(start_date, end_date, ucp)

        for cls, _orig in clases:
            cls.get_festivos = get_festivos

        def restaurar():
            for cls, orig in clases:
                if orig is None:
                    try:
                        delattr(cls, "get_festivos")
                    except AttributeError:
                        pass
                else:
                    cls.get_festivos = orig

        pila.callback(restaurar)

    def _parchear_actualizacion(self, pila: ExitStack) -> None:
        cambios = []
        for nombre, mod in _modulos_proyecto():
            for attr in ("full_update_csv", "full_update_csv_diario"):
                if callable(getattr(mod, attr, None)):
                    cambios.append((mod, attr, getattr(mod, attr)))
                    setattr(mod, attr, _no_op_actualizacion)

        def restaurar():
            for mod, attr, val in reversed(cambios):
                setattr(mod, attr, val)

        pila.callback(restaurar)

    def _aislar_file_handlers(self, pila: ExitStack) -> None:
        """Quita temporalmente los FileHandler que escriben fuera del sandbox (p. ej. logs/ del repo)."""
        quitados = []
        loggers = [logging.getLogger()] + [
            lg for lg in logging.Logger.manager.loggerDict.values() if isinstance(lg, logging.Logger)
        ]
        for lg in loggers:
            for h in list(lg.handlers):
                if isinstance(h, logging.FileHandler):
                    if _dentro_de(Path(h.baseFilename), self.dir):
                        continue
                    lg.removeHandler(h)
                    quitados.append((lg, h))

        def restaurar():
            # cerrar los FileHandler creados dentro del sandbox (en Windows impiden borrarlo)
            todos = [logging.getLogger()] + [
                lg for lg in logging.Logger.manager.loggerDict.values() if isinstance(lg, logging.Logger)
            ]
            for lg in todos:
                for h in list(lg.handlers):
                    if isinstance(h, logging.FileHandler) and _dentro_de(Path(h.baseFilename), self.dir):
                        lg.removeHandler(h)
                        h.close()
            for lg, h in quitados:
                lg.addHandler(h)

        pila.callback(restaurar)

    # -- datos ---------------------------------------------------------------
    def preparar_corte(self, ucp: str, corte: date) -> date:
        """Reescribe el `datos.csv` del sandbox con FECHA <= corte. Devuelve la última fecha real."""
        base = self.dir / "data" / "raw" / ucp
        filas, ultima = recortar_datos(base / "datos.csv.completo", base / "datos.csv", corte)
        if filas == 0 or ultima is None:
            raise ValueError(f"{ucp}: no hay datos <= {corte}")
        return ultima

    # -- flujo ---------------------------------------------------------------
    def _request(self, ucp: str, corte: date, n_days: int, force_retrain: bool):
        return self._api.PredictRequest(
            ucp=ucp, end_date=corte.isoformat(), n_days=int(n_days), force_retrain=force_retrain,
        )

    def _correr(self, ucp: str, corte: date, n_days: int, force_retrain: bool) -> pd.DataFrame:
        """Ejecuta el flujo y devuelve el DataFrame de predicciones (fecha, demanda_predicha, P1..P24)."""
        api = self._api
        ultima = self.preparar_corte(ucp, corte)
        with _silencio(self.silenciar):
            if self.usar_flujo_api:
                resp = asyncio.run(api.run_predict_flow(self._request(ucp, corte, n_days, force_retrain)))
                # la API redondea a 2 decimales
                preds = pd.DataFrame(
                    [p.model_dump() if hasattr(p, "model_dump") else dict(p) for p in resp.predictions]
                )
                preds["fecha"] = pd.to_datetime(preds["fecha"])
            else:
                preds = self._flujo_directo(ucp, corte, n_days, force_retrain)
        preds.attrs["ultima_fecha_historico"] = ultima
        return preds

    def _flujo_directo(self, ucp: str, corte: date, n_days: int, force_retrain: bool) -> pd.DataFrame:
        """Pasos 1–4 de `run_predict_flow` con los mismos argumentos, sin la predicción de prueba de 30 días."""
        api = self._api
        df_feat, _ = api.run_automated_pipeline(
            power_data_path=f"data/raw/{ucp}/datos.csv",
            weather_data_path=f"data/raw/{ucp}/clima_new.csv",
            start_date="2015-01-01",
            end_date=corte.isoformat(),
            output_dir=Path(f"data/features/{ucp}"),
            ucp=ucp,
        )
        model_path, _ = api.train_model_if_needed(df_with_features=df_feat, ucp=ucp, force_retrain=force_retrain)
        api.train_hourly_disaggregation_if_needed(df_feat, ucp, force_retrain=force_retrain)
        temp = f"data/features/{ucp}/temp_api_features.csv"
        df_feat.to_csv(temp, index=False)
        fp = api.ForecastPipeline(
            model_path=str(model_path),
            historical_data_path=temp,
            festivos_path="config/festivos.json",
            enable_hourly_disaggregation=True,
            raw_climate_path=f"data/raw/{ucp}/clima_new.csv",
            ucp=ucp,
        )
        preds = fp.predict_next_n_days(n_days=int(n_days))
        if os.path.exists(temp):
            os.remove(temp)
        preds = preds.copy()
        preds["fecha"] = pd.to_datetime(preds["fecha"])
        return preds

    def entrenar(self, ucp: str, corte: date, n_days: int = 8) -> Curves:
        """Entrena modelo diario + desagregador horario con datos <= corte (force_retrain=True).

        Devuelve las curvas de los `n_days` días > corte que produce esa misma llamada.
        """
        t0 = time.perf_counter()
        preds = self._correr(ucp, corte, n_days, force_retrain=True)
        seg = time.perf_counter() - t0
        self.modelo_corte[ucp] = corte
        self.tiempos.append({"operacion": "entrenar", "ucp": ucp, "corte": corte.isoformat(), "segundos": seg})
        return self._a_curvas(preds, corte, n_days)

    def predecir(self, ucp: str, corte: date, n_days: int = 8) -> Curves:
        """Predice `n_days` días > corte con los modelos ya entrenados en el sandbox (force_retrain=False)."""
        mc = self.modelo_corte.get(ucp)
        if mc is None:
            raise RuntimeError(f"{ucp}: no hay modelos entrenados en el sandbox (llame entrenar primero)")
        if mc > corte:
            raise ValueError(f"{ucp}: modelos entrenados con corte {mc} posterior al corte pedido {corte} (fuga)")
        t0 = time.perf_counter()
        preds = self._correr(ucp, corte, n_days, force_retrain=False)
        seg = time.perf_counter() - t0
        self.tiempos.append({"operacion": "predecir", "ucp": ucp, "corte": corte.isoformat(), "segundos": seg})
        return self._a_curvas(preds, corte, n_days)

    def proveedor(self, ucp: str, modelo_corte: Optional[date] = None) -> pipeline.BaseProvider:
        """`provider(ucp, end_date, n_days) -> Curves` para `scripts/backtest_adjustments.py`.

        Si `modelo_corte` no es None y difiere del corte de los modelos actuales del
        sandbox, se reentrena con ese corte antes de predecir.
        """
        ucp_fijo = ucp

        def provider(ucp: str, end_date: date, n_days: int) -> Curves:
            if ucp != ucp_fijo:
                raise ValueError(f"proveedor creado para {ucp_fijo}, llamado con {ucp}")
            if modelo_corte is not None and self.modelo_corte.get(ucp) != modelo_corte:
                self.entrenar(ucp, modelo_corte, n_days)
            return self.predecir(ucp, end_date, n_days)

        return provider

    @staticmethod
    def _a_curvas(preds: pd.DataFrame, corte: date, n_days: int) -> Curves:
        """DataFrame de predicciones → Curves; verifica que empiece en corte+1 y sea consecutivo."""
        periodos = [f"P{i}" for i in range(1, 25)]
        faltan = [p for p in periodos if p not in preds.columns]
        if faltan:
            raise ValueError(f"predicción sin columnas horarias: {faltan[:3]}...")
        p = preds.sort_values("fecha")
        fechas = [pd.Timestamp(f).date() for f in p["fecha"]]
        esperadas = [corte + timedelta(days=i) for i in range(1, int(n_days) + 1)]
        if fechas != esperadas:
            raise ValueError(
                f"fechas predichas {fechas[:1]}..{fechas[-1:]} != esperadas {esperadas[0]}..{esperadas[-1]} "
                f"(última fecha del histórico: {preds.attrs.get('ultima_fecha_historico')})"
            )
        out: Curves = {}
        for f, (_, fila) in zip(fechas, p.iterrows()):
            v = fila[periodos].to_numpy(dtype=float)
            if not np.isfinite(v).all():
                raise ValueError(f"curva no finita para {f}")
            out[f.isoformat()] = v.tolist()
        return out


def _dentro_de(ruta: Path, raiz: Path) -> bool:
    try:
        ruta.resolve().relative_to(raiz.resolve())
        return True
    except (ValueError, OSError):
        return False


def _no_op_actualizacion(*args, **kwargs):
    """Sustituto de `full_update_csv`: no red, no escribe data/raw."""
    return None


class _cambiar_dir:
    def __init__(self, destino: Path):
        self.destino = destino
        self.previo: Optional[str] = None

    def __enter__(self):
        self.previo = os.getcwd()
        os.chdir(self.destino)
        return self

    def __exit__(self, *exc):
        os.chdir(self.previo)


class _silencio:
    """Silencia stdout y logging (INFO) del flujo, que es muy verboso."""

    def __init__(self, activo: bool):
        self.activo = activo

    def __enter__(self):
        if not self.activo:
            return self
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
        self._nivel = logging.root.manager.disable
        logging.disable(logging.WARNING)
        return self

    def __exit__(self, *exc):
        if not self.activo:
            return
        sys.stdout.close()
        sys.stdout = self._stdout
        logging.disable(self._nivel)


# ---------------------------------------------------------------------------
# CLI: tiempos + hashes
# ---------------------------------------------------------------------------

def _corte_default(ucp: str) -> date:
    """Sábado ~8 semanas antes del último dato real del UCP (deja semanas reales para comparar)."""
    reales = data.curvas_reales(data.cargar_historico(ucp, data_dir=ROOT / "data" / "raw"))
    ultima = max(date.fromisoformat(f) for f in reales)
    d = ultima - timedelta(weeks=8)
    return d - timedelta(days=(d.weekday() - 5) % 7)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ucp", nargs="+", default=list(UCPS_DEFAULT))
    ap.add_argument("--corte", default=None, help="YYYY-MM-DD (default: sábado ~8 semanas antes del último dato)")
    ap.add_argument("--n-days", type=int, default=8)
    ap.add_argument("--flujo-api", action="store_true", help="usar run_predict_flow completo")
    ap.add_argument("--json", default=None, help="ruta donde guardar tiempos/hashes en JSON")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    antes = hash_arbol(ROOT)
    resultado: Dict = {"ucps": {}, "hash_archivos": len(antes)}
    with SandboxModelo(args.ucp, usar_flujo_api=args.flujo_api) as sb:
        for ucp in args.ucp:
            corte = date.fromisoformat(args.corte) if args.corte else _corte_default(ucp)
            corte_pred = corte + timedelta(days=7)
            c1 = sb.entrenar(ucp, corte, args.n_days)
            c2 = sb.predecir(ucp, corte_pred, args.n_days)
            t = [r for r in sb.tiempos if r["ucp"] == ucp]
            # error rápido contra el real (sanidad)
            reales = data.curvas_reales(data.cargar_historico(ucp, data_dir=ROOT / "data" / "raw"))
            def wape(curvas):
                num = den = 0.0
                for f, v in curvas.items():
                    if f in reales:
                        r = np.asarray(reales[f], float)
                        num += float(np.abs(np.asarray(v) - r).sum())
                        den += float(r.sum())
                return round(100 * num / den, 2) if den else None
            resultado["ucps"][ucp] = {
                "corte_entreno": corte.isoformat(),
                "corte_prediccion": corte_pred.isoformat(),
                "fechas_entreno": [min(c1), max(c1)],
                "fechas_prediccion": [min(c2), max(c2)],
                "wape_entreno_pct": wape(c1),
                "wape_prediccion_pct": wape(c2),
                "tiempos": t,
            }
            logger.info("%s: %s", ucp, json.dumps(resultado["ucps"][ucp], default=str))
        resultado["rutas_redirigidas"] = sb.rutas_redirigidas
    despues = hash_arbol(ROOT)
    resultado["diferencias_hash"] = diferencias_hash(antes, despues)
    resultado["hash_intacto"] = antes == despues
    print(json.dumps(resultado, indent=2, default=str, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(resultado, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return 0 if resultado["hash_intacto"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
