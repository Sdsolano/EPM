"""Persistencia de corridas de ajustes.

- `Store`: interfaz (Protocol) que usa el router.
- `InMemoryStore`: para tests y desarrollo sin base.
- `PostgresStore`: psycopg2 sobre el esquema `ajustes` (sql/001_init.sql + sql/002_versionado.sql),
  conexión por `AJUSTES_DATABASE_URL`. Una conexión por operación (mismo patrón que
  fastapi_factores/app/db.py); cada operación es una transacción.
- `get_store()`: devuelve un PostgresStore si hay `AJUSTES_DATABASE_URL`, si no `None`.

Versionado y concurrencia (optimista):
- `corrida.version` empieza en 1 y sube en cada `replace_records` y `approve`. Cada versión guarda un
  snapshot completo de los records en `corrida_version` (historial reconstruible).
- `replace_records` y `approve` reciben `version_esperada` y la verifican con la fila de la corrida
  bloqueada (`FOR UPDATE`); si no coincide lanzan `VersionConflictError` (el router responde 409).
- `approve` re-evalúa los bloqueos (callback `check`) sobre los records leídos dentro del lock.
- Una sola corrida `aprobada` vigente por (ucp, semana_inicio): al aprobar, la anterior pasa a
  `reemplazada` (`reemplazada_por` = id nueva). En Postgres además hay índice único parcial.

Forma de `get_run`: {"meta": {...}, "records": [StepRecord, ...], "eventos": [{...}, ...]}.
meta: id, ucp, semana_inicio, fecha_corte, fecha_emision (ISO "YYYY-MM-DD"), modelo_version,
params_iniciales (params del request original; en Postgres columna `corrida.params`), extras, estado ("borrador" | "aprobada" | "reemplazada"), version, creado_por, created_at,
approved_at, approved_by, reemplazada_por, reemplazada_at.

Evento: {"usuario", "accion", "paso" (nombre o None), "motivo", "detalle" (dict), "version"}.
- `create_run(meta, records, evento=None)` inserta el evento (p. ej. `crear`) en la misma transacción.
- `reemplazar` lleva la versión (sin cambio) de la corrida reemplazada, igual en InMemory y Postgres.
- `approve(..., check, detalle)`: `check` corre bajo lock antes de insertar el evento `aprobar`; puede
  completar `detalle` (mismo dict) o lanzar una excepción propia (se propaga y no se aprueba).
"""

from __future__ import annotations

import copy
import json
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, runtime_checkable

from src.adjustments.types import Alert, StepRecord, parse_date, to_jsonable

ENV_DSN = "AJUSTES_DATABASE_URL"
ESTADO_BORRADOR = "borrador"
ESTADO_APROBADA = "aprobada"
ESTADO_REEMPLAZADA = "reemplazada"

_META_FIELDS = (
    "ucp", "semana_inicio", "fecha_corte", "fecha_emision",
    "modelo_version", "params_iniciales", "extras", "creado_por",
)
_META_UPDATABLE = ("modelo_version",)

BlockerCheck = Callable[[List[StepRecord]], List[Alert]]


class StoreError(Exception):
    """Error base del store de ajustes."""


class RunNotFoundError(StoreError):
    """La corrida no existe."""


class RunApprovedError(StoreError):
    """La corrida ya está aprobada (o reemplazada) y no se puede modificar."""


class VersionConflictError(StoreError):
    """`version_esperada` no coincide con la versión actual de la corrida."""

    def __init__(self, run_id: str, esperada: int, actual: int) -> None:
        super().__init__(f"Corrida {run_id}: versión esperada {esperada}, actual {actual}")
        self.esperada = esperada
        self.actual = actual


class ApprovalBlockedError(StoreError):
    """La corrida tiene bloqueos de aprobación (re-evaluados dentro del lock)."""

    def __init__(self, bloqueos: List[Alert]) -> None:
        super().__init__(f"{len(bloqueos)} bloqueo(s) de aprobación")
        self.bloqueos = bloqueos


@runtime_checkable
class Store(Protocol):
    def create_run(self, meta: Dict[str, Any], records: List[StepRecord],
                   evento: Optional[Dict[str, Any]] = None) -> str: ...
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]: ...
    def replace_records(self, run_id: str, records: List[StepRecord], evento: Dict[str, Any],
                        version_esperada: int,
                        meta_updates: Optional[Dict[str, Any]] = None) -> int: ...
    def approve(self, run_id: str, usuario: str, motivo: Optional[str], version_esperada: int,
                check: Optional[BlockerCheck] = None,
                detalle: Optional[Dict[str, Any]] = None) -> int: ...
    def add_event(self, run_id: str, evento: Dict[str, Any]) -> None: ...
    def get_approved(self, ucp: str, semana_inicio: Any) -> Optional[Dict[str, Any]]: ...
    def list_versions(self, run_id: str) -> Optional[List[Dict[str, Any]]]: ...
    def get_version(self, run_id: str, version: int) -> Optional[Dict[str, Any]]: ...
    def list_runs(self, ucp: Optional[str] = None, desde: Optional[date] = None,
                  hasta: Optional[date] = None, limite: int = 200) -> List[Dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _iso(v: Any) -> Any:
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return v


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: _iso(meta.get(k)) for k in _META_FIELDS}
    out["params_iniciales"] = to_jsonable(copy.deepcopy(out.get("params_iniciales") or {}))
    out["extras"] = to_jsonable(copy.deepcopy(out.get("extras") or {}))
    for k in ("ucp", "semana_inicio", "fecha_corte", "fecha_emision"):
        if not out.get(k):
            raise ValueError(f"meta.{k} es obligatorio")
    return out


def _normalize_meta_updates(meta_updates: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    updates = dict(meta_updates or {})
    invalidos = sorted(set(updates) - set(_META_UPDATABLE))
    if invalidos:
        raise ValueError(f"meta no actualizable: {invalidos}")
    return updates


def _normalize_event(evento: Dict[str, Any]) -> Dict[str, Any]:
    accion = evento.get("accion")
    if not accion:
        raise ValueError("evento.accion es obligatorio")
    paso = evento.get("paso")
    return {
        "usuario": evento.get("usuario"),
        "accion": str(accion),
        "paso": str(paso) if paso is not None else None,
        "motivo": evento.get("motivo"),
        "detalle": to_jsonable(copy.deepcopy(evento.get("detalle") or {})),
    }


def _validate_run_id(run_id: str) -> Optional[str]:
    try:
        return str(uuid.UUID(str(run_id)))
    except (ValueError, TypeError, AttributeError):
        return None


def _fecha_o_none(value: Any) -> Optional[date]:
    if value is None:
        return None
    try:
        return parse_date(value)
    except (TypeError, ValueError):
        return None


MAX_LIMITE_RUNS = 500  # tope del endpoint /aprendizaje; el store no acepta límites mayores


def _limite(limite: Any) -> int:
    try:
        n = int(limite)
    except (TypeError, ValueError):
        return 200
    return min(MAX_LIMITE_RUNS, max(1, n))


def _records_json(records: List[StepRecord]) -> List[Dict[str, Any]]:
    """Records como JSON estricto (NaN/inf → None; Postgres no acepta NaN en jsonb)."""
    return [to_jsonable(r.to_dict()) for r in records]


def record_to_columns(rec: StepRecord) -> Dict[str, Any]:
    """StepRecord -> columnas de ajustes.corrida_paso (valores JSON estrictos)."""
    d = to_jsonable(rec.to_dict())
    result = d.get("result") or {}
    return {
        "orden": int(d["orden"]),
        "paso": d["paso"],
        "estado": d["estado"],
        "params": d.get("params") or {},
        "curvas": result.get("curves") or {},
        "curvas_calculadas": result.get("curvas_calculadas") or {},
        "diagnostico": result.get("diagnostics") or {},
        "alertas": result.get("alerts") or [],
    }


def columns_to_record(row: Dict[str, Any]) -> StepRecord:
    return StepRecord.from_dict({
        "orden": int(row["orden"]),
        "paso": row["paso"],
        "estado": row["estado"],
        "params": row.get("params") or {},
        "result": {
            "curves": row.get("curvas") or {},
            "curvas_calculadas": row.get("curvas_calculadas") or {},
            "diagnostics": row.get("diagnostico") or {},
            "alerts": row.get("alertas") or [],
        },
    })


def _clone_records(records: List[StepRecord]) -> List[StepRecord]:
    # ida y vuelta por JSON estricto: emula la persistencia y desacopla de objetos del llamador
    return [StepRecord.from_dict(json.loads(json.dumps(d, allow_nan=False))) for d in _records_json(records)]


def _records_solo_lectura(records: List[StepRecord]) -> List[StepRecord]:
    """Copia superficial para `list_runs`: el consumidor (aprendizaje) no muta los records.

    Evita el clon por JSON (caro para cientos de corridas). Los objetos anidados
    (`result.curves`, `params`, ...) se comparten con el store: quien llame a
    `list_runs` debe tratarlos como inmutables.
    """
    return [copy.copy(r) for r in records]


# ---------------------------------------------------------------------------
# InMemoryStore
# ---------------------------------------------------------------------------

class InMemoryStore:
    def __init__(self) -> None:
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def create_run(self, meta: Dict[str, Any], records: List[StepRecord],
                   evento: Optional[Dict[str, Any]] = None) -> str:
        m = _normalize_meta(meta)
        ev = _normalize_event(evento) if evento is not None else None  # validar antes de mutar
        run_id = str(uuid.uuid4())
        m.update({
            "id": run_id, "estado": ESTADO_BORRADOR, "version": 1, "created_at": _now().isoformat(),
            "approved_at": None, "approved_by": None, "reemplazada_por": None, "reemplazada_at": None,
        })
        with self._lock:
            self._runs[run_id] = {"meta": m, "records": _clone_records(records), "eventos": [],
                                  "versiones": []}
            self._snapshot(self._runs[run_id], m["creado_por"], "crear")
            if ev is not None:
                self._append_event(self._runs[run_id], ev)
        return run_id

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            run = self._runs.get(str(run_id))
            if run is None:
                return None
            return self._public(run)

    @staticmethod
    def _public(run: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "meta": copy.deepcopy(run["meta"]),
            "records": _clone_records(run["records"]),
            "eventos": copy.deepcopy(run["eventos"]),
        }

    def _get_mutable(self, run_id: str) -> Dict[str, Any]:
        run = self._runs.get(str(run_id))
        if run is None:
            raise RunNotFoundError(run_id)
        return run

    @staticmethod
    def _check_writable(run: Dict[str, Any], run_id: str, version_esperada: int) -> None:
        if run["meta"]["estado"] != ESTADO_BORRADOR:
            raise RunApprovedError(run_id)
        if int(version_esperada) != run["meta"]["version"]:
            raise VersionConflictError(run_id, int(version_esperada), run["meta"]["version"])

    @staticmethod
    def _snapshot(run: Dict[str, Any], creado_por: Optional[str], motivo: Optional[str]) -> None:
        run["versiones"].append({
            "version": run["meta"]["version"], "records": _records_json(run["records"]),
            "creado_por": creado_por, "motivo": motivo, "created_at": _now().isoformat(),
        })

    def _append_event(self, run: Dict[str, Any], evento: Dict[str, Any]) -> None:
        ev = _normalize_event(evento)
        ev["version"] = run["meta"]["version"]
        ev["ts"] = _now().isoformat()
        run["eventos"].append(ev)

    def replace_records(self, run_id: str, records: List[StepRecord], evento: Dict[str, Any],
                        version_esperada: int, meta_updates: Optional[Dict[str, Any]] = None) -> int:
        ev = _normalize_event(evento)  # validar antes de mutar
        updates = _normalize_meta_updates(meta_updates)
        with self._lock:
            run = self._get_mutable(run_id)
            self._check_writable(run, run_id, version_esperada)
            run["records"] = _clone_records(records)
            run["meta"].update(updates)
            run["meta"]["version"] += 1
            self._snapshot(run, ev["usuario"], ev["motivo"] or ev["accion"])
            self._append_event(run, ev)
            return run["meta"]["version"]

    def approve(self, run_id: str, usuario: str, motivo: Optional[str], version_esperada: int,
                check: Optional[BlockerCheck] = None, detalle: Optional[Dict[str, Any]] = None) -> int:
        with self._lock:
            run = self._get_mutable(run_id)
            self._check_writable(run, run_id, version_esperada)
            if check is not None:
                bloqueos = list(check(_clone_records(run["records"])))
                if bloqueos:
                    raise ApprovalBlockedError(bloqueos)
            meta = run["meta"]
            ahora = _now().isoformat()
            for otro_id, otro in self._runs.items():
                om = otro["meta"]
                if (otro_id != run_id and om["estado"] == ESTADO_APROBADA
                        and om["ucp"] == meta["ucp"] and om["semana_inicio"] == meta["semana_inicio"]):
                    om.update({"estado": ESTADO_REEMPLAZADA, "reemplazada_por": run_id, "reemplazada_at": ahora})
                    self._append_event(otro, {"usuario": usuario, "accion": "reemplazar",
                                              "detalle": {"reemplazada_por": run_id}})
            meta.update({"estado": ESTADO_APROBADA, "approved_at": ahora, "approved_by": usuario})
            meta["version"] += 1
            self._snapshot(run, usuario, motivo or "aprobar")
            self._append_event(run, {"usuario": usuario, "accion": "aprobar", "motivo": motivo,
                                     "detalle": detalle if detalle is not None else {}})
            return meta["version"]

    def add_event(self, run_id: str, evento: Dict[str, Any]) -> None:
        with self._lock:
            self._append_event(self._get_mutable(run_id), evento)

    def get_approved(self, ucp: str, semana_inicio: Any) -> Optional[Dict[str, Any]]:
        semana = _iso(semana_inicio)
        with self._lock:
            for run in self._runs.values():
                m = run["meta"]
                if m["estado"] == ESTADO_APROBADA and m["ucp"] == ucp and m["semana_inicio"] == semana:
                    return self._public(run)
        return None

    def list_versions(self, run_id: str) -> Optional[List[Dict[str, Any]]]:
        with self._lock:
            run = self._runs.get(str(run_id))
            if run is None:
                return None
            return [{k: v for k, v in ver.items() if k != "records"} for ver in run["versiones"]]

    def get_version(self, run_id: str, version: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            run = self._runs.get(str(run_id))
            if run is None:
                return None
            for ver in run["versiones"]:
                if ver["version"] == int(version):
                    out = copy.deepcopy(ver)
                    out["records"] = [StepRecord.from_dict(d) for d in out["records"]]
                    return out
        return None

    def list_runs(self, ucp: Optional[str] = None, desde: Optional[date] = None,
                  hasta: Optional[date] = None, limite: int = 200) -> List[Dict[str, Any]]:
        """Corridas filtradas por ucp/semana, más reciente primero (solo lectura).

        `limite` se acota a 1..500. Los records van como copia superficial
        (`_records_solo_lectura`): tratarlos como inmutables.
        """
        d_ini, d_fin = _fecha_o_none(desde), _fecha_o_none(hasta)
        with self._lock:
            seleccionados = []
            for run in self._runs.values():
                m = run["meta"]
                if ucp is not None and m.get("ucp") != ucp:
                    continue
                semana = _fecha_o_none(m.get("semana_inicio"))
                if d_ini is not None and (semana is None or semana < d_ini):
                    continue
                if d_fin is not None and (semana is None or semana > d_fin):
                    continue
                seleccionados.append(run)
            seleccionados.sort(
                key=lambda r: (str(r["meta"].get("semana_inicio") or ""),
                               str(r["meta"].get("created_at") or ""),
                               str(r["meta"].get("id") or "")),
                reverse=True,
            )
            return [{"meta": copy.deepcopy(r["meta"]),
                     "records": _records_solo_lectura(r["records"])}
                    for r in seleccionados[:_limite(limite)]]


# ---------------------------------------------------------------------------
# PostgresStore
# ---------------------------------------------------------------------------

_RUN_COLUMNS = """
    id, ucp, semana_inicio, fecha_corte, fecha_emision, modelo_version, params AS params_iniciales, extras,
    estado, version, creado_por, created_at, approved_at, approved_by, reemplazada_por, reemplazada_at
"""


class PostgresStore:
    def __init__(self, dsn: Optional[str] = None) -> None:
        self.dsn = dsn or os.getenv(ENV_DSN)
        if not self.dsn:
            raise RuntimeError(f"{ENV_DSN} no está definido")

    @contextmanager
    def _tx(self) -> Iterator[Any]:
        import psycopg2

        conn = psycopg2.connect(self.dsn)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _json(v: Any) -> Any:
        from psycopg2.extras import Json

        return Json(v, dumps=lambda o: json.dumps(o, allow_nan=False))

    def _insert_steps(self, cur: Any, run_id: str, records: List[StepRecord]) -> None:
        for rec in records:
            c = record_to_columns(rec)
            cur.execute(
                """
                INSERT INTO ajustes.corrida_paso
                    (corrida_id, orden, paso, estado, params, curvas, curvas_calculadas,
                     diagnostico, alertas, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                """,
                (run_id, c["orden"], c["paso"], c["estado"], self._json(c["params"]),
                 self._json(c["curvas"]), self._json(c["curvas_calculadas"]),
                 self._json(c["diagnostico"]), self._json(c["alertas"])),
            )

    def _insert_version(self, cur: Any, run_id: str, version: int, records: List[StepRecord],
                        creado_por: Optional[str], motivo: Optional[str]) -> None:
        cur.execute(
            """
            INSERT INTO ajustes.corrida_version (corrida_id, version, records, creado_por, motivo)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (run_id, version, self._json(_records_json(records)), creado_por, motivo),
        )

    def _insert_event(self, cur: Any, run_id: str, evento: Dict[str, Any], version: Optional[int]) -> None:
        ev = _normalize_event(evento)
        cur.execute(
            """
            INSERT INTO ajustes.corrida_evento (corrida_id, usuario, accion, paso, motivo, detalle, version)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (run_id, ev["usuario"], ev["accion"], ev["paso"], ev["motivo"], self._json(ev["detalle"]), version),
        )

    def _lock_run(self, cur: Any, run_id: str) -> Dict[str, Any]:
        """Bloquea la fila de la corrida (FOR UPDATE) y devuelve id normalizado, estado, versión, ucp, semana."""
        rid = _validate_run_id(run_id)
        if rid is None:
            raise RunNotFoundError(run_id)
        cur.execute(
            "SELECT estado, version, ucp, semana_inicio FROM ajustes.corrida WHERE id = %s FOR UPDATE",
            (rid,),
        )
        row = cur.fetchone()
        if row is None:
            raise RunNotFoundError(run_id)
        return {"id": rid, "estado": row[0], "version": row[1], "ucp": row[2], "semana_inicio": row[3]}

    @staticmethod
    def _check_writable(locked: Dict[str, Any], run_id: str, version_esperada: int) -> None:
        if locked["estado"] != ESTADO_BORRADOR:
            raise RunApprovedError(run_id)
        if int(version_esperada) != locked["version"]:
            raise VersionConflictError(run_id, int(version_esperada), locked["version"])

    _STEP_COLUMNS = (
        "orden, paso, estado, params, curvas, curvas_calculadas, diagnostico, alertas"
    )

    @staticmethod
    def _read_steps(cur: Any, rid: str) -> List[StepRecord]:
        cur.execute(
            f"""
            SELECT {PostgresStore._STEP_COLUMNS}
            FROM ajustes.corrida_paso WHERE corrida_id = %s ORDER BY orden
            """,
            (rid,),
        )
        cols = [c[0] for c in cur.description]
        return [columns_to_record(dict(zip(cols, r))) for r in cur.fetchall()]

    @classmethod
    def _read_steps_by_ids(cls, cur: Any, run_ids: List[str]) -> Dict[str, List[StepRecord]]:
        """Pasos de varias corridas en UNA query (`ANY`), agrupados en Python."""
        if not run_ids:
            return {}
        cur.execute(
            f"""
            SELECT corrida_id, {cls._STEP_COLUMNS}
            FROM ajustes.corrida_paso WHERE corrida_id = ANY(%s::uuid[]) ORDER BY corrida_id, orden
            """,
            (list(run_ids),),
        )
        cols = [c[0] for c in cur.description]
        por_run: Dict[str, List[StepRecord]] = {rid: [] for rid in run_ids}
        for row in cur.fetchall():
            d = dict(zip(cols, row))
            por_run.setdefault(str(d["corrida_id"]), []).append(columns_to_record(d))
        return por_run

    @staticmethod
    def _row_to_meta(raw: Dict[str, Any]) -> Dict[str, Any]:
        meta = {k: _iso(v) for k, v in raw.items()}
        meta["id"] = str(raw["id"])
        meta["reemplazada_por"] = str(raw["reemplazada_por"]) if raw["reemplazada_por"] else None
        meta["params_iniciales"] = raw["params_iniciales"] or {}
        meta["extras"] = raw["extras"] or {}
        return meta

    def create_run(self, meta: Dict[str, Any], records: List[StepRecord],
                   evento: Optional[Dict[str, Any]] = None) -> str:
        m = _normalize_meta(meta)
        if evento is not None:
            _normalize_event(evento)  # validar antes de abrir la transacción
        run_id = str(uuid.uuid4())
        with self._tx() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ajustes.corrida
                    (id, ucp, semana_inicio, fecha_corte, fecha_emision, modelo_version, params, extras,
                     estado, version, creado_por)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 1, %s)
                """,
                (run_id, m["ucp"], m["semana_inicio"], m["fecha_corte"], m["fecha_emision"],
                 m["modelo_version"], self._json(m["params_iniciales"]), self._json(m["extras"]),
                 ESTADO_BORRADOR, m["creado_por"]),
            )
            self._insert_steps(cur, run_id, records)
            self._insert_version(cur, run_id, 1, records, m["creado_por"], "crear")
            if evento is not None:
                self._insert_event(cur, run_id, evento, 1)
        return run_id

    def _read_run(self, cur: Any, where: str, args: tuple) -> Optional[Dict[str, Any]]:
        cur.execute(f"SELECT {_RUN_COLUMNS} FROM ajustes.corrida WHERE {where}", args)
        row = cur.fetchone()
        if row is None:
            return None
        cols = [c[0] for c in cur.description]
        raw = dict(zip(cols, row))
        meta = self._row_to_meta(raw)
        rid = meta["id"]
        records = self._read_steps(cur, rid)
        cur.execute(
            """
            SELECT usuario, accion, paso, motivo, detalle, version, ts
            FROM ajustes.corrida_evento WHERE corrida_id = %s ORDER BY ts, id
            """,
            (rid,),
        )
        ecols = [c[0] for c in cur.description]
        eventos = [{k: _iso(v) for k, v in zip(ecols, r)} for r in cur.fetchall()]
        return {"meta": meta, "records": records, "eventos": eventos}

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        rid = _validate_run_id(run_id)
        if rid is None:
            return None
        with self._tx() as conn, conn.cursor() as cur:
            return self._read_run(cur, "id = %s", (rid,))

    def get_approved(self, ucp: str, semana_inicio: Any) -> Optional[Dict[str, Any]]:
        with self._tx() as conn, conn.cursor() as cur:
            return self._read_run(cur, "ucp = %s AND semana_inicio = %s AND estado = %s",
                                  (ucp, _iso(semana_inicio), ESTADO_APROBADA))

    def replace_records(self, run_id: str, records: List[StepRecord], evento: Dict[str, Any],
                        version_esperada: int, meta_updates: Optional[Dict[str, Any]] = None) -> int:
        ev = _normalize_event(evento)  # validar antes de abrir la transacción
        updates = _normalize_meta_updates(meta_updates)
        with self._tx() as conn, conn.cursor() as cur:
            locked = self._lock_run(cur, run_id)
            self._check_writable(locked, run_id, version_esperada)
            rid, nueva = locked["id"], locked["version"] + 1
            cur.execute("DELETE FROM ajustes.corrida_paso WHERE corrida_id = %s", (rid,))
            self._insert_steps(cur, rid, records)
            sets, args = ["version = %s"], [nueva]
            for k, v in updates.items():
                sets.append(f"{k} = %s")  # k validado contra _META_UPDATABLE
                args.append(v)
            cur.execute(f"UPDATE ajustes.corrida SET {', '.join(sets)} WHERE id = %s", (*args, rid))
            self._insert_version(cur, rid, nueva, records, ev["usuario"], ev["motivo"] or ev["accion"])
            self._insert_event(cur, rid, ev, nueva)
        return nueva

    def approve(self, run_id: str, usuario: str, motivo: Optional[str], version_esperada: int,
                check: Optional[BlockerCheck] = None, detalle: Optional[Dict[str, Any]] = None) -> int:
        import psycopg2

        try:
            with self._tx() as conn, conn.cursor() as cur:
                locked = self._lock_run(cur, run_id)
                self._check_writable(locked, run_id, version_esperada)
                rid, nueva = locked["id"], locked["version"] + 1
                records = self._read_steps(cur, rid)  # = versión esperada (fila bloqueada)
                if check is not None:
                    bloqueos = list(check(records))
                    if bloqueos:
                        raise ApprovalBlockedError(bloqueos)
                # serializa aprobaciones concurrentes de la misma (ucp, semana)
                cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                            (f"ajustes:{locked['ucp']}:{_iso(locked['semana_inicio'])}",))
                cur.execute(
                    """
                    UPDATE ajustes.corrida
                    SET estado = %s, reemplazada_por = %s, reemplazada_at = now()
                    WHERE ucp = %s AND semana_inicio = %s AND estado = %s AND id <> %s
                    RETURNING id, version
                    """,
                    (ESTADO_REEMPLAZADA, rid, locked["ucp"], locked["semana_inicio"], ESTADO_APROBADA, rid),
                )
                for otro_id, otra_version in cur.fetchall():
                    self._insert_event(cur, str(otro_id), {"usuario": usuario, "accion": "reemplazar",
                                                           "detalle": {"reemplazada_por": rid}}, otra_version)
                cur.execute(
                    """
                    UPDATE ajustes.corrida
                    SET estado = %s, approved_at = now(), approved_by = %s, version = %s
                    WHERE id = %s
                    """,
                    (ESTADO_APROBADA, usuario, nueva, rid),
                )
                self._insert_version(cur, rid, nueva, records, usuario, motivo or "aprobar")
                self._insert_event(cur, rid, {"usuario": usuario, "accion": "aprobar", "motivo": motivo,
                                              "detalle": detalle if detalle is not None else {}}, nueva)
        except psycopg2.errors.UniqueViolation as e:  # respaldo del advisory lock
            raise VersionConflictError(run_id, int(version_esperada), -1) from e
        return nueva

    def add_event(self, run_id: str, evento: Dict[str, Any]) -> None:
        with self._tx() as conn, conn.cursor() as cur:
            locked = self._lock_run(cur, run_id)
            self._insert_event(cur, locked["id"], evento, locked["version"])

    def list_versions(self, run_id: str) -> Optional[List[Dict[str, Any]]]:
        rid = _validate_run_id(run_id)
        if rid is None:
            return None
        with self._tx() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM ajustes.corrida WHERE id = %s", (rid,))
            if cur.fetchone() is None:
                return None
            cur.execute(
                """
                SELECT version, creado_por, motivo, created_at
                FROM ajustes.corrida_version WHERE corrida_id = %s ORDER BY version
                """,
                (rid,),
            )
            cols = [c[0] for c in cur.description]
            return [{k: _iso(v) for k, v in zip(cols, r)} for r in cur.fetchall()]

    def get_version(self, run_id: str, version: int) -> Optional[Dict[str, Any]]:
        rid = _validate_run_id(run_id)
        if rid is None:
            return None
        with self._tx() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT version, records, creado_por, motivo, created_at
                FROM ajustes.corrida_version WHERE corrida_id = %s AND version = %s
                """,
                (rid, int(version)),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "version": row[0], "records": [StepRecord.from_dict(d) for d in row[1] or []],
                "creado_por": row[2], "motivo": row[3], "created_at": _iso(row[4]),
            }

    def list_runs(self, ucp: Optional[str] = None, desde: Optional[date] = None,
                  hasta: Optional[date] = None, limite: int = 200) -> List[Dict[str, Any]]:
        """Corridas filtradas por ucp/semana, más reciente primero (solo lectura).

        Dos queries: (1) corridas con filtros/orden/límite y (2) todos sus pasos
        con `corrida_id = ANY(...)` agrupados en Python. No lee `corrida_evento`.
        """
        d_ini, d_fin = _fecha_o_none(desde), _fecha_o_none(hasta)
        clauses: List[str] = []
        args: List[Any] = []
        if ucp is not None:
            clauses.append("ucp = %s")
            args.append(ucp)
        if d_ini is not None:
            clauses.append("semana_inicio >= %s")
            args.append(d_ini)
        if d_fin is not None:
            clauses.append("semana_inicio <= %s")
            args.append(d_fin)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._tx() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_RUN_COLUMNS} FROM ajustes.corrida{where} "
                "ORDER BY semana_inicio DESC, created_at DESC, id DESC LIMIT %s",
                (*args, _limite(limite)),
            )
            cols = [c[0] for c in cur.description]
            metas = [self._row_to_meta(dict(zip(cols, row))) for row in cur.fetchall()]
            pasos = self._read_steps_by_ids(cur, [m["id"] for m in metas])
            return [{"meta": meta, "records": pasos.get(meta["id"], [])} for meta in metas]


def get_store() -> Optional[Store]:
    """PostgresStore si `AJUSTES_DATABASE_URL` está definido; si no, None (el router responde 503)."""
    dsn = os.getenv(ENV_DSN)
    if not dsn:
        return None
    return PostgresStore(dsn)
