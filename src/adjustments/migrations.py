"""Aplica las migraciones SQL del módulo de ajustes (src/adjustments/sql/*.sql) de forma idempotente.

Pensado para el arranque del servicio (entorno demo): las migraciones usan IF NOT EXISTS, así que se
pueden re-aplicar; un advisory lock evita carreras si arrancan varias instancias a la vez.
Para uso manual sigue existiendo scripts/apply_adjustments_schema.py.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).resolve().parent / "sql"
_LOCK_KEY = 7_204_001  # clave arbitraria del advisory lock


def aplicar_migraciones(dsn: str, sql_dir: Path = SQL_DIR, intentos: int = 15, espera_s: float = 2.0) -> List[str]:
    """Aplica en orden lexicográfico cada .sql (una transacción por archivo). Reintenta la conexión
    (la base puede tardar en estar lista al arrancar). Devuelve los archivos aplicados."""
    import psycopg2

    archivos = sorted(Path(sql_dir).glob("*.sql"))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos .sql en {sql_dir}")
    ultimo: Exception | None = None
    for i in range(max(1, intentos)):
        try:
            conn = psycopg2.connect(dsn)
            break
        except psycopg2.OperationalError as e:
            ultimo = e
            if i + 1 < intentos:
                time.sleep(espera_s)
    else:
        raise RuntimeError(f"No se pudo conectar a la base tras {intentos} intentos: {ultimo}")
    aplicados: List[str] = []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s)", (_LOCK_KEY,))
        conn.commit()
        try:
            for path in archivos:
                with conn:  # commit por archivo, rollback si falla
                    with conn.cursor() as cur:
                        cur.execute(path.read_text(encoding="utf-8"))
                aplicados.append(path.name)
        finally:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
            conn.commit()
    finally:
        conn.close()
    logger.info("[ajustes] migraciones aplicadas: %s", ", ".join(aplicados))
    return aplicados
