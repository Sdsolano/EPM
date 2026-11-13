"""Aplica los archivos .sql de src/adjustments/sql/ en orden lexicográfico.

Uso (desde la raíz del repo):
    AJUSTES_DATABASE_URL=postgresql://user:pass@host:5432/db .venv/bin/python scripts/apply_adjustments_schema.py
    .venv/bin/python scripts/apply_adjustments_schema.py --dsn postgresql://...

Los .sql son idempotentes (IF NOT EXISTS), así que se puede correr varias veces.
Cada archivo se aplica en su propia transacción.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

SQL_DIR = Path(__file__).resolve().parent.parent / "src" / "adjustments" / "sql"


def apply_schema(dsn: str, sql_dir: Path = SQL_DIR) -> list[str]:
    files = sorted(sql_dir.glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"No hay archivos .sql en {sql_dir}")
    applied = []
    conn = psycopg2.connect(dsn)
    try:
        for path in files:
            with conn:  # commit por archivo, rollback si falla
                with conn.cursor() as cur:
                    cur.execute(path.read_text(encoding="utf-8"))
            applied.append(path.name)
            print(f"  aplicado: {path.name}")
    finally:
        conn.close()
    return applied


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dsn", default=None, help="URL de Postgres (por defecto AJUSTES_DATABASE_URL)")
    args = parser.parse_args()
    dsn = args.dsn or os.getenv("AJUSTES_DATABASE_URL")
    if not dsn:
        print("ERROR: falta AJUSTES_DATABASE_URL (o --dsn)", file=sys.stderr)
        return 2
    print(f"Aplicando esquema de ajustes desde {SQL_DIR} ...")
    apply_schema(dsn)
    print("Listo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
