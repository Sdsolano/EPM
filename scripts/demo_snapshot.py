"""Arma el snapshot de datos/modelos que necesita la API en modo demo (solo lectura).

Uso (desde la raíz del repo):
    .venv/bin/python scripts/demo_snapshot.py                # crea demo_snapshot/
    .venv/bin/python scripts/demo_snapshot.py --dest /ruta   # otro directorio
    .venv/bin/python scripts/demo_snapshot.py --verify       # solo verifica un snapshot existente contra su manifiesto

El snapshot replica el layout del repo (data/, models/, reports/) para copiarse tal cual a /app en la imagen
(`COPY demo_snapshot/ /app/`) o a un volumen. Incluye MANIFEST.json con ruta, bytes y sha256 de cada archivo.

IMPORTANTE: contiene datos reales de las UCP. El repo EPM es público: `demo_snapshot/` está en .gitignore y NO
debe commitearse. Se usa como contexto de build local o se sube a un volumen. Este script no escribe nada
fuera de --dest y solo LEE data/ y models/ (no los modifica).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
UCPS = ("Antioquia", "Atlantico")
MANIFEST = "MANIFEST.json"

# Archivos globales de modelos (los de cada UCP se copian completos: models/<ucp>/).
MODELOS_GLOBALES = (
    "models/hourly_disaggregator.pkl",
    "models/special_days_disaggregator.pkl",
    "models/registry/champion_model.joblib",
    "models/registry/registry_metadata.json",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _archivos(src_root: Path, ucps: Tuple[str, ...]) -> Tuple[List[Path], List[str]]:
    """Rutas relativas a src_root a incluir y lista de avisos por faltantes opcionales."""
    rels: List[Path] = []
    avisos: List[str] = []
    for ucp in ucps:
        for sub in (f"data/raw/{ucp}", f"models/{ucp}"):
            d = src_root / sub
            if not d.is_dir():
                raise FileNotFoundError(f"Falta {d} (requerido)")
            rels += [p.relative_to(src_root) for p in sorted(d.rglob("*")) if p.is_file() and p.name != ".DS_Store"]
        feat = Path(f"data/features/{ucp}/data_with_features_latest.csv")
        if (src_root / feat).is_file():
            rels.append(feat)
        else:
            avisos.append(f"falta {feat} (el endpoint /health?ucp={ucp} lo marcará 'missing')")
        curvas = Path(f"reports/ajustes/curvas_modelo_k1_{ucp}.csv")
        if (src_root / curvas).is_file():
            rels.append(curvas)
        else:
            avisos.append(f"no hay {curvas} en esta máquina (opcional: base alternativa sin correr el modelo)")
    for g in MODELOS_GLOBALES:
        if (src_root / g).is_file():
            rels.append(Path(g))
        else:
            avisos.append(f"falta {g} (opcional)")
    return rels, avisos


def construir(dest: Path, src_root: Path = ROOT, ucps: Tuple[str, ...] = UCPS) -> Dict:
    dest = dest.resolve()
    if dest == src_root.resolve() or src_root.resolve() in dest.parents and dest.name in {"data", "models"}:
        raise ValueError("--dest no puede ser la raíz del repo ni data/ o models/")
    rels, avisos = _archivos(src_root, ucps)
    if dest.exists():
        shutil.rmtree(dest)  # el snapshot es reproducible: se reconstruye entero
    entradas = []
    for rel in rels:
        destino = dest / rel
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_root / rel, destino)
        entradas.append({"ruta": rel.as_posix(), "bytes": destino.stat().st_size, "sha256": sha256(destino)})
    manifiesto = {
        "creado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ucps": list(ucps),
        "total_bytes": sum(e["bytes"] for e in entradas),
        "avisos": avisos,
        "archivos": entradas,
    }
    (dest / MANIFEST).write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifiesto


def verificar(dest: Path) -> List[str]:
    """Devuelve la lista de problemas (vacía = snapshot íntegro)."""
    mf = dest / MANIFEST
    if not mf.is_file():
        return [f"no existe {mf}"]
    problemas = []
    for e in json.loads(mf.read_text(encoding="utf-8"))["archivos"]:
        p = dest / e["ruta"]
        if not p.is_file():
            problemas.append(f"falta {e['ruta']}")
        elif sha256(p) != e["sha256"]:
            problemas.append(f"hash distinto {e['ruta']}")
    return problemas


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default=str(ROOT / "demo_snapshot"))
    ap.add_argument("--verify", action="store_true", help="solo verificar el snapshot existente")
    args = ap.parse_args(argv)
    dest = Path(args.dest)
    if not args.verify:
        mf = construir(dest)
        print(f"Snapshot en {dest}: {len(mf['archivos'])} archivos, {mf['total_bytes'] / 1e6:.1f} MB")
        for a in mf["avisos"]:
            print(f"  aviso: {a}")
    problemas = verificar(dest)
    for p in problemas:
        print(f"  ERROR: {p}", file=sys.stderr)
    print("Verificación:", "OK" if not problemas else f"{len(problemas)} problemas")
    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())
