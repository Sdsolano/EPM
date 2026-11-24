#!/usr/bin/env bash
# Arma el contexto de build de la demo: static_demo/ (front Sphere) y demo_snapshot/ (datos+modelos reales).
# Uso (desde la raiz de EPM):  SPHERE_DIR=../Sphere scripts/demo_build.sh
# Requiere node/npm en PATH y el venv de EPM. Ninguna de las dos carpetas va a git.
set -euo pipefail
cd "$(dirname "$0")/.."
SPHERE_DIR="${SPHERE_DIR:-../Sphere}"
PY="${PYTHON:-.venv/bin/python}"
[ -d "$SPHERE_DIR" ] || { echo "No existe SPHERE_DIR=$SPHERE_DIR"; exit 1; }
[ -z "$(git -C "$SPHERE_DIR" status --porcelain)" ] || { echo "Sphere tiene cambios sin commitear; abortando"; exit 1; }
echo "Compilando Sphere en la rama: $(git -C "$SPHERE_DIR" branch --show-current) (debe ser test)"
( cd "$SPHERE_DIR" && npm ci && npm run build:demo )
rm -rf static_demo && cp -R "$SPHERE_DIR/dist" static_demo
# DEMO_DATA_OVERLAY (opcional): directorio con datos/modelos extendidos que sobrescriben los del repo SOLO dentro del snapshot.
OVERLAY_ARGS=()
if [ -n "${DEMO_DATA_OVERLAY:-}" ]; then OVERLAY_ARGS=(--overlay "$DEMO_DATA_OVERLAY"); echo "Snapshot con overlay: $DEMO_DATA_OVERLAY"; fi
rm -rf demo_snapshot && "$PY" scripts/demo_snapshot.py --dest demo_snapshot ${OVERLAY_ARGS[@]+"${OVERLAY_ARGS[@]}"}
"$PY" scripts/demo_snapshot.py --verify --dest demo_snapshot
echo "Listo: static_demo/ y demo_snapshot/ generados. Siguiente: docker compose up --build  |  railway up --no-gitignore"
