#!/bin/bash
# Script de inicio para Railway
# Expande la variable PORT correctamente

PORT=${PORT:-8000}
exec uvicorn src.api.main:app --host 0.0.0.0 --port $PORT

