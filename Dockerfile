# Imagen de la DEMO (EPM + front Sphere). Contexto de build: raiz de EPM con static_demo/ y demo_snapshot/
# ya generados (los arma scripts/demo_build.sh; NO estan en git porque contienen datos reales).
# Etapa 1: dependencias Python en un venv
FROM python:3.11-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Etapa 2: runtime
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY --from=deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    DEMO_MODE=1 DEMO_STATIC_DIR=/app/static_demo
WORKDIR /app
# Codigo
COPY src/ /app/src/
COPY config/ /app/config/
# Snapshot de datos/modelos (layout del repo: data/, models/, reports/) y build del front
COPY demo_snapshot/ /app/
COPY static_demo/ /app/static_demo/
# La API lee data/ y models/ relativos al cwd (/app): apuntan al snapshot copiado arriba.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fsS http://localhost:${PORT:-8000}/health || exit 1
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
