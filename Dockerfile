# Dockerfile para EPM Energy Demand Forecasting API
FROM python:3.10-slim

# Instalar dependencias del sistema necesarias para LightGBM y XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Configurar variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements primero (para aprovechar cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer el puerto (Railway lo sobrescribirá con $PORT)
EXPOSE 8000

# Comando de inicio (Railway sobrescribirá el puerto con $PORT)
# El código en main.py ya maneja el puerto dinámico
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}

