FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Opcional: utilidades mínimas
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Instala torch CPU desde el índice oficial y luego el resto de deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 && \
    pip install --no-cache-dir -r requirements.txt

# Copia el código
COPY . .

# La app escucha en 8080; el host publica 80->8080 (ya lo hace tu deploy.yml)
ENV PORT=8080
EXPOSE 8080

# Servidor de producción
CMD ["gunicorn","-w","2","-k","gthread","-b","0.0.0.0:8080","app:app"]
