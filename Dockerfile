FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Seguridad: usuario no root
RUN useradd -m appuser
WORKDIR /app

# Instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia código y estáticos
COPY app.py .
COPY static ./static

EXPOSE 8080
USER appuser

# Servimos con gunicorn (2 workers con threads, suficiente para t2.micro)
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-b", "0.0.0.0:8080", "app:app"]

