# DevOps + AI Demo (Free Tier, Determinista), Prueba

Este proyecto está listo para laboratorio **sin opcionales**: misma imagen, mismo contenedor, mismo pipeline.  
Objetivo: app Flask con análisis de sentimiento (VADER), empaquetada en Docker, desplegada vía GitHub Actions a una EC2 **t2.micro** (capa gratuita).

## Pre-requisitos (fijos)
1. **EC2 t2.micro** (Amazon Linux 2023), Security Group con:
   - TCP 80 abierto (HTTP) para acceso web.
   - TCP 22 abierto desde tu IP (SSH).
2. **Par de claves** (PEM) para SSH (copia privada en secreto `EC2_SSH_KEY`).
3. **Docker Hub** con token (para subir la imagen).

## Estructura
```
.
├── app.py
├── requirements.txt
├── Dockerfile
├── static/
│   └── index.html
└── .github/
    └── workflows/
        └── deploy.yml
```

## Secrets obligatorios en el repositorio
- `DOCKERHUB_USERNAME` → tu usuario de Docker Hub.
- `DOCKERHUB_TOKEN` → token de acceso de Docker Hub.
- `EC2_HOST` → DNS público o IP de la instancia EC2.
- `EC2_SSH_KEY` → contenido del archivo .pem (clave privada).

## Cómo funciona el pipeline
- En cada `push` a `main`:
  1. **Build** de imagen Docker: `${DOCKERHUB_USERNAME}/ai-demo:<SHA7>` y `:latest`.
  2. **Push** de ambas etiquetas a Docker Hub.
  3. **SSH a EC2** y despliegue idempotente:
     - Instala Docker si no existe y asegura el servicio activo.
     - `docker pull` de `:latest`.
     - Reemplaza el contenedor `ai-demo` y expone puerto 80.

Acceso: `http://<EC2_HOST>/`

## Comandos locales de prueba (opcional fuera del pipeline)
```bash
docker build -t ai-demo:local .
docker run --rm -p 8080:8080 ai-demo:local
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{"text":"I love this!"}'
