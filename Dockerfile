FROM nginx:stable

# Copia los archivos HTML al directorio de Nginx
COPY ./html /usr/share/nginx/html

# Opcional: exponer el puerto
EXPOSE 80
