# DOCKERFILE FRONTEND
# 1. BASE IMAGE
FROM python:3.11-slim

# Evita prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# 2. INSTALACIÓN DEPENDENCIAS DE SISTEMA (Vital para OpenCV)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Directorio de trabajo
WORKDIR /app

# 4. Copiamos PRIMERO los requirements (Optimización de caché de Docker)
# Si cambias tu código pero no tus librerías, Docker se saltará el paso de instalación (ahorra tiempo)
COPY requirements_frontend.txt .


# 5. Instalación las librerías de Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements_frontend.txt

# 6. Copia del resto del código de la carpeta actual al contenedor
COPY . .

# 7. Exponemos el puerto de Uvicorn
EXPOSE 8501

# 8. Comando para ejecutar streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
