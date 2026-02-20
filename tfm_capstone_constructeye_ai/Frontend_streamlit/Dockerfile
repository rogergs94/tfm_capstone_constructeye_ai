# FRONTEND DOCKERFILE
# 1. BASE IMAGE
FROM python:3.11-slim

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
# Python won't try to write .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Force the stdout and stderr streams to be unbuffered
ENV PYTHONUNBUFFERED=1

# 2. SYSTEM DEPENDENCIES (Essential for OpenCV and image processing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Work Directory
WORKDIR /app

# 4. Copy requirements (Adjusted for root context)
# We copy from the specific folder so Docker finds the file
COPY Frontend_streamlit/requirements_frontend.txt .

# 5. Install Python Libraries
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements_frontend.txt \
    && rm -rf /root/.cache/pip

# 6. Copy frontend source code only
# This copies the contents of Frontend_streamlit into /app in the container
COPY Frontend_streamlit/ .

# 7. Expose Streamlit port
EXPOSE 8501

# 8. Command to run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
