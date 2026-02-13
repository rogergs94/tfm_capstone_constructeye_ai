# BACKEND DOCKERFILE
# 1. BASE IMAGE
FROM python:3.11-slim

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
# Python won't try to write .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Force the stdout and stderr streams to be unbuffered
ENV PYTHONUNBUFFERED=1

# 2. SYSTEM DEPENDENCIES (Essential for OpenCV and Boto3)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Work Directory
WORKDIR /app

# 4. Copy requirements FIRST (Docker Cache Optimization)
COPY requirements_app.txt .

# 5. Install Python Libraries
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements_app.txt \
    && rm -rf /root/.cache/pip

# 6. Copy the rest of the application code
COPY . .

# 7. Expose Uvicorn port
EXPOSE 8000

# 8. Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]