# 🟩 Base NVIDIA CUDA + cuDNN (EXISTANTE ET COMPATIBLE)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Dépendances système
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    curl \
    wget \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Dépendances Python
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Code de l'application
COPY clipai_runpod_engine /app/clipai_runpod_engine
COPY . .

# Commande de démarrage RunPod
CMD ["python3", "-u", "-m", "clipai_runpod_engine.handler"]


