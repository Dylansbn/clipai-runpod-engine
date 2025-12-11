# 🔥 Image NVIDIA optimisée pour RunPod + Whisper
FROM nvidia/cuda:12.1.1-cudnn9-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# ================================
# 🔧 Dépendances système
# ================================
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    ffmpeg \
    wget curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# ================================
# 📦 Dépendances Python
# ================================
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ================================
# 📁 Code backend
# ================================
COPY clipai_runpod_engine /app/clipai_runpod_engine

# ================================
# ⚡ Pré-chargement modèle Whisper (optionnel, accélère première requête)
# ================================
RUN python3 - <<EOF
from faster_whisper import WhisperModel
WhisperModel("medium", device="cuda", compute_type="float16")
EOF

# ================================
# 🚀 Commande de démarrage RunPod
# ================================
CMD ["python3", "-u", "-m", "clipai_runpod_engine.handler"]

