FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

# --- FIX CUDNN FOR RUNPOD SERVERLESS ---
# RunPod Serverless préinstalle cuDNN 9 → incompatible avec faster-whisper.
# On réinstalle cuDNN 8 et on supprime cuDNN 9.
RUN apt-get update && \
    apt-get install -y libcudnn8 libcudnn8-dev && \
    rm -f /usr/lib/x86_64-linux-gnu/libcudnn* && \
    ln -s /usr/lib/x86_64-linux-gnu/libcudnn8.so /usr/lib/x86_64-linux-gnu/libcudnn.so

# Dépendances système
RUN apt-get update && apt-get install -y \
    tzdata \
    ffmpeg \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Dépendances Python
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Code de l'application
COPY clipai_runpod_engine /app/clipai_runpod_engine

# Préchargement Whisper (medium)
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('medium')"

# Copier le reste (README, etc.)
COPY . .

# Commande de démarrage
CMD ["python3", "-u", "-m", "clipai_runpod_engine.handler"]
t(...)
CMD ["python3", "-u", "-m", "clipai_runpod_engine.handler"]
