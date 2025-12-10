FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

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

# Code source
COPY clipai_runpod_engine /app/clipai_runpod_engine

# Précharge Whisper (accélère le premier job)
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('medium')"

# Copier le reste
COPY . .

# Commande de démarrage
CMD ["python3", "-u", "-m", "clipai_runpod_engine.handler"]

