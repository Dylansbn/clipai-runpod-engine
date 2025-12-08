FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

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

COPY requirements.txt .

# 🔥 FIX NUMPY + TORCH COMPATIBILITY
RUN pip install --upgrade pip
RUN pip install "numpy<2.0"

# Puis installation normale
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD bash -c "\
    uvicorn clipai_runpod_engine.handler:app --host 0.0.0.0 --port 8000 & \
    python3 -m clipai_runpod_engine.engine.worker \
"
