FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Fix pip + install numpy compatible avec Torch (SUPER IMPORTANT)
RUN pip install --upgrade pip
RUN pip install "numpy==1.26.4"

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Installer toutes les libs après avoir fixé NumPy
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD bash -c "\
    uvicorn clipai_runpod_engine.handler:app --host 0.0.0.0 --port 8000 & \
    python3 -m clipai_runpod_engine.engine.worker \
"

