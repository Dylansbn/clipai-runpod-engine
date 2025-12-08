FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# ---- System deps ----
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ---- Upgrade pip ----
RUN pip3 install --upgrade pip setuptools wheel

# ---- Fix PyTorch + NumPy ----
RUN pip3 install numpy==1.26.4
RUN pip3 install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# ---- Install other Python deps ----
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ---- Copy code ----
COPY . .

# ---- Start API + Worker ----
CMD bash -c "\
    uvicorn clipai_runpod_engine.handler:app --host 0.0.0.0 --port 8000 & \
    python3 -m clipai_runpod_engine.engine.worker \
"
