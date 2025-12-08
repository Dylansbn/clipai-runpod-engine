FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# -------------------------
# Install system packages
# -------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Set working directory
# -------------------------
WORKDIR /app

# -------------------------
# Install Python dependencies
# -------------------------
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# -------------------------
# Copy project files
# -------------------------
COPY . .

# -------------------------
# Start API first, then worker
# IMPORTANT: sleep 2 ensures Uvicorn starts BEFORE worker
# -------------------------
CMD bash -c "\
    uvicorn clipai_runpod_engine.handler:app --host 0.0.0.0 --port 8000 & \
    sleep 2 && \
    python3 -m clipai_runpod_engine.engine.worker \
"
