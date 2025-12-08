FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

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
RUN pip3 install --no-cache-dir -r requirements.txt

# ---------------------------
# Properly package your code
# ---------------------------
COPY . /app/

RUN mkdir -p /app/clipai_runpod_engine && \
    mv /app/engine /app/clipai_runpod_engine/ && \
    mv /app/job_queue /app/clipai_runpod_engine/ && \
    mv /app/shared /app/clipai_runpod_engine/ && \
    mv /app/subs /app/clipai_runpod_engine/ && \
    mv /app/shorts /app/clipai_runpod_engine/ && \
    mv /app/uploads /app/clipai_runpod_engine/

CMD ["python3", "-m", "clipai_runpod_engine.engine.worker"]
