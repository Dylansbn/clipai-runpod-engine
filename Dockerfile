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

COPY . .

EXPOSE 8000

# Lance le worker en arrière-plan, puis l’API FastAPI
CMD bash -c "python3 -m clipai_runpod_engine.engine.worker & uvicorn handler:app --host 0.0.0.0 --port 8000"
