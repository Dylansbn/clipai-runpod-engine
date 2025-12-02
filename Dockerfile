FROM runpod/serverless:3.0.0

# On travaille dans /app
WORKDIR /app

# Copie des dépendances
COPY requirements.txt .

# Installation des libs Python + ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt

# Copie du code
COPY . .

# On lance le worker RunPod
CMD ["python", "-u", "handler.py"]
