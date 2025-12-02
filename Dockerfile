FROM python:3.10-slim

WORKDIR /app

# Installer FFmpeg
RUN apt update && apt install -y ffmpeg && apt clean

# Copier requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code
COPY . .

# Exposer le port (obligatoire pour RunPod même si pas utilisé)
EXPOSE 8000

# Lancer le handler RunPod
CMD ["python", "handler.py"]
