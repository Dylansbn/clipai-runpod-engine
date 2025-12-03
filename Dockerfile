FROM python:3.10-slim

# ========== INSTALL SYSTEM DEPS ==========
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ========== INSTALL PYTHON DEPENDENCIES ==========
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# yt-dlp obligatoire pour TikTok / YouTube / Vimeo protégé
RUN pip install --no-cache-dir yt-dlp

# ========== COPY PROJECT ==========
COPY . .

# ========== LAUNCH ==========
CMD ["python", "handler.py"]
