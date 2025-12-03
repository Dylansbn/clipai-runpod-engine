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

# yt-dlp + pysubs2 au cas où ils ne sont pas dans requirements.txt
RUN pip install --no-cache-dir yt-dlp pysubs2

# ========== COPY PROJECT ==========
COPY . .

# ========== LAUNCH ==========
CMD ["python", "handler.py"]
