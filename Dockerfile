FROM runpod/serverless:gpu

WORKDIR /app

# Copier les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

# Handler pour RunPod
CMD ["python", "-u", "handler.py"]
