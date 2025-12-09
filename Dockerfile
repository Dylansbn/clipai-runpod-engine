FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

# Installation des dépendances système (AJOUT de git !)
RUN apt-get update && apt-get install -y \
    tzdata \
    ffmpeg \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================================
# 🚀 ÉTAPE D'OPTIMISATION DU DÉMARRAGE (CORRECTIF FINAL)
# On utilise la méthode de chargement simple du modèle pour forcer le téléchargement.
# La bibliothèque gère les chemins de cache.
# ==========================================================
# Copie temporaire du code pour que l'importation fonctionne
COPY clipai_runpod_engine /app/clipai_runpod_engine 
# Force le téléchargement et le caching du modèle 'medium'.
# Nous utilisons la classe de modèle elle-même pour déclencher le téléchargement sans nécessiter de GPU.
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('medium')"
# ==========================================================

# Copie du reste du code et du script d'entrée
COPY . .
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Utilisation du script d'entrée pour lancer les deux processus
CMD ["/app/entrypoint.sh"]