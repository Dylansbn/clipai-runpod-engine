FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

# Installation des dépendances système
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
# 🚀 ÉTAPE D'OPTIMISATION : PRÉ-TÉLÉCHARGEMENT DU MODÈLE WHISPER (CORRIGE LE DÉLAI DE 24 MIN)
# ==========================================================
# Le modèle "medium" est celui utilisé par défaut sur RunPod. 
# Cette commande télécharge et met en cache les 3.44GB de modèle, une seule fois.
RUN python3 -c "from faster_whisper import download_model; download_model('medium', local_model_path='/root/.cache/whisper')"

# Copie du reste du code et du script d'entrée
COPY . .
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Utilisation du script d'entrée pour lancer les deux processus
CMD ["/app/entrypoint.sh"]