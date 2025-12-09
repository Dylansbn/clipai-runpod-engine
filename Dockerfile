FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive
# ... (Vos installations apt-get)
WORKDIR /app
# ... (Installation de requirements.txt)

# ==========================================================
# 🚀 ÉTAPE D'OPTIMISATION DU DÉMARRAGE (CORRECTIF 24 MIN)
# Nouvelle et dernière tentative : Utilisation de la fonction download_model.
# ==========================================================
# Copie temporaire du code pour que Python trouve le chemin de faster-whisper
COPY clipai_runpod_engine /app/clipai_runpod_engine
# Lance la fonction de téléchargement directement
RUN python3 -c "from faster_whisper.utils import download_model; download_model('medium', '/root/.cache/faster-whisper')"
# ==========================================================

# Copie du reste du code et du script d'entrée
COPY . .
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
CMD ["/app/entrypoint.sh"]