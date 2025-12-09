FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive
# ... (Vos installations apt-get)
WORKDIR /app
# ... (Installation de requirements.txt)

# ==========================================================
# 🚀 ÉTAPE D'OPTIMISATION DU DÉMARRAGE (CORRIGE LE DÉLAI DE 24 MIN)
# Nouvelle tentative : on lance simplement le Worker pour une fois, 
# ce qui force le téléchargement du modèle 'medium' dans le cache.
# ==========================================================
# Copie temporaire du code nécessaire au téléchargement
COPY clipai_runpod_engine /app/clipai_runpod_engine
# Lance une commande Python simple qui utilise le modèle
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('medium')"
# ==========================================================

# Copie du reste du code et du script d'entrée
COPY . .
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
CMD ["/app/entrypoint.sh"]