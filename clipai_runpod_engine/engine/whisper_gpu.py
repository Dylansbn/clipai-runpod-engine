# Fichier: clipai_runpod_engine/engine/whisper_gpu.py

import os
import platform

from faster_whisper import WhisperModel

# ==========================================================
# 🚀 AJOUT D'UNE INITIALISATION GLOBALE
# ==========================================================
# Ceci charge le modèle UNE FOIS au démarrage du Worker (RUN).
# Le modèle a été téléchargé pendant le BUILD (Dockerfile).
SYSTEM = platform.system().lower()

if SYSTEM == "darwin":
    # Cas Mac (pour le développement local)
    WHISPER_MODEL = WhisperModel("small", device="cpu", compute_type="int8")
    print("⚠️ Modèle 'small' chargé pour CPU (Développement Local)")
else:
    # Cas LINUX + CUDA (RunPod Serverless)
    # Le modèle 'medium' est déjà sur le disque grâce au Dockerfile
    WHISPER_MODEL = WhisperModel("medium", device="cuda", compute_type="float16")
    print("⚡ Modèle 'medium' chargé pour Whisper GPU (Production)")
# ==========================================================


def transcribe_gpu(video_path):
    """
    Fonction de transcription. Utilise le modèle global WHISPER_MODEL.
    """
    
    # La logique de détection de plateforme est désormais inutile ici, car
    # le modèle est initialisé une seule fois de manière globale
    
    segments, _ = WHISPER_MODEL.transcribe(video_path) 
    
    results = []
    for seg in segments:
        results.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        })

    return results

# NOTE: La fonction de pré-téléchargement n'est plus nécessaire dans le Worker
# car le modèle est initialisé de manière globale.