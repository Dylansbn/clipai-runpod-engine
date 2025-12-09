import os
import platform

from faster_whisper import WhisperModel

# ==========================================================
# 🚀 INITIALISATION GLOBALE DU MODÈLE (CORRECTIF DÉMARRAGE)
# ==========================================================
# Le modèle est chargé UNE SEULE FOIS au démarrage du Worker.
# Le modèle 'medium' est téléchargé sur le disque durant la phase BUILD du Dockerfile.

SYSTEM = platform.system().lower()

# Chemin où le modèle 'medium' est stocké par le Dockerfile
MODEL_CACHE_PATH = "/root/.cache/faster-whisper/medium" 

WHISPER_MODEL = None

try:
    if SYSTEM == "darwin":
        # Cas Mac (pour le développement local)
        WHISPER_MODEL = WhisperModel("small", device="cpu", compute_type="int8")
        print("⚠️ Modèle 'small' chargé pour CPU (Développement Local)")
    else:
        # Cas LINUX + CUDA (RunPod Serverless)
        # On utilise le chemin local car le modèle a été téléchargé pendant le BUILD
        WHISPER_MODEL = WhisperModel(MODEL_CACHE_PATH, device="cuda", compute_type="float16")
        print("⚡ Modèle 'medium' chargé pour Whisper GPU (Production)")

except Exception as e:
    # Ceci capture les erreurs si le modèle n'est pas trouvé ou si le GPU est inaccessible au démarrage
    print(f"FATAL ERROR: Échec du chargement du modèle Whisper : {e}")
    # En production, cela mènera à un crash immédiat, signalant un problème de configuration/build.
    raise RuntimeError("Impossible d'initialiser le modèle Whisper GPU.")


def transcribe_gpu(video_path):
    """
    Fonction de transcription. Utilise le modèle global WHISPER_MODEL.
    """
    
    # Vérification de sécurité, bien que le modèle doive être initialisé en haut
    if WHISPER_MODEL is None:
        raise RuntimeError("Le modèle Whisper n'a pas pu être chargé au démarrage du Worker.")

    print("🎧 Démarrage de la transcription...")
    
    # ------------------------------
    # Démarrage de la transcription
    # ------------------------------
    # La logique de détection de plateforme est gérée par l'initialisation globale.
    
    segments, _ = WHISPER_MODEL.transcribe(video_path) 
    
    results = []
    for seg in segments:
        results.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        })

    return results