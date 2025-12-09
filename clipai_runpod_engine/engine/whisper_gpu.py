# Fichier: clipai_runpod_engine/engine/whisper_gpu.py
import os
import platform
from faster_whisper import WhisperModel

SYSTEM = platform.system().lower()
WHISPER_MODEL = None

try:
    if SYSTEM == "darwin":
        WHISPER_MODEL = WhisperModel("small", device="cpu", compute_type="int8")
        print("⚠️ Modèle 'small' chargé pour CPU (Développement Local)")
    else:
        # Cas LINUX + CUDA (RunPod Serverless)
        # En passant juste le nom "medium", faster-whisper cherche dans le cache par défaut (/root/.cache).
        WHISPER_MODEL = WhisperModel("medium", device="cuda", compute_type="float16")
        print("⚡ Modèle 'medium' chargé pour Whisper GPU (Production)")

except Exception as e:
    print(f"FATAL ERROR: Échec du chargement du modèle Whisper : {e}")
    # Cette erreur est ce que vous voyez dans les logs.
    raise RuntimeError("Impossible d'initialiser le modèle Whisper GPU.")


def transcribe_gpu(video_path):
    # Utilise le modèle global WHISPER_MODEL déjà chargé.
    if WHISPER_MODEL is None:
        raise RuntimeError("Le modèle Whisper n'a pas pu être chargé au démarrage du Worker.")

    print("🎧 Démarrage de la transcription...")
    segments, _ = WHISPER_MODEL.transcribe(video_path) 
    
    results = []
    for seg in segments:
        results.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        })
    return results