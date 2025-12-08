import os
import platform

from faster_whisper import WhisperModel


def transcribe_gpu(video_path):
    """
    Fonction intelligente :
    - Sur Mac : utilise CPU automatiquement
    - Sur RunPod : utilise le GPU CUDA
    """

    system = platform.system().lower()
    print(f"🧠 Plateforme détectée : {system}")

    # ------------------------------
    # 1️⃣ Cas MAC (Aucun GPU NVIDIA)
    # ------------------------------
    if system == "darwin":
        print("⚠️ Aucun GPU NVIDIA → utilisation du CPU pour Whisper")
        model = WhisperModel("small", device="cpu", compute_type="int8")
    else:
        # ------------------------------
        # 2️⃣ Cas LINUX + CUDA (RunPod)
        # ------------------------------
        print("⚡ Whisper GPU activé (CUDA)")
        model = WhisperModel("medium", device="cuda", compute_type="float16")

    segments, _ = model.transcribe(video_path)
    results = []

    for seg in segments:
        results.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        })

    return results
