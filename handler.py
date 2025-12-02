import os
import uuid
import traceback
from pathlib import Path
from typing import Any, Dict

import requests
import runpod

from processor import generate_shorts, UPLOADS_DIR


# ===============================
#  UTILITAIRE : téléchargement vidéo
# ===============================

def download_video_to_uploads(url: str) -> str:
    """
    Télécharge proprement une vidéo HTTP/HTTPS dans UPLOADS_DIR.
    Retourne le chemin local complet.
    """
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # Détection de l’extension si possible
    ext = ".mp4"
    filename_raw = url.split("/")[-1]

    if "." in filename_raw:
        ext = "." + filename_raw.split(".")[-1].split("?")[0]

    filename = f"input_{uuid.uuid4().hex}{ext}"
    dest = UPLOADS_DIR / filename

    print(f"⬇️ Téléchargement vidéo : {url}")

    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Erreur téléchargement vidéo : {e}")

    with dest.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"✅ Vidéo téléchargée dans {dest}")

    # Taille du fichier pour debug
    try:
        size = os.path.getsize(dest)
        print(f"📏 Taille du fichier téléchargé : {size} octets")
    except:
        print("⚠️ Impossible de lire la taille du fichier")

    return str(dest)


# ===============================
#  HANDLER PRINCIPAL RUNPOD
# ===============================

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Structure du event :

    Ping :
    {
      "input": { "task": "ping" }
    }

    Traitement vidéo :
    {
      "input": {
        "task": "process",
        "video_url": "https://…",
        "num_clips": 8,
        "min_duration": 20,
        "max_duration": 45
      }
    }
    """

    try:
        inp = event.get("input") or {}
        task = inp.get("task", "ping")

        # -------------------------
        # 1️⃣ Ping de test
        # -------------------------
        if task == "ping":
            print("🏓 Ping reçu.")
            return {
                "status": "ok",
                "message": "clipai-runpod-engine is alive 🟢"
            }

        # -------------------------
        # 2️⃣ Traitement vidéo complet
        # -------------------------
        if task == "process":

            video_url = inp.get("video_url")
            if not video_url:
                return {
                    "status": "error",
                    "error": "Missing 'video_url' in input."
                }

            num_clips = int(inp.get("num_clips", 8))
            min_duration = float(inp.get("min_duration", 20))
            max_duration = float(inp.get("max_duration", 45))

            # 🔽 Téléchargement
            local_path = download_video_to_uploads(video_url)

            # 🎥 IA pipeline
            clips = generate_shorts(
                input_video_path=local_path,
                num_clips=num_clips,
                min_duration=min_duration,
                max_duration=max_duration,
            )

            return {
                "status": "done",
                "clips": clips
            }

        # -------------------------
        # 3️⃣ Task inconnue
        # -------------------------
        return {
            "status": "error",
            "error": f"Unknown task '{task}'"
        }

    except Exception as e:
        print("🔥 ERREUR DANS HANDLER :", e)
        print(traceback.format_exc())

        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ===============================
#  START RUNPOD WORKER
# ===============================

runpod.serverless.start({"handler": handler})
