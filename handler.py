import os
import uuid
import traceback
from pathlib import Path
from typing import Any, Dict

import requests
import runpod

from processor import generate_shorts, UPLOADS_DIR


# ===============================
#  UTILITAIRE : téléchargement
# ===============================

def download_video_to_uploads(url: str) -> str:
    """
    Télécharge la vidéo depuis une URL HTTP(s) dans UPLOADS_DIR.
    Retourne le chemin local.
    """
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    ext = ".mp4"
    if "." in url.split("/")[-1]:
        # essaie de garder l'extension originale
        ext = "." + url.split("/")[-1].split(".")[-1].split("?")[0]

    filename = f"input_{uuid.uuid4().hex}{ext}"
    dest = UPLOADS_DIR / filename

    print(f"⬇️  Téléchargement vidéo depuis {url}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    with dest.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"✅ Vidéo téléchargée : {dest}")
    return str(dest)


# ===============================
#  HANDLER PRINCIPAL RUNPOD
# ===============================

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    event = {
      "input": {
        "task": "ping"
      }
    }
    ou
    {
      "input": {
        "task": "process",
        "video_url": "https://....mp4",
        "num_clips": 8,
        "min_duration": 20,
        "max_duration": 45
      }
    }
    """
    try:
        inp = event.get("input") or {}
        task = inp.get("task", "ping")

        # 1) Test rapide
        if task == "ping":
            return {
                "status": "ok",
                "message": "clipai-runpod-engine is alive ✅"
            }

        # 2) Traitement vidéo
        if task == "process":
            video_url = inp.get("video_url")
            if not video_url:
                return {
                    "status": "error",
                    "error": "Missing 'video_url' in input."
                }

            num_clips = int(inp.get("num_clips", 8))
            min_duration = float(inp.get("min_duration", 20.0))
            max_duration = float(inp.get("max_duration", 45.0))

            # Téléchargement
            local_path = download_video_to_uploads(video_url)

            # Pipeline IA
            clips = generate_shorts(
                input_video_path=local_path,
                num_clips=num_clips,
                min_duration=min_duration,
                max_duration=max_duration,
            )

            # Réponse JSON-friendly
            return {
                "status": "done",
                "clips": clips,
            }

        # 3) Autre task inconnue
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


# Lancement du worker RunPod
runpod.serverless.start({"handler": handler})
