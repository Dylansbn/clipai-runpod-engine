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
    Retourne le chemin local complet.
    """
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # Détection extension si possible
    ext = ".mp4"
    filename_raw = url.split("/")[-1]
    if "." in filename_raw:
        ext = "." + filename_raw.split(".")[-1].split("?")[0]

    filename = f"input_{uuid.uuid4().hex}{ext}"
    dest = UPLOADS_DIR / filename

    print(f"⬇️  Téléchargement vidéo depuis : {url}")

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

        # -------------------------
        # 1️⃣ Ping de test
        # -------------------------
        if task == "ping":
            return {
                "status": "ok",
                "message": "clipai-runpod-engine is alive ✅"
            }

        # -------------------------
        # 2️⃣ Traitement vidéo
        # -------------------------
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

            # Téléchargement vidéo
            local_path = download_video_to_uploads(video_url)

            # 👍 AJOUT : afficher la taille du fichier téléchargé
            try:
                file_size = os.path.getsize(local_path)
                print(f"📏 Taille du fichier téléchargé : {file_size} octets")
            except:
                print("⚠️ Impossible de lire la taille du fichier")

            # -------------------------
            # Pipeline IA
            # -------------------------
            clips = generate_shorts(
                input_video_path=local_path,
                num_clips=num_clips,
                min_duration=min_duration,
                max_duration=max_duration,
            )

            return {
                "status": "done",
                "clips": clips,
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


# Lancement du worker RunPod
runpod.serverless.start({"handler": handler})
