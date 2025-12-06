import os
import traceback
from typing import Any, Dict

import runpod

from processor import (
    download_video,
    generate_shorts,
)

# ===============================
#  UTILITAIRE DEBUG : ffprobe
# ===============================

def debug_probe(path: str) -> Dict[str, Any]:
    """Petit helper pour inspecter une vidéo via ffprobe."""
    import subprocess

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        path,
    ]

    try:
        out = subprocess.check_output(cmd).decode("utf-8")
        return {
            "file_path": path,
            "size_bytes": os.path.getsize(path),
            "ffprobe_json": out,
        }
    except Exception as e:
        return {
            "error": "ffprobe failed",
            "traceback": str(e),
        }


# ===============================
#  HANDLER PRINCIPAL RUNPOD
# ===============================

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    event = {
      "input": {
         "url": "...",        # ou "video_url"
         "task": "...",       # optionnel : process | ping | debug_*
         "num_clips": 3,
         "min_duration": 6,
         "max_duration": 25
      }
    }

    Avec /runsync, RunPod renverra :
    {
      "status": "COMPLETED",
      "output": { ...ce que cette fonction retourne... }
    }
    """

    print("📩 EVENT REÇU :", event)

    try:
        inp = event.get("input") or {}

        # On accepte à la fois "video_url" et "url"
        url = inp.get("video_url") or inp.get("url")

        # Déduction de la task si absente :
        #  - URL présente => process
        #  - pas d’URL    => ping
        task = inp.get("task")
        if not task:
            task = "process" if url else "ping"

        num_clips = int(inp.get("num_clips", 8))
        min_duration = float(inp.get("min_duration", 20))
        max_duration = float(inp.get("max_duration", 45))

        # 1️⃣ PING
        if task == "ping":
            resp = {
                "status": "ok",
                "message": "ClipAI Engine is alive 🔥",
                "version": "serverless-pro",
            }
            print("🔵 RÉPONSE HANDLER :", resp)
            return resp

        # 2️⃣ DEBUG DOWNLOAD
        if task == "debug_download":
            if not url:
                resp = {"status": "error", "error": "Missing 'url' / 'video_url'"}
                print("🔵 RÉPONSE HANDLER :", resp)
                return resp

            local_path = download_video(url)
            resp = {
                "status": "downloaded",
                "path": local_path,
                "size_bytes": os.path.getsize(local_path),
            }
            print("🔵 RÉPONSE HANDLER :", resp)
            return resp

        # 3️⃣ DEBUG PROBE
        if task == "debug_probe":
            if not url:
                resp = {"status": "error", "error": "Missing 'url' / 'video_url'"}
                print("🔵 RÉPONSE HANDLER :", resp)
                return resp

            local_path = download_video(url)
            ff = debug_probe(local_path)

            resp = {
                "status": "probe_ok",
                "file": local_path,
                "probe": ff,
            }
            print("🔵 RÉPONSE HANDLER :", resp)
            return resp

        # 4️⃣ PIPELINE COMPLET = SHORTS
        if task == "process":
            if not url:
                resp = {"status": "error", "error": "Missing 'url' / 'video_url'"}
                print("🔵 RÉPONSE HANDLER :", resp)
                return resp

            print("⬇️ Téléchargement…", url)
            local_path = download_video(url)

            print("🎥 Génération Shorts…")
            clips = generate_shorts(
                input_video_path=local_path,
                num_clips=num_clips,
                min_duration=min_duration,
                max_duration=max_duration,
            )

            resp = {
                "status": "done",
                "clips": clips,
            }
            print("🔵 RÉPONSE HANDLER :", resp)
            return resp

        # 5️⃣ TASK INCONNUE
        resp = {
            "status": "error",
            "error": f"Unknown task '{task}'",
        }
        print("🔵 RÉPONSE HANDLER :", resp)
        return resp

    except Exception as e:
        print("🔥 ERREUR handler:", e)
        print(traceback.format_exc())

        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ===============================
#  DÉMARRAGE RUNPOD
# ===============================

runpod.serverless.start({"handler": handler})
