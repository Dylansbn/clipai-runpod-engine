import os
import traceback
from typing import Any, Dict

import runpod

from processor import (
    download_video,
    generate_shorts,
)


# ===============================
#  HANDLER PRINCIPAL RUNPOD
# ===============================

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compatible avec TON frontend et Curl :

    event = {
        "input": {
            "url": "...",
            "video_url": "...",
            "task": "process" | "ping" | "debug_download" | "debug_probe",
            "num_clips": 3,
            "min_duration": 6,
            "max_duration": 25
        }
    }
    """

    print("📩 EVENT REÇU :", event)

    try:
        inp = event.get("input", {})

        # On accepte "url" ou "video_url"
        url = inp.get("video_url") or inp.get("url")

        # Task par défaut
        task = inp.get("task")
        if not task:
            task = "process" if url else "ping"

        num_clips = int(inp.get("num_clips", 3))
        min_duration = float(inp.get("min_duration", 6))
        max_duration = float(inp.get("max_duration", 25))

        # -------------------------
        # 1️⃣ TEST : ping simple
        # -------------------------
        if task == "ping":
            return {
                "status": "ok",
                "message": "ClipAI Engine Alive 🔥",
                "version": "serverless-pro"
            }

        # -------------------------
        # 2️⃣ Téléchargement simple
        # -------------------------
        if task == "debug_download":
            if not url:
                return {"status": "error", "error": "Missing URL"}

            local_path = download_video(url)
            return {
                "status": "downloaded",
                "local_path": local_path,
                "size_bytes": os.path.getsize(local_path)
            }

        # -------------------------
        # 3️⃣ PIPELINE COMPLET
        # -------------------------
        if task == "process":
            if not url:
                return {"status": "error", "error": "Missing URL"}

            print("⬇️ Téléchargement…", url)
            local_path = download_video(url)

            print("🎥 Génération des shorts…")
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
        # 4️⃣ TASK INCONNUE
        # -------------------------
        return {"status": "error", "error": f"Unknown task: {task}"}

    except Exception as e:
        print("🔥 ERREUR handler :", e)
        print(traceback.format_exc())

        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# ===============================
#  RUNPOD ENTRYPOINT
# ===============================
runpod.serverless.start({"handler": handler})
