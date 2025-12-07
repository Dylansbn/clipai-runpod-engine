import os
import traceback
from typing import Any, Dict

import runpod

from processor import (
    download_video,
    generate_shorts,
)

# ============================================
#  HANDLER PRINCIPAL — VERSION STABLE (RUNSYNC)
# ============================================

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    print("📩 EVENT REÇU :", event)

    try:
        inp = event.get("input", {})
        if not isinstance(inp, dict):
            return {"status": "error", "error": "Invalid input payload"}

        # -------------------------
        # Champs reçus
        # -------------------------
        url = inp.get("video_url") or inp.get("url")
        task = inp.get("task") or ("process" if url else "ping")

        num_clips = int(inp.get("num_clips", 3))

        # DURÉES KLAP v3 (stables)
        min_duration = float(inp.get("min_duration", 10))
        max_duration = float(inp.get("max_duration", 50))

        print(f"🔧 Task: {task}")
        print(f"🎞 URL: {url}")
        print(f"🎬 Clips demandés: {num_clips} ({min_duration}s → {max_duration}s)")

        # ============================================
        # 1️⃣ PING
        # ============================================
        if task == "ping":
            return {
                "status": "ok",
                "message": "ClipAI Engine Alive 🔥",
                "version": "stable-runsync"
            }

        # ============================================
        # 2️⃣ DEBUG DOWNLOAD
        # ============================================
        if task == "debug_download":
            if not url:
                return {"status": "error", "error": "Missing URL"}

            local_path = download_video(url)
            size = os.path.getsize(local_path)

            return {
                "status": "downloaded",
                "local_path": local_path,
                "size_bytes": size
            }

        # ============================================
        # 3️⃣ PROCESS (pipeline complet)
        # ============================================
        if task == "process":
            if not url:
                return {"status": "error", "error": "Missing URL"}

            print("⬇️ Téléchargement…", url)
            local_path = download_video(url)

            print("🎥 Génération des shorts…")
            clips = generate_shorts(
                video_path=local_path,
                num_clips=num_clips,
            )

            print(f"✅ {len(clips)} clips générés")

            return {
                "status": "done",
                "clips": clips
            }

        # ============================================
        # 4️⃣ Task inconnue
        # ============================================
        return {"status": "error", "error": f"Unknown task: {task}"}        

    except Exception as e:
        print("🔥 ERREUR handler :", e)
        print(traceback.format_exc())

        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# ENTRYPOINT RUNSYNC
runpod.serverless.start({"handler": handler})
