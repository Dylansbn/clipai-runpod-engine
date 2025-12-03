import traceback
from typing import Any, Dict

import runpod
from processor import generate_shorts, download_video


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = event.get("input", {})
        task = inp.get("task", "ping")

        # 1) TEST
        if task == "ping":
            return {
                "status": "ok",
                "message": "clipai-runpod-engine is alive 🔥"
            }

        # 2) TRAITEMENT VIDÉO
        if task == "process":
            url = inp.get("video_url")
            if not url:
                return {"status": "error", "error": "Missing video_url"}

            num_clips = int(inp.get("num_clips", 8))
            min_d = float(inp.get("min_duration", 20))
            max_d = float(inp.get("max_duration", 45))

            # Téléchargement robuste
            local_path = download_video(url)

            # Génération shorts
            clips = generate_shorts(
                input_video_path=local_path,
                num_clips=num_clips,
                min_duration=min_d,
                max_duration=max_d
            )

            return {"status": "done", "clips": clips}

        # 3) TASK INCONNU
        return {"status": "error", "error": f"Unknown task '{task}'"}

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# Start RunPod
runpod.serverless.start({"handler": handler})
