import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import requests
import runpod

from processor import generate_shorts, SHORTS_DIR, SUBS_DIR

# ============== UTIL ==============
def download_video_to_tmp(url: str) -> str:
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()

    suffix = ".mp4"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(tmp_fd, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return tmp_path


# ============== HANDLER RUNPOD ==============
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    event["input"] attendu :
    {
      "video_url": "https://...",
      "num_clips": 8,
      "min_duration": 20,
      "max_duration": 45,
      "language": "fr"
    }
    """

    inp = event.get("input", {}) or {}

    video_url = inp.get("video_url")
    if not video_url:
        return {"error": "Missing 'video_url' in input."}

    num_clips = int(inp.get("num_clips", 8))
    min_duration = float(inp.get("min_duration", 20.0))
    max_duration = float(inp.get("max_duration", 45.0))
    language = inp.get("language", "fr")

    try:
        local_video_path = download_video_to_tmp(video_url)

        shorts = generate_shorts(
            input_video_path=local_video_path,
            num_clips=num_clips,
            min_duration=min_duration,
            max_duration=max_duration,
            language=language,
        )

        # Pour l'instant, on renvoie uniquement les chemins locaux.
        # Étape suivante possible : uploader les shorts vers S3 / Cloudflare et retourner les URLs.
        serialized_shorts = []
        for s in shorts:
            serialized_shorts.append(
                {
                    "index": s["index"],
                    "title": s["title"],
                    "reason": s["reason"],
                    "start": s["start"],
                    "end": s["end"],
                    "video_path": s["video_path"],
                    "subs_path": s["subs_path"],
                }
            )

        return {
            "success": True,
            "shorts": serialized_shorts,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# Démarrage serverless RunPod
runpod.serverless.start({"handler": handler})
