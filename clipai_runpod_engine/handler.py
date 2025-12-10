import uuid
import traceback
from typing import Any, Dict

import runpod
from clipai_runpod_engine.engine.worker import process_job


def handler(event: Dict[str, Any]):
    print("📩 EVENT:", event)

    try:
        inp = event.get("input", {})
        video_url = inp.get("video_url")
        num_clips = int(inp.get("num_clips", 3))

        if not video_url:
            return {"status": "error", "error": "Missing video_url"}

        job_id = str(uuid.uuid4())
        print(f"🚀 JOB START: {job_id}")

        output = process_job(job_id, video_url, num_clips)

        return {
            "status": "success",
            "job_id": job_id,
            "output": output
        }

    except Exception as e:
        print("🔥 SERVERLESS ERROR")
        print(traceback.format_exc())
        return {"status": "error", "error": str(e)}

# Très important : démarre le mode serverless
runpod.serverless.start({"handler": handler})
