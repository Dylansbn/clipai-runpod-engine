import runpod
import uuid
from job_queue.file_queue import push_job


def handler(event):
    """
    Handler RunPod Serverless
    → Reçoit une requête
    → Crée un job
    → L’ajoute dans la queue
    → Le worker GPU le traitera
    """
    
    print("📩 EVENT REÇU :", event)

    inp = event.get("input", {})

    video_url = inp.get("video_url")
    num_clips = int(inp.get("num_clips", 3))

    if not video_url:
        return {
            "status": "error",
            "message": "Missing video_url"
        }

    # Créer l'ID unique du job
    job_id = str(uuid.uuid4())

    # Ajouter le job dans ta file JSON
    push_job({
        "job_id": job_id,
        "video_url": video_url,
        "num_clips": num_clips
    })

    print(f"📌 Job créé : {job_id}")

    # Répond immédiatement (serverless)
    return {
        "status": "queued",
        "job_id": job_id
    }


# Lancer le serveur RunPod
runpod.serverless.start({"handler": handler})
