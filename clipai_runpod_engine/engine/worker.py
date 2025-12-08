# ============================================================
# WORKER GPU — VERSION KLAP PRO COMPLÈTEMENT CORRIGÉE
# ============================================================

import time
import traceback

# IMPORT RELATIF (obligatoire avec un package Python)
from ..job_queue.file_queue import pop_job

# Imports internes du moteur
from .video_analyzer import analyze_video, download_video
from .whisper_gpu import transcribe_gpu
from .clip_selector import select_clips
from .render import render_clips
from .storage import upload_results


def worker_loop():
    print("🚀 Worker GPU démarré — moteur KLAP PRO opérationnel\n")

    while True:
        # ------------------------------------------------------------
        # 1️⃣ Récupérer un job dans la file
        # ------------------------------------------------------------
        job = pop_job()

        if not job:
            time.sleep(1)
            continue

        job_id = job["job_id"]
        video_url = job["video_url"]
        num_clips = job["num_clips"]

        print("\n==============================================")
        print(f"🎬 NOUVEAU JOB : {job_id}")
        print("==============================================")
        print(f"📹 URL vidéo : {video_url}")

        try:
            # ------------------------------------------------------------
            # 2️⃣ Téléchargement vidéo
            # ------------------------------------------------------------
            print("⬇️ Téléchargement de la vidéo...")
            local_path = download_video(video_url)

            # ------------------------------------------------------------
            # 3️⃣ Analyse vidéo : peaks & énergie visuelle
            # ------------------------------------------------------------
            print("📊 Analyse vidéo...")
            analysis = analyze_video(local_path)

            # ------------------------------------------------------------
            # 4️⃣ Transcription Whisper GPU
            # ------------------------------------------------------------
            print("🎧 Transcription (Whisper GPU)...")
            segments = transcribe_gpu(local_path)

            # ------------------------------------------------------------
            # 5️⃣ Sélection IA (texte + analyse visuelle)
            # ------------------------------------------------------------
            print("🧠 Sélection des meilleurs moments...")
            clips = select_clips(segments, analysis, num_clips)

            # ------------------------------------------------------------
            # 6️⃣ Rendu vidéo + sous-titres KLAP PRO
            # ------------------------------------------------------------
            print("🎬 Rendu des clips (NVENC + sous-titres)...")
            outputs = render_clips(local_path, clips, segments)

            # ------------------------------------------------------------
            # 7️⃣ Upload final vers Cloudflare R2
            # ------------------------------------------------------------
            print("☁️ Upload R2 des clips rendus...")
            urls = upload_results(job_id, outputs)

            print(f"✅ JOB TERMINÉ → {job_id}")
            print(f"🌐 URLs générées : {urls}\n")

        except Exception as e:
            print(f"🔥 ERREUR dans le job {job_id} !")
            print(traceback.format_exc())

        # Petite pause entre deux jobs
        time.sleep(0.5)


# Mode exécution directe
if __name__ == "__main__":
    worker_loop()
