import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any

import pysubs2
from yt_dlp import YoutubeDL
from openai import OpenAI


# ==============================
#  CONFIG
# ==============================

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
SHORTS_DIR = BASE_DIR / "shorts"
SUBS_DIR = BASE_DIR / "subs"

for d in (UPLOADS_DIR, SHORTS_DIR, SUBS_DIR):
    d.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ==============================
# 0. DOWNLOAD ROBUSTE (yt-dlp)
# ==============================

def download_video(url: str) -> str:
    """Télécharge TOUTES les vidéos (YouTube, TikTok, Vimeo, mp4 direct…)."""
    print(f"⬇️ Téléchargement vidéo via yt-dlp : {url}")

    output_file = UPLOADS_DIR / f"input_{os.urandom(4).hex()}.mp4"
    ydl_opts = {
        "outtmpl": str(output_file),
        "format": "mp4/best/bestvideo+bestaudio",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise Exception(f"Erreur téléchargement vidéo (yt-dlp) : {str(e)}")

    if not output_file.exists():
        raise Exception("Téléchargement échoué : fichier non trouvé après yt-dlp")

    print(f"✅ Vidéo téléchargée → {output_file}")
    return str(output_file)


# ==============================
# 1. TRANSCRIPTION WHISPER API
# ==============================

def transcribe_with_whisper(video_path: str) -> Dict[str, Any]:
    print("🎙️ Envoi vidéo → Whisper API ...")

    with open(video_path, "rb") as f:
        res = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json"
        )

    segments = [
        {
            "start": float(s.start),
            "end": float(s.end),
            "text": s.text.strip()
        }
        for s in res.segments
    ]

    return {
        "text": res.text.strip(),
        "segments": segments
    }


# ==============================
# 2. IA VIRALE GPT
# ==============================

def select_viral_segments(
    segments: List[Dict[str, Any]],
    num_clips: int,
    min_duration: float,
    max_duration: float
):
    if not segments:
        return []

    transcript = "\n".join(
        f"[{s['start']:.2f}->{s['end']:.2f}] {s['text']}"
        for s in segments
    )[:15000]

    system_prompt = (
        "Tu es expert TikTok. Sélectionne les meilleurs moments viraux. "
        "Réponds en strict JSON."
    )

    user_prompt = f"""
Transcription :

{transcript}

Choisis {num_clips} clips de {min_duration} à {max_duration} secondes.

Format JSON :
{{
 "clips":[
   {{"start":0,"end":20,"title":"Titre","reason":"..."}}
 ]
}}
"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    data = json.loads(completion.choices[0].message.content)
    clips = data.get("clips", [])

    final = []
    for c in clips:
        try:
            if float(c["end"]) > float(c["start"]):
                final.append({
                    "start": float(c["start"]),
                    "end": float(c["end"]),
                    "title": c.get("title", "Clip"),
                    "reason": c.get("reason", "")
                })
        except:
            pass

    return final


# ==============================
# 3. GÉNÉRATION SOUS-TITRES .ASS
# ==============================

def build_karaoke_text(text, start_sec, end_sec):
    words = text.split()
    if not words:
        return ""
    duration_ms = max((end_sec - start_sec) * 1000, 1)
    per = max(int(duration_ms / len(words)), 1)
    return " ".join([f"{{\\k{per}}}{w}" for w in words])


def generate_ass_subs_for_clip(clip_start, clip_end, segments, subs_path):
    subs = pysubs2.SSAFile()

    style = pysubs2.SSAStyle()
    style.name = "Klap"
    style.fontname = "Poppins"
    style.fontsize = 60
    style.bold = True
    style.outline = 4
    style.primarycolor = pysubs2.Color(255, 255, 0)
    style.outlinecolor = pysubs2.Color(0, 0, 0)
    style.alignment = 2
    subs.styles[style.name] = style

    for seg in segments:
        if seg["end"] <= clip_start or seg["start"] >= clip_end:
            continue

        start = max(seg["start"], clip_start) - clip_start
        end = min(seg["end"], clip_end) - clip_start

        ev = pysubs2.SSAEvent()
        ev.start = int(start * 1000)
        ev.end = int(end * 1000)
        ev.style = "Klap"
        ev.text = build_karaoke_text(seg["text"], seg["start"], seg["end"])
        subs.events.append(ev)

    subs.save(str(subs_path))


# ==============================
# 4. FFMPEG (CUT + CROP + SUBS)
# ==============================

def ffmpeg_extract_and_style(src, out, subs, start, end):
    duration = max(end - start, 0.5)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", src,
        "-t", str(duration),
        "-vf", f"scale=-2:1920,crop=1080:1920,subtitles='{subs}'",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "160k",
        out
    ]

    subprocess.run(cmd, check=True)


# ==============================
# 5. PIPELINE GLOBAL
# ==============================

def generate_shorts(input_video_path, num_clips=8, min_duration=20, max_duration=45):

    tr = transcribe_with_whisper(input_video_path)
    segments = tr["segments"]

    viral = select_viral_segments(segments, num_clips, min_duration, max_duration)

    results = []

    for i, c in enumerate(viral, start=1):
        out_vid = SHORTS_DIR / f"short_{i:02d}.mp4"
        out_ass = SUBS_DIR / f"short_{i:02d}.ass"

        generate_ass_subs_for_clip(c["start"], c["end"], segments, out_ass)
        ffmpeg_extract_and_style(input_video_path, str(out_vid), str(out_ass), c["start"], c["end"])

        results.append({
            "index": i,
            "title": c["title"],
            "start": c["start"],
            "end": c["end"],
            "video_path": str(out_vid),
            "subs_path": str(out_ass)
        })

    return results
