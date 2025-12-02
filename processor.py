import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any

import pysubs2
from openai import OpenAI

# ==============================
#  CONFIG & DOSSIERS
# ==============================

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
SHORTS_DIR = BASE_DIR / "shorts"
SUBS_DIR = BASE_DIR / "subs"

for d in (UPLOADS_DIR, SHORTS_DIR, SUBS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ==============================
#  CLIENT OPENAI
# ==============================

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT_ID")
)

# ==============================
# 1. TRANSCRIPTION — WHISPER API (CORRIGÉ)
# ==============================

def transcribe_with_whisper(video_path: str) -> Dict[str, Any]:
    """
    Utilise Whisper API officielle.
    CORRECTION : segments = objets → utiliser seg.start et seg.end
    """
    print("🎙️ Envoi vidéo → Whisper API ...")

    with open(video_path, "rb") as f:
        res = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json"
        )

    # CORRECTION ICI : résout l'erreur "object is not subscriptable"
    segments = []
    for seg in res.segments:
        segments.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip()
        })

    print(f"🧩 Segments détectés : {len(segments)}")

    return {
        "text": res.text.strip(),
        "segments": segments
    }

# ==============================
# 2. IA VIRALE — GPT-4.1-mini
# ==============================

def select_viral_segments(
    segments: List[Dict[str, Any]],
    num_clips: int = 8,
    min_duration: float = 20,
    max_duration: float = 45,
    language: str = "fr"
) -> List[Dict[str, Any]]:

    if not segments:
        return []

    transcript_for_ai = [
        f"[{s['start']:.2f}→{s['end']:.2f}] {s['text']}"
        for s in segments
    ]
    joined = "\n".join(transcript_for_ai)[:15000]

    system_prompt = (
        "Tu es un expert TikTok/YouTube Shorts. "
        "Sélectionne les moments les plus viraux. "
        "Réponds STRICTEMENT en JSON."
    )

    user_prompt = f"""
Transcription complète :

{joined}

Sélectionne les {num_clips} meilleurs moments (durée {min_duration}–{max_duration}s).
Réponds en JSON :

{{
  "clips": [
    {{"start": 12.5, "end": 34.1, "title": "Hook", "reason": "Pourquoi c'est viral"}}
  ]
}}
"""

    print("🤖 Appel IA virale...")

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )

    raw = response.output[0].content[0].text.strip()

    print("📥 JSON IA reçu (début) :", raw[:300])

    try:
        clips = json.loads(raw)["clips"]
    except:
        clips = []

    final = []
    for c in clips:
        try:
            if float(c["end"]) > float(c["start"]):
                final.append(c)
        except:
            pass

    print(f"🔥 Clips retenus : {len(final)}")
    return final

# ==============================
# 3. SOUS-TITRES KLAP — ASS + KARAOKÉ
# ==============================

def build_karaoke_text(text: str, start: float, end: float) -> str:
    words = text.strip().split()
    if not words:
        return ""

    duration = (end - start) * 1000
    per_word = max(int(duration / len(words)), 1)

    return " ".join([f"{{\\k{per_word}}}{w}" for w in words])


def generate_ass_subs_for_clip(start: float, end: float, segments, subs_path: Path):
    subs = pysubs2.SSAFile()

    style = pysubs2.SSAStyle()
    style.name = "Klap"
    style.fontname = "Poppins"
    style.fontsize = 58
    style.bold = True
    style.outline = 4
    style.alignment = 2
    style.primarycolor = pysubs2.Color(255, 255, 0)
    style.outlinecolor = pysubs2.Color(0, 0, 0)

    subs.styles["Klap"] = style

    for seg in segments:
        if seg["end"] <= start or seg["start"] >= end:
            continue

        local_start = max(seg["start"], start) - start
        local_end = min(seg["end"], end) - start

        kar = build_karaoke_text(seg["text"], seg["start"], seg["end"])

        event = pysubs2.SSAEvent(
            start=int(local_start * 1000),
            end=int(local_end * 1000),
            text=kar,
            style="Klap"
        )
        subs.events.append(event)

    subs.save(str(subs_path))

# ==============================
# 4. FFMPEG — FORMAT 9:16 + SUBS
# ==============================

def ffmpeg_extract_and_style(video: Path, out_vid: Path, subs: Path, start: float, end: float):
    duration = max(end - start, 0.5)

    vf = f"scale=-2:1920,crop=1080:1920,subtitles='{subs}'"

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start}",
        "-i", str(video),
        "-t", f"{duration}",
        "-vf", vf,
        "-preset", "veryfast",
        "-c:v", "libx264",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "160k",
        str(out_vid)
    ]

    print("🎬 FFmpeg :", " ".join(cmd))
    subprocess.run(cmd, check=True)

# ==============================
# 5. PIPELINE GLOBAL
# ==============================

def generate_shorts(input_video_path: str, num_clips=8, min_duration=20, max_duration=45):

    video = Path(input_video_path)
    if not video.exists():
        raise FileNotFoundError(video)

    print("🚀 Lancement pipeline sur :", video)

    # 1. Transcription
    transcription = transcribe_with_whisper(str(video))
    segments = transcription["segments"]

    # 2. IA virale
    clips = select_viral_segments(segments, num_clips, min_duration, max_duration)

    results = []

    for i, c in enumerate(clips, start=1):
        out_vid = SHORTS_DIR / f"short_{i:02d}.mp4"
        out_ass = SUBS_DIR / f"short_{i:02d}.ass"

        print(f"🎯 Clip {i} : {c['start']} → {c['end']}")

        generate_ass_subs_for_clip(c["start"], c["end"], segments, out_ass)
        ffmpeg_extract_and_style(video, out_vid, out_ass, c["start"], c["end"])

        results.append({
            "index": i,
            "title": c.get("title", ""),
            "reason": c.get("reason", ""),
            "video_path": str(out_vid),
            "subs_path": str(out_ass)
        })

    print("🎉 Pipeline terminé :", len(results), "shorts générés")
    return results
