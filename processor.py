import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any

import pysubs2
import ffmpeg               # ffmpeg-python
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
# 1. TRANSCRIPTION WHISPER — FIX WAV
# ==============================

def transcribe_with_whisper(video_path: str) -> Dict[str, Any]:
    """
    Convertit la vidéo en WAV (format compatible) avant l’envoi à Whisper API.
    Corrige l’erreur : "audio file cannot be decoded".
    """

    print("🎧 Conversion vidéo → WAV pour Whisper...")

    wav_path = str(Path(video_path).with_suffix(".wav"))

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        wav_path
    ]

    subprocess.run(cmd, check=True)

    print(f"🔊 WAV généré : {wav_path}")

    with open(wav_path, "rb") as f:
        res = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json"
        )

    segments = []
    for s in res.segments:
        segments.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": s.text.strip()
        })

    return {
        "text": res.text.strip(),
        "segments": segments
    }


# ==============================
# 2. IA VIRALE (GPT MINIMAL)
# ==============================

def select_viral_segments(
    segments: List[Dict[str, Any]],
    num_clips: int = 8,
    min_duration: float = 20.0,
    max_duration: float = 45.0,
    language: str = "fr"
) -> List[Dict[str, Any]]:

    if not segments:
        return []

    transcript_for_ai = [
        f"[{s['start']:.2f} → {s['end']:.2f}] {s['text']}"
        for s in segments
    ]
    joined = "\n".join(transcript_for_ai)[:15000]

    system_prompt = (
        "Tu es un expert TikTok/Shorts. "
        "Sélectionne les moments les plus viraux. "
        f"Durée entre {min_duration} et {max_duration} secondes. "
        "Réponds STRICTEMENT en JSON."
    )

    user_prompt = f"""
Transcription :

{joined}

Réponds sous forme JSON :
{{
  "clips": [
    {{"start": 12.1, "end": 34.5, "title": "Titre", "reason": "Pourquoi"}} 
  ]
}}
"""

    print("🤖 GPT sélection virale...")

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    raw = completion.choices[0].message["content"]
    print("📝 JSON IA :", raw[:400])

    try:
        clips = json.loads(raw)["clips"]
    except:
        clips = []

    final = []
    for c in clips:
        try:
            if float(c["end"]) > float(c["start"]):
                final.append({
                    "start": float(c["start"]),
                    "end": float(c["end"]),
                    "title": c.get("title", ""),
                    "reason": c.get("reason", "")
                })
        except:
            pass

    print("✅ Clips viraux :", len(final))
    return final


# ==============================
# 3. SOUS-TITRES STYLE KLAP (KARAOKE)
# ==============================

def build_karaoke_text(text: str, start_sec: float, end_sec: float) -> str:
    words = [w for w in text.strip().split() if w]
    if not words:
        return ""

    duration_ms = max(int((end_sec - start_sec) * 1000), 1)
    per_word = max(duration_ms // len(words), 1)

    return " ".join([f"{{\\k{per_word}}}{w}" for w in words])


def generate_ass_subs_for_clip(
    clip_start: float,
    clip_end: float,
    segments: List[Dict[str, Any]],
    subs_path: Path,
):

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

        kar = build_karaoke_text(seg["text"], seg["start"], seg["end"])
        if not kar:
            continue

        ev = pysubs2.SSAEvent()
        ev.start = int(start * 1000)
        ev.end = int(end * 1000)
        ev.style = "Klap"
        ev.text = kar
        subs.events.append(ev)

    subs.save(str(subs_path))


# ==============================
# 4. FFMPEG : CROP + SUBS + EXPORT
# ==============================

def ffmpeg_extract_and_style(input_video: Path, output_video: Path, subs: Path, start: float, end: float):
    duration = max(end - start, 0.5)

    vf = f"scale=-2:1920,crop=1080:1920,subtitles='{subs}'"

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start}",
        "-i", str(input_video),
        "-t", f"{duration}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "160k",
        str(output_video)
    ]

    print("🎬 FFmpeg :", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ==============================
# 5. PIPELINE GLOBAL
# ==============================

def generate_shorts(input_video_path: str, num_clips: int = 8, min_duration: float = 20, max_duration: float = 45):

    video = Path(input_video_path)
    if not video.exists():
        raise FileNotFoundError(video)

    print("🚀 Lancement pipeline…")

    # 1 — Whisper API (avec conversion WAV)
    transcription = transcribe_with_whisper(str(video))
    segments = transcription["segments"]

    # 2 — Clips IA viraux
    viral = select_viral_segments(segments, num_clips, min_duration, max_duration)

    outputs = []

    # 3 — Génération vidéo finale
    for i, clip in enumerate(viral, start=1):
        out_vid = SHORTS_DIR / f"short_{i:02d}.mp4"
        out_ass = SUBS_DIR / f"short_{i:02d}.ass"

        print(f"▶️ Clip {i} | {clip['start']} → {clip['end']}")

        generate_ass_subs_for_clip(clip["start"], clip["end"], segments, out_ass)

        ffmpeg_extract_and_style(video, out_vid, out_ass, clip["start"], clip["end"])

        outputs.append({
            "index": i,
            "title": clip.get("title", ""),
            "reason": clip.get("reason", ""),
            "video_path": str(out_vid),
            "subs_path": str(out_ass)
        })

    print("🎉 Shorts générés :", len(outputs))
    return outputs
