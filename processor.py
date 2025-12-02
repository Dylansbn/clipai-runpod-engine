import os
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
import whisper
import pysubs2
from openai import OpenAI
import subprocess

# ==============================
# CONFIG BASE
# ==============================
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
SHORTS_DIR = BASE_DIR / "shorts"
SUBS_DIR = BASE_DIR / "subs"

for d in (UPLOADS_DIR, SHORTS_DIR, SUBS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ==============================
# OPENAI CLIENT (project-based)
# ==============================
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT_ID"),
    organization=os.getenv("OPENAI_ORG_ID") or None,
)

# ==============================
# 1. TRANSCRIPTION WHISPER (GPU)
# ==============================
_WHISPER_MODEL = None

def get_whisper_model(model_name: str = "small") -> whisper.Whisper:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _WHISPER_MODEL = whisper.load_model(model_name, device=device)
    return _WHISPER_MODEL


def transcribe_with_whisper(video_path: str, model_name: str = "small") -> Dict[str, Any]:
    model = get_whisper_model(model_name)
    result = model.transcribe(video_path, verbose=False)

    segments = [
        {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg["text"].strip(),
        }
        for seg in result.get("segments", [])
    ]

    return {
        "text": result.get("text", "").strip(),
        "segments": segments,
    }

# ==============================
# 2. IA VIRALE (GPT-4.1)
# ==============================
def select_viral_segments(
    segments: List[Dict[str, Any]],
    num_clips: int = 8,
    min_duration: float = 20.0,
    max_duration: float = 45.0,
    language: str = "fr",
) -> List[Dict[str, Any]]:

    transcript_for_ai = [
        f"[{s['start']:.2f} → {s['end']:.2f}] {s['text']}"
        for s in segments
    ]

    joined = "\n".join(transcript_for_ai)
    # Limite de sécurité pour ne pas exploser le prompt
    joined = joined[:15000]

    system_prompt = (
        "Tu es un expert TikTok/YouTube Shorts. "
        "Tu sélectionnes les moments les plus viraux, avec hook fort, émotions, punchlines. "
        f"Durée cible des extraits : {min_duration}-{max_duration} secondes. "
        "Réponds uniquement en JSON strict, sans texte autour."
    )

    user_prompt = f"""
Transcription avec timecodes :

{joined}

Tâche :
- Choisir les {num_clips} meilleurs moments viraux.
- Durée entre {min_duration} et {max_duration} secondes.
- Hook fort au début de chaque clip.
- Ne coupe pas au milieu d'un mot.
- Réponds en JSON strict de la forme :
{{
  "clips": [
    {{
      "start": float,
      "end": float,
      "title": "Titre court",
      "reason": "Pourquoi ce moment est viral"
    }}
  ]
}}

Réponds en langue : {language}.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = response.output[0].content[0].text

    try:
        data = json.loads(raw)
        clips = data.get("clips", [])
    except Exception:
        clips = []

    final_clips: List[Dict[str, Any]] = []
    for c in clips:
        try:
            start = float(c["start"])
            end = float(c["end"])
            if end <= start:
                continue
            if (end - start) < min_duration or (end - start) > max_duration:
                # on peut filtrer léger, mais on reste souple
                pass

            final_clips.append(
                {
                    "start": start,
                    "end": end,
                    "title": c.get("title", "Clip viral"),
                    "reason": c.get("reason", ""),
                }
            )
        except Exception:
            continue

    return final_clips

# ==============================
# 3. SOUS-TITRES KARAOKÉ STYLE KLAP
# ==============================
def build_karaoke_text(text: str, start_sec: float, end_sec: float) -> str:
    words = [w for w in text.strip().split() if w]
    if not words:
        return ""

    duration_ms = max(int((end_sec - start_sec) * 1000), 1)
    per_word = max(duration_ms // len(words), 1)

    parts = [f"{{\\k{per_word}}}{w}" for w in words]
    return " ".join(parts)


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
    style.fontsize = 64
    style.bold = True
    style.borderstyle = 3           # fond opaque
    style.outline = 6               # gros contour noir
    style.shadow = 3
    style.primarycolor = pysubs2.Color.from_hex("#FFD400")  # jaune
    style.outlinecolor = pysubs2.Color.from_hex("#000000")  # noir
    style.backcolor = pysubs2.Color.from_hex("#00000080")   # fond noir semi-transparent
    style.marginv = 40
    style.alignment = 2             # bas centre
    subs.styles[style.name] = style

    for seg in segments:
        s_start = seg["start"]
        s_end = seg["end"]

        if s_end <= clip_start or s_start >= clip_end:
            continue

        local_start = max(s_start, clip_start) - clip_start
        local_end = min(s_end, clip_end) - clip_start

        if local_end <= local_start:
            continue

        kar = build_karaoke_text(seg["text"], s_start, s_end)
        if not kar:
            continue

        ev = pysubs2.SSAEvent()
        ev.start = int(local_start * 1000)
        ev.end = int(local_end * 1000)
        ev.style = "Klap"
        ev.text = kar

        subs.events.append(ev)

    subs.save(str(subs_path), format="ass")

# ==============================
# 4. FFMPEG GPU : CUT + 9:16 + SUBS
# ==============================
def ffmpeg_extract_and_style(
    input_video: Path,
    output_video: Path,
    subs_path: Path,
    start: float,
    end: float,
):
    duration = max(end - start, 0.5)

    # Filtre 9:16 + sous-titres
    vf = (
        "scale=w=-1:h=1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
        f"subtitles='{subs_path}'"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(input_video),
        "-t",
        f"{duration:.3f}",
        "-vf",
        vf,
        "-c:v",
        "h264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        str(output_video),
    ]

    subprocess.run(cmd, check=True)

# ==============================
# 5. PIPELINE GLOBAL : 1 vidéo → N shorts
# ==============================
def generate_shorts(
    input_video_path: str,
    num_clips: int = 8,
    min_duration: float = 20.0,
    max_duration: float = 45.0,
    language: str = "fr",
) -> List[Dict[str, Any]]:

    video = Path(input_video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    # 1) Transcription
    transcription = transcribe_with_whisper(str(video))
    segments = transcription["segments"]

    if not segments:
        return []

    # 2) IA virale pour choisir les meilleurs moments
    viral_clips = select_viral_segments(
        segments=segments,
        num_clips=num_clips,
        min_duration=min_duration,
        max_duration=max_duration,
        language=language,
    )

    results: List[Dict[str, Any]] = []

    # 3) Génération des shorts
    for i, clip in enumerate(viral_clips, start=1):
        start = clip["start"]
        end = clip["end"]

        out_video = SHORTS_DIR / f"short_{i:02d}.mp4"
        out_subs = SUBS_DIR / f"short_{i:02d}.ass"

        generate_ass_subs_for_clip(start, end, segments, out_subs)

        ffmpeg_extract_and_style(
            input_video=video,
            output_video=out_video,
            subs_path=out_subs,
            start=start,
            end=end,
        )

        results.append(
            {
                "index": i,
                "title": clip["title"],
                "reason": clip.get("reason", ""),
                "start": start,
                "end": end,
                "video_path": str(out_video),
                "subs_path": str(out_subs),
            }
        )

    return results
