import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any

import whisper              # openai-whisper
import pysubs2
import ffmpeg               # ffmpeg-python, utilisé si besoin
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

if client.api_key is None:
    print("⚠️  OPENAI_API_KEY manquant dans les variables d'environnement.")

# ==============================
# 1. TRANSCRIPTION WHISPER
# ==============================

_WHISPER_MODEL = None  # cache global pour éviter de recharger à chaque job


def get_whisper_model(model_name: str = "small"):
    """
    Charge le modèle Whisper une seule fois (cache global).
    Pas d'annotation de type en whisper.Whisper => évite le bug AttributeError.
    """
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        print(f"🧠 Chargement du modèle Whisper '{model_name}'...")
        _WHISPER_MODEL = whisper.load_model(model_name)
        print("✅ Whisper chargé.")
    return _WHISPER_MODEL


def transcribe_with_whisper(video_path: str, model_name: str = "small") -> Dict[str, Any]:
    """
    Transcription de la vidéo avec timecodes par segment.
    """
    model = get_whisper_model(model_name)
    print(f"🎙️ Transcription de {video_path} ...")
    result = model.transcribe(video_path, verbose=False)

    segments = [
        {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg["text"].strip()
        }
        for seg in result.get("segments", [])
    ]

    return {
        "text": result.get("text", "").strip(),
        "segments": segments,
    }

# ==============================
# 2. IA VIRALE (GPT-4.1-mini)
# ==============================


def select_viral_segments(
    segments: List[Dict[str, Any]],
    num_clips: int = 8,
    min_duration: float = 20.0,
    max_duration: float = 45.0,
    language: str = "fr"
) -> List[Dict[str, Any]]:
    """
    Utilise GPT pour choisir les meilleurs moments viraux.
    Retourne une liste de dicts : {start, end, title, reason}
    """

    if not segments:
        return []

    transcript_for_ai = [
        f"[{s['start']:.2f} → {s['end']:.2f}] {s['text']}"
        for s in segments
    ]
    joined = "\n".join(transcript_for_ai)[:15000]

    system_prompt = (
        "Tu es un expert TikTok/YouTube Shorts. "
        "Tu sélectionnes les moments les plus viraux, avec un hook fort au début. "
        f"Chaque clip doit durer entre {min_duration} et {max_duration} secondes. "
        "Réponds STRICTEMENT en JSON."
    )

    user_prompt = f"""
Transcription avec timecodes :

{joined}

Tâche :
- Choisir les {num_clips} meilleurs moments viraux.
- Durée entre {min_duration} et {max_duration} secondes.
- Hook fort dès les 3 premières secondes.
- Ne coupe pas au milieu d'un mot.
- Réponds uniquement en JSON de la forme :

{{
  "clips": [
    {{"start": 12.5, "end": 34.0, "title": "Titre accrocheur", "reason": "Pourquoi c'est viral"}},
    ...
  ]
}}

Réponds en {language}.
"""

    print("🤖 Appel GPT pour sélection virale...")
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_text = response.output[0].content[0].text
    # Debug léger dans les logs
    print("🔎 Réponse brute GPT (début) :", raw_text[:500])

    try:
        data = json.loads(raw_text)
        clips = data.get("clips", [])
    except Exception as e:
        print("⚠️ Impossible de parser le JSON renvoyé par GPT :", e)
        clips = []

    final_clips: List[Dict[str, Any]] = []

    for c in clips:
        try:
            start = float(c["start"])
            end = float(c["end"])
            if end <= start:
                continue

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

    print(f"✅ Clips viraux sélectionnés : {len(final_clips)}")
    return final_clips

# ==============================
# 3. SOUS-TITRES KARAOKÉ (STYLE KLAP)
# ==============================


def build_karaoke_text(text: str, start_sec: float, end_sec: float) -> str:
    """
    Convertit une phrase en séquence karaoke \kxxx pour ASS.
    """
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
    """
    Crée un fichier .ass avec un style centré, jaune, outline noir, karaoke.
    """
    subs = pysubs2.SSAFile()

    style = pysubs2.SSAStyle()
    style.name = "Klap"
    style.fontname = "Poppins"
    style.fontsize = 58
    style.bold = True
    style.borderstyle = 1
    style.outline = 4
    style.primarycolor = pysubs2.Color(255, 255, 0)  # jaune
    style.outlinecolor = pysubs2.Color(0, 0, 0)      # noir
    style.shadow = 0
    style.alignment = 2  # centre bas
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
# 4. FFMPEG : CUT + FORMAT 9:16 + SUBS
# ==============================


def ffmpeg_extract_and_style(
    input_video: Path,
    output_video: Path,
    subs_path: Path,
    start: float,
    end: float,
):
    """
    Utilise ffmpeg pour :
    - couper au bon timing
    - mettre au format 9:16 (1080x1920)
    - incruster les sous-titres ASS
    """
    duration = max(end - start, 0.5)

    # filtre vidéo : mise à l'échelle puis crop 9:16 + sous-titres
    vf = f"scale=-2:1920,crop=1080:1920,subtitles='{subs_path}'"

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
        "libx264",
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

    print("🎬 Commande ffmpeg :", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ==============================
# 5. PIPELINE GLOBAL
# ==============================


def generate_shorts(
    input_video_path: str,
    num_clips: int = 8,
    min_duration: float = 20.0,
    max_duration: float = 45.0,
) -> List[Dict[str, Any]]:
    """
    Pipeline complet :
    - transcription Whisper
    - sélection IA virale
    - génération sous-titres .ass
    - ffmpeg 9:16 avec sous-titres incrustés
    """
    video = Path(input_video_path)
    if not video.exists():
        raise FileNotFoundError(video)

    print(f"🚀 Démarrage pipeline sur {video}")

    # 1. Transcription
    t = transcribe_with_whisper(str(video))
    segs = t["segments"]

    # 2. IA virale (GPT)
    viral_clips = select_viral_segments(
        segments=segs,
        num_clips=num_clips,
        min_duration=min_duration,
        max_duration=max_duration,
    )

    results: List[Dict[str, Any]] = []

    # 3. Génération des shorts
    for i, clip in enumerate(viral_clips, start=1):
        start = clip["start"]
        end = clip["end"]

        out_video = SHORTS_DIR / f"short_{i:02d}.mp4"
        out_subs = SUBS_DIR / f"short_{i:02d}.ass"

        print(f"▶️ Clip {i} | {start:.2f}s → {end:.2f}s")

        # Sous-titres
        generate_ass_subs_for_clip(start, end, segs, out_subs)

        # ffmpeg
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

    print("✅ Pipeline terminé, shorts générés :", len(results))
    return results
