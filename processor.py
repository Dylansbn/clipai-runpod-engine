import os
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urlparse

import pysubs2
import yt_dlp
import boto3
import requests
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
    project=os.getenv("OPENAI_PROJECT_ID"),
)

# ==============================
#  CLIENT R2 (S3 compatible)
# ==============================

def get_r2_client():
    """Retourne un client S3 vers Cloudflare R2 si tout est configuré, sinon None."""
    access = os.getenv("R2_ACCESS_KEY_ID")
    secret = os.getenv("R2_SECRET_ACCESS_KEY")
    endpoint = os.getenv("R2_ENDPOINT_URL")

    if not (access and secret and endpoint):
        print("ℹ️ R2 non configuré (variables manquantes), on garde les chemins locaux.")
        return None

    session = boto3.session.Session()
    s3 = session.client(
        service_name="s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name="auto",
    )
    return s3


def upload_to_r2(local_path: str, key_prefix: str = "") -> str:
    """
    Upload un fichier vers R2.
    - key_prefix : ex. 'shorts/' ou 'subs/'
    Retourne soit l'URL publique (si possible), soit le chemin local si R2 n'est pas configuré.
    """
    s3 = get_r2_client()
    bucket = os.getenv("R2_BUCKET_NAME")

    if not s3 or not bucket:
        # Pas de R2 → on renvoie juste le chemin local
        return local_path

    local = Path(local_path)
    key_prefix = key_prefix.strip("/")
    if key_prefix:
        key = f"{key_prefix}/{local.name}"
    else:
        key = local.name

    print(f"☁️ Upload vers R2 : bucket={bucket}, key={key}")
    content_type = "video/mp4" if local.suffix.lower() == ".mp4" else "text/plain"

    s3.upload_file(
        Filename=str(local),
        Bucket=bucket,
        Key=key,
        ExtraArgs={"ContentType": content_type},
    )

    public_base = os.getenv("R2_PUBLIC_BASE_URL")
    if public_base:
        url = f"{public_base.rstrip('/')}/{key}"
    else:
        # URL S3-style (fonctionnera si tu rends le bucket public)
        endpoint = os.getenv("R2_ENDPOINT_URL", "").rstrip("/")
        url = f"{endpoint}/{bucket}/{key}"

    print(f"✅ Upload R2 OK → {url}")
    return url


# ==============================
# 0. TÉLÉCHARGEMENT ROBUSTE (YouTube / TikTok / Vimeo / HTTP direct)
# ==============================

def download_video(url: str) -> str:
    """
    Télécharge une vidéo depuis :
    - une URL directe (.mp4, Cloudflare R2, S3…)
    - ou une plateforme (YouTube, TikTok, etc.) via yt-dlp.

    Retourne le chemin local du .mp4 dans uploads/.
    """
    print(f"⬇️ [DOWNLOAD] URL : {url}")
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()

    # Fichier de destination local
    dest = UPLOADS_DIR / f"input_{os.urandom(4).hex()}.mp4"

    # 1️⃣ Cas : URL directe (R2, S3, .mp4)
    if url.endswith(".mp4") or "r2.cloudflarestorage.com" in host:
        print("📥 Téléchargement direct HTTP (R2 / .mp4) ...")
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            print(f"❌ Téléchargement direct HTTP a échoué : {e}")
            raise Exception(f"Téléchargement direct HTTP a échoué : {e}")

    else:
        # 2️⃣ Cas : plateforme (YouTube / TikTok / etc.) → yt-dlp
        try:
            ua = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )

            ydl_opts = {
                "outtmpl": str(dest),
                "format": "bv*+ba/best/b",
                "merge_output_format": "mp4",
                "noplaylist": True,
                "quiet": True,
                "geo_bypass": True,
                "geo_bypass_country": "US",
                "nocheckcertificate": True,
                "retries": 3,
                "user_agent": ua,
                "http_headers": {
                    "User-Agent": ua,
                    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
                },
            }

            # TikTok un peu plus sensible → Referer
            if "tiktok.com" in host:
                ydl_opts["http_headers"]["Referer"] = "https://www.tiktok.com/"

            print("📥 Tentative yt-dlp ...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            print(f"⚠️ yt-dlp a échoué : {e}")
            raise Exception(f"yt-dlp a échoué : {e}")

    # 3️⃣ Vérification du fichier téléchargé
    if not dest.exists() or dest.stat().st_size < 100_000:
        raise Exception("Téléchargement échoué ou fichier trop petit pour Whisper.")

    # 4️⃣ Vérif rapide via ffprobe
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        str(dest),
    ]
    probe = subprocess.run(
        probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if probe.returncode != 0:
        raise Exception("ffprobe: fichier vidéo illisible ou non supporté.")

    print(f"✅ Vidéo téléchargée OK → {dest} ({dest.stat().st_size} bytes)")
    return str(dest)


# ==============================
# 1. TRANSCRIPTION WHISPER API
# ==============================

def transcribe_with_whisper(video_path: str) -> Dict[str, Any]:
    """Transcription via Whisper API (pas de modèle local)."""
    print("🎙️ Envoi vidéo → Whisper API ...")

    with open(video_path, "rb") as f:
        res = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
        )

    segments = []
    for s in res.segments:
        segments.append(
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip(),
            }
        )

    print(f"📌 Segments générés : {len(segments)}")

    return {"text": res.text.strip(), "segments": segments}


# ==============================
# 2. IA VIRALE (GPT) — sélection des clips
# ==============================

def select_viral_segments(
    segments: List[Dict[str, Any]],
    num_clips: int = 8,
    min_duration: float = 20.0,
    max_duration: float = 45.0,
) -> List[Dict[str, Any]]:
    if not segments:
        return []

    transcript = "\n".join(
        f"[{s['start']:.2f} → {s['end']:.2f}] {s['text']}" for s in segments
    )[:15000]

    system_prompt = (
        "Tu es un expert TikTok/Shorts. "
        "Tu choisis les moments les plus viraux avec un hook fort. "
        "Réponds STRICTEMENT en JSON : "
        "{\"clips\": [{\"start\": x, \"end\": y, \"title\": \"\", \"reason\": \"\"}]}"
    )

    user_prompt = f"""
Transcription :

{transcript}

Choisis {num_clips} clips de {min_duration}-{max_duration} secondes.
Réponds en JSON strict.
"""

    print("🤖 Appel GPT...")
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    raw = completion.choices[0].message.content
    print("🔎 JSON IA reçu (début) :", raw[:300])

    try:
        data = json.loads(raw)
        clips = data.get("clips", [])
    except Exception:
        clips = []

    final = []
    for c in clips:
        try:
            s = float(c["start"])
            e = float(c["end"])
            if e > s and (e - s) >= min_duration:
                final.append(
                    {
                        "start": s,
                        "end": e,
                        "title": c.get("title", "Clip viral"),
                        "reason": c.get("reason", ""),
                    }
                )
        except Exception:
            pass

    print(f"✅ Clips retenus : {len(final)}")
    return final


# ==============================
# 3. SOUS-TITRES KLAP PRO
# ==============================

def _sanitize_ass_text(text: str) -> str:
    cleaned = text.replace("\n", " ").replace("\r", " ")
    cleaned = cleaned.replace("{", "(").replace("}", ")")
    return " ".join(cleaned.split())


def _split_into_lines(words: List[str], max_words_per_line: int = 7) -> List[List[str]]:
    if len(words) <= max_words_per_line:
        return [words]
    mid = len(words) // 2
    return [words[:mid], words[mid:]]


def build_karaoke_text(text: str, start: float, end: float) -> str:
    clean = _sanitize_ass_text(text)
    words = clean.split()
    if not words:
        return ""

    duration_ms = max((end - start) * 1000, 1)
    per_word = max(int(duration_ms / len(words)), 1)

    lines = _split_into_lines(words, max_words_per_line=7)
    ass_lines = []

    for line_words in lines:
        parts = [f"{{\\k{per_word}}}{w}" for w in line_words]
        ass_lines.append(" ".join(parts))

    prefix = r"{\an2\fad(80,120)}"  # align bas centre + fade in/out
    return prefix + r"\N".join(ass_lines)


def generate_ass_subs_for_clip(
    clip_start: float,
    clip_end: float,
    segments: List[Dict[str, Any]],
    subs_path: Path,
):
    subs = pysubs2.SSAFile()

    style = pysubs2.SSAStyle()
    style.name = "KlapMain"
    style.fontname = "Poppins"
    style.fontsize = 64
    style.bold = True

    style.primarycolor = pysubs2.Color(255, 255, 255)  # blanc
    style.secondarycolor = pysubs2.Color(255, 220, 0)  # jaune karaoké
    style.outlinecolor = pysubs2.Color(0, 0, 0)        # noir
    style.backcolor = pysubs2.Color(0, 0, 0, 0)

    style.outline = 5
    style.shadow = 0
    style.borderstyle = 1

    style.marginl = 40
    style.marginr = 40
    style.marginv = 120
    style.alignment = 2  # bas-centre

    subs.styles[style.name] = style

    for seg in segments:
        if seg["end"] <= clip_start or seg["start"] >= clip_end:
            continue

        start = max(seg["start"], clip_start) - clip_start
        end = min(seg["end"], clip_end) - clip_start

        ktext = build_karaoke_text(seg["text"], seg["start"], seg["end"])
        if not ktext:
            continue

        ev = pysubs2.SSAEvent()
        ev.start = int(start * 1000)
        ev.end = int(end * 1000)
        ev.style = "KlapMain"
        ev.text = ktext
        subs.events.append(ev)

    subs.save(str(subs_path))


# ==============================
# 4. FFMPEG : CUT + 9:16 + SUBTITLES
# ==============================

def ffmpeg_extract_and_style(
    src: str,
    out: str,
    subs: str,
    start: float,
    end: float,
):
    duration = max(end - start, 0.5)

    vf = f"scale=-2:1920,crop=1080:1920,subtitles='{subs}'"

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        src,
        "-t",
        str(duration),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        out,
    ]

    print("🎬 FFmpeg :", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ==============================
# 5. PIPELINE GLOBAL
# ==============================

def generate_shorts(
    input_video_path: str,
    num_clips: int = 8,
    min_duration: float = 20,
    max_duration: float = 45,
):
    """
    Pipeline complet :
    - transcription Whisper
    - sélection IA virale
    - génération .ASS
    - FFmpeg 9:16
    - upload R2 (si configuré)
    """
    video = Path(input_video_path)
    if not video.exists():
        raise FileNotFoundError(video)

    print("🚀 Lancement pipeline IA…")

    # 1. Transcription Whisper
    tr = transcribe_with_whisper(str(video))
    segments = tr["segments"]

    # 2. IA virale
    viral = select_viral_segments(segments, num_clips, min_duration, max_duration)

    results = []

    # 3. Génération des shorts
    for i, c in enumerate(viral, start=1):
        out_vid = SHORTS_DIR / f"short_{i:02d}.mp4"
        out_ass = SUBS_DIR / f"short_{i:02d}.ass"

        print(f"▶️ Clip {i} | {c['start']} → {c['end']}")

        generate_ass_subs_for_clip(c["start"], c["end"], segments, out_ass)
        ffmpeg_extract_and_style(str(video), str(out_vid), str(out_ass), c["start"], c["end"])

        # Upload vers R2 (si configuré)
        video_url = upload_to_r2(str(out_vid), key_prefix="shorts")
        subs_url = upload_to_r2(str(out_ass), key_prefix="subs")

        results.append(
            {
                "index": i,
                "title": c["title"],
                "reason": c["reason"],
                "start": c["start"],
                "end": c["end"],
                "video_path": str(out_vid),
                "subs_path": str(out_ass),
                "video_url": video_url,
                "subs_url": subs_url,
            }
        )

    print("🎉 Shorts générés :", len(results))
    return results
