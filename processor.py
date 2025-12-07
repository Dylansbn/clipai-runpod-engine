import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any
from urllib.parse import urlparse

import pysubs2
import yt_dlp
import boto3
import requests
from openai import OpenAI

# ==========================================
# DOSSIERS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
SHORTS_DIR = BASE_DIR / "shorts"
SUBS_DIR = BASE_DIR / "subs"

for d in (UPLOADS_DIR, SHORTS_DIR, SUBS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ==========================================
# OPENAI
# ==========================================

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT_ID"),
)

# ==========================================
# CLOUDFLARE R2
# ==========================================

def get_r2():
    if not (
        os.getenv("R2_ACCESS_KEY_ID")
        and os.getenv("R2_SECRET_ACCESS_KEY")
        and os.getenv("R2_ENDPOINT_URL")
    ):
        print("⚠️ R2 OFF → local mode")
        return None

    return boto3.client(
        "s3",
        endpoint_url=os.getenv("R2_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
    )


def upload_to_r2(local_path: str, prefix=""):
    s3 = get_r2()
    bucket = os.getenv("R2_BUCKET_NAME")

    if not s3 or not bucket:
        return local_path

    file = Path(local_path)
    key = f"{prefix}/{file.name}" if prefix else file.name

    s3.upload_file(
        Filename=str(file),
        Bucket=bucket,
        Key=key,
        ExtraArgs={"ContentType": "video/mp4" if file.suffix == ".mp4" else "text/plain"},
    )

    base = os.getenv("R2_PUBLIC_BASE_URL").rstrip("/")
    return f"{base}/{key}"


# ==========================================
# 0. DOWNLOAD VIDEO
# ==========================================

def download_video(url: str) -> str:
    print(f"⬇️ DOWNLOAD → {url}")

    dest = UPLOADS_DIR / f"input_{os.urandom(4).hex()}.mp4"
    parsed = urlparse(url)

    # Télécharger direct si déjà mp4
    if url.endswith(".mp4") or "r2.dev" in parsed.netloc:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
    else:
        # Mode yt-dlp
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        ydl_opts = {
            "outtmpl": str(dest),
            "format": "bv*+ba/best",
            "merge_output_format": "mp4",
            "quiet": True,
            "user_agent": ua,
            "http_headers": {"User-Agent": ua},
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    if not dest.exists() or dest.stat().st_size < 200_000:
        raise Exception("Téléchargement incomplet")

    return str(dest)


# ==========================================
# AUDIO → WHISPER
# ==========================================

def extract_audio(video: str):
    audio = str(Path(video).with_suffix(".mp3"))

    cmd = [
        "ffmpeg", "-y",
        "-i", video,
        "-vn",
        "-acodec", "libmp3lame",
        "-b:a", "64k",
        audio
    ]

    subprocess.run(cmd, check=True)
    return audio


def transcribe(video: str):
    audio = extract_audio(video)

    with open(audio, "rb") as f:
        res = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
        )

    segments = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
                for s in res.segments]

    return segments


# ==========================================
# GPT SELECTION — KLAP EXACT
# ==========================================

def select_clips(segments, num_clips=3):
    transcript = "\n".join(
        f"[{s['start']:.2f} → {s['end']:.2f}] {s['text']}"
        for s in segments
    )[:15000]

    system_prompt = (
        "Tu es KLAP.APP (version exacte). "
        "Tu dois sélectionner EXACTEMENT les meilleurs moments viraux. "
        "Pas de clips trop courts. "
        "Chaque clip doit durer ENTRE 18 ET 35 SECONDES (comme KLAP). "
        "Toujours renvoyer JSON strict."
    )

    user_prompt = f"""
Transcription :
{transcript}

Sélectionne {num_clips} clips viraux.
"""

    res = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    data = json.loads(res.choices[0].message.content)

    clips = []
    for c in data["clips"]:
        s = float(c["start"])
        e = float(c["end"])
        dur = e - s

        # Ajustements exact KLAP
        if dur < 18:
            e = s + 18
        if dur > 35:
            e = s + 35

        clips.append({
            "start": s,
            "end": e,
            "title": c.get("title", "Clip viral"),
            "reason": c.get("reason", "")
        })

    return clips


# ==========================================
# SUBTITLES — KLAP EXACT STYLE
# ==========================================

def build_ass(text, start, end):
    words = text.strip().split()
    if not words:
        return ""

    duration_ms = max(int((end - start) * 1000), 200)
    per_word = max(duration_ms // len(words), 15)

    line = " ".join([f"{{\\k{per_word}}}{w}" for w in words])

    return (
        r"{\an2\fs50\bord2\shad0\1c&Hffffff&\3c&H202020&\fad(80,80)}" +
        line
    )


def make_ass(clip_start, clip_end, segments, path):
    subs = pysubs2.SSAFile()

    style = pysubs2.SSAStyle()
    style.name = "Main"
    style.fontname = "Poppins"
    style.fontsize = 50
    style.bold = False
    style.marginv = 140
    style.alignment = 2
    subs.styles["Main"] = style

    for seg in segments:
        if seg["end"] <= clip_start or seg["start"] >= clip_end:
            continue

        st = max(seg["start"], clip_start) - clip_start
        en = min(seg["end"], clip_end) - clip_start

        subs.events.append(
            pysubs2.SSAEvent(
                start=int(st * 1000),
                end=int(en * 1000),
                style="Main",
                text=build_ass(seg["text"], st, en),
            )
        )

    subs.save(path)


# ==========================================
# FFMPEG EXPORT
# ==========================================

def render_clip(src, out, subs, start, end):
    dur = max(end - start, 1)

    vf = f"scale=-2:1920,crop=1080:1920,subtitles='{subs}'"

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-i", src,
        "-t", str(dur),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        out,
    ]

    subprocess.run(cmd, check=True)


# ==========================================
# PIPELINE KLAP EXACT
# ==========================================

def generate_shorts(video_path, num_clips=3):
    segments = transcribe(video_path)
    clips = select_clips(segments, num_clips)

    results = []
    for i, c in enumerate(clips, 1):
        out_vid = SHORTS_DIR / f"short_{i:02d}.mp4"
        out_ass = SUBS_DIR / f"short_{i:02d}.ass"

        make_ass(c["start"], c["end"], segments, out_ass)
        render_clip(video_path, out_vid, out_ass, c["start"], c["end"])

        results.append({
            "index": i,
            "title": c["title"],
            "reason": c["reason"],
            "video_url": upload_to_r2(str(out_vid), "shorts"),
            "subs_url": upload_to_r2(str(out_ass), "subs"),
        })

    return results
