import os
import json
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import pysubs2
import yt_dlp
import boto3
import requests


# ==========================================
# DOSSIERS (inchangés)
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
SHORTS_DIR = BASE_DIR / "shorts"
SUBS_DIR = BASE_DIR / "subs"

for d in (UPLOADS_DIR, SHORTS_DIR, SUBS_DIR):
    d.mkdir(parents=True, exist_ok=True)


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
        ExtraArgs={
            "ContentType": "video/mp4"
            if file.suffix == ".mp4"
            else "text/plain"
        },
    )

    base = os.getenv("R2_PUBLIC_BASE_URL").rstrip("/")
    return f"{base}/{key}"


# ==========================================
# DOWNLOAD VIDEO (conservé)
# ==========================================

def download_video(url: str) -> str:
    print(f"⬇️ DOWNLOAD → {url}")

    dest = UPLOADS_DIR / f"input_{os.urandom(4).hex()}.mp4"
    parsed = urlparse(url)

    if url.endswith(".mp4") or "r2.dev" in parsed.netloc:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
    else:
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
# LES SOUS-TITRES KLAP (conservés)
# ==========================================

def build_ass(text, start, end):
    words = text.strip().split()
    if not words:
        return ""

    duration_ms = max(int((end - start) * 1000), 200)
    per_word = max(duration_ms // len(words), 15)

    line = " ".join([f"{{\\k{per_word}}}{w}" for w in words])

    return (
        r"{\an2\fs50\bord2\shad0\1c&Hffffff&\3c&H202020&\fad(80,80)}"
        + line
    )


def make_ass(clip_start, clip_end, segments, path):
    subs = pysubs2.SSAFile()

    style = pysubs2.SSAStyle()
    style.name = "Main"
    style.fontname = "Poppins"
    style.fontsize = 50
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
# RENDU FFmpeg CPU (debug uniquement)
# ==========================================

def render_clip(src, out, subs, start, end):
    dur = max(end - start, 1)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        src,
        "-t",
        str(dur),
        "-vf",
        f"scale=-2:1920,crop=1080:1920,subtitles='{subs}'",
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

    subprocess.run(cmd, check=True)


# ==========================================
# PIPELINE (désactivé)
# ==========================================

def generate_shorts(*args, **kwargs):
    raise Exception(
        "🚫 generate_shorts() n'est plus utilisé dans la version PRO. "
        "Le worker GPU utilise engine/*.py"
    )
