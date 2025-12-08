import subprocess
from pathlib import Path

from .subtitles import make_ass_krap


def render_clips(src, clips, segments):
    """
    Génère chaque clip :
    - produit un fichier .ass (sous-titres)
    - fait le rendu ffmpeg NVENC
    - retourne la liste des chemins des vidéos produites
    """

    outputs = []

    for i, c in enumerate(clips, 1):
        start = c["start"]
        end = c["end"]
        duration = end - start

        out_video = Path(f"short_{i}.mp4")
        out_ass = Path(f"short_{i}.ass")

        print(f"📝 Génération .ass pour le clip {i}…")
        make_ass_krap(start, end, segments, out_ass)

        print(f"🎬 Rendu clip {i}… ({start} → {end})")

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", src,
            "-t", str(duration),
            "-vf", f"scale=-2:1920,crop=1080:1920,subtitles='{out_ass}'",
            "-c:v", "h264_nvenc",
            "-preset", "fast",
            "-qp", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            str(out_video)
        ]

        subprocess.run(cmd, check=True)
        outputs.append(str(out_video))

    return outputs
