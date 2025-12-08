# ============================================================
# SUBTITLES ENGINE — KLAP PRO STYLE
# ============================================================

import pysubs2
from pathlib import Path

# ------------------------------------------------------------
# 1. Construire un événement par mot (zoom + couleur)
# ------------------------------------------------------------

def build_ass_krap(words, local_start, local_end):
    events = []
    duration = local_end - local_start
    per_word = duration / max(len(words), 1)
    t = local_start

    for w in words:
        start = t
        end = t + per_word
        t += per_word

        # Mot actif : zoom + couleur bleue KLAP
        text = (
            r"{\fscx120\fscy120\1c&H20DAF2&}"
            + w +
            r"{\fscx100\fscy100\1c&HFFFFFF&}"
        )

        events.append({
            "start": start,
            "end": end,
            "text": text
        })

    return events


# ------------------------------------------------------------
# 2. Générer un fichier .ass complet (header + style + events)
# ------------------------------------------------------------

def make_ass_krap(clip_start, clip_end, segments, output_path):
    """
    Génère un fichier .ass au style Klap :
    - texte centré bas
    - outline épais
    - couleur dynamique
    - zoom sur chaque mot
    """

    output_path = Path(output_path)

    with open(output_path, "w", encoding="utf-8") as f:

        # ---------- HEADER ----------
        f.write(
            """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main, Poppins, 64, &H00FFFFFF, &H0000FFFF, &H00111111, &H96000000, 0, 0, 0, 0, 100, 100, 0, 0, 1, 6, 0, 2, 60, 60, 110, 1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        )

        # ---------- EVENTS ----------
        for seg in segments:
            if seg["end"] <= clip_start or seg["start"] >= clip_end:
                continue

            text = seg["text"].strip()
            words = text.split()
            if not words:
                continue

            local_start = max(seg["start"], clip_start) - clip_start
            local_end = min(seg["end"], clip_end) - clip_start

            events = build_ass_krap(words, local_start, local_end)

            for evt in events:
                start = evt["start"]
                end = evt["end"]
                ev_text = evt["text"]

                f.write(
                    f"Dialogue: 0,{start:.3f},{end:.3f},Main,,0,0,0,,{ev_text}\n"
                )
