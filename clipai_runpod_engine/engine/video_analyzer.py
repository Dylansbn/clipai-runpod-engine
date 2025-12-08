import cv2
import numpy as np
import requests
from pathlib import Path


def download_video(url):
    """Télécharge une vidéo depuis URL vers un fichier local."""
    dest = Path(f"video_{np.random.randint(999999)}.mp4")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

    return str(dest)


def analyze_video(path):
    """Analyse visuelle simple : détection de pics (moments forts)."""
    cap = cv2.VideoCapture(path)
    scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = float(np.std(gray))  # contraste / mouvement
        scores.append(score)

    cap.release()

    peaks = detect_peaks(scores)

    return {
        "visual_energy": scores,
        "peaks": peaks,
    }


def detect_peaks(values, threshold_ratio=1.4):
    """Détecte les pics visuels simples."""
    peaks = []
    median = np.median(values) if values else 0

    for i, v in enumerate(values):
        if v > median * threshold_ratio:
            peaks.append(i)

    return peaks
