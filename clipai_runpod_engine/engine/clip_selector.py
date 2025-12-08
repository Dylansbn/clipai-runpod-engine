import json
from openai import OpenAI

client = OpenAI()

def select_clips(segments, analysis, num_clips):
    transcript = "\n".join(
        f"[{s['start']} → {s['end']}] {s['text']}"
        for s in segments
    )

    peaks = analysis["peaks"]

    system_prompt = """
Tu es un moteur expert en viralité.
Analyse :
- les pics visuels (peaks)
- le texte
- le rythme du discours
Retourne EXACTEMENT les clips les plus viraux.
"""

    user = json.dumps({
        "transcript": transcript,
        "visual_peaks": peaks,
        "num_clips": num_clips
    })

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ]
    )

    data = json.loads(res.choices[0].message.content)
    return data["clips"]
