import json
from pathlib import Path

QUEUE_PATH = Path("queue_jobs.json")

if not QUEUE_PATH.exists():
    with open(QUEUE_PATH, "w") as f:
        json.dump([], f)

def push_job(job_data):
    jobs = _load()
    jobs.append(job_data)
    _save(jobs)
    print(f"📥 Job ajouté : {job_data['job_id']}")

def pop_job():
    jobs = _load()
    if not jobs:
        return None
    job = jobs.pop(0)
    _save(jobs)
    print(f"📤 Job traité : {job['job_id']}")
    return job

def _load():
    with open(QUEUE_PATH) as f:
        return json.load(f)

def _save(jobs):
    with open(QUEUE_PATH, "w") as f:
        json.dump(jobs, f, indent=2)
