import boto3
import os
from pathlib import Path

def upload_results(job_id, paths):
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("R2_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
    )

    bucket = os.getenv("R2_BUCKET_NAME")
    base = os.getenv("R2_PUBLIC_BASE_URL").rstrip("/")

    urls = []

    for p in paths:
        file = Path(p)
        key = f"{job_id}/{file.name}"

        s3.upload_file(
            Filename=str(file),
            Bucket=bucket,
            Key=key,
            ExtraArgs={"ContentType": "video/mp4"},
        )

        urls.append(f"{base}/{key}")

    return urls
