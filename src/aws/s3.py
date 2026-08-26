"""Private S3 storage for FinSight's deployable generated artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import boto3


def s3_client():
    """Create an S3 client using the configured local AWS profile or IAM role."""
    session = boto3.Session(
        profile_name=os.getenv("AWS_PROFILE") or None,
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )
    return session.client("s3")


def bucket_name() -> str:
    """Return the configured bucket name or explain the missing configuration."""
    bucket = os.getenv("S3_BUCKET_NAME")
    if not bucket:
        raise RuntimeError("S3_BUCKET_NAME is not configured.")
    return bucket


def artifact_keys() -> dict[str, str]:
    """Map local generated files to private S3 object keys."""
    processed_prefix = os.getenv("S3_PROCESSED_PREFIX", "processed").strip("/")
    sentiment_prefix = os.getenv("S3_SENTIMENT_PREFIX", "analytics/sentiment").strip("/")
    return {
        "faiss_index/transcript_chunks.faiss": f"{processed_prefix}/faiss_index/transcript_chunks.faiss",
        "faiss_index/transcript_chunks.metadata.jsonl": (
            f"{processed_prefix}/faiss_index/transcript_chunks.metadata.jsonl"
        ),
        "quarterly_sentiment.csv": f"{sentiment_prefix}/quarterly_sentiment.csv",
    }


def upload_artifacts(processed_directory: Path) -> list[str]:
    """Upload FAISS and sentiment artifacts to the configured private bucket."""
    client = s3_client()
    bucket = bucket_name()
    uploaded: list[str] = []

    for relative_path, key in artifact_keys().items():
        local_path = processed_directory / relative_path
        if not local_path.exists():
            raise FileNotFoundError(f"Missing generated artifact: {local_path}")
        client.upload_file(str(local_path), bucket, key)
        uploaded.append(f"s3://{bucket}/{key}")
    return uploaded


def download_artifacts(processed_directory: Path) -> list[Path]:
    """Download the deployment artifacts to the local runtime directory."""
    client = s3_client()
    bucket = bucket_name()
    downloaded: list[Path] = []

    for relative_path, key in artifact_keys().items():
        local_path = processed_directory / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(local_path))
        downloaded.append(local_path)
    return downloaded
