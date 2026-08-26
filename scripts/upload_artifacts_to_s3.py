"""Upload generated FinSight deployment artifacts to private S3 storage.

Run from the project root:
    python scripts/upload_artifacts_to_s3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.aws.s3 import upload_artifacts


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    uploaded = upload_artifacts(PROJECT_ROOT / "data" / "processed")
    print("Uploaded FinSight deployment artifacts:")
    for location in uploaded:
        print(f"- {location}")


if __name__ == "__main__":
    main()
