"""Build FinSight's local retrieval-ready transcript chunk file.

Run from the project root:
    python scripts/build_chunks.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.chunk_transcripts import iter_chunks
from src.ingestion.load_transcripts import load_transcript_calls


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "120"))
    output_path = PROJECT_ROOT / "data" / "processed" / "transcript_chunks.jsonl"

    calls = load_transcript_calls(use_sample_data=False)
    if calls.empty:
        raise SystemExit(
            "No raw transcripts found. Add the text files under "
            "data/raw/META_EarningsCallTranscripts/ and run this command again."
        )

    chunks = list(iter_chunks(calls, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(chunks):,} chunks from {len(calls):,} calls to "
        f"{output_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
