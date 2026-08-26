"""Score FinSight earnings calls with FinBERT and save quarterly tone data.

Run from the project root:
    python scripts/build_sentiment_scores.py
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.sentiment import score_calls
from src.ingestion.load_transcripts import load_transcript_calls


def main() -> None:
    calls = load_transcript_calls(use_sample_data=False)
    if calls.empty:
        raise SystemExit("No raw calls found. Add transcripts before building sentiment scores.")

    print(f"Scoring prepared remarks from {len(calls)} earnings calls with FinBERT...")
    scores = score_calls(calls)
    output_path = PROJECT_ROOT / "data" / "processed" / "quarterly_sentiment.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, index=False)
    print(f"Wrote {len(scores)} quarterly sentiment scores to {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
