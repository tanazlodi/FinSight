"""Utilities for reading FinSight's local earnings-call corpus.

The app starts with a tiny demonstration corpus in ``data/sample``. When a
Kaggle download is available, place it under ``data/raw/meta_transcripts``.
This loader accepts individual text files as well as CSV, JSON, JSONL and
pickle files, mapping them into one consistent schema.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "meta_demo_transcripts.json"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "meta_transcripts"

REQUIRED_COLUMNS = ["ticker", "date", "quarter", "year", "transcript"]
COLUMN_ALIASES = {
    "ticker": ("ticker", "symbol", "stock", "company_ticker"),
    "date": ("date", "call_date", "earnings_date", "datetime"),
    "quarter": ("quarter", "fiscal_quarter", "qtr"),
    "year": ("year", "fiscal_year", "fy"),
    "transcript": ("transcript", "content", "text", "full_text", "call_transcript"),
    "title": ("title", "call_title", "filename", "file_name"),
}


def _read_file(path: Path) -> pd.DataFrame:
    """Read a supported file and return one or more raw transcript records."""
    if path.suffix.lower() == ".txt":
        # Kaggle's Meta corpus stores one earnings call per text file. Metadata
        # is inferred from the filename/path where possible; normalize_transcripts
        # supplies safe fallbacks for anything not encoded in the filename.
        filename = path.stem
        # Treat underscores and hyphens as word separators so filenames such as
        # ``META_Q1_2024.txt`` are parsed as reliably as natural-language names.
        searchable_name = re.sub(r"[_-]+", " ", " ".join(path.parts))
        year_match = re.search(r"(?<!\d)(19|20)\d{2}(?!\d)", searchable_name)
        quarter_match = re.search(
            r"(?:\bQ\s*([1-4])\b|\b([1-4])\s*Q\b|\bQUARTER\s*([1-4])\b)",
            searchable_name,
            flags=re.IGNORECASE,
        )
        quarter_number = next((group for group in quarter_match.groups() if group), None) if quarter_match else None

        return pd.DataFrame(
            [{
                "ticker": "META",
                "year": int(year_match.group()) if year_match else None,
                "quarter": f"Q{quarter_number}" if quarter_number else "Unknown",
                "title": filename,
                "transcript": path.read_text(encoding="utf-8", errors="replace"),
            }]
        )

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported transcript file type: {path.suffix}")


def _find_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def normalize_transcripts(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only the source fields the UI and later RAG pipeline need."""
    result = pd.DataFrame(index=frame.index)

    for target, aliases in COLUMN_ALIASES.items():
        source = _find_column(frame, aliases)
        if source is not None:
            result[target] = frame[source]

    if "transcript" not in result:
        raise ValueError(
            "No transcript text column found. Expected one of: "
            + ", ".join(COLUMN_ALIASES["transcript"])
        )

    result["ticker"] = result.get("ticker", pd.Series("META", index=result.index)).fillna("META").astype(str).str.upper()
    result["date"] = pd.to_datetime(result.get("date", pd.Series(pd.NaT, index=result.index)), errors="coerce")
    result["year"] = result.get("year", result["date"].dt.year).fillna(result["date"].dt.year)
    result["quarter"] = result.get("quarter", pd.Series("Unknown", index=result.index)).fillna("Unknown").astype(str)
    result["title"] = result.get("title", pd.Series("Earnings call", index=result.index)).fillna("Earnings call").astype(str)
    result["transcript"] = result["transcript"].fillna("").astype(str).str.strip()

    result = result[result["transcript"].str.len() > 0].copy()
    result["source_id"] = [f"{ticker}-{year}-{quarter}-{index}" for index, (ticker, year, quarter) in enumerate(zip(result["ticker"], result["year"], result["quarter"]))]
    return result[REQUIRED_COLUMNS + ["title", "source_id"]].sort_values("date", ascending=False)


def load_transcript_calls(use_sample_data: bool = True) -> pd.DataFrame:
    """Load the raw corpus when present, otherwise load the UI demo corpus."""
    raw_files = [
        path
        for extension in ("*.txt", "*.csv", "*.json", "*.jsonl", "*.pkl", "*.pickle")
        for path in RAW_DATA_DIR.rglob(extension)
    ] if RAW_DATA_DIR.exists() else []

    if raw_files:
        frames = [_read_file(path) for path in raw_files]
        return normalize_transcripts(pd.concat(frames, ignore_index=True))

    if use_sample_data and SAMPLE_PATH.exists():
        with SAMPLE_PATH.open(encoding="utf-8") as file:
            return normalize_transcripts(pd.DataFrame(json.load(file)))

    return pd.DataFrame(columns=REQUIRED_COLUMNS + ["title", "source_id"])
