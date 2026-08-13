"""Split normalized earnings calls into retrieval-ready passages."""

from __future__ import annotations

import re
from collections.abc import Iterator

import pandas as pd


def _sentences(text: str) -> list[str]:
    """Split readable transcript text into simple sentence-sized units."""
    clean_text = re.sub(r"\s+", " ", text).strip()
    if not clean_text:
        return []
    return re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", clean_text)


def _word_count(text: str) -> int:
    return len(text.split())


def _split_long_sentence(sentence: str, max_words: int) -> list[str]:
    """Prevent one unusually long paragraph from exceeding the chunk limit."""
    words = sentence.split()
    return [" ".join(words[start : start + max_words]) for start in range(0, len(words), max_words)]


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    """Create word-limited chunks with a small repeated context overlap.

    Chunks end on sentence boundaries whenever possible. The overlap prevents a
    sentence at the boundary from losing the context supplied by the preceding
    passage during later vector search.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be at least 0 and smaller than chunk_size")

    units: list[str] = []
    for sentence in _sentences(text):
        units.extend(_split_long_sentence(sentence, chunk_size))

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for unit in units:
        unit_words = _word_count(unit)
        if current and current_words + unit_words > chunk_size:
            chunks.append(" ".join(current))

            # Reuse enough of the tail to give the next chunk continuity.
            overlap: list[str] = []
            overlap_words = 0
            for previous_unit in reversed(current):
                overlap.insert(0, previous_unit)
                overlap_words += _word_count(previous_unit)
                if overlap_words >= chunk_overlap:
                    break
            current = overlap
            current_words = overlap_words

        current.append(unit)
        current_words += unit_words

    if current:
        chunks.append(" ".join(current))
    return chunks


def iter_chunks(
    calls: pd.DataFrame,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> Iterator[dict[str, object]]:
    """Yield a source-citable record for every chunk in a call DataFrame."""
    for call in calls.itertuples(index=False):
        for chunk_index, text in enumerate(
            chunk_text(call.transcript, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            yield {
                "chunk_id": f"{call.source_id}-chunk-{chunk_index}",
                "source_id": call.source_id,
                "ticker": call.ticker,
                "date": call.date.isoformat() if pd.notna(call.date) else None,
                "quarter": call.quarter,
                "year": int(call.year) if pd.notna(call.year) else None,
                "title": call.title,
                "chunk_index": chunk_index,
                "text": text,
                "word_count": _word_count(text),
            }
