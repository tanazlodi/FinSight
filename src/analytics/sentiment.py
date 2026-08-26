"""Quarterly prepared-remarks tone scoring with ProsusAI FinBERT."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


FINBERT_MODEL_ID = "ProsusAI/finbert"


def prepared_remarks(transcript: str) -> str:
    """Keep remarks before the analyst Q&A section when that heading exists.

    This is a lightweight approximation of management tone: operator language
    may remain, but analyst questions in the later Q&A section are excluded.
    """
    q_and_a_heading = re.search(
        r"\b(?:questions?[\s-]*(?:and|&)[\s-]*answers?|q[\s-]*(?:and|&)[\s-]*a)"
        r"(?:\s+session)?\b",
        transcript,
        flags=re.IGNORECASE,
    )
    return transcript[: q_and_a_heading.start()] if q_and_a_heading else transcript


def text_windows(text: str, max_words: int = 220) -> list[str]:
    """Split a call into FinBERT-safe sentence windows below its token limit."""
    clean_text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", clean_text)
    windows: list[str] = []
    current: list[str] = []
    current_count = 0

    for sentence in sentences:
        words = sentence.split()
        # Split rare unusually long transcript paragraphs safely.
        pieces = [words[start : start + max_words] for start in range(0, len(words), max_words)]
        for piece in pieces:
            if current and current_count + len(piece) > max_words:
                windows.append(" ".join(current))
                current, current_count = [], 0
            current.append(" ".join(piece))
            current_count += len(piece)

    if current:
        windows.append(" ".join(current))
    return [window for window in windows if window.strip()]


def score_calls(calls: pd.DataFrame, batch_size: int = 8) -> pd.DataFrame:
    """Score each call and aggregate FinBERT probabilities by fiscal quarter."""
    try:
        from transformers import pipeline
    except ImportError as error:
        raise RuntimeError(
            "FinBERT or one of its Python dependencies could not be imported. "
            "Run `python -m pip install --force-reinstall --no-cache-dir sympy transformers torch`."
        ) from error

    # Use CPU deliberately: this offline analysis runs once and avoids
    # laptop-specific Metal/MPS graph-cache behavior.
    classifier = pipeline(
        "text-classification", model=FINBERT_MODEL_ID, top_k=None, device=-1
    )
    rows: list[dict[str, object]] = []

    for call in calls.itertuples(index=False):
        windows = text_windows(prepared_remarks(call.transcript))
        if not windows:
            continue

        probabilities = {"positive": [], "negative": [], "neutral": []}
        for start in range(0, len(windows), batch_size):
            batch_scores = classifier(
                windows[start : start + batch_size],
                truncation=True,
                max_length=512,
            )
            for window_scores in batch_scores:
                for score in window_scores:
                    label = score["label"].lower()
                    if label in probabilities:
                        probabilities[label].append(float(score["score"]))

        positive = sum(probabilities["positive"]) / len(probabilities["positive"])
        negative = sum(probabilities["negative"]) / len(probabilities["negative"])
        neutral = sum(probabilities["neutral"]) / len(probabilities["neutral"])
        rows.append(
            {
                "ticker": call.ticker,
                "date": call.date,
                "quarter": call.quarter,
                "year": int(call.year),
                "source_id": call.source_id,
                "prepared_remark_windows": len(windows),
                "positive_score": positive,
                "negative_score": negative,
                "neutral_score": neutral,
                # Net tone makes a compact trend line: positive probability
                # minus negative probability, from -1 (negative) to +1.
                "net_tone_score": positive - negative,
            }
        )

    return pd.DataFrame(rows).sort_values("date")


def sentiment_chart(scores: pd.DataFrame, ticker: str):
    """Return a quarter-by-quarter Plotly trend figure for one company."""
    import plotly.graph_objects as go

    data = scores[scores["ticker"] == ticker].sort_values("date")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["net_tone_score"],
            mode="lines+markers",
            line={"color": "#0F766E", "width": 2.5},
            marker={"size": 7},
            customdata=data[["quarter", "year", "positive_score", "negative_score"]],
            hovertemplate=(
                "%{customdata[0]} %{customdata[1]}<br>Net tone: %{y:.2f}"
                "<br>Positive: %{customdata[2]:.0%}<br>Negative: %{customdata[3]:.0%}<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=0, line_dash="dot", line_color="#94A3B8")
    figure.update_layout(
        title=f"{ticker} — Management tone by quarter",
        xaxis_title="Earnings-call date",
        yaxis_title="Net tone score (positive − negative)",
        template="plotly_white",
        height=360,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        showlegend=False,
    )
    return figure
