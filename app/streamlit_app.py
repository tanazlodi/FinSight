import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.load_transcripts import load_transcript_calls

load_dotenv()

st.set_page_config(
    page_title="FinSight",
    page_icon="📈",
    layout="wide",
)

TICKERS = os.getenv("DEFAULT_TICKERS", "META").split(",")
TICKERS = [ticker.strip().upper() for ticker in TICKERS]
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.04"))


@st.cache_data(ttl=3600)
def load_price_data(ticker: str) -> pd.DataFrame:
    """Download two years of daily adjusted price history."""
    prices = yf.download(
        ticker,
        period="2y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if prices.empty:
        return pd.DataFrame()

    # yfinance may return MultiIndex columns for some versions.
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)

    return prices.dropna()


def calculate_risk_metrics(prices: pd.DataFrame) -> dict[str, float]:
    """Calculate simple annualized return, volatility, and Sharpe ratio."""
    daily_returns = prices["Close"].pct_change().dropna()

    if daily_returns.empty:
        return {
            "current_price": np.nan,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe_ratio": np.nan,
        }

    trading_days = 252
    annual_return = (1 + daily_returns.mean()) ** trading_days - 1
    annual_volatility = daily_returns.std() * np.sqrt(trading_days)
    sharpe_ratio = (
        (annual_return - RISK_FREE_RATE) / annual_volatility
        if annual_volatility > 0
        else np.nan
    )

    return {
        "current_price": float(prices["Close"].iloc[-1]),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe_ratio": float(sharpe_ratio),
    }


def price_chart(prices: pd.DataFrame, ticker: str) -> go.Figure:
    chart = go.Figure()

    chart.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices["Close"],
            mode="lines",
            name=ticker,
            line={"color": "#4F46E5", "width": 2.5},
            hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>",
        )
    )

    chart.update_layout(
        title=f"{ticker} — Adjusted Closing Price",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=430,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        showlegend=False,
    )

    return chart


@st.cache_data
def load_calls() -> pd.DataFrame:
    """Load cached transcript metadata for the dashboard preview."""
    use_sample_data = os.getenv("USE_SAMPLE_DATA", "true").lower() == "true"
    return load_transcript_calls(use_sample_data=use_sample_data)


def main() -> None:
    st.title("FinSight")
    st.caption("AI-powered equity research from earnings-call transcripts and market data.")

    with st.sidebar:
        st.header("Research settings")
        ticker = st.selectbox("Select a company", TICKERS)
        st.divider()
        st.caption("Transcript search and cited AI answers are coming next.")

    with st.spinner(f"Loading {ticker} market data..."):
        prices = load_price_data(ticker)

    if prices.empty:
        st.error(
            f"Could not load market data for {ticker}. "
            "Check your internet connection and try again."
        )
        st.stop()

    metrics = calculate_risk_metrics(prices)

    st.subheader(f"{ticker} market overview")
    price_col, return_col, volatility_col, sharpe_col = st.columns(4)

    price_col.metric("Latest price", f"${metrics['current_price']:,.2f}")
    return_col.metric("Annualized return", f"{metrics['annual_return']:.1%}")
    volatility_col.metric(
        "Annualized volatility",
        f"{metrics['annual_volatility']:.1%}",
    )
    sharpe_col.metric("Sharpe ratio", f"{metrics['sharpe_ratio']:.2f}")

    st.plotly_chart(price_chart(prices, ticker), use_container_width=True)

    st.subheader("Earnings-call corpus")
    calls = load_calls()
    calls = calls[calls["ticker"] == ticker].copy()

    if calls.empty:
        st.warning(
            "No transcripts are loaded yet. Add the downloaded dataset under "
            "`data/raw/META_EarningsCallTranscripts/` and restart the app."
        )
    else:
        is_demo = os.getenv("USE_SAMPLE_DATA", "true").lower() == "true" and not (
            PROJECT_ROOT / "data" / "raw" / "META_EarningsCallTranscripts"
        ).exists()
        if is_demo:
            st.info(
                "You are viewing fictional preview text. It verifies the interface only; "
                "replace it with the permitted Kaggle corpus before building RAG."
            )

        call_labels = {
            row.source_id: f"{row.quarter} {int(row.year)} — {row.title}"
            for row in calls.itertuples()
        }
        selected_id = st.selectbox("Browse a call", list(call_labels), format_func=call_labels.get)
        selected_call = calls.loc[calls["source_id"] == selected_id].iloc[0]
        st.caption(f"Call date: {selected_call['date'].date() if pd.notna(selected_call['date']) else 'Unknown'}")
        st.text_area(
            "Transcript preview",
            selected_call["transcript"],
            height=180,
            disabled=True,
        )

    st.subheader("What FinSight will answer")
    st.info(
            'Next milestone: “What did management say about margin pressure this quarter?” '
            "FinSight will retrieve relevant transcript passages and show the sources "
            "behind its answer."
    )

    with st.expander("About these metrics"):
        st.write(
            "Annualized return and volatility are calculated from the last two years "
            "of daily adjusted closing prices. The Sharpe ratio uses the risk-free "
            f"rate in your `.env` file ({RISK_FREE_RATE:.1%})."
        )

    st.caption(
        "For educational and portfolio-demonstration purposes only. "
        "Not investment advice."
    )


if __name__ == "__main__":
    main()
