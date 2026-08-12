# Data sources

## Demo corpus

`data/sample/meta_demo_transcripts.json` contains short fictional text solely
to verify the interface and ingestion pipeline. It must not be presented as an
actual company transcript.

## Intended transcript corpus

For the portfolio build, use the Kaggle **META Earnings Call Transcripts**
dataset. Download it manually, retain its attribution and license information,
and place its files in `data/raw/meta_transcripts/`. The raw data is ignored by
Git so it is not redistributed by this repository.

## Price data

The app uses `yfinance` for educational/demo price charts. It is not intended
as a production market-data service.
