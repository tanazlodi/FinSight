# FinSight

FinSight is an equity-research assistant that will use retrieval-augmented
generation (RAG) to answer questions from cited earnings-call passages.

## Run locally

```bash
source .venv/bin/activate
python -m streamlit run app/streamlit_app.py
```

The first working screen shows Meta price history, basic risk metrics, and a
local earnings-call corpus preview. The bundled preview text is fictional and
exists only to make the interface testable before a permitted transcript corpus
is downloaded.

## Add the intended transcript dataset

1. Download the META Earnings Call Transcripts dataset from Kaggle.
2. Unzip it inside `data/raw/META_EarningsCallTranscripts/`.
3. Restart Streamlit.

The loader accepts individual `.txt` files as well as CSV, JSON, JSONL, and
pickle files. It normalizes common field names such as `ticker`, `date`,
`quarter`, and `transcript` into one schema for the upcoming chunking and
vector-index steps. For text files, it uses the filename/path to infer the
quarter and year when those values are included there.

Read [docs/data_sources.md](docs/data_sources.md) before adding data.

## Build retrieval chunks

After adding the raw text corpus, run:

```bash
python scripts/build_chunks.py
```

This writes `data/processed/transcript_chunks.jsonl`: one source-citable,
overlapping text passage per line. This local artifact is ignored by Git and
will be the input to the embedding and vector-search stage.

## Build the local vector index

Configure an AWS profile with permission to invoke Amazon Bedrock, then install
the remaining packages and build the index:

```bash
python -m pip install -r requirements.txt
python scripts/build_vector_index.py
```

The script sends each chunk to Amazon Bedrock Titan Text Embeddings V2 and
writes a local FAISS index plus matching citation metadata under
`data/processed/faiss_index/`. The local index is ignored by Git.
