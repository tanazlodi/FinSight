"""Embed FinSight chunks with Bedrock Titan and create a local FAISS index.

Run from the project root after AWS credentials and Bedrock access are ready:
    python scripts/build_vector_index.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.embeddings import BedrockEmbedder
from src.rag.vector_store import FaissVectorStore


def read_chunks(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(
            "No processed chunks found. Run `python scripts/build_chunks.py` first."
        )
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    chunks = read_chunks(PROJECT_ROOT / "data" / "processed" / "transcript_chunks.jsonl")
    embedder = BedrockEmbedder()

    print(f"Embedding {len(chunks):,} transcript chunks with {embedder.model_id}...")
    vectors = embedder.embed_many(str(chunk["text"]) for chunk in chunks)
    store = FaissVectorStore.from_vectors(vectors, chunks)

    index_directory = PROJECT_ROOT / "data" / "processed" / "faiss_index"
    store.save(index_directory)
    print(f"Saved {store.index.ntotal:,} vectors to {index_directory.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
