"""Local FAISS vector-store helpers for FinSight development."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class FaissVectorStore:
    """Persist normalized vectors and their transcript-citation metadata locally."""

    INDEX_FILENAME = "transcript_chunks.faiss"
    METADATA_FILENAME = "transcript_chunks.metadata.jsonl"

    def __init__(self, index: faiss.Index, metadata: list[dict[str, object]]) -> None:
        if index.ntotal != len(metadata):
            raise ValueError("The FAISS index and metadata must have the same number of records")
        self.index = index
        self.metadata = metadata

    @classmethod
    def from_vectors(
        cls, vectors: np.ndarray, metadata: list[dict[str, object]]
    ) -> "FaissVectorStore":
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("vectors must be a non-empty two-dimensional array")
        if len(metadata) != vectors.shape[0]:
            raise ValueError("Each vector requires one metadata record")

        # Titan output is normalized. Inner-product search on normalized vectors
        # equals cosine-similarity search, which ranks semantic similarity.
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors.astype(np.float32)))
        return cls(index=index, metadata=metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict[str, object]]:
        """Return the most semantically similar chunks and their citation metadata."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results: list[dict[str, object]] = []
        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue
            result = dict(self.metadata[index])
            result["similarity_score"] = float(score)
            results.append(result)
        return results

    def save(self, directory: Path) -> None:
        """Save the binary index and matching JSONL metadata side by side."""
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / self.INDEX_FILENAME))
        with (directory / self.METADATA_FILENAME).open("w", encoding="utf-8") as file:
            for record in self.metadata:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: Path) -> "FaissVectorStore":
        """Load an index previously created by ``save``."""
        index_path = directory / cls.INDEX_FILENAME
        metadata_path = directory / cls.METADATA_FILENAME
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"No saved FAISS index found in {directory}")

        with metadata_path.open(encoding="utf-8") as file:
            metadata = [json.loads(line) for line in file if line.strip()]
        return cls(index=faiss.read_index(str(index_path)), metadata=metadata)
