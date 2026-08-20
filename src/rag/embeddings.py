"""Amazon Bedrock Titan embedding client for FinSight transcript chunks."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable

import boto3
import numpy as np


DEFAULT_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_DIMENSIONS = 1024


class BedrockEmbedder:
    """Create normalized float embeddings with Amazon Titan Text Embeddings V2."""

    def __init__(
        self,
        region_name: str | None = None,
        profile_name: str | None = None,
        model_id: str | None = None,
        dimensions: int = DEFAULT_DIMENSIONS,
    ) -> None:
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.profile_name = profile_name or os.getenv("AWS_PROFILE")
        self.model_id = model_id or os.getenv("BEDROCK_EMBEDDING_MODEL_ID", DEFAULT_MODEL_ID)
        self.dimensions = dimensions

        # boto3 uses the standard AWS credential chain. In local development it
        # will use the named AWS CLI profile selected in the .env file.
        session = boto3.Session(profile_name=self.profile_name, region_name=self.region_name)
        self.client = session.client("bedrock-runtime")

    def embed(self, text: str) -> np.ndarray:
        """Embed one string and return a normalized float32 vector."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {
                    "inputText": text,
                    "dimensions": self.dimensions,
                    "normalize": True,
                }
            ),
        )
        payload = json.loads(response["body"].read())
        vector = np.asarray(payload["embedding"], dtype=np.float32)
        if vector.shape != (self.dimensions,):
            raise RuntimeError(
                f"Expected a {self.dimensions}-dimension vector, received {vector.shape}"
            )
        return vector

    def embed_many(self, texts: Iterable[str]) -> np.ndarray:
        """Embed texts one at a time; small corpus size keeps this simple and reliable."""
        vectors = [self.embed(text) for text in texts]
        if not vectors:
            return np.empty((0, self.dimensions), dtype=np.float32)
        return np.vstack(vectors)
