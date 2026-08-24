"""Grounded answer generation from FinSight's retrieved transcript evidence."""

from __future__ import annotations

import os
from collections.abc import Sequence

import boto3


DEFAULT_CHAT_MODEL_ID = "amazon.nova-lite-v1:0"


def _source_label(result: dict[str, object], number: int) -> str:
    """Create the stable source label referenced in the generated answer."""
    return f"S{number}"


def _format_context(results: Sequence[dict[str, object]]) -> str:
    """Format retrieved chunks into an explicitly delimited model context."""
    passages: list[str] = []
    for number, result in enumerate(results, start=1):
        source = _source_label(result, number)
        metadata = (
            f"{result.get('ticker', 'Unknown')} {result.get('quarter', 'Unknown')} "
            f"{result.get('year', 'Unknown')} | {result.get('title', 'Earnings call')}"
        )
        passages.append(
            f"[SOURCE {source}: {metadata}]\n{result['text']}\n[END SOURCE {source}]"
        )
    return "\n\n".join(passages)


class GroundedAnswerGenerator:
    """Use Amazon Nova Lite to answer only from retrieved source passages."""

    def __init__(
        self,
        region_name: str | None = None,
        profile_name: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.profile_name = profile_name or os.getenv("AWS_PROFILE")
        self.model_id = model_id or os.getenv("BEDROCK_CHAT_MODEL_ID", DEFAULT_CHAT_MODEL_ID)
        session = boto3.Session(profile_name=self.profile_name, region_name=self.region_name)
        self.client = session.client("bedrock-runtime")

    def answer(self, question: str, results: Sequence[dict[str, object]]) -> str:
        """Return a concise answer with source labels such as ``[S1]``."""
        if not question.strip():
            raise ValueError("Cannot answer an empty question")
        if not results:
            raise ValueError("Cannot answer without retrieved source passages")

        system_prompt = """You are FinSight, an evidence-grounded equity research assistant.
Answer using only the supplied earnings-call sources. Do not use outside knowledge,
make investment recommendations, or infer facts that are not directly supported.
Give a concise answer in 2–4 sentences. Cite every factual claim with one or more
source labels in square brackets, for example [S1] or [S1][S2]. If the sources do
not answer the question, say so plainly and do not speculate."""
        user_prompt = (
            f"Question: {question}\n\n"
            f"Retrieved earnings-call sources:\n{_format_context(results)}"
        )

        response = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": 400, "temperature": 0.1, "topP": 0.9},
        )
        return response["output"]["message"]["content"][0]["text"].strip()
