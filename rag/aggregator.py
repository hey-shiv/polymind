from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from config import load_personality_cards
from rag.types import RetrievedChunk


WORD_RE = re.compile(r"\b\w+\b")


def compute_basic_metrics(answer: str, citations: list[dict[str, Any]]) -> dict[str, float]:
    words = WORD_RE.findall(answer.lower())
    unique_ratio = len(set(words)) / max(1, len(words))
    source_count = len(citations)
    grounding_score = min(1.0, source_count / 4.0)

    return {
        "word_count": float(len(words)),
        "lexical_diversity": float(unique_ratio),
        "grounding_score": float(grounding_score),
        "source_count": float(source_count),
    }


def extract_takeaway(answer: str) -> str:
    for line in reversed([line.strip() for line in answer.splitlines() if line.strip()]):
        if line.lower().startswith("takeaway:"):
            return line.split(":", 1)[1].strip()

    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    return sentences[-1].strip() if sentences and sentences[-1].strip() else ""


class ResponseAggregator:
    def __init__(self, personality_cards: dict[str, dict[str, Any]] | None = None):
        self.personality_cards = personality_cards or load_personality_cards()

    def aggregate(
        self,
        query: str,
        expanded_query: str,
        raw_responses: list[dict[str, Any]],
        trace: dict[str, Any],
    ) -> dict[str, Any]:
        persona_cards: list[dict[str, Any]] = []

        for item in raw_responses:
            personality_id = item["personality_id"]
            persona = self.personality_cards[personality_id]
            answer = str(item["answer"]).strip()

            citations = [
                chunk.to_dict() if isinstance(chunk, RetrievedChunk) else dict(chunk)
                for chunk in item["chunks"]
            ]
            metrics = compute_basic_metrics(answer, citations)

            persona_cards.append(
                {
                    "personality_id": personality_id,
                    "display_name": persona["display_name"],
                    "model_name": item["model_name"],
                    "answer": answer,
                    "takeaway": extract_takeaway(answer),
                    "citations": citations,
                    "metrics": metrics,
                    "accent_color": persona.get("accent_color", "#444444"),
                    "card_class": persona.get("card_class", personality_id),
                }
            )

        return {
            "query": query,
            "expanded_query": expanded_query,
            "cards": persona_cards,
            "trace": trace,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
