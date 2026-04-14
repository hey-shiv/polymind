from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


class FakeEmbeddingModel:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, **_kwargs):
        self.calls += 1
        vectors: List[np.ndarray] = []
        for text in texts:
            text = str(text).lower()
            leadership = text.count("leadership")
            mastery = text.count("mastery")
            practice = text.count("practice")
            strategy = text.count("strategy")
            length = max(len(text.split()), 1)
            vector = np.array(
                [
                    leadership + 1.0,
                    mastery + 1.0,
                    practice + 1.0,
                    strategy + 1.0,
                    float(length),
                ],
                dtype=np.float32,
            )
            vector = vector / (np.linalg.norm(vector) + 1e-8)
            vectors.append(vector)
        return np.vstack(vectors)


class FakeGenerator:
    def __call__(self, prompt: str, max_new_tokens: int = 0, temperature: float = 0.2):
        return self.generate_text(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

    def generate_text(self, prompt: str, max_new_tokens: int = 0, temperature: float = 0.2):
        prompt_lower = prompt.lower()
        if "rewritten question:" in prompt_lower:
            return "How does Greene connect leadership and social intelligence?"
        if "updated summary:" in prompt_lower:
            return "The conversation focused on leadership, apprenticeship, and social intelligence."
        if "hypothetical answer:" in prompt_lower:
            return "Mastery grows through apprenticeship, observation, and strategic social intelligence."
        if "connect these ideas" in prompt_lower:
            return (
                "Greene links leadership and social intelligence through apprenticeship and observation "
                "[Chapter 01 | Chunk 000]"
            )
        arm = "grounded"
        if "patient teacher" in prompt_lower:
            arm = "teacher"
        elif "connect themes" in prompt_lower:
            arm = "synth"
        return (
            f"{arm} answer about mastery, leadership, and practice "
            "[Chapter 01 | Chunk 000]"
        )


def write_tmp_book(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "CHAPTER 1: Apprenticeship",
                "Dr. Greene argues that mastery begins with practice. The U.S. example still holds.",
                "Leadership grows through observation and social intelligence.",
                "CHAPTER 2: Strategy",
                "Strategy depends on patient practice and adaptive thinking.",
            ]
        ),
        encoding="utf-8",
    )
