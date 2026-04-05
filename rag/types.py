from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryRequest:
    query: str
    debate_mode: bool = False
    style_strength: float = 0.7
    top_k: int = 4
    session_id: str | None = None


@dataclass
class RetrievedChunk:
    chunk_id: str
    personality_id: str
    book_id: str
    title: str
    chapter: str
    text: str
    source_label: str
    score: float = 0.0
    start_offset: int | None = None
    end_offset: int | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any], score: float = 0.0) -> "RetrievedChunk":
        return cls(
            chunk_id=str(record.get("chunk_id", "")),
            personality_id=str(record.get("personality_id", "")),
            book_id=str(record.get("book_id", "")),
            title=str(record.get("title", "")),
            chapter=str(record.get("chapter", "")),
            text=str(record.get("text", "")),
            source_label=str(record.get("source_label", "")),
            score=float(score if score is not None else record.get("score", 0.0)),
            start_offset=record.get("start_offset"),
            end_offset=record.get("end_offset"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "personality_id": self.personality_id,
            "book_id": self.book_id,
            "title": self.title,
            "chapter": self.chapter,
            "text": self.text,
            "source_label": self.source_label,
            "score": self.score,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
        }


@dataclass
class RAGBundle:
    personality_id: str
    query: str
    expanded_query: str
    chunks: list[RetrievedChunk]
    context_text: str
    retrieval_backend: str = "unknown"


@dataclass
class PersonaResponse:
    personality_id: str
    display_name: str
    model_name: str
    answer: str
    citations: list[RetrievedChunk] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
