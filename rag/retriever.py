from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Protocol

import numpy as np

from config import DEFAULT_BOOTSTRAP_CORPUS, SETTINGS, Settings, load_personality_cards
from rag.types import RAGBundle, RetrievedChunk

try:
    import faiss
except ImportError:  # pragma: no cover - optional dependency.
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency.
    SentenceTransformer = None


WORD_RE = re.compile(r"[A-Za-z0-9']+")


class RetrieverProtocol(Protocol):
    backend_name: str

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        ...


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_bootstrap_records() -> list[dict]:
    personality_cards = load_personality_cards()
    records: list[dict] = []

    for personality_id, entries in DEFAULT_BOOTSTRAP_CORPUS.items():
        display_name = personality_cards[personality_id]["display_name"]
        for index, entry in enumerate(entries):
            records.append(
                {
                    "chunk_id": f"{personality_id}_bootstrap_{index:04d}",
                    "personality_id": personality_id,
                    "book_id": "bootstrap_notes",
                    "title": entry["title"],
                    "chapter": entry["chapter"],
                    "text": entry["text"],
                    "source_label": f"{display_name} / {entry['title']} / {entry['chapter']}",
                    "start_offset": None,
                    "end_offset": None,
                }
            )

    return records


def load_chunk_records(chunks_path: Path = SETTINGS.chunks_path) -> list[dict]:
    records = _load_jsonl(chunks_path)
    if records:
        return records
    return build_bootstrap_records()


class LexicalRetriever:
    backend_name = "lexical"

    def __init__(self, records: list[dict]):
        self.records = [dict(record) for record in records]
        self.record_counters = [
            Counter(
                tokenize(
                    " ".join(
                        [
                            str(record.get("title", "")),
                            str(record.get("chapter", "")),
                            str(record.get("source_label", "")),
                            str(record.get("text", "")),
                        ]
                    )
                )
            )
            for record in self.records
        ]

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        query_counter = Counter(tokenize(query))
        scored: list[tuple[float, dict]] = []

        for record, token_counter in zip(self.records, self.record_counters):
            overlap = sum(min(query_counter[token], token_counter.get(token, 0)) for token in query_counter)
            if overlap == 0 and query_counter:
                continue

            length_penalty = math.sqrt(max(1, sum(token_counter.values())))
            score = (overlap / length_penalty) if query_counter else 0.0
            scored.append((score, record))

        if not scored:
            scored = [(0.0, record) for record in self.records]

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk.from_record(record, score=score)
            for score, record in scored[: max(1, top_k)]
        ]


class FaissRetriever:
    backend_name = "faiss"

    def __init__(self, index_path: Path, metadata_path: Path, model_dir: Path):
        if faiss is None or SentenceTransformer is None:
            raise ImportError(
                "FAISS retrieval requires `faiss-cpu` and `sentence-transformers`."
            )

        self.index = faiss.read_index(str(index_path))
        self.records = _load_jsonl(metadata_path)
        self.model = SentenceTransformer(str(model_dir))

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        vector = self.model.encode([query], normalize_embeddings=True)
        vector = np.asarray(vector, dtype="float32")
        scores, indices = self.index.search(vector, top_k)

        results: list[RetrievedChunk] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or index >= len(self.records):
                continue
            results.append(RetrievedChunk.from_record(self.records[index], score=float(score)))
        return results


def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No retrieved context available."

    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(f"[Source {index}] {chunk.source_label}\n{chunk.text}")
    return "\n\n".join(blocks)


class PersonaRAGService:
    def __init__(self, retrievers: dict[str, RetrieverProtocol]):
        self.retrievers = retrievers

    def build_bundle(
        self,
        personality_id: str,
        query: str,
        expanded_query: str,
        top_k: int = 4,
    ) -> RAGBundle:
        retrieval_query = expanded_query or query
        retriever = self.retrievers[personality_id]
        chunks = retriever.search(retrieval_query, top_k=top_k)
        return RAGBundle(
            personality_id=personality_id,
            query=query,
            expanded_query=expanded_query,
            chunks=chunks,
            context_text=format_context(chunks),
            retrieval_backend=retriever.backend_name,
        )

    def build_all(
        self,
        query: str,
        expanded_query: str,
        top_k: int,
        personality_ids: tuple[str, ...] = SETTINGS.persona_ids,
    ) -> dict[str, RAGBundle]:
        return {
            personality_id: self.build_bundle(
                personality_id=personality_id,
                query=query,
                expanded_query=expanded_query,
                top_k=top_k,
            )
            for personality_id in personality_ids
        }


def build_personality_retrievers(settings: Settings = SETTINGS) -> dict[str, RetrieverProtocol]:
    personality_cards = load_personality_cards(settings.personality_cards_path)
    records = load_chunk_records(settings.chunks_path)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["personality_id"])].append(record)

    bootstrap_grouped: dict[str, list[dict]] = defaultdict(list)
    for record in build_bootstrap_records():
        bootstrap_grouped[str(record["personality_id"])].append(record)

    retrievers: dict[str, RetrieverProtocol] = {}
    for personality_id in personality_cards:
        personality_records = grouped.get(personality_id) or bootstrap_grouped[personality_id]
        index_path = settings.embeddings_dir / f"{personality_id}.faiss"
        metadata_path = settings.embeddings_dir / f"{personality_id}_metadata.jsonl"
        use_faiss = (
            index_path.exists()
            and metadata_path.exists()
            and settings.embedding_model_dir.exists()
            and faiss is not None
            and SentenceTransformer is not None
        )

        if use_faiss:
            try:
                retrievers[personality_id] = FaissRetriever(
                    index_path=index_path,
                    metadata_path=metadata_path,
                    model_dir=settings.embedding_model_dir,
                )
                continue
            except Exception:
                pass

        retrievers[personality_id] = LexicalRetriever(personality_records)

    return retrievers
