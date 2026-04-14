from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import (
    ChunkRecord,
    PolymindConfig,
    RetrievalInspection,
    RetrievalResult,
    StageProfile,
)
from .embeddings import cosine_similarity, detect_device

LOGGER = logging.getLogger("polymind.retrieval")


class SimpleIndex:
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def search(self, query_embeddings: np.ndarray, k: int):
        query = np.asarray(query_embeddings, dtype=np.float32)
        scores = np.matmul(query, self.embeddings.T)
        top_indices = np.argsort(-scores, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top_indices, axis=1)
        return top_scores.astype(np.float32), top_indices.astype(np.int64)


def build_index(embeddings: np.ndarray):
    if embeddings.size == 0:
        return SimpleIndex(np.zeros((0, 1), dtype=np.float32))
    try:
        import faiss  # type: ignore
    except ImportError:
        LOGGER.info("faiss not installed; using numpy fallback index.")
        return SimpleIndex(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index


def save_index(index, path: str | Path) -> None:
    output_path = Path(path)
    try:
        import faiss  # type: ignore

        if hasattr(index, "ntotal"):
            faiss.write_index(index, str(output_path))
            return
    except ImportError:
        pass

    with output_path.open("wb") as handle:
        pickle.dump(index, handle)


def load_index(path: str | Path):
    index_path = Path(path)
    try:
        import faiss  # type: ignore

        index = faiss.read_index(str(index_path))
        faiss.write_index(index, str(index_path))
        return index
    except ImportError:
        with index_path.open("rb") as handle:
            return pickle.load(handle)


def _call_generator(generator, prompt: str, max_new_tokens: int) -> str:
    if generator is None:
        return ""
    if callable(generator):
        return generator(prompt, max_new_tokens=max_new_tokens)
    if hasattr(generator, "generate_text"):
        return generator.generate_text(prompt, max_new_tokens=max_new_tokens)
    if hasattr(generator, "generate"):
        return generator.generate(prompt, max_new_tokens=max_new_tokens)
    return ""


def expand_query(
    query: str,
    generator=None,
    config: Optional[PolymindConfig] = None,
) -> str:
    config = config or PolymindConfig()
    if not config.enable_hyde:
        return ""
    prompt = (
        "Write a short hypothetical answer paragraph that would likely appear in the book.\n"
        f"Question: {query}\n"
        "Hypothetical answer:"
    )
    try:
        return _call_generator(generator, prompt, config.token_budget("hyde")).strip()
    except Exception:
        return ""


def _embed_query(text: str, model) -> np.ndarray:
    vector = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    return np.asarray(vector, dtype=np.float32)


def retrieve_chunks(
    query: str,
    rewritten_query: str,
    chunks: Sequence[ChunkRecord],
    chunk_embeddings: np.ndarray,
    index,
    embed_model,
    config: PolymindConfig,
    generator=None,
    top_k: Optional[int] = None,
) -> Tuple[List[RetrievalResult], RetrievalInspection, StageProfile]:
    start_time = time.perf_counter()
    resolved_top_k = top_k or config.top_k
    hyde_query = expand_query(rewritten_query, generator=generator, config=config)

    query_embedding = _embed_query(rewritten_query, embed_model)
    if hyde_query:
        hyde_embedding = _embed_query(hyde_query, embed_model)
        search_embedding = ((query_embedding + hyde_embedding) / 2.0).astype(np.float32)
    else:
        search_embedding = query_embedding

    search_count = min(len(chunks), max(resolved_top_k, resolved_top_k * config.initial_retrieval_multiplier))
    scores, indices = index.search(np.asarray([search_embedding], dtype=np.float32), search_count)

    raw_top_k: List[Dict[str, Any]] = []
    selected_results: List[RetrievalResult] = []
    chapter_counts: Dict[int, int] = {}
    dedup_events: List[str] = []
    diversity_events: List[str] = []
    selected_embeddings: List[np.ndarray] = []

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[int(idx)]
        raw_top_k.append(
            {
                "chunk_id": chunk.chunk_id,
                "chapter_id": chunk.chapter_id,
                "chapter_title": chunk.chapter_title,
                "score": float(score),
            }
        )

        duplicate = False
        chunk_embedding = chunk_embeddings[int(idx)]
        for chosen_embedding, chosen in zip(selected_embeddings, selected_results):
            similarity = cosine_similarity(chunk_embedding, chosen_embedding)
            if similarity > config.retrieval_dedup_threshold:
                dedup_events.append(
                    f"Dropped chunk {chunk.chunk_id} as near-duplicate of {chosen.chunk_id} ({similarity:.3f})."
                )
                duplicate = True
                break
        if duplicate:
            continue

        if chapter_counts.get(chunk.chapter_id, 0) >= config.chapter_diversity_limit:
            diversity_events.append(
                f"Skipped chunk {chunk.chunk_id} to preserve chapter diversity for chapter {chunk.chapter_id}."
            )
            continue

        chapter_counts[chunk.chapter_id] = chapter_counts.get(chunk.chapter_id, 0) + 1
        selected_embeddings.append(chunk_embedding)
        selected_results.append(
            RetrievalResult(
                chunk_id=chunk.chunk_id,
                chapter_id=chunk.chapter_id,
                chapter_title=chunk.chapter_title,
                text=chunk.text,
                score=float(score),
                rank=len(selected_results) + 1,
                source_reason="hyde+query" if hyde_query else "query",
                metadata={"preview": chunk.text[:180]},
            )
        )
        if len(selected_results) >= resolved_top_k:
            break

    inspection = RetrievalInspection(
        query=query,
        rewritten_query=rewritten_query,
        hyde_query=hyde_query,
        raw_top_k=raw_top_k,
        selected_results=[asdict(result) for result in selected_results],
        chapter_counts=chapter_counts,
        dedup_events=dedup_events,
        diversity_events=diversity_events,
    )
    profile = StageProfile(
        stage="retrieve_chunks",
        runtime_s=round(time.perf_counter() - start_time, 4),
        device=detect_device(),
        peak_memory_mb=0.0,
        notes=f"top_k={resolved_top_k}",
    )
    return selected_results, inspection, profile


def retrieve_for_concepts(
    concept_a: str,
    concept_b: str,
    chunks: Sequence[ChunkRecord],
    chunk_embeddings: np.ndarray,
    index,
    embed_model,
    config: PolymindConfig,
    generator=None,
) -> Tuple[List[RetrievalResult], Dict[str, RetrievalInspection]]:
    left_results, left_inspection, _ = retrieve_chunks(
        query=concept_a,
        rewritten_query=concept_a,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        index=index,
        embed_model=embed_model,
        config=config,
        generator=generator,
        top_k=config.top_k,
    )
    right_results, right_inspection, _ = retrieve_chunks(
        query=concept_b,
        rewritten_query=concept_b,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        index=index,
        embed_model=embed_model,
        config=config,
        generator=generator,
        top_k=config.top_k,
    )
    merged: List[RetrievalResult] = []
    seen = set()
    for result in left_results + right_results:
        if result.chunk_id in seen:
            continue
        merged.append(result)
        seen.add(result.chunk_id)
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[: config.top_k], {"concept_a": left_inspection, "concept_b": right_inspection}


def format_retrieval_context(results: Sequence[RetrievalResult]) -> str:
    context_blocks = []
    for result in results:
        context_blocks.append(
            f"[Chapter {result.chapter_id:02d} | Chunk {result.chunk_id:03d} | Score {result.score:.3f}]\n"
            f"{result.text}"
        )
    return "\n\n".join(context_blocks)
