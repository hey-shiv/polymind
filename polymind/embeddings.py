from __future__ import annotations

import logging
import time
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .config import ChunkRecord, PolymindConfig, StageProfile

LOGGER = logging.getLogger("polymind.embeddings")


def detect_device() -> str:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        return "cpu"
    return "cpu"


def load_embedding_model(model_name: str, device: str | None = None):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for real embeddings. Install requirements.txt in Colab."
        ) from exc

    resolved_device = device or detect_device()
    LOGGER.info("Loading embedding model %s on %s", model_name, resolved_device)
    return SentenceTransformer(model_name, device=resolved_device)


def _tqdm(iterable: Iterable, desc: str):
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm(iterable, desc=desc)
    except Exception:
        return iterable


def _move_model_to_cpu(model) -> None:
    if hasattr(model, "to"):
        try:
            model.to("cpu")
        except Exception:
            LOGGER.info("Embedding model does not support .to('cpu'); continuing.")


def _clear_cuda_cache() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def embed_texts(
    texts: Sequence[str],
    model,
    config: PolymindConfig,
) -> Tuple[np.ndarray, StageProfile]:
    start_time = time.perf_counter()
    all_vectors: List[np.ndarray] = []
    peak_memory_mb = 0.0
    current_device = detect_device()
    for start in _tqdm(range(0, len(texts), config.embedding_batch_size), desc="Embedding batches"):
        batch = list(texts[start : start + config.embedding_batch_size])
        if not batch:
            continue
        try:
            vectors = model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=len(batch),
                show_progress_bar=False,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            LOGGER.warning("CUDA OOM during embedding batch; falling back to CPU.")
            _clear_cuda_cache()
            current_device = "cpu"
            _move_model_to_cpu(model)
            vectors = model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=max(8, min(16, len(batch))),
                show_progress_bar=False,
            )
        all_vectors.append(np.asarray(vectors, dtype=np.float32))

        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                peak_memory_mb = max(
                    peak_memory_mb,
                    float(torch.cuda.max_memory_allocated() / (1024**2)),
                )
                torch.cuda.empty_cache()
        except Exception:
            pass

    embeddings = np.vstack(all_vectors) if all_vectors else np.zeros((0, 384), dtype=np.float32)
    profile = StageProfile(
        stage="embed_chunks",
        runtime_s=round(time.perf_counter() - start_time, 4),
        device=current_device,
        peak_memory_mb=round(peak_memory_mb, 2),
    )
    return embeddings, profile


def embed_chunks(
    chunks: List[ChunkRecord],
    model,
    config: PolymindConfig,
) -> Tuple[np.ndarray, StageProfile]:
    texts = [chunk.text for chunk in chunks]
    return embed_texts(texts, model, config)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denominator)
