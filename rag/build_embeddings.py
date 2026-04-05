from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from config import SETTINGS, ensure_local_layout
from rag.retriever import load_chunk_records

try:
    import faiss
except ImportError:  # pragma: no cover - optional dependency.
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency.
    SentenceTransformer = None


def _require_embedding_dependencies() -> None:
    if faiss is None or SentenceTransformer is None:
        raise ImportError(
            "Building FAISS indices requires `faiss-cpu` and `sentence-transformers`. "
            "Install the updated requirements first."
        )


def _write_metadata(records: list[dict], out_meta_path: Path) -> None:
    out_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with out_meta_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_faiss_index(
    records: list[dict],
    embedding_model: SentenceTransformer,
    out_index_path: Path,
    out_meta_path: Path,
) -> int:
    texts = [record["text"] for record in records]
    vectors = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    vectors = np.asarray(vectors, dtype="float32")

    dim = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    out_index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_index_path))
    _write_metadata(records, out_meta_path)
    return len(records)


def build_all_indices(
    embedding_model_dir: Path = SETTINGS.embedding_model_dir,
    embeddings_dir: Path = SETTINGS.embeddings_dir,
) -> dict[str, int]:
    _require_embedding_dependencies()
    ensure_local_layout()

    records = load_chunk_records(SETTINGS.chunks_path)
    if not records:
        raise ValueError("No chunk records were found. Run `python -m rag.ingest_books` first.")

    model = SentenceTransformer(str(embedding_model_dir))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["personality_id"])].append(record)

    stats: dict[str, int] = {}
    stats["global"] = build_faiss_index(
        records=records,
        embedding_model=model,
        out_index_path=embeddings_dir / "global.faiss",
        out_meta_path=embeddings_dir / "global_metadata.jsonl",
    )

    for personality_id, personality_records in grouped.items():
        stats[personality_id] = build_faiss_index(
            records=personality_records,
            embedding_model=model,
            out_index_path=embeddings_dir / f"{personality_id}.faiss",
            out_meta_path=embeddings_dir / f"{personality_id}_metadata.jsonl",
        )

    stats_path = embeddings_dir / "stats.json"
    corpus_counts = Counter(record["personality_id"] for record in records)
    payload = {
        "records_indexed": int(len(records)),
        "records_per_personality": dict(corpus_counts),
        "embedding_model_dir": str(embedding_model_dir),
        "built_indices": stats,
    }
    stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build global and per-personality FAISS indices.")
    parser.add_argument(
        "--embedding-model-dir",
        type=Path,
        default=SETTINGS.embedding_model_dir,
        help="Local directory for the embedding model.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=SETTINGS.embeddings_dir,
        help="Directory where FAISS indices and metadata should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_all_indices(
        embedding_model_dir=args.embedding_model_dir,
        embeddings_dir=args.embeddings_dir,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
