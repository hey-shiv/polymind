"""TinyStories dataset utilities for the mini LLM."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Union

import numpy as np
import torch

from download_dataset import DATA_PATH, ensure_dataset_exists
from src.tokenizer import (
    EOS_TOKEN,
    TOKENIZER_PATH,
    encode,
    get_special_token_id,
    get_vocab_size,
    load_tokenizer,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TOKEN_IDS_PATH = DATA_DIR / "dataset_tokens.uint16.bin"
TOKEN_META_PATH = DATA_DIR / "dataset_tokens_meta.json"
TOKEN_DTYPE = np.uint16


@dataclass(frozen=True)
class DatasetBundle:
    data_path: Path
    tokenizer_path: Path
    token_ids_path: Path
    raw_text_bytes: int
    total_tokens: int
    vocab_size: int
    train_data: np.memmap
    val_data: np.memmap


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _iter_stories(data_path: Path) -> Iterator[str]:
    story_lines: list[str] = []

    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                story_lines.append(line.rstrip("\n"))
                continue

            if story_lines:
                yield "\n".join(story_lines)
                story_lines.clear()

    if story_lines:
        yield "\n".join(story_lines)


def _load_cache_meta(meta_path: Path) -> dict[str, object] | None:
    if not meta_path.exists():
        return None

    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _cache_is_valid(data_path: Path, tokenizer_path: Path, meta_path: Path) -> bool:
    meta = _load_cache_meta(meta_path)
    if meta is None:
        return False

    token_ids_path = Path(str(meta.get("token_ids_path", "")))
    if not token_ids_path.exists():
        return False

    data_stat = data_path.stat()
    tokenizer_hash = _file_sha256(tokenizer_path)

    return (
        meta.get("data_path") == str(data_path.resolve())
        and meta.get("data_size") == data_stat.st_size
        and meta.get("data_mtime_ns") == data_stat.st_mtime_ns
        and meta.get("tokenizer_path") == str(tokenizer_path.resolve())
        and meta.get("tokenizer_sha256") == tokenizer_hash
        and meta.get("dtype") == "uint16"
    )


def _build_token_cache(
    data_path: Path,
    tokenizer_path: Path,
    token_ids_path: Path,
    meta_path: Path,
) -> dict[str, object]:
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = get_vocab_size(tokenizer)
    if vocab_size >= np.iinfo(TOKEN_DTYPE).max:
        raise ValueError(
            f"Vocabulary size {vocab_size} does not fit in {TOKEN_DTYPE.__name__}."
        )

    eos_id = get_special_token_id(tokenizer, EOS_TOKEN)
    if eos_id is None:
        raise ValueError(f"{EOS_TOKEN} is missing from the tokenizer vocabulary.")

    tmp_token_ids_path = token_ids_path.with_suffix(".tmp")
    tmp_meta_path = meta_path.with_suffix(".tmp")
    token_ids_path.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    story_count = 0

    with tmp_token_ids_path.open("wb") as handle:
        for story in _iter_stories(data_path):
            token_ids = encode(story, tokenizer, add_eos=True)
            if not token_ids:
                continue

            np.asarray(token_ids, dtype=TOKEN_DTYPE).tofile(handle)
            total_tokens += len(token_ids)
            story_count += 1

            if story_count % 50_000 == 0:
                print(
                    f"Tokenized {story_count:,} stories "
                    f"({total_tokens:,} tokens written)..."
                )

    data_stat = data_path.stat()
    meta = {
        "data_path": str(data_path.resolve()),
        "data_size": data_stat.st_size,
        "data_mtime_ns": data_stat.st_mtime_ns,
        "tokenizer_path": str(tokenizer_path.resolve()),
        "tokenizer_sha256": _file_sha256(tokenizer_path),
        "token_ids_path": str(token_ids_path.resolve()),
        "dtype": "uint16",
        "total_tokens": total_tokens,
        "story_count": story_count,
        "vocab_size": vocab_size,
        "eos_token_id": eos_id,
    }

    with tmp_meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    tmp_token_ids_path.replace(token_ids_path)
    tmp_meta_path.replace(meta_path)
    return meta


def ensure_token_cache(
    data_path: Union[str, Path] = DATA_PATH,
    tokenizer_path: Union[str, Path] = TOKENIZER_PATH,
    token_ids_path: Union[str, Path] = TOKEN_IDS_PATH,
    meta_path: Union[str, Path] = TOKEN_META_PATH,
) -> tuple[Path, dict[str, object]]:
    """Download TinyStories if needed and cache token ids for fast training."""
    resolved_data_path = ensure_dataset_exists(Path(data_path))
    resolved_tokenizer_path = Path(tokenizer_path)
    resolved_token_ids_path = Path(token_ids_path)
    resolved_meta_path = Path(meta_path)

    if not resolved_tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {resolved_tokenizer_path}. "
            "Run `python train_tokenizer.py` first."
        )

    if _cache_is_valid(resolved_data_path, resolved_tokenizer_path, resolved_meta_path):
        meta = _load_cache_meta(resolved_meta_path)
        if meta is None:
            raise RuntimeError("Token cache metadata disappeared while loading.")
        return resolved_token_ids_path, meta

    print("Building BPE token cache for TinyStories...")
    meta = _build_token_cache(
        data_path=resolved_data_path,
        tokenizer_path=resolved_tokenizer_path,
        token_ids_path=resolved_token_ids_path,
        meta_path=resolved_meta_path,
    )
    return resolved_token_ids_path, meta


def load_dataset(
    data_path: Union[str, Path] = DATA_PATH,
    tokenizer_path: Union[str, Path] = TOKENIZER_PATH,
    val_ratio: float = 0.1,
) -> DatasetBundle:
    """Load TinyStories token ids and return train/validation splits."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    resolved_data_path = Path(data_path)
    resolved_tokenizer_path = Path(tokenizer_path)
    token_ids_path, meta = ensure_token_cache(
        data_path=resolved_data_path,
        tokenizer_path=resolved_tokenizer_path,
    )

    total_tokens = int(meta["total_tokens"])
    data = np.memmap(token_ids_path, dtype=TOKEN_DTYPE, mode="r", shape=(total_tokens,))

    split_idx = int((1.0 - val_ratio) * total_tokens)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    return DatasetBundle(
        data_path=Path(meta["data_path"]),
        tokenizer_path=Path(meta["tokenizer_path"]),
        token_ids_path=Path(meta["token_ids_path"]),
        raw_text_bytes=int(meta["data_size"]),
        total_tokens=total_tokens,
        vocab_size=int(meta["vocab_size"]),
        train_data=train_data,
        val_data=val_data,
    )


def get_batch(
    data: Union[np.ndarray, torch.Tensor],
    batch_size: int,
    context_length: int,
    device: Union[str, torch.device],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of input and next-token target sequences."""
    if len(data) <= context_length:
        raise ValueError("data must be longer than context_length.")

    max_start = len(data) - context_length - 1
    start_indices = torch.randint(max_start + 1, (batch_size,)).tolist()

    if isinstance(data, torch.Tensor):
        x = torch.stack([data[i : i + context_length] for i in start_indices])
        y = torch.stack([data[i + 1 : i + context_length + 1] for i in start_indices])
        return x.to(device), y.to(device)

    x_np = np.stack(
        [np.asarray(data[i : i + context_length], dtype=np.int64) for i in start_indices]
    )
    y_np = np.stack(
        [
            np.asarray(data[i + 1 : i + context_length + 1], dtype=np.int64)
            for i in start_indices
        ]
    )
    return torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)
