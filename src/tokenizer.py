"""Byte-pair tokenizer utilities for the mini LLM."""

from pathlib import Path
from typing import Iterable, Union

from tokenizers import Tokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT_DIR / "tokenizer.json"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"


def load_tokenizer(tokenizer_path: Union[str, Path] = TOKENIZER_PATH) -> Tokenizer:
    """Load the trained tokenizer from disk."""
    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {path}. Run `python train_tokenizer.py` first."
        )
    return Tokenizer.from_file(str(path))


def get_vocab_size(tokenizer: Tokenizer) -> int:
    """Return the tokenizer vocabulary size."""
    return tokenizer.get_vocab_size()


def get_special_token_id(tokenizer: Tokenizer, token: str) -> int | None:
    """Look up the integer id for a special token if it exists."""
    return tokenizer.token_to_id(token)


def encode(
    text: str,
    tokenizer: Tokenizer,
    add_bos: bool = False,
    add_eos: bool = False,
) -> list[int]:
    """Convert text into BPE token ids using the trained tokenizer."""
    token_ids = tokenizer.encode(text).ids

    if add_bos:
        bos_id = get_special_token_id(tokenizer, BOS_TOKEN)
        if bos_id is None:
            raise ValueError(f"{BOS_TOKEN} is missing from the tokenizer vocabulary.")
        token_ids = [bos_id, *token_ids]

    if add_eos:
        eos_id = get_special_token_id(tokenizer, EOS_TOKEN)
        if eos_id is None:
            raise ValueError(f"{EOS_TOKEN} is missing from the tokenizer vocabulary.")
        token_ids = [*token_ids, eos_id]

    return token_ids


def decode(
    tokens: Iterable[int],
    tokenizer: Tokenizer,
    skip_special_tokens: bool = True,
) -> str:
    """Convert token ids back into text using the trained tokenizer."""
    return tokenizer.decode(list(tokens), skip_special_tokens=skip_special_tokens)
