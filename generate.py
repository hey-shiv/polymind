import argparse
from pathlib import Path
from typing import Tuple

import torch
from tokenizers import Tokenizer

from src.device import get_default_device
from src.model import MiniLLM
from src.tokenizer import EOS_TOKEN, TOKENIZER_PATH, decode, encode, load_tokenizer


DEFAULT_MODEL_PATH = Path(__file__).resolve().with_name("model.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with the trained mini LLM.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to continue.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Number of tokens to generate after the prompt.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the saved model checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    return parser.parse_args()


def _resolve_tokenizer_path(checkpoint: dict[str, object], model_path: Path) -> Path:
    checkpoint_tokenizer_path = checkpoint.get("tokenizer_path")
    if isinstance(checkpoint_tokenizer_path, str):
        candidate = Path(checkpoint_tokenizer_path)
        if not candidate.is_absolute():
            candidate = model_path.parent / candidate
        if candidate.exists():
            return candidate

    if TOKENIZER_PATH.exists():
        return TOKENIZER_PATH

    raise FileNotFoundError(
        "Tokenizer not found next to the checkpoint or at the repository root."
    )


def load_model(model_path: Path, device: torch.device) -> Tuple[MiniLLM, Tokenizer]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {model_path}. Run `python train.py` first."
        )

    checkpoint = torch.load(model_path, map_location=device)
    if (
        not isinstance(checkpoint, dict)
        or "config" not in checkpoint
        or "model_state_dict" not in checkpoint
    ):
        raise ValueError(
            "Unsupported checkpoint format. Expected a dict with `config` and "
            "`model_state_dict`."
        )

    config = checkpoint["config"]
    model = MiniLLM(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer_path = _resolve_tokenizer_path(checkpoint, model_path)
    tokenizer = load_tokenizer(tokenizer_path)
    return model, tokenizer


@torch.no_grad()
def generate_text(
    model: MiniLLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> str:
    if not prompt:
        raise ValueError("prompt must not be empty.")
    if temperature < 0:
        raise ValueError("temperature must be >= 0.")

    input_ids = encode(prompt, tokenizer)
    if not input_ids:
        raise ValueError("Prompt was tokenized into an empty sequence.")

    tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    for _ in range(max_new_tokens):
        context = tokens[:, -model.max_seq_len :]
        logits, _ = model(context)
        next_token_logits = logits[:, -1, :]

        if temperature == 0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_token], dim=1)

        if eos_id is not None and next_token.item() == eos_id:
            break

    return decode(tokens[0].tolist(), tokenizer, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be >= 0.")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = get_default_device()
    model, tokenizer = load_model(args.model_path, device)
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
    )
    print(output)


if __name__ == "__main__":
    main()
