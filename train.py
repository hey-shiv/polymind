from pathlib import Path

import torch

from src.dataset import get_batch, load_dataset
from src.device import get_default_device
from src.model import MiniLLM


# Training configuration
BATCH_SIZE = 64
CONTEXT_LENGTH = 256
TRAIN_STEPS = 3000
LOG_INTERVAL = 100
LEARNING_RATE = 3e-4
MODEL_PATH = Path(__file__).resolve().with_name("model.pt")


# Model configuration
D_MODEL = 256
N_LAYERS = 4
N_HEADS = 8
N_KV_HEADS = 2
FFN_HIDDEN_DIM = 680
DROPOUT = 0.2


def main() -> None:
    torch.manual_seed(1337)
    device = get_default_device()

    dataset = load_dataset()
    train_data = dataset.train_data
    val_data = dataset.val_data
    vocab_size = dataset.vocab_size

    print(f"Device: {device}")
    print(f"Dataset file: {dataset.data_path}")
    print(f"Dataset size: {dataset.raw_text_bytes / (1024 ** 3):.2f} GB")
    print(f"Tokenizer: {dataset.tokenizer_path}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")

    model = MiniLLM(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        ffn_hidden_dim=FFN_HIDDEN_DIM,
        max_seq_len=CONTEXT_LENGTH,
        dropout=DROPOUT,
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(1, TRAIN_STEPS + 1):
        xb, yb = get_batch(
            train_data,
            batch_size=BATCH_SIZE,
            context_length=CONTEXT_LENGTH,
            device=device,
        )

        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % LOG_INTERVAL == 0:
            print(f"step {step:4d} | train loss {loss.item():.4f}")

    checkpoint = {
        "model_state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        "config": {
            "vocab_size": vocab_size,
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "n_kv_heads": N_KV_HEADS,
            "ffn_hidden_dim": FFN_HIDDEN_DIM,
            "max_seq_len": CONTEXT_LENGTH,
            "dropout": DROPOUT,
        },
        "tokenizer_path": "tokenizer.json",
        "dataset_path": "data/dataset.txt",
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"Saved model checkpoint to {MODEL_PATH}")


if __name__ == "__main__":
    main()
