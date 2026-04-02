from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "roneneldan/TinyStories"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_PATH = DATA_DIR / "dataset.txt"


def ensure_dataset_exists(data_path: Path = DATA_PATH) -> Path:
    """Download TinyStories once and store it as a plain text file."""
    path = Path(data_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        print(f"Dataset already exists at {path}")
        return path

    print("Downloading TinyStories dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")

    print("Writing dataset to file...")
    with path.open("w", encoding="utf-8") as handle:
        for example in dataset:
            text = example["text"].strip()
            if text:
                handle.write(text)
                handle.write("\n\n")

    print(f"Dataset saved to {path}")
    return path


def main() -> None:
    ensure_dataset_exists()


if __name__ == "__main__":
    main()
