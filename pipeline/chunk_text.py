import json
from pathlib import Path

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def process_books(input_dir, output_file):
    dataset = []

    for file in Path(input_dir).glob("*.txt"):
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        chunks = chunk_text(text)

        for chunk in chunks:
            dataset.append({
                "text": chunk,
                "source": file.name
            })

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)


if __name__ == "__main__":
    process_books(
        "data/processed/books",
        "data/processed/chunks.json"
    )