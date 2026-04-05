from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from config import SETTINGS, ensure_local_layout


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"Page\s+\d+\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def split_into_chunks(
    text: str,
    target_words: int = 320,
    overlap_words: int = 60,
) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for paragraph in paragraphs:
        paragraph_word_count = len(paragraph.split())

        if current and current_words + paragraph_word_count > target_words:
            chunk_text = "\n\n".join(current).strip()
            chunks.append(chunk_text)

            overlap_buffer = chunk_text.split()[-overlap_words:]
            overlap_text = " ".join(overlap_buffer).strip()
            current = [overlap_text, paragraph] if overlap_text else [paragraph]
            current_words = len(overlap_text.split()) + paragraph_word_count
            continue

        current.append(paragraph)
        current_words += paragraph_word_count

    if current:
        chunks.append("\n\n".join(current).strip())

    return chunks


def infer_book_id(path: Path) -> str:
    return path.stem.lower().replace(" ", "_")


def build_chunk_records(
    personality_id: str,
    book_id: str,
    title: str,
    text: str,
) -> list[dict[str, Any]]:
    chunk_texts = split_into_chunks(text)
    records: list[dict[str, Any]] = []

    search_start = 0
    for index, chunk_text in enumerate(chunk_texts):
        start_offset = text.find(chunk_text, search_start)
        if start_offset < 0:
            start_offset = search_start
        end_offset = start_offset + len(chunk_text)
        search_start = end_offset

        records.append(
            {
                "chunk_id": f"{personality_id}_{book_id}_{index:04d}",
                "personality_id": personality_id,
                "book_id": book_id,
                "title": title,
                "chapter": "",
                "text": chunk_text,
                "source_label": f"{personality_id} / {title} / chunk {index + 1}",
                "start_offset": start_offset,
                "end_offset": end_offset,
            }
        )

    return records


def ingest_books(
    raw_books_dir: Path = SETTINGS.raw_books_dir,
    output_path: Path = SETTINGS.chunks_path,
) -> int:
    ensure_local_layout()
    records_written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for personality_dir in sorted(raw_books_dir.iterdir()):
            if not personality_dir.is_dir():
                continue

            personality_id = personality_dir.name
            for text_file in sorted(personality_dir.rglob("*.txt")):
                text = clean_text(text_file.read_text(encoding="utf-8", errors="ignore"))
                if not text:
                    continue

                book_id = infer_book_id(text_file)
                title = text_file.stem

                for record in build_chunk_records(personality_id, book_id, title, text):
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records_written += 1

    return records_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest persona-organized books into chunk records.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=SETTINGS.raw_books_dir,
        help="Directory containing raw books organized by personality.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=SETTINGS.chunks_path,
        help="Where to write the chunked JSONL corpus.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = ingest_books(raw_books_dir=args.input_dir, output_path=args.output_path)
    print(f"Wrote {total} chunk records to {args.output_path}")


if __name__ == "__main__":
    main()
