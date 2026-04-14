from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .config import BookSection, ChunkInspectionRow, ChunkRecord, PolymindConfig

LOGGER = logging.getLogger("polymind.chunking")

ABBREVIATIONS = {
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "sr.",
    "jr.",
    "st.",
    "vs.",
    "etc.",
    "u.s.",
    "u.k.",
    "e.g.",
    "i.e.",
}


@dataclass
class SentenceSpan:
    text: str
    start: int
    end: int


def split_into_sentences(text: str) -> List[SentenceSpan]:
    try:
        import nltk  # type: ignore

        try:
            tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        except LookupError:
            nltk.download("punkt", quiet=True)
            tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        raw_sentences = tokenizer.tokenize(text)
    except Exception:
        protected = text
        for abbr in ABBREVIATIONS:
            protected = re.sub(
                re.escape(abbr),
                lambda match: match.group(0).replace(".", "<prd>"),
                protected,
                flags=re.IGNORECASE,
            )
        protected = re.sub(
            r"\b(?:[A-Z]\.){2,}",
            lambda match: match.group(0).replace(".", "<prd>"),
            protected,
        )
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", protected)
        raw_sentences = [sentence.replace("<prd>", ".") for sentence in raw_sentences]

    spans: List[SentenceSpan] = []
    cursor = 0
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        start = text.find(sentence, cursor)
        if start < 0:
            start = text.find(sentence)
        if start < 0:
            continue
        end = start + len(sentence)
        spans.append(SentenceSpan(text=sentence, start=start, end=end))
        cursor = end
    return spans


def _overlap_sentences(buffer: Sequence[SentenceSpan], overlap_words: int) -> List[SentenceSpan]:
    selected: List[SentenceSpan] = []
    running_words = 0
    for sentence in reversed(buffer):
        selected.insert(0, sentence)
        running_words += len(sentence.text.split())
        if running_words >= overlap_words:
            break
    return selected


def semantic_chunk_book(
    sections: List[BookSection],
    config: PolymindConfig,
) -> List[ChunkRecord]:
    chunks: List[ChunkRecord] = []
    chunk_id = 0
    for section in sections:
        sentences = split_into_sentences(section.text)
        buffer: List[SentenceSpan] = []
        buffer_words = 0
        for sentence in sentences:
            buffer.append(sentence)
            buffer_words += len(sentence.text.split())
            if buffer_words < config.chunk_target_words:
                continue
            chunk_text = " ".join(item.text for item in buffer).strip()
            start_char = section.start_char + buffer[0].start
            end_char = section.start_char + buffer[-1].end
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    chapter_id=section.chapter_id,
                    chapter_title=section.chapter_title,
                    text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    sentence_count=len(buffer),
                    word_count=len(chunk_text.split()),
                    source_path=section.source_path,
                    metadata={"section_start_char": section.start_char},
                )
            )
            chunk_id += 1
            buffer = _overlap_sentences(buffer, config.chunk_overlap_words)
            buffer_words = sum(len(item.text.split()) for item in buffer)

        if buffer:
            chunk_text = " ".join(item.text for item in buffer).strip()
            if chunk_text:
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        chapter_id=section.chapter_id,
                        chapter_title=section.chapter_title,
                        text=chunk_text,
                        start_char=section.start_char + buffer[0].start,
                        end_char=section.start_char + buffer[-1].end,
                        sentence_count=len(buffer),
                        word_count=len(chunk_text.split()),
                        source_path=section.source_path,
                        metadata={"section_start_char": section.start_char},
                    )
                )
                chunk_id += 1
    return chunks


def inspect_chunks(
    chunks: List[ChunkRecord],
    config: PolymindConfig,
    limit: int = 10,
) -> List[ChunkInspectionRow]:
    rows: List[ChunkInspectionRow] = []
    previous_end = None
    for chunk in chunks[:limit]:
        warnings: List[str] = []
        if chunk.word_count < config.min_chunk_words:
            warnings.append("under_target")
        if chunk.word_count > config.max_chunk_words:
            warnings.append("over_target")
        if chunk.sentence_count < 2:
            warnings.append("low_sentence_count")
        if chunk.start_char >= chunk.end_char:
            warnings.append("invalid_offsets")
        if previous_end is not None and chunk.start_char < previous_end:
            warnings.append("overlapping_offsets")
        overlap_ratio = 0.0
        if previous_end is not None and chunk.end_char > chunk.start_char:
            overlap_ratio = max(0.0, min(1.0, (previous_end - chunk.start_char) / (chunk.end_char - chunk.start_char)))
        rows.append(
            ChunkInspectionRow(
                chunk_id=chunk.chunk_id,
                chapter_id=chunk.chapter_id,
                chapter_title=chunk.chapter_title,
                word_count=chunk.word_count,
                sentence_count=chunk.sentence_count,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                overlap_ratio=round(overlap_ratio, 4),
                warnings=warnings,
                preview=chunk.text[:180],
            )
        )
        previous_end = chunk.end_char
    return rows
