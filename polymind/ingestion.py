from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .config import BookSection, PolymindConfig

LOGGER = logging.getLogger("polymind.ingestion")

CHAPTER_PATTERNS = [
    r"(?im)^(chapter\s+\d+[:.\- ]+.*)$",
    r"(?im)^([ivxlcdm]+\.\s+[A-Z][^\n]{5,})$",
    r"(?im)^(\d+\.\s+[A-Z][^\n]{5,})$",
    r"(?im)^([A-Z][A-Z0-9 ,:'\-]{8,})$",
]


def read_book_text(path: str | Path) -> str:
    book_path = Path(path)
    suffix = book_path.suffix.lower()
    if suffix == ".txt":
        return book_path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        return extract_pdf_text(book_path)
    raise ValueError(f"Unsupported book format: {book_path.suffix}")


def extract_pdf_text(path: str | Path) -> str:
    book_path = Path(path)
    try:
        import fitz  # type: ignore
    except ImportError:
        fitz = None

    if fitz is not None:
        text_parts: List[str] = []
        with fitz.open(book_path) as document:
            for page in document:
                text_parts.append(page.get_text("text"))
        return "\n".join(text_parts)

    from pypdf import PdfReader

    reader = PdfReader(str(book_path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def extract_font_headings(path: str | Path, size_threshold: float = 14.0) -> List[Tuple[int, str, int]]:
    book_path = Path(path)
    try:
        import fitz  # type: ignore
    except ImportError:
        LOGGER.info("PyMuPDF not available, skipping font-based heading extraction.")
        return []

    headings: List[Tuple[int, str, int]] = []
    char_cursor = 0
    with fitz.open(book_path) as document:
        for page in document:
            page_text = page.get_text("text")
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    line_text = "".join(span.get("text", "") for span in spans).strip()
                    max_size = max(float(span.get("size", 0.0)) for span in spans)
                    if max_size < size_threshold or len(line_text) < 5:
                        continue
                    local_offset = page_text.find(line_text)
                    if local_offset >= 0:
                        headings.append((char_cursor + local_offset, line_text, page.number + 1))
            char_cursor += len(page_text) + 1
    return headings


def clean_text(text: str) -> str:
    replacements = {
        "\u00a0": " ",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\u2014": " - ",
        "\u2013": " - ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^[ \t]+", "", text)
    return text.strip()


def _normalize_heading(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip()).lower()


def detect_chapters(
    text: str,
    font_headings: Optional[Iterable[Tuple[int, str, int]]] = None,
    dedup_window: int = 50,
) -> List[Tuple[int, str, int]]:
    candidates: List[Tuple[int, str, int]] = []
    for pattern in CHAPTER_PATTERNS:
        for match in re.finditer(pattern, text):
            title = match.group(1).strip()
            if len(title.split()) < 2:
                continue
            candidates.append((match.start(), title, 0))

    if font_headings:
        candidates.extend((start, title, page) for start, title, page in font_headings)

    candidates.sort(key=lambda item: item[0])
    deduped: List[Tuple[int, str, int]] = []
    for start, title, page in candidates:
        norm = _normalize_heading(title)
        if len(norm) < 5:
            continue
        is_duplicate = False
        for existing_start, existing_title, _ in deduped:
            if abs(existing_start - start) <= dedup_window:
                existing_norm = _normalize_heading(existing_title)
                if norm == existing_norm or norm in existing_norm or existing_norm in norm:
                    is_duplicate = True
                    break
        if not is_duplicate:
            deduped.append((start, title, page))
    return deduped


def build_book_sections(
    text: str,
    chapters: List[Tuple[int, str, int]],
    source_path: str | Path = "",
) -> List[BookSection]:
    if not chapters:
        return [
            BookSection(
                chapter_id=0,
                chapter_title="Book",
                text=text,
                start_char=0,
                end_char=len(text),
                source_path=str(source_path),
            )
        ]

    sections: List[BookSection] = []
    sorted_chapters = sorted(chapters, key=lambda item: item[0])
    for idx, (start, title, _page) in enumerate(sorted_chapters):
        end = sorted_chapters[idx + 1][0] if idx + 1 < len(sorted_chapters) else len(text)
        section_text = text[start:end].strip()
        if not section_text:
            continue
        sections.append(
            BookSection(
                chapter_id=idx + 1,
                chapter_title=title,
                text=section_text,
                start_char=start,
                end_char=end,
                source_path=str(source_path),
            )
        )
    return sections


def ingest_book(
    book_path: str | Path,
    config: Optional[PolymindConfig] = None,
) -> Dict[str, object]:
    config = config or PolymindConfig()
    book_path = Path(book_path)
    LOGGER.info("Loading book from %s", book_path)
    raw_text = read_book_text(book_path)
    cleaned_text = clean_text(raw_text)
    font_headings = extract_font_headings(book_path) if book_path.suffix.lower() == ".pdf" else []
    chapters = detect_chapters(cleaned_text, font_headings=font_headings)
    sections = build_book_sections(cleaned_text, chapters, source_path=book_path)
    metadata = {
        "book_path": str(book_path),
        "raw_characters": len(raw_text),
        "clean_characters": len(cleaned_text),
        "chapter_count": len(chapters),
        "section_count": len(sections),
    }
    return {
        "raw_text": raw_text,
        "clean_text": cleaned_text,
        "font_headings": font_headings,
        "chapters": chapters,
        "sections": sections,
        "metadata": metadata,
    }


def save_sections(path: str | Path, sections: List[BookSection]) -> None:
    payload = [
        {
            "chapter_id": section.chapter_id,
            "chapter_title": section.chapter_title,
            "text": section.text,
            "start_char": section.start_char,
            "end_char": section.end_char,
            "source_path": section.source_path,
        }
        for section in sections
    ]
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
