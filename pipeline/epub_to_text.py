"""Utilities for extracting clean text from EPUB and PDF books.

This module is designed for dataset-building workflows where we want clean,
readable book text and we would rather skip noisy sections than pollute the
training corpus.
"""

from __future__ import annotations

import argparse
import html
import logging
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

try:
    import ebooklib
    from ebooklib import epub
except ImportError:  # pragma: no cover - handled at runtime.
    ebooklib = None
    epub = None

try:
    from bs4 import BeautifulSoup
    from bs4.dammit import UnicodeDammit
except ImportError:  # pragma: no cover - handled at runtime.
    BeautifulSoup = None
    UnicodeDammit = None

try:
    import fitz
except ImportError:  # pragma: no cover - optional PDF fallback.
    fitz = None


LOGGER_NAME = "pipeline.epub_to_text"
PREVIEW_LENGTH = 1000
PRINTABLE_ASCII_START = 33
PRINTABLE_ASCII_END = 126
PRINTABLE_ASCII_SPAN = PRINTABLE_ASCII_END - PRINTABLE_ASCII_START + 1
ROTATED_ASCII_SHIFT = 29

TEXT_BLOCK_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre")
DROP_HTML_TAGS = ("script", "style", "noscript", "svg", "math", "head", "title", "meta", "link")
SKIP_FILE_KEYWORDS = {
    "titlepage",
    "toc",
    "tableofcontents",
    "contents",
    "nav",
    "metadata",
    "cover",
    "copyright",
    "colophon",
    "imprint",
}
BOILERPLATE_PHRASES = {
    "table of contents",
    "contents",
    "copyright",
    "all rights reserved",
}
READABILITY_TOKENS = (
    " the ",
    " and ",
    " of ",
    " to ",
    " in ",
    " that ",
    " with ",
    " for ",
    " chapter ",
    " book ",
    " business ",
    " company ",
    " published ",
    " copyright ",
    " introduction ",
)
CHARACTER_REPLACEMENTS = str.maketrans(
    {
        "\u00a0": " ",
        "\u00ad": "",
        "\u200b": "",
        "\ufeff": "",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": " - ",
        "\u2014": " - ",
        "\u2026": "...",
        "\u00b6": "'",
        "\u2032": "'",
        "\u2033": '"',
        "\u00ae": "",
        "\u2122": "",
        "\ufffd": " ",
    }
)


class BookExtractionError(RuntimeError):
    """Raised when a book cannot be parsed into readable text."""


@dataclass
class ExtractionResult:
    """Structured result for a processed book."""

    source_path: Path
    text: str
    skipped_items: list[str] = field(default_factory=list)
    output_path: Optional[Path] = None


def configure_logger(level: int = logging.INFO) -> logging.Logger:
    """Create a reusable logger for extraction runs."""

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def _require_epub_dependencies() -> None:
    """Fail with a clear message when EPUB dependencies are missing."""

    if ebooklib is None or epub is None or BeautifulSoup is None or UnicodeDammit is None:
        raise ImportError(
            "EPUB extraction requires 'ebooklib' and 'beautifulsoup4'. "
            "Install them with: pip install ebooklib beautifulsoup4"
        )


def _require_pdf_dependency() -> None:
    """Fail with a clear message when the PDF fallback dependency is missing."""

    if fitz is None:
        raise ImportError(
            "PDF fallback extraction requires 'PyMuPDF'. "
            "Install it with: pip install PyMuPDF"
        )


def decode_content_bytes(raw_bytes: bytes) -> str:
    """Decode HTML bytes with a UTF-8-first strategy and a latin-1 fallback."""

    if not raw_bytes:
        return ""

    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    # UnicodeDammit is useful for messy real-world EPUBs that declare the
    # wrong encoding or contain a mix of encodings.
    if UnicodeDammit is not None:
        detected = UnicodeDammit(raw_bytes, ["utf-8", "cp1252", "latin-1"])
        if detected.unicode_markup:
            return detected.unicode_markup

    return raw_bytes.decode("latin-1", errors="ignore")


def rotate_printable_ascii(text: str, shift: int = ROTATED_ASCII_SHIFT) -> str:
    """Decode text that has been rotated across printable ASCII characters."""

    rotated_characters: list[str] = []

    for char in text:
        code_point = ord(char)

        # Preserve whitespace exactly so paragraph structure stays intact.
        if char.isspace():
            rotated_characters.append(char)
            continue

        if PRINTABLE_ASCII_START <= code_point <= PRINTABLE_ASCII_END:
            normalized = code_point - PRINTABLE_ASCII_START
            rotated = (normalized + shift) % PRINTABLE_ASCII_SPAN
            rotated_characters.append(chr(rotated + PRINTABLE_ASCII_START))
        else:
            rotated_characters.append(char)

    return "".join(rotated_characters)


def readability_score(text: str) -> float:
    """Score how readable a text fragment looks after cleanup."""

    if not text:
        return float("-inf")

    lowered = f" {text.lower()} "
    alpha_count = sum(character.isalpha() for character in text)
    whitespace_count = sum(character.isspace() for character in text)
    suspicious_count = sum(character in "\\[]{}|~^_" for character in text)
    word_count = len(re.findall(r"[A-Za-z']+", text))
    long_token_count = len(re.findall(r"[A-Za-z]{25,}", text))
    common_token_hits = sum(lowered.count(token) for token in READABILITY_TOKENS)

    length = max(len(text), 1)
    alpha_ratio = alpha_count / length
    whitespace_ratio = whitespace_count / length
    suspicious_ratio = suspicious_count / length

    return (
        common_token_hits * 3.0
        + word_count * 0.2
        + alpha_ratio * 4.0
        + whitespace_ratio * 2.5
        - suspicious_ratio * 6.0
        - long_token_count * 1.5
    )


def maybe_decode_obfuscated_text(text: str) -> str:
    """Recover text from printable-ASCII rotation when it improves readability.

    Some EPUBs generated from PDF-to-HTML tools contain text like
    '&RS\\ULJKW...' instead of 'Copyright...'. We decode that pattern only when
    the rotated text scores as meaningfully more readable.
    """

    decoded = rotate_printable_ascii(text)
    score_delta = readability_score(decoded) - readability_score(text)
    looks_obfuscated = bool(re.search(r"[\\$%&+/0-9]{2,}", text)) and any(
        character.isalpha() for character in text
    )

    if score_delta > 2.0:
        return decoded

    if looks_obfuscated and score_delta > 0.5:
        return decoded

    return text


def normalize_text(text: str) -> str:
    """Normalize text so downstream filtering works on clean strings."""

    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(CHARACTER_REPLACEMENTS)
    text = maybe_decode_obfuscated_text(text)
    text = unicodedata.normalize("NFKC", text)

    # Repair a few common spacing issues after decoding.
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_meaningful_text(text: str) -> bool:
    """Decide whether a block is readable enough to keep."""

    cleaned = normalize_text(text)
    lowered = cleaned.lower()

    if not cleaned:
        return False

    if lowered in BOILERPLATE_PHRASES:
        return False

    if re.fullmatch(r"(page\s*)?\d+", lowered):
        return False

    if re.fullmatch(r"[ivxlcdm]+", lowered):
        return False

    if len(cleaned) < 3:
        return False

    words = re.findall(r"[A-Za-z']+", cleaned)
    if not words:
        return False

    alpha_count = sum(character.isalpha() for character in cleaned)
    alpha_ratio = alpha_count / max(len(cleaned), 1)
    word_count = len(words)
    longest_word = max(len(word) for word in words)
    whitespace_ratio = sum(character.isspace() for character in cleaned) / max(len(cleaned), 1)

    # Allow short headings, but reject long unreadable blocks with no spacing.
    if word_count <= 8 and alpha_ratio >= 0.55 and longest_word <= 20:
        return True

    if word_count < 5:
        return False

    if whitespace_ratio < 0.04 and longest_word > 25:
        return False

    if readability_score(cleaned) < 1.5:
        return False

    return True


def deduplicate_consecutive_blocks(blocks: Iterable[str]) -> list[str]:
    """Drop duplicate neighboring blocks that sometimes appear in EPUB HTML."""

    deduplicated: list[str] = []

    for block in blocks:
        if not deduplicated or deduplicated[-1] != block:
            deduplicated.append(block)

    return deduplicated


def should_skip_epub_item(item_name: str) -> tuple[bool, str]:
    """Skip front matter and navigation files that do not add training value."""

    lowered_name = item_name.lower()

    for keyword in SKIP_FILE_KEYWORDS:
        if keyword in lowered_name:
            return True, f"matched skip keyword '{keyword}'"

    return False, ""


def iter_ordered_document_items(book) -> Iterable:
    """Yield EPUB document items in spine order, then any remaining documents."""

    manifest_items = {
        item.get_id(): item
        for item in book.get_items()
        if item.get_type() == ebooklib.ITEM_DOCUMENT
    }
    yielded_ids: set[str] = set()

    for spine_entry in getattr(book, "spine", []):
        item_id = spine_entry[0] if isinstance(spine_entry, (tuple, list)) else spine_entry
        item = manifest_items.get(item_id)

        if item is None or item_id in yielded_ids:
            continue

        yielded_ids.add(item_id)
        yield item

    for item_id in sorted(manifest_items):
        if item_id not in yielded_ids:
            yield manifest_items[item_id]


def extract_blocks_from_html(html_text: str) -> list[str]:
    """Extract readable text blocks from one EPUB HTML document."""

    soup = BeautifulSoup(html_text, "html.parser")

    # Remove HTML elements that never contain useful book text.
    for tag in soup.find_all(DROP_HTML_TAGS):
        tag.decompose()

    for tag in soup.find_all("nav"):
        tag.decompose()

    body = soup.body or soup
    blocks: list[str] = []

    for element in body.find_all(TEXT_BLOCK_TAGS):
        # Skip wrapper nodes that contain nested text blocks to avoid duplicates.
        if element.find(TEXT_BLOCK_TAGS):
            continue

        raw_text = element.get_text(separator=" ", strip=True)
        cleaned_text = normalize_text(raw_text)

        if is_meaningful_text(cleaned_text):
            blocks.append(cleaned_text)

    if blocks:
        return deduplicate_consecutive_blocks(blocks)

    # Fallback for books with minimal semantic markup.
    for line in body.get_text(separator="\n").splitlines():
        cleaned_line = normalize_text(line)

        if is_meaningful_text(cleaned_line):
            blocks.append(cleaned_line)

    return deduplicate_consecutive_blocks(blocks)


def validate_epub_archive(epub_path: Path) -> None:
    """Catch invalid or partially corrupted EPUB archives before parsing."""

    try:
        with zipfile.ZipFile(epub_path) as archive:
            corrupted_member = archive.testzip()
            if corrupted_member is not None:
                raise BookExtractionError(
                    f"EPUB archive is corrupted. First bad member: {corrupted_member}"
                )
    except zipfile.BadZipFile as error:
        raise BookExtractionError(f"Invalid EPUB archive: {epub_path}") from error


def extract_epub_book(epub_path: str | Path, logger: Optional[logging.Logger] = None) -> ExtractionResult:
    """Extract clean text from a single EPUB file."""

    _require_epub_dependencies()

    path = Path(epub_path)
    logger = logger or configure_logger()
    skipped_items: list[str] = []
    extracted_sections: list[str] = []

    if not path.exists():
        raise BookExtractionError(f"EPUB file not found: {path}")

    validate_epub_archive(path)

    try:
        book = epub.read_epub(str(path))
    except Exception as error:  # pragma: no cover - depends on ebooklib runtime.
        raise BookExtractionError(f"Failed to read EPUB '{path.name}': {error}") from error

    for item in iter_ordered_document_items(book):
        item_name = item.get_name()
        should_skip, reason = should_skip_epub_item(item_name)

        if should_skip:
            logger.info("Skipping EPUB item '%s' in %s: %s", item_name, path.name, reason)
            skipped_items.append(f"{item_name}: {reason}")
            continue

        try:
            html_text = decode_content_bytes(item.get_content())
            blocks = extract_blocks_from_html(html_text)
        except Exception as error:
            logger.warning(
                "Skipping EPUB item '%s' in %s because it could not be parsed: %s",
                item_name,
                path.name,
                error,
            )
            skipped_items.append(f"{item_name}: parse failure ({error})")
            continue

        if not blocks:
            logger.info(
                "Skipping EPUB item '%s' in %s: no meaningful readable text found",
                item_name,
                path.name,
            )
            skipped_items.append(f"{item_name}: no meaningful readable text")
            continue

        extracted_sections.append("\n\n".join(blocks))

    clean_text = normalize_text("\n\n".join(section for section in extracted_sections if section))

    if not clean_text:
        raise BookExtractionError(f"No readable text could be extracted from '{path.name}'.")

    return ExtractionResult(source_path=path, text=clean_text, skipped_items=skipped_items)


def clean_pdf_page_text(page_text: str) -> str:
    """Clean a single PDF page worth of extracted text."""

    lines: list[str] = []

    for raw_line in page_text.splitlines():
        cleaned_line = normalize_text(raw_line)

        if is_meaningful_text(cleaned_line):
            lines.append(cleaned_line)

    return "\n".join(deduplicate_consecutive_blocks(lines)).strip()


def pdf_to_text(pdf_path: str | Path, logger: Optional[logging.Logger] = None) -> str:
    """Fallback extractor for PDFs using PyMuPDF."""

    _require_pdf_dependency()

    path = Path(pdf_path)
    logger = logger or configure_logger()

    if not path.exists():
        raise BookExtractionError(f"PDF file not found: {path}")

    extracted_pages: list[str] = []

    try:
        with fitz.open(path) as document:
            for page_index, page in enumerate(document, start=1):
                page_text = clean_pdf_page_text(page.get_text("text", sort=True))

                if not page_text:
                    logger.info("Skipping PDF page %s in %s: no readable text", page_index, path.name)
                    continue

                extracted_pages.append(page_text)
    except Exception as error:  # pragma: no cover - depends on PyMuPDF runtime.
        raise BookExtractionError(f"Failed to read PDF '{path.name}': {error}") from error

    final_text = normalize_text("\n\n".join(extracted_pages))

    if not final_text:
        raise BookExtractionError(f"No readable text could be extracted from '{path.name}'.")

    return final_text


def save_text_output(
    text: str,
    source_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Save extracted text to a .txt file next to the source or in an output folder."""

    source = Path(source_path)
    destination_dir = Path(output_dir) if output_dir is not None else source.parent
    destination_dir.mkdir(parents=True, exist_ok=True)

    output_path = destination_dir / f"{source.stem}.txt"
    output_path.write_text(text, encoding="utf-8")

    return output_path


def extract_book_text(
    book_path: str | Path,
    save_txt: bool = False,
    output_dir: str | Path | None = None,
    fallback_pdf_path: str | Path | None = None,
    logger: Optional[logging.Logger] = None,
) -> ExtractionResult:
    """Extract text from a single book file and optionally save it as .txt."""

    path = Path(book_path)
    logger = logger or configure_logger()

    if path.suffix.lower() == ".epub":
        try:
            result = extract_epub_book(path, logger=logger)
        except BookExtractionError:
            if fallback_pdf_path is None:
                raise

            fallback_path = Path(fallback_pdf_path)
            logger.warning(
                "EPUB extraction failed for %s. Falling back to PDF: %s",
                path.name,
                fallback_path.name,
            )
            result = ExtractionResult(
                source_path=path,
                text=pdf_to_text(fallback_path, logger=logger),
                skipped_items=[f"EPUB fallback used PDF source: {fallback_path}"],
            )
    elif path.suffix.lower() == ".pdf":
        result = ExtractionResult(source_path=path, text=pdf_to_text(path, logger=logger))
    else:
        raise BookExtractionError(f"Unsupported file type: {path.suffix or '<no extension>'}")

    if save_txt:
        result.output_path = save_text_output(result.text, result.source_path, output_dir=output_dir)
        logger.info("Saved extracted text to %s", result.output_path)

    return result


def epub_to_text(
    file_path: str | Path,
    save_txt: bool = False,
    output_dir: str | Path | None = None,
    fallback_pdf_path: str | Path | None = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Backward-compatible helper that returns clean EPUB text as a string."""

    return extract_book_text(
        file_path,
        save_txt=save_txt,
        output_dir=output_dir,
        fallback_pdf_path=fallback_pdf_path,
        logger=logger,
    ).text


def iter_supported_books(folder_path: str | Path, recursive: bool = True) -> list[Path]:
    """Collect all supported book files from a folder."""

    folder = Path(folder_path)
    pattern = "**/*" if recursive else "*"

    return sorted(
        path
        for path in folder.glob(pattern)
        if path.is_file() and path.suffix.lower() in {".epub", ".pdf"}
    )


def process_books_in_folder(
    folder_path: str | Path,
    save_txt: bool = False,
    output_dir: str | Path | None = None,
    recursive: bool = True,
    logger: Optional[logging.Logger] = None,
) -> list[ExtractionResult]:
    """Process every supported book in a folder and continue past failures."""

    folder = Path(folder_path)
    logger = logger or configure_logger()

    if not folder.exists():
        raise BookExtractionError(f"Folder not found: {folder}")

    results: list[ExtractionResult] = []

    for book_path in iter_supported_books(folder, recursive=recursive):
        try:
            result = extract_book_text(
                book_path,
                save_txt=save_txt,
                output_dir=output_dir,
                logger=logger,
            )
        except Exception as error:
            logger.error("Failed to extract %s: %s", book_path.name, error)
            continue

        logger.info("Extracted %s characters from %s", len(result.text), book_path.name)
        results.append(result)

    return results


def build_preview(text: str, preview_length: int = PREVIEW_LENGTH) -> str:
    """Create a short preview string for manual verification."""

    return text[:preview_length]


def print_preview(text: str, source_path: str | Path, preview_length: int = PREVIEW_LENGTH) -> None:
    """Print the first 1000 characters so extraction quality can be checked quickly."""

    print(f"\n=== Preview: {Path(source_path).name} ===")
    print(build_preview(text, preview_length=preview_length))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for single-book or folder-wide extraction."""

    parser = argparse.ArgumentParser(description="Extract clean text from EPUB or PDF books.")
    parser.add_argument("input_path", help="Path to a single book file or a folder of books.")
    parser.add_argument(
        "--output-dir",
        help="Directory where .txt outputs should be saved. Defaults to the source folder.",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save extracted text as .txt files.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recurse into subfolders when the input path is a directory.",
    )
    parser.add_argument(
        "--fallback-pdf",
        help="Optional PDF file to use if EPUB extraction fails.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level, for example INFO or DEBUG.",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    arguments = parse_args()
    log_level = getattr(logging, str(arguments.log_level).upper(), logging.INFO)
    logger = configure_logger(log_level)
    input_path = Path(arguments.input_path)

    if input_path.is_dir():
        results = process_books_in_folder(
            input_path,
            save_txt=arguments.save_txt,
            output_dir=arguments.output_dir,
            recursive=not arguments.no_recursive,
            logger=logger,
        )

        for result in results:
            print_preview(result.text, result.source_path)
    else:
        result = extract_book_text(
            input_path,
            save_txt=arguments.save_txt,
            output_dir=arguments.output_dir,
            fallback_pdf_path=arguments.fallback_pdf,
            logger=logger,
        )
        print_preview(result.text, result.source_path)


if __name__ == "__main__":
    main()
