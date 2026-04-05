from __future__ import annotations

import re
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "POLYMIND_COLAB_INSTRUCTION_MANUAL.md"
OUTPUT_PATH = ROOT / "POLYMIND_COLAB_INSTRUCTION_MANUAL.pdf"

PAGE_WIDTH = 595
PAGE_HEIGHT = 842
LEFT_MARGIN = 50
RIGHT_MARGIN = 50
TOP_MARGIN = 56
BOTTOM_MARGIN = 50
BODY_FONT_SIZE = 11
HEADING_FONT_SIZE = 16
SUBHEADING_FONT_SIZE = 13
LINE_HEIGHT = 15
SECTION_GAP = 10
TEXT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
CHARS_PER_LINE = 88


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def markdown_to_blocks(markdown_text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    in_code_block = False
    code_buffer: list[str] = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()

        if line.startswith("```"):
            if in_code_block:
                blocks.append(("code", "\n".join(code_buffer).rstrip()))
                code_buffer = []
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_buffer.append(line)
            continue

        stripped = line.strip()
        if not stripped:
            blocks.append(("blank", ""))
            continue

        if stripped.startswith("# "):
            blocks.append(("heading1", stripped[2:].strip()))
            continue

        if stripped.startswith("## "):
            blocks.append(("heading2", stripped[3:].strip()))
            continue

        if stripped.startswith("- "):
            blocks.append(("bullet", stripped[2:].strip()))
            continue

        if re.match(r"^\d+\.\s+", stripped):
            blocks.append(("number", stripped))
            continue

        blocks.append(("paragraph", stripped))

    return blocks


def wrap_block(block_type: str, text: str) -> list[tuple[str, str]]:
    if block_type == "blank":
        return [("blank", "")]
    if block_type == "heading1":
        return [("heading1", text)]
    if block_type == "heading2":
        return [("heading2", text)]
    if block_type == "code":
        lines = text.splitlines() or [""]
        return [("code", line) for line in lines]
    if block_type == "bullet":
        wrapped = textwrap.wrap(text, width=CHARS_PER_LINE - 4) or [text]
        return [("bullet", wrapped[0])] + [("bullet_cont", line) for line in wrapped[1:]]
    if block_type == "number":
        match = re.match(r"^(\d+\.)\s+(.*)$", text)
        prefix = match.group(1) if match else "1."
        body = match.group(2) if match else text
        wrapped = textwrap.wrap(body, width=CHARS_PER_LINE - len(prefix) - 2) or [body]
        return [("number", f"{prefix} {wrapped[0]}")] + [
            ("number_cont", line) for line in wrapped[1:]
        ]

    wrapped = textwrap.wrap(text, width=CHARS_PER_LINE) or [text]
    return [("paragraph", line) for line in wrapped]


def prepare_lines(markdown_text: str) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    for block_type, text in markdown_to_blocks(markdown_text):
        lines.extend(wrap_block(block_type, text))
    return lines


def line_height_for(kind: str) -> int:
    if kind == "heading1":
        return 22
    if kind == "heading2":
        return 18
    if kind in {"blank"}:
        return 8
    return LINE_HEIGHT


def font_for(kind: str) -> tuple[str, int]:
    if kind == "heading1":
        return ("F1", HEADING_FONT_SIZE)
    if kind == "heading2":
        return ("F1", SUBHEADING_FONT_SIZE)
    if kind == "code":
        return ("F2", 10)
    return ("F1", BODY_FONT_SIZE)


def x_position_for(kind: str) -> int:
    if kind == "bullet":
        return LEFT_MARGIN
    if kind == "bullet_cont":
        return LEFT_MARGIN + 16
    if kind == "number":
        return LEFT_MARGIN
    if kind == "number_cont":
        return LEFT_MARGIN + 20
    if kind == "code":
        return LEFT_MARGIN + 12
    return LEFT_MARGIN


def paginate(lines: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    pages: list[list[tuple[str, str]]] = []
    current_page: list[tuple[str, str]] = []
    y = PAGE_HEIGHT - TOP_MARGIN

    for item in lines:
        kind, _ = item
        needed = line_height_for(kind)
        if y - needed < BOTTOM_MARGIN and current_page:
            pages.append(current_page)
            current_page = []
            y = PAGE_HEIGHT - TOP_MARGIN

        current_page.append(item)
        y -= needed

        if kind in {"heading1", "heading2"}:
            y -= 2
        elif kind == "code":
            y -= 1

    if current_page:
        pages.append(current_page)

    return pages


def build_page_stream(page_lines: list[tuple[str, str]], page_number: int, total_pages: int) -> str:
    commands: list[str] = []
    y = PAGE_HEIGHT - TOP_MARGIN

    for kind, text in page_lines:
        if kind == "blank":
            y -= line_height_for(kind)
            continue

        font_name, font_size = font_for(kind)
        x = x_position_for(kind)

        if kind == "bullet":
            bullet_y = y
            commands.append(f"BT /{font_name} {font_size} Tf 0 g 38 {bullet_y} Td (\\267) Tj ET")
        elif kind == "number":
            pass

        if kind == "code":
            escaped = escape_pdf_text(text)
            commands.append(
                f"BT /{font_name} {font_size} Tf 0.1 0.1 0.1 rg {x} {y} Td ({escaped}) Tj ET"
            )
        elif kind == "heading1":
            escaped = escape_pdf_text(text)
            commands.append(
                f"BT /{font_name} {font_size} Tf 0.05 0.05 0.05 rg {x} {y} Td ({escaped}) Tj ET"
            )
        elif kind == "heading2":
            escaped = escape_pdf_text(text)
            commands.append(
                f"BT /{font_name} {font_size} Tf 0.12 0.12 0.12 rg {x} {y} Td ({escaped}) Tj ET"
            )
        else:
            escaped = escape_pdf_text(text)
            commands.append(
                f"BT /{font_name} {font_size} Tf 0.1 0.1 0.1 rg {x} {y} Td ({escaped}) Tj ET"
            )

        y -= line_height_for(kind)

    footer = f"Page {page_number} of {total_pages}"
    commands.append(
        f"BT /F1 9 Tf 0.35 0.35 0.35 rg {PAGE_WIDTH - RIGHT_MARGIN - 60} 24 Td ({escape_pdf_text(footer)}) Tj ET"
    )
    return "\n".join(commands)


def build_pdf(pages: list[list[tuple[str, str]]]) -> bytes:
    objects: list[bytes] = []

    def add_object(payload: str | bytes) -> int:
        data = payload.encode("latin-1") if isinstance(payload, str) else payload
        objects.append(data)
        return len(objects)

    catalog_id = add_object("<< /Type /Catalog /Pages 2 0 R >>")
    # Placeholder for pages tree, replaced later.
    pages_tree_id = add_object("")
    font_helvetica_id = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_courier_id = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    page_object_ids: list[int] = []
    content_object_ids: list[int] = []

    total_pages = len(pages)
    for page_number, page_lines in enumerate(pages, start=1):
        stream = build_page_stream(page_lines, page_number, total_pages)
        content_id = add_object(
            f"<< /Length {len(stream.encode('latin-1'))} >>\nstream\n{stream}\nendstream"
        )
        content_object_ids.append(content_id)
        page_id = add_object(
            (
                "<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Contents {content_id} 0 R "
                f"/Resources << /Font << /F1 {font_helvetica_id} 0 R /F2 {font_courier_id} 0 R >> >> >>"
            )
        )
        page_object_ids.append(page_id)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_object_ids)
    objects[pages_tree_id - 1] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_object_ids)} >>".encode("latin-1")
    )

    buffer = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]

    for index, obj in enumerate(objects, start=1):
        offsets.append(len(buffer))
        buffer.extend(f"{index} 0 obj\n".encode("latin-1"))
        buffer.extend(obj)
        buffer.extend(b"\nendobj\n")

    xref_offset = len(buffer)
    buffer.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    buffer.extend(b"0000000000 65535 f \n")

    for offset in offsets[1:]:
        buffer.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))

    buffer.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("latin-1")
    )
    return bytes(buffer)


def main() -> None:
    markdown_text = SOURCE_PATH.read_text(encoding="utf-8")
    lines = prepare_lines(markdown_text)
    pages = paginate(lines)
    OUTPUT_PATH.write_bytes(build_pdf(pages))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
