from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
GUIDE_PATH = ROOT / "POLYMIND_V2_COLAB_GUIDE.md"
PDF_PATH = ROOT / "POLYMIND_V2_COLAB_GUIDE.pdf"
NOTEBOOK_PATH = ROOT / "notebooks" / "polymind_v2_colab.ipynb"
BENCHMARK_PATH = ROOT / "benchmarks" / "mastery_eval_set.json"
TEMPLATE_PATH = ROOT / "scripts" / "pandoc_template.tex"
VARIABLES_PATH = ROOT / "scripts" / "pandoc_variables.yaml"

MAX_CODE_LINE_LENGTH = 88
MIN_PAGE_COUNT = 90


def iter_markdown_code_lines(text: str):
    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            yield line


def validate_guide_structure(text: str) -> None:
    sections = re.findall(r"^#\s+\d+\.\s+.+$", text, flags=re.M)
    assert len(sections) == 23, f"Expected 23 top-level sections, found {len(sections)}"

    parts = re.split(r"^#\s+\d+\.\s+.+$", text, flags=re.M)[1:]
    for index, part in enumerate(parts, start=1):
        for heading in [
            "## Intuition",
            "## Colab-ready code cell",
            "## Expected output",
            "## Debugging tips",
        ]:
            assert heading in part, f"Section {index} is missing {heading}"

    required_phrases = [
        "System introspection",
        "Design tradeoffs",
        "Failure modes and recovery",
        "Experimentation framework",
        "Benchmarking",
        "Performance and cost",
    ]
    lowered = text.lower()
    for phrase in required_phrases:
        assert phrase.lower() in lowered, f"Guide is missing integrated theme: {phrase}"


def validate_markdown_code_lines(text: str) -> None:
    long_lines = [
        line for line in iter_markdown_code_lines(text)
        if len(line) > MAX_CODE_LINE_LENGTH
    ]
    assert not long_lines, f"Guide has code lines over {MAX_CODE_LINE_LENGTH} chars"


def validate_notebook() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code = "\n\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for line in cell.source.splitlines():
            assert len(line) <= MAX_CODE_LINE_LENGTH, (
                f"Notebook line exceeds {MAX_CODE_LINE_LENGTH}: {line}"
            )

    required_markers = [
        "chunk_target_words",
        "temperature",
        "primary_model",
        "baseline_dense",
        "\"mode\": \"cached\"",
        "system.policy.learning_curve()",
    ]
    for marker in required_markers:
        assert marker in code, f"Notebook is missing required experiment marker: {marker}"


def validate_benchmark_fixture() -> None:
    data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    assert len(data) >= 20, "Benchmark fixture needs at least 20 questions"
    tags = {tag for item in data for tag in item["tags"]}
    for required_tag in ["single-hop", "cross-chapter", "follow-up", "concept-link"]:
        assert required_tag in tags, f"Benchmark fixture missing tag: {required_tag}"
    for item in data:
        assert item["expected_chapter_ids"], f"{item['id']} is missing chapter ids"
        assert item["expected_keywords"], f"{item['id']} is missing keywords"


def validate_pdf() -> None:
    reader = PdfReader(str(PDF_PATH))
    assert len(reader.pages) >= MIN_PAGE_COUNT, (
        f"Guide PDF must be at least {MIN_PAGE_COUNT} pages, got {len(reader.pages)}"
    )


def validate_export_artifacts() -> None:
    assert TEMPLATE_PATH.exists(), f"Missing pandoc template: {TEMPLATE_PATH}"
    assert VARIABLES_PATH.exists(), f"Missing pandoc variables: {VARIABLES_PATH}"


def main() -> None:
    text = GUIDE_PATH.read_text(encoding="utf-8")
    validate_guide_structure(text)
    validate_markdown_code_lines(text)
    validate_notebook()
    validate_benchmark_fixture()
    validate_export_artifacts()
    validate_pdf()
    print("Guide validation passed.")


if __name__ == "__main__":
    main()
