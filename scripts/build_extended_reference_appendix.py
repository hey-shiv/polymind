from __future__ import annotations

import re
from pathlib import Path
from textwrap import fill

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
GUIDE_PATH = ROOT / "POLYMIND_V2_COLAB_GUIDE.md"
SOURCE_PDF = ROOT / "POLYMIND_DEEP_READING_AI_GUIDE.original.pdf"
MARKER = "\n<!-- EXTENDED_REFERENCE_APPENDIX -->\n"

APPENDIX_RANGES = [
    (
        "# Appendix F. Extended Foundations And Architecture Notes",
        [
            ("## F.1 Reading-system foundations", 9, 15),
            ("## F.2 Architecture and Colab framing", 18, 24),
            ("## F.3 Model and transformer background", 29, 35),
        ],
    ),
    (
        "# Appendix G. Extended Retrieval, Generation, And RL Notes",
        [
            ("## G.1 Book processing and chunking details", 33, 44),
            ("## G.2 Embeddings, FAISS, and retrieval details", 45, 52),
            ("## G.3 Generation, reward, and policy details", 58, 79),
        ],
    ),
    (
        "# Appendix H. Extended Evaluation, Optimization, And Assembly Notes",
        [
            ("## H.1 Evaluation and acceptance guidance", 82, 87),
            ("## H.2 Optimization and future improvements", 88, 91),
            ("## H.3 Final assembly and workflow recap", 92, 95),
        ],
    ),
]

NOISE_PATTERNS = [
    r"\bdef\s+\w+\(",
    r"\bclass\s+\w+",
    r"\breturn\s+\w",
    r"\bself\.",
    r"plt\.",
    r"AutoTokenizer",
    r"AutoModelForSeq2SeqLM",
    r"SentenceTransformer\(",
    r"pipeline\(",
    r"faiss\.",
    r"BanditArmStats\(",
    r"__post_init__",
    r"\bindex_book\(",
    r"\bask\(",
    r"\bsummarize\(",
    r"\blink_concepts\(",
]


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("ﬁ", "fi")
    text = text.replace("ﬂ", "fl")
    text = re.sub(r"Polymind\s+\d+", " ", text)
    text = re.sub(r"\b\d{2}\s+SECTION\s+\d+\b", " ", text)
    text = re.sub(r"SECTION SNAPSHOT", "Section snapshot:", text)
    text = re.sub(r"KEY TAKEAWAYS", "Key takeaways:", text)
    text = re.sub(
        r"Python Code.*?(?=\d+\.\d+|Key takeaways:|Section snapshot:|$)",
        " ",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"Diagram / Text.*?(?=\d+\.\d+|Key takeaways:|Section snapshot:|$)",
        " ",
        text,
        flags=re.S,
    )
    text = re.sub(r"\+\-+\+.*", " ", text)
    text = re.sub(r"\|\s+v\s+\|", " ", text)
    text = re.sub(r"\s1\s+", " • ", text)
    text = re.sub(r"\b\d+\.\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_paragraphs(text: str, sentences_per_paragraph: int = 5) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    paragraphs: list[str] = []
    bucket: list[str] = []
    for sentence in sentences:
        bucket.append(sentence)
        if len(bucket) >= sentences_per_paragraph:
            paragraphs.append(" ".join(bucket))
            bucket = []
    if bucket:
        paragraphs.append(" ".join(bucket))
    return paragraphs


def is_noise_paragraph(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if sum(char.isalpha() for char in stripped) < 45:
        return True
    if re.search(r"[\(\)\{\}\[\]=]", stripped) and len(
        re.findall(r"[\(\)\{\}\[\]=]", stripped)
    ) >= 8:
        return True
    if stripped.count("->") >= 2:
        return True
    if "| Mastery PDF" in stripped or "Conversation Memory" in stripped:
        return True
    return any(re.search(pattern, stripped) for pattern in NOISE_PATTERNS)


def wrap_paragraph(text: str) -> str:
    return fill(text, width=92)


def extract_range(reader: PdfReader, start: int, end: int) -> str:
    text = "\n".join((reader.pages[i].extract_text() or "") for i in range(start, end + 1))
    return clean_text(text)


def build_appendix() -> str:
    reader = PdfReader(str(SOURCE_PDF))
    lines: list[str] = []
    for appendix_title, sections in APPENDIX_RANGES:
        lines.append(appendix_title)
        lines.append("")
        lines.append(
            wrap_paragraph(
                "These extended reference notes deepen the same Polymind V2 system "
                "from a longer-form teaching angle. They are kept in appendix form "
                "so the main 23-section spine stays clean while readers who want "
                "more architecture, retrieval, RL, evaluation, and workflow detail "
                "can keep reading without leaving the handbook."
            )
        )
        lines.append("")
        for heading, start, end in sections:
            lines.append(heading)
            lines.append("")
            text = extract_range(reader, start, end)
            for paragraph in chunk_paragraphs(text):
                if is_noise_paragraph(paragraph):
                    continue
                lines.append(wrap_paragraph(paragraph))
                lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    guide = GUIDE_PATH.read_text(encoding="utf-8")
    appendix = build_appendix()
    if MARKER in guide:
        guide = guide.split(MARKER, 1)[0].rstrip() + MARKER + "\n" + appendix
    else:
        guide = guide.rstrip() + MARKER + "\n" + appendix
    GUIDE_PATH.write_text(guide, encoding="utf-8")
    print("Injected extended reference appendices into guide.")


if __name__ == "__main__":
    main()
