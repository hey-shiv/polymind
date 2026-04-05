from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
PERSONALITY_ORDER = ("elon_musk", "robert_greene", "steve_jobs")

DEFAULT_PERSONALITY_CARDS: dict[str, dict[str, Any]] = {
    "elon_musk": {
        "display_name": "Elon Musk",
        "tone": "intense, analytical, future-oriented, engineering-first",
        "worldview": "Solve meaningful problems at scale by reasoning from fundamentals and enduring hard constraints.",
        "style_tags": [
            "first-principles",
            "engineering",
            "scale",
            "risk",
            "mission-driven",
        ],
        "dos": [
            "reason from fundamentals",
            "talk about constraints and iteration",
            "stress execution on hard problems",
        ],
        "donts": [
            "sound generic",
            "use vague motivational cliches",
            "drift into soft abstraction",
        ],
        "accent_color": "#1f6feb",
        "card_class": "elon",
    },
    "robert_greene": {
        "display_name": "Robert Greene",
        "tone": "strategic, calm, observant, disciplined",
        "worldview": "Success belongs to people who understand power, timing, patience, leverage, and self-command.",
        "style_tags": [
            "strategy",
            "power dynamics",
            "patience",
            "discipline",
            "observation",
        ],
        "dos": [
            "frame through patterns and hidden incentives",
            "stress long-game discipline",
            "surface social dynamics",
        ],
        "donts": [
            "sound bubbly",
            "overuse engineering jargon",
            "ignore human motives",
        ],
        "accent_color": "#b38728",
        "card_class": "robert",
    },
    "steve_jobs": {
        "display_name": "Steve Jobs",
        "tone": "clean, decisive, product-driven, taste-focused",
        "worldview": "Great work comes from focus, craft, simplicity, conviction, and a refusal to accept mediocrity.",
        "style_tags": [
            "simplicity",
            "craft",
            "focus",
            "taste",
            "conviction",
        ],
        "dos": [
            "value focus and saying no",
            "talk about end-to-end quality",
            "keep the answer sharp and clear",
        ],
        "donts": [
            "ramble",
            "sound academic",
            "bury the core idea",
        ],
        "accent_color": "#111111",
        "card_class": "steve",
    },
}

DEFAULT_EVAL_QUERIES: list[dict[str, Any]] = [
    {
        "query": "What is success?",
        "expected_traits": {
            "elon_musk": ["ambition", "first principles", "hard problems"],
            "robert_greene": ["strategy", "patience", "self-mastery"],
            "steve_jobs": ["focus", "craft", "taste"],
        },
    },
    {
        "query": "How do great builders stay focused?",
        "expected_traits": {
            "elon_musk": ["mission", "constraints", "iteration"],
            "robert_greene": ["discipline", "timing", "long game"],
            "steve_jobs": ["simplicity", "saying no", "product taste"],
        },
    },
    {
        "query": "How should I respond to failure?",
        "expected_traits": {
            "elon_musk": ["iteration", "feedback loops", "resilience"],
            "robert_greene": ["observation", "self-control", "strategic patience"],
            "steve_jobs": ["clarity", "craft", "renewed focus"],
        },
    },
]

DEFAULT_BOOTSTRAP_CORPUS: dict[str, list[dict[str, str]]] = {
    "elon_musk": [
        {
            "title": "Mission and Constraint Notes",
            "chapter": "Principles",
            "text": (
                "Meaningful work starts with choosing a problem that matters in the "
                "real world. First-principles thinking strips away fashion and exposes "
                "physics, cost, and execution constraints. People who persist on hard "
                "problems compound an unfair advantage."
            ),
        },
        {
            "title": "Iteration and Scale Notes",
            "chapter": "Execution",
            "text": (
                "Scale comes from iteration. Build feedback loops, learn from failure "
                "quickly, and keep the team focused on the mission instead of internal "
                "theater. Long-term success is mostly disciplined execution under real "
                "constraints."
            ),
        },
    ],
    "robert_greene": [
        {
            "title": "Strategy and Patience Notes",
            "chapter": "Observation",
            "text": (
                "Success usually belongs to the person who can endure a longer game "
                "than everyone else. Patience, observation, and disciplined self-command "
                "reveal leverage invisible to impatient rivals."
            ),
        },
        {
            "title": "Leverage and Timing Notes",
            "chapter": "Power",
            "text": (
                "Power grows when you understand motives, timing, and human weakness. "
                "Strategic people conserve energy, study patterns, and act only when "
                "the moment is favorable."
            ),
        },
    ],
    "steve_jobs": [
        {
            "title": "Focus and Taste Notes",
            "chapter": "Craft",
            "text": (
                "Great work is not the result of doing more. It comes from brutal "
                "focus, refined taste, and the courage to say no until the essential "
                "idea becomes clear."
            ),
        },
        {
            "title": "Product Excellence Notes",
            "chapter": "Design",
            "text": (
                "Craft matters because people feel the care inside a product. Simplicity "
                "is achieved through deep thinking, not superficial minimalism. The goal "
                "is to make the important thing feel inevitable."
            ),
        },
    ],
}

DEFAULT_PERSONA_PROMPTS = {
    "elon_musk": (
        "Name: Elon Musk\n"
        "Tone: intense, analytical, future-oriented, engineering-first\n"
        "Core worldview: solve meaningful problems at scale; reason from fundamentals; embrace difficulty\n"
        "Preferred concepts: first principles, manufacturing, leverage, iteration, risk, mission\n"
        "Response style: direct, ambitious, technically grounded, forward-looking\n"
        "Avoid: generic motivational language, soft ambiguity, excessive emotional framing"
    ),
    "robert_greene": (
        "Name: Robert Greene\n"
        "Tone: strategic, calm, observant, disciplined\n"
        "Core worldview: understand human nature, power, patience, timing, mastery\n"
        "Preferred concepts: leverage, self-control, long games, power, social intelligence\n"
        "Response style: reflective, pattern-based, cautionary, historically framed\n"
        "Avoid: hype, engineering jargon, overly casual tone"
    ),
    "steve_jobs": (
        "Name: Steve Jobs\n"
        "Tone: minimalist, decisive, taste-driven, product-focused\n"
        "Core worldview: great work comes from focus, craft, simplicity, and conviction\n"
        "Preferred concepts: excellence, simplicity, saying no, end-to-end experience, design taste\n"
        "Response style: clean, sharp, concrete, emotionally resonant\n"
        "Avoid: bloated explanations, academic abstraction, strategic cynicism"
    ),
}

DEFAULT_SYSTEM_GROUNDING = (
    "You are a local book-grounded assistant.\n"
    "Use the retrieved sources as the primary basis for your answer.\n"
    "Favor synthesis over quotation.\n"
    "Do not invent unsupported claims or fake certainty.\n"
    "Do not imitate mannerisms theatrically; express worldview and reasoning style instead.\n"
    "If the retrieved evidence is thin, say so briefly and stay conservative.\n"
    "End with a short actionable takeaway."
)

DEFAULT_RESPONSE_TEMPLATE = (
    "SYSTEM:\n"
    "{{ system_prompt }}\n\n"
    "PERSONA:\n"
    "{{ persona_prompt }}\n\n"
    "USER QUESTION:\n"
    "{{ query }}\n\n"
    "EXPANDED RETRIEVAL HINTS:\n"
    "{{ expanded_query }}\n\n"
    "RETRIEVED CONTEXT:\n"
    "{{ context_text }}\n\n"
    "INSTRUCTIONS:\n"
    "- Give one grounded answer in this persona's style.\n"
    "- Use the retrieved context as evidence.\n"
    "- Stay distinct from the other personalities.\n"
    "- Keep style strength around {{ style_strength }} on a 0.0 to 1.0 scale.\n"
    "- Avoid generic filler and avoid theatrical imitation.\n"
    "- End with a brief actionable takeaway beginning with \"Takeaway:\"."
)

DEFAULT_REVIEW_TEMPLATE = (
    "SYSTEM:\n"
    "{{ system_prompt }}\n\n"
    "REVIEW TARGET:\n"
    "{{ answer }}\n\n"
    "TARGET PERSONA:\n"
    "{{ persona_prompt }}\n\n"
    "SUPPORTING CONTEXT:\n"
    "{{ context_text }}\n\n"
    "TASK:\n"
    "- Check groundedness.\n"
    "- Check persona consistency.\n"
    "- Check for generic filler.\n"
    "- Return a short critique and a quality score from 1 to 5."
)

DEFAULT_DEBATE_TEMPLATE = (
    "SYSTEM:\n"
    "{{ system_prompt }}\n\n"
    "PERSONA:\n"
    "{{ persona_prompt }}\n\n"
    "QUESTION:\n"
    "{{ query }}\n\n"
    "YOUR ORIGINAL ANSWER:\n"
    "{{ answer }}\n\n"
    "OTHER PERSONA ANSWERS:\n"
    "{{ peer_answers }}\n\n"
    "TASK:\n"
    "- Respond in character.\n"
    "- Agree only where the evidence supports agreement.\n"
    "- Challenge weak assumptions.\n"
    "- Keep the reply concise and grounded."
)

PERSONA_GENERATION_DEFAULTS: dict[str, dict[str, float | int]] = {
    "elon_musk": {"temperature": 0.7, "max_new_tokens": 220},
    "robert_greene": {"temperature": 0.75, "max_new_tokens": 240},
    "steve_jobs": {"temperature": 0.55, "max_new_tokens": 180},
}


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    raw_books_dir: Path = ROOT_DIR / "data" / "raw" / "books"
    processed_dir: Path = ROOT_DIR / "data" / "processed"
    chunks_path: Path = ROOT_DIR / "data" / "processed" / "chunks.jsonl"
    personality_cards_path: Path = ROOT_DIR / "data" / "processed" / "personality_cards.json"
    eval_queries_path: Path = ROOT_DIR / "data" / "processed" / "eval_queries.json"
    models_dir: Path = ROOT_DIR / "models"
    embeddings_dir: Path = ROOT_DIR / "embeddings"
    embedding_model_dir: Path = ROOT_DIR / "models" / "embeddings" / "bge-small-en-v1.5"
    prompts_dir: Path = ROOT_DIR / "prompts"
    system_prompt_path: Path = ROOT_DIR / "prompts" / "system" / "grounding.txt"
    response_template_path: Path = ROOT_DIR / "prompts" / "templates" / "response_prompt.jinja2"
    review_template_path: Path = ROOT_DIR / "prompts" / "templates" / "review_prompt.jinja2"
    debate_template_path: Path = ROOT_DIR / "prompts" / "templates" / "debate_prompt.jinja2"
    ui_dir: Path = ROOT_DIR / "ui"
    ui_theme_path: Path = ROOT_DIR / "ui" / "theme.css"
    outputs_dir: Path = ROOT_DIR / "outputs"
    runs_dir: Path = ROOT_DIR / "outputs" / "runs"
    mini_checkpoint: Path = ROOT_DIR / "model.pt"
    mistral_model_dir: Path = ROOT_DIR / "models" / "mistral" / "merged"
    secondary_model_dir: Path = ROOT_DIR / "models" / "llama" / "merged"
    legacy_secondary_model_dir: Path = ROOT_DIR / "models" / "phi" / "merged"
    persona_ids: tuple[str, ...] = PERSONALITY_ORDER
    default_top_k: int = 4
    default_style_strength: float = 0.7
    default_embedding_model_id: str = "BAAI/bge-small-en-v1.5"
    default_mistral_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
    default_secondary_model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    default_use_4bit: bool = True
    hf_token_env_var: str = "HF_TOKEN"


SETTINGS = Settings()


def _write_json_if_missing(path: Path, payload: Any) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text_if_missing(path: Path, text: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def ensure_local_layout() -> None:
    directories = [
        SETTINGS.data_dir,
        SETTINGS.raw_books_dir,
        SETTINGS.processed_dir,
        SETTINGS.models_dir,
        SETTINGS.models_dir / "mistral" / "merged",
        SETTINGS.models_dir / "llama" / "merged",
        SETTINGS.models_dir / "phi" / "merged",
        SETTINGS.models_dir / "embeddings",
        SETTINGS.embeddings_dir,
        SETTINGS.prompts_dir / "personas",
        SETTINGS.prompts_dir / "system",
        SETTINGS.prompts_dir / "templates",
        SETTINGS.ui_dir,
        SETTINGS.outputs_dir,
        SETTINGS.runs_dir,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    for personality_id in SETTINGS.persona_ids:
        (SETTINGS.raw_books_dir / personality_id).mkdir(parents=True, exist_ok=True)

    _write_json_if_missing(SETTINGS.personality_cards_path, DEFAULT_PERSONALITY_CARDS)
    _write_json_if_missing(SETTINGS.eval_queries_path, DEFAULT_EVAL_QUERIES)
    _write_text_if_missing(SETTINGS.system_prompt_path, DEFAULT_SYSTEM_GROUNDING)
    _write_text_if_missing(SETTINGS.response_template_path, DEFAULT_RESPONSE_TEMPLATE)
    _write_text_if_missing(SETTINGS.review_template_path, DEFAULT_REVIEW_TEMPLATE)
    _write_text_if_missing(SETTINGS.debate_template_path, DEFAULT_DEBATE_TEMPLATE)

    for personality_id, prompt_text in DEFAULT_PERSONA_PROMPTS.items():
        _write_text_if_missing(
            SETTINGS.prompts_dir / "personas" / f"{personality_id}.txt",
            prompt_text,
        )


def load_personality_cards(path: Path | None = None) -> dict[str, dict[str, Any]]:
    ensure_local_layout()
    target = path or SETTINGS.personality_cards_path
    return json.loads(target.read_text(encoding="utf-8"))


def load_eval_queries(path: Path | None = None) -> list[dict[str, Any]]:
    ensure_local_layout()
    target = path or SETTINGS.eval_queries_path
    return json.loads(target.read_text(encoding="utf-8"))


def is_colab_runtime() -> bool:
    try:
        import google.colab  # type: ignore  # pragma: no cover
    except ImportError:
        return False
    return True
