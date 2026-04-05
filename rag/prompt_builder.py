from __future__ import annotations

from pathlib import Path

from config import SETTINGS, Settings, ensure_local_layout, load_personality_cards
from rag.types import RAGBundle

try:
    from jinja2 import Template
except ImportError:  # pragma: no cover - optional dependency.
    Template = None


class PromptBuilder:
    def __init__(self, settings: Settings = SETTINGS):
        ensure_local_layout()
        self.settings = settings
        self.system_prompt = settings.system_prompt_path.read_text(encoding="utf-8").strip()
        self.template_text = settings.response_template_path.read_text(encoding="utf-8")
        self.personality_cards = load_personality_cards(settings.personality_cards_path)

    def _render_template(self, **values: str) -> str:
        if Template is not None:
            return Template(self.template_text).render(**values)

        rendered = self.template_text
        for key, value in values.items():
            rendered = rendered.replace(f"{{{{ {key} }}}}", value)
        return rendered

    def _persona_prompt_path(self, personality_id: str) -> Path:
        return self.settings.prompts_dir / "personas" / f"{personality_id}.txt"

    def build_persona_prompt(self, personality_id: str, style_strength: float) -> str:
        persona = self.personality_cards[personality_id]
        prompt_path = self._persona_prompt_path(personality_id)
        base_prompt = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists() else ""

        structured_prompt = (
            f"Style tags: {', '.join(persona['style_tags'])}\n"
            f"Do: {', '.join(persona['dos'])}\n"
            f"Do not: {', '.join(persona['donts'])}\n"
            f"Style strength: {style_strength:.2f} on a 0.0 to 1.0 scale.\n"
            "Do not imitate mannerisms theatrically. Aim for worldview and reasoning style more than surface mimicry."
        )

        return f"{base_prompt}\n\n{structured_prompt}".strip()

    def build(self, query: str, bundle: RAGBundle, style_strength: float = 0.7) -> str:
        return self._render_template(
            system_prompt=self.system_prompt,
            persona_prompt=self.build_persona_prompt(bundle.personality_id, style_strength),
            query=query.strip(),
            expanded_query=(bundle.expanded_query or query).strip(),
            context_text=bundle.context_text.strip(),
            style_strength=f"{style_strength:.2f}",
        )
