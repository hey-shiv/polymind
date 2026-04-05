from __future__ import annotations

from typing import Any


def render_placeholder_card(title: str, body: str) -> str:
    return f"## {title}\n\n{body}"


def render_persona_card(card: dict[str, Any]) -> str:
    citations = card.get("citations", [])
    source_lines = []
    for index, citation in enumerate(citations[:3], start=1):
        label = citation.get("source_label", "Unknown source")
        score = float(citation.get("score", 0.0))
        source_lines.append(f"{index}. {label} ({score:.2f})")

    metrics = card.get("metrics", {})
    grounding = metrics.get("grounding_score", 0.0)
    source_count = int(metrics.get("source_count", len(citations)))

    sources_block = "\n".join(source_lines) if source_lines else "1. No sources available yet."
    takeaway = card.get("takeaway") or "Keep the answer grounded and inspect the sources."

    return (
        f"## {card['display_name']}\n\n"
        f"`Model: {card['model_name']}` `Sources: {source_count}` `Grounding: {grounding:.2f}`\n\n"
        f"{card['answer']}\n\n"
        f"**Takeaway:** {takeaway}\n\n"
        f"**Sources**\n"
        f"{sources_block}"
    )
