from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .config import MemoryInspection, MemoryTurn, PolymindConfig

LOGGER = logging.getLogger("polymind.memory")


def _call_generator(generator, prompt: str, max_new_tokens: int) -> str:
    if generator is None:
        return ""
    if callable(generator):
        return generator(prompt, max_new_tokens=max_new_tokens, temperature=0.2)
    if hasattr(generator, "generate_text"):
        return generator.generate_text(prompt, max_new_tokens=max_new_tokens, temperature=0.2)
    if hasattr(generator, "generate"):
        return generator.generate(prompt, max_new_tokens=max_new_tokens)
    return ""


@dataclass
class ConversationMemory:
    config: PolymindConfig
    generator: Optional[object] = None
    summary: str = ""
    recent_turns: List[MemoryTurn] = field(default_factory=list)
    max_recent_turns: int = 3
    compression_events: int = 0
    last_rewrite_context: str = ""

    def _compress_to_summary(self, turns: List[MemoryTurn]) -> None:
        if not turns:
            return
        transcript = "\n".join(
            f"Q: {turn.question}\nA: {turn.answer}\nArm: {turn.selected_arm}\nReward: {turn.reward:.3f}"
            for turn in turns
        )
        prompt = (
            "Summarize the older conversation turns for retrieval memory. Keep the summary factual,\n"
            "compact, and focused on what the user asked and what the book evidence established.\n\n"
            f"Existing summary:\n{self.summary or '(none)'}\n\n"
            f"New turns to compress:\n{transcript}\n\n"
            "Updated summary:"
        )
        compressed = _call_generator(
            self.generator,
            prompt=prompt,
            max_new_tokens=self.config.token_budget("memory_summary"),
        ).strip()
        if compressed:
            self.summary = compressed
        else:
            appended = " ".join(f"{turn.question}: {turn.answer[:160]}" for turn in turns)
            self.summary = f"{self.summary} {appended}".strip()
        self.compression_events += 1

    def add_turn(self, turn: MemoryTurn) -> None:
        self.recent_turns.append(turn)
        if len(self.recent_turns) > self.max_recent_turns:
            overflow = self.recent_turns[:-self.max_recent_turns]
            self.recent_turns = self.recent_turns[-self.max_recent_turns :]
            self._compress_to_summary(overflow)

    def build_memory_summary(self) -> str:
        lines: List[str] = []
        if self.summary:
            lines.append(f"Older summary: {self.summary}")
        for idx, turn in enumerate(self.recent_turns, start=1):
            lines.append(
                f"Turn {idx}: Q={turn.question} | Rewritten={turn.rewritten_question} | "
                f"Arm={turn.selected_arm} | Reward={turn.reward:.3f}"
            )
            lines.append(f"Answer: {turn.answer[:240]}")
        return "\n".join(lines).strip()

    def inspect_memory(self) -> MemoryInspection:
        return MemoryInspection(
            summary=self.summary,
            recent_turns=list(self.recent_turns),
            compression_events=self.compression_events,
            rewrite_context=self.last_rewrite_context,
        )


def rewrite_followup_query(
    question: str,
    memory: ConversationMemory,
    generator=None,
) -> str:
    if not memory.recent_turns and not memory.summary:
        memory.last_rewrite_context = ""
        return question
    context = memory.build_memory_summary()
    prompt = (
        "Rewrite the follow-up question so it is self-contained for retrieval.\n"
        "Preserve the user's intent and resolve pronouns or references using the memory context.\n\n"
        f"Memory context:\n{context}\n\n"
        f"Follow-up question: {question}\n\n"
        "Rewritten question:"
    )
    rewritten = _call_generator(
        generator or memory.generator,
        prompt=prompt,
        max_new_tokens=memory.config.token_budget("rewrite"),
    ).strip()
    memory.last_rewrite_context = context
    return rewritten or question
