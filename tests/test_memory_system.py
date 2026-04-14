from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from polymind.config import ChunkRecord, MemoryTurn, PolymindConfig
from polymind.memory import ConversationMemory, rewrite_followup_query
from polymind.retrieval import build_index
from polymind.system import PolymindSystem
from tests.helpers import FakeEmbeddingModel, FakeGenerator, write_tmp_book


class MemoryAndSystemTests(unittest.TestCase):
    def test_memory_compression_populates_summary(self):
        config = PolymindConfig()
        memory = ConversationMemory(config=config, generator=FakeGenerator(), max_recent_turns=2)
        for idx in range(3):
            memory.add_turn(
                MemoryTurn(
                    question=f"Q{idx}",
                    rewritten_question=f"RQ{idx}",
                    answer=f"A{idx}",
                    chunk_ids=[idx],
                    selected_arm="grounded_concise",
                    reward=0.5,
                )
            )
        inspection = memory.inspect_memory()
        self.assertTrue(inspection.summary)
        self.assertEqual(inspection.compression_events, 1)

    def test_rewrite_followup_query_uses_generator(self):
        config = PolymindConfig()
        memory = ConversationMemory(config=config, generator=FakeGenerator())
        memory.add_turn(
            MemoryTurn(
                question="What is mastery?",
                rewritten_question="What is mastery?",
                answer="A grounded answer.",
                chunk_ids=[0],
                selected_arm="grounded_concise",
                reward=0.7,
            )
        )
        rewritten = rewrite_followup_query("How does that relate to leadership?", memory, generator=FakeGenerator())
        self.assertIn("leadership", rewritten.lower())
        self.assertTrue(memory.last_rewrite_context)

    def test_system_exposes_inspection_methods_and_rewritten_question(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            book_path = tmp_path / "book.txt"
            write_tmp_book(book_path)
            config = PolymindConfig(project_root=tmp_path, chunk_target_words=10, chunk_overlap_words=2, top_k=2)
            system = PolymindSystem(
                config=config,
                embed_model=FakeEmbeddingModel(),
                reward_model=FakeEmbeddingModel(),
                generator=FakeGenerator(),
            )
            system.index_book(book_path)
            answer = system.ask("How does that relate to leadership?")
            self.assertIn("rewritten_question", answer)
            self.assertTrue(system.inspect_chunks())
            self.assertIsNotNone(system.inspect_retrieval())
            self.assertIsNotNone(system.inspect_rewards())
            self.assertTrue(system.inspect_memory().recent_turns)


if __name__ == "__main__":
    unittest.main()
