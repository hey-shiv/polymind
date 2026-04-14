from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from polymind.config import AnswerCandidate, ChunkRecord, PolymindConfig, RetrievalResult
from polymind.ingestion import detect_chapters
from polymind.retrieval import build_index, retrieve_chunks
from polymind.rl import EpsilonGreedyBanditPolicy, select_best_answer_with_policy
from tests.helpers import FakeEmbeddingModel, FakeGenerator


class IngestionRetrievalRLTests(unittest.TestCase):
    def test_detect_chapters_dedups_nearby_matches(self):
        text = "CHAPTER 1: Apprenticeship\n\nChapter 1: Apprenticeship\n\nContent begins here."
        chapters = detect_chapters(
            text,
            font_headings=[(0, "CHAPTER 1: Apprenticeship", 1), (2, "Chapter 1: Apprenticeship", 1)],
        )
        self.assertEqual(len(chapters), 1)

    def test_retrieval_returns_expected_chapter(self):
        config = PolymindConfig(top_k=2)
        chunks = [
            ChunkRecord(0, 1, "Apprenticeship", "Leadership and mastery grow through practice.", 0, 50, 1, 7),
            ChunkRecord(1, 2, "Strategy", "Strategy depends on adaptive thinking and observation.", 51, 110, 1, 8),
        ]
        embed_model = FakeEmbeddingModel()
        embeddings = embed_model.encode([chunk.text for chunk in chunks], convert_to_numpy=True, normalize_embeddings=True)
        index = build_index(embeddings)
        results, inspection, _ = retrieve_chunks(
            query="How does leadership connect to mastery?",
            rewritten_query="How does leadership connect to mastery?",
            chunks=chunks,
            chunk_embeddings=embeddings,
            index=index,
            embed_model=embed_model,
            config=config,
            generator=FakeGenerator(),
        )
        self.assertEqual(results[0].chapter_id, 1)
        self.assertTrue(inspection.selected_results)

    def test_select_best_answer_computes_rewards_once_per_candidate(self):
        config = PolymindConfig()
        embed_model = FakeEmbeddingModel()
        policy = EpsilonGreedyBanditPolicy(config)
        results = [
            RetrievalResult(0, 1, "Apprenticeship", "Leadership and mastery grow through practice.", 0.9, 1, "query")
        ]
        candidates = [
            AnswerCandidate("grounded_concise", "p", "leadership mastery practice", 0.2, 64, ["[Chapter 01 | Chunk 000]"]),
            AnswerCandidate("teacher_explainer", "p", "leadership strategy practice", 0.4, 64, ["[Chapter 01 | Chunk 000]"]),
        ]
        _selected, _breakdowns, _inspection = select_best_answer_with_policy(
            candidates=candidates,
            retrieved_results=results,
            model=embed_model,
            policy=policy,
            config=config,
        )
        self.assertEqual(embed_model.calls, 4)

    def test_policy_from_json_is_missing_arm_safe(self):
        config = PolymindConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.json"
            path.write_text('{"arms": {"grounded_concise": {"pulls": 2, "wins": 1, "total_reward": 0.8}}}', encoding="utf-8")
            policy = EpsilonGreedyBanditPolicy.from_json(path, config)
            self.assertIn("synthesizer", policy.stats)
            self.assertEqual(policy.stats["synthesizer"].pulls, 0)
            self.assertLessEqual(policy.epsilon, config.max_epsilon)


if __name__ == "__main__":
    unittest.main()
