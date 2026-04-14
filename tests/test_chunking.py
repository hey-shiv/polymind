from __future__ import annotations

import unittest

from polymind.chunking import inspect_chunks, semantic_chunk_book, split_into_sentences
from polymind.config import BookSection, PolymindConfig


class ChunkingTests(unittest.TestCase):
    def test_sentence_splitter_handles_abbreviations(self):
        text = "Dr. Greene teaches patiently. The U.S. example still matters. Practice follows."
        spans = split_into_sentences(text)
        self.assertEqual(len(spans), 3)
        self.assertTrue(spans[0].text.startswith("Dr. Greene"))
        self.assertTrue(spans[1].text.startswith("The U.S."))

    def test_semantic_chunk_offsets_are_monotonic(self):
        config = PolymindConfig(chunk_target_words=8, chunk_overlap_words=2, min_chunk_words=1, max_chunk_words=20)
        section = BookSection(
            chapter_id=1,
            chapter_title="Chapter 1",
            text="Practice builds mastery. Leadership requires patience. Strategy follows repetition.",
            start_char=100,
            end_char=180,
            source_path="book.txt",
        )
        chunks = semantic_chunk_book([section], config)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertLess(chunks[0].start_char, chunks[0].end_char)
        self.assertLessEqual(chunks[0].start_char, chunks[1].start_char)
        self.assertLessEqual(chunks[0].end_char, chunks[1].end_char)

    def test_inspect_chunks_surfaces_quality_warnings(self):
        config = PolymindConfig(min_chunk_words=10, max_chunk_words=20)
        rows = inspect_chunks([], config)
        self.assertEqual(rows, [])

        fake_chunk = semantic_chunk_book(
            [
                BookSection(
                    chapter_id=1,
                    chapter_title="Chapter 1",
                    text="Short text only.",
                    start_char=0,
                    end_char=16,
                    source_path="book.txt",
                )
            ],
            PolymindConfig(chunk_target_words=5, chunk_overlap_words=0, min_chunk_words=10, max_chunk_words=20),
        )
        inspected = inspect_chunks(fake_chunk, PolymindConfig(min_chunk_words=10, max_chunk_words=20))
        self.assertIn("under_target", inspected[0].warnings)


if __name__ == "__main__":
    unittest.main()
