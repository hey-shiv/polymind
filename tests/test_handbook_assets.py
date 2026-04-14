from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
GUIDE_PATH = ROOT / "POLYMIND_V2_COLAB_GUIDE.md"
NOTEBOOK_PATH = ROOT / "notebooks" / "polymind_v2_colab.ipynb"
BENCHMARK_PATH = ROOT / "benchmarks" / "mastery_eval_set.json"
TEMPLATE_PATH = ROOT / "scripts" / "pandoc_template.tex"
VARIABLES_PATH = ROOT / "scripts" / "pandoc_variables.yaml"
MAX_CODE_LINE_LENGTH = 88


def iter_markdown_code_lines(text: str):
    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            yield line


class HandbookAssetTests(unittest.TestCase):
    def test_guide_has_23_sections_and_required_blocks(self):
        text = GUIDE_PATH.read_text(encoding="utf-8")
        sections = re.findall(r"^#\s+\d+\.\s+.+$", text, flags=re.M)
        self.assertEqual(len(sections), 23)

        parts = re.split(r"^#\s+\d+\.\s+.+$", text, flags=re.M)[1:]
        for part in parts:
            self.assertIn("## Intuition", part)
            self.assertIn("## Colab-ready code cell", part)
            self.assertIn("## Expected output", part)
            self.assertIn("## Debugging tips", part)

    def test_guide_code_lines_fit_pdf_width_budget(self):
        text = GUIDE_PATH.read_text(encoding="utf-8")
        offenders = [
            line for line in iter_markdown_code_lines(text)
            if len(line) > MAX_CODE_LINE_LENGTH
        ]
        self.assertEqual(offenders, [])

    def test_notebook_covers_full_experiment_matrix(self):
        notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        code = "\n\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        for required in [
            "chunk_target_words",
            "temperature",
            "primary_model",
            "baseline_dense",
            "\"mode\": \"cached\"",
            "system.policy.learning_curve()",
        ]:
            self.assertIn(required, code)

        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            for line in cell.source.splitlines():
                self.assertLessEqual(len(line), MAX_CODE_LINE_LENGTH)

    def test_mastery_benchmark_fixture_is_non_trivial(self):
        data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(data), 20)
        tags = {tag for item in data for tag in item["tags"]}
        for required in ["single-hop", "cross-chapter", "follow-up", "concept-link"]:
            self.assertIn(required, tags)
        for item in data:
            self.assertIn("expected_chapter_ids", item)
            self.assertIn("expected_keywords", item)
            self.assertTrue(item["expected_chapter_ids"])
            self.assertTrue(item["expected_keywords"])

    def test_export_design_artifacts_exist(self):
        self.assertTrue(TEMPLATE_PATH.exists())
        self.assertTrue(VARIABLES_PATH.exists())


if __name__ == "__main__":
    unittest.main()
