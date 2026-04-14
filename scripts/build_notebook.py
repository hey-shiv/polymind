from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


MAX_CODE_LINE_LENGTH = 88


def markdown_cell(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code_cell(code: str):
    source = dedent(code).strip() + "\n"
    for line in source.splitlines():
        if len(line) > MAX_CODE_LINE_LENGTH:
            raise ValueError(
                f"Notebook code line exceeds {MAX_CODE_LINE_LENGTH} chars: {line}"
            )
    return nbf.v4.new_code_cell(source)


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        markdown_cell(
            """
            # Polymind V2: RL-Enhanced Deep Reading AI System

            This notebook is the Colab orchestration layer for the `polymind`
            package. It stays restart-safe by mounting Google Drive, reusing
            persisted artifacts, and delegating all real logic to typed Python
            modules rather than notebook-only helper code.
            """
        ),
        code_cell(
            """
            !pip install -q -r /content/drive/MyDrive/polymind_v2/requirements.txt
            """
        ),
        code_cell(
            """
            from google.colab import drive

            drive.mount("/content/drive")
            """
        ),
        code_cell(
            """
            import json
            import logging
            import random
            import time
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import torch

            from polymind import PolymindConfig, PolymindSystem

            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

            PROJECT_ROOT = Path("/content/drive/MyDrive/polymind_v2")
            DATA_DIR = PROJECT_ROOT / "data"
            ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
            BENCHMARK_PATH = PROJECT_ROOT / "benchmarks" / "mastery_eval_set.json"

            PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)

            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            )
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            print({"device": DEVICE, "project_root": str(PROJECT_ROOT)})
            """
        ),
        code_cell(
            """
            config = PolymindConfig(project_root=PROJECT_ROOT)
            system = PolymindSystem(config=config)
            """
        ),
        markdown_cell(
            """
            ## Data, Artifacts, and Benchmarks

            Copy `Mastery.pdf` into `DATA_DIR` and the benchmark fixture into
            `BENCHMARK_PATH` once. Every later session can load the same
            artifacts, benchmark questions, and policy statistics.
            """
        ),
        code_cell(
            """
            book_path = DATA_DIR / "Mastery.pdf"
            with open(BENCHMARK_PATH, "r", encoding="utf-8") as handle:
                benchmark_set = json.load(handle)

            print(
                {
                    "book_exists": book_path.exists(),
                    "benchmark_questions": len(benchmark_set),
                }
            )
            """
        ),
        code_cell(
            """
            result = system.index_book(book_path)
            print(result["status"])
            print(config.chunks_path())
            print(config.index_path())
            """
        ),
        code_cell(
            """
            chunk_preview = system.inspect_chunks(limit=8)
            chunk_df = pd.DataFrame([row.__dict__ for row in chunk_preview])
            display(chunk_df)
            """
        ),
        code_cell(
            """
            retrieval_debug = system.inspect_retrieval(
                "How does Greene connect apprenticeship and social intelligence?",
                top_k=8,
            )
            display(retrieval_debug["inspection"])
            display(pd.DataFrame(retrieval_debug["inspection"].selected_results))
            """
        ),
        code_cell(
            """
            answer = system.ask("How does Greene connect mastery and leadership?")
            print(answer["rewritten_question"])
            print(answer["selected_arm"])
            print(answer["answer"])
            display(pd.DataFrame(answer["candidate_rewards"]))
            display(system.inspect_rewards())
            """
        ),
        code_cell(
            """
            memory_debug = system.inspect_memory()
            display(memory_debug)

            deep_summary = system.summarize(
                "Summarize Greene's apprenticeship model.",
                level="deep",
            )
            print(deep_summary["summary"])

            linked = system.link_concepts("leadership", "social intelligence")
            print(linked["answer"])
            """
        ),
        markdown_cell(
            """
            ## Experimentation Framework

            The next cells sweep retrieval, chunking, answer-arm behavior,
            model choice, and runtime caching. They reuse the same benchmark
            fixture so the comparisons stay consistent.
            """
        ),
        code_cell(
            """
            questions = [item["question"] for item in benchmark_set[:6]]
            top_k_rows = []
            for top_k in config.experiment_grid["top_k"]:
                for question in questions:
                    result = system.ask(question, top_k=top_k, use_rl=True)
                    rewards = result["candidate_rewards"]
                    top_reward = max(
                        (row["total_reward"] for row in rewards),
                        default=0.0,
                    )
                    top_k_rows.append(
                        {
                            "question": question,
                            "top_k": top_k,
                            "selected_arm": result["selected_arm"],
                            "reward": top_reward,
                            "citations": len(result["verified_citations"]),
                        }
                    )

            top_k_df = pd.DataFrame(top_k_rows)
            display(top_k_df)
            reward_curve = top_k_df.groupby("top_k")["reward"].mean()
            reward_curve.plot(kind="bar", title="Average reward by top_k")
            plt.show()
            """
        ),
        code_cell(
            """
            chunk_rows = []
            for target_words in config.experiment_grid["chunk_target_words"]:
                alt_config = PolymindConfig(
                    project_root=PROJECT_ROOT,
                    chunk_target_words=target_words,
                    chunk_overlap_words=config.chunk_overlap_words,
                )
                alt_system = PolymindSystem(config=alt_config)
                alt_system.index_book(book_path, force_recompute=True)
                question = benchmark_set[0]["question"]
                result = alt_system.ask(question, use_rl=True)
                rewards = result["candidate_rewards"]
                top_reward = max(
                    (row["total_reward"] for row in rewards),
                    default=0.0,
                )
                chunk_rows.append(
                    {
                        "chunk_target_words": target_words,
                        "chunk_count": len(alt_system.chunks),
                        "reward": top_reward,
                        "citations": len(result["verified_citations"]),
                    }
                )

            chunk_df = pd.DataFrame(chunk_rows)
            display(chunk_df)
            chunk_df.plot(
                x="chunk_target_words",
                y="reward",
                kind="line",
                marker="o",
                title="Chunk size vs reward",
            )
            plt.show()
            """
        ),
        code_cell(
            """
            arm_rows = []
            for question in questions:
                result = system.ask(question, use_rl=True)
                reward_map = {
                    row["arm_name"]: row["total_reward"]
                    for row in result["candidate_rewards"]
                }
                for arm_name, arm_cfg in config.answer_arms.items():
                    arm_rows.append(
                        {
                            "question": question,
                            "arm_name": arm_name,
                            "temperature": arm_cfg["temperature"],
                            "reward": reward_map.get(arm_name, 0.0),
                        }
                    )

            arm_df = pd.DataFrame(arm_rows)
            display(arm_df)
            arm_df.plot.scatter(
                x="temperature",
                y="reward",
                title="Temperature vs reward",
            )
            plt.show()
            """
        ),
        code_cell(
            """
            curve_df = pd.DataFrame(system.policy.learning_curve())
            display(curve_df)
            curve_df.plot(
                x="arm_name",
                y="average_reward",
                kind="bar",
                title="Bandit learning curve",
            )
            plt.show()
            """
        ),
        code_cell(
            """
            baseline_rows = []
            for question in questions:
                for use_rl in [False, True]:
                    result = system.ask(question, use_rl=use_rl)
                    rewards = result["candidate_rewards"]
                    top_reward = max(
                        (row["total_reward"] for row in rewards),
                        default=0.0,
                    )
                    baseline_rows.append(
                        {
                            "question": question,
                            "mode": "rl" if use_rl else "baseline",
                            "selected_arm": result["selected_arm"],
                            "reward": top_reward,
                            "citations": len(result["verified_citations"]),
                            "answer_length": len(result["answer"].split()),
                        }
                    )

            baseline_df = pd.DataFrame(baseline_rows)
            display(baseline_df)
            """
        ),
        code_cell(
            """
            model_rows = []
            model_choices = [
                ("google/flan-t5-base", "google/flan-t5-base"),
                ("google/flan-t5-xl", "google/flan-t5-base"),
            ]
            for primary_model, fallback_model in model_choices:
                alt_config = PolymindConfig(
                    project_root=PROJECT_ROOT,
                    generator_model_name=primary_model,
                    generator_fallback_model_name=fallback_model,
                )
                alt_system = PolymindSystem(config=alt_config)
                alt_system.load_from_artifacts()
                result = alt_system.ask(benchmark_set[1]["question"], use_rl=True)
                rewards = result["candidate_rewards"]
                top_reward = max(
                    (row["total_reward"] for row in rewards),
                    default=0.0,
                )
                model_rows.append(
                    {
                        "primary_model": primary_model,
                        "reward": top_reward,
                        "citations": len(result["verified_citations"]),
                        "answer_length": len(result["answer"].split()),
                    }
                )

            model_df = pd.DataFrame(model_rows)
            display(model_df)
            """
        ),
        code_cell(
            """
            ablation_rows = []
            ablation_modes = [
                {
                    "label": "baseline_dense",
                    "enable_hyde": False,
                    "retrieval_dedup_threshold": 1.01,
                    "chapter_diversity_limit": 99,
                },
                {
                    "label": "plus_hyde",
                    "enable_hyde": True,
                    "retrieval_dedup_threshold": 1.01,
                    "chapter_diversity_limit": 99,
                },
                {
                    "label": "plus_hyde_dedup",
                    "enable_hyde": True,
                    "retrieval_dedup_threshold": 0.92,
                    "chapter_diversity_limit": 99,
                },
                {
                    "label": "full_stack",
                    "enable_hyde": True,
                    "retrieval_dedup_threshold": 0.92,
                    "chapter_diversity_limit": 2,
                },
            ]
            ablation_question = benchmark_set[2]["question"]
            for mode in ablation_modes:
                alt_config = PolymindConfig(
                    project_root=PROJECT_ROOT,
                    enable_hyde=mode["enable_hyde"],
                    retrieval_dedup_threshold=mode[
                        "retrieval_dedup_threshold"
                    ],
                    chapter_diversity_limit=mode["chapter_diversity_limit"],
                )
                alt_system = PolymindSystem(config=alt_config)
                alt_system.load_from_artifacts()
                debug = alt_system.inspect_retrieval(
                    ablation_question,
                    top_k=8,
                )
                inspection = debug["inspection"]
                ablation_rows.append(
                    {
                        "mode": mode["label"],
                        "selected_chunks": len(debug["results"]),
                        "dedup_events": len(inspection.dedup_events),
                        "diversity_events": len(inspection.diversity_events),
                    }
                )

            ablation_df = pd.DataFrame(ablation_rows)
            display(ablation_df)
            """
        ),
        code_cell(
            """
            profile_rows = []
            cold_config = PolymindConfig(project_root=PROJECT_ROOT)
            cold_system = PolymindSystem(config=cold_config)

            cold_start = time.perf_counter()
            cold_system.index_book(book_path, force_recompute=True)
            cold_runtime = time.perf_counter() - cold_start

            cached_start = time.perf_counter()
            cached_loaded = cold_system.load_from_artifacts()
            cached_runtime = time.perf_counter() - cached_start

            profile_rows.append(
                {
                    "mode": "cold",
                    "runtime_s": round(cold_runtime, 3),
                    "loaded": False,
                }
            )
            profile_rows.append(
                {
                    "mode": "cached",
                    "runtime_s": round(cached_runtime, 3),
                    "loaded": cached_loaded,
                }
            )

            profile_df = pd.DataFrame(profile_rows)
            display(profile_df)
            """
        ),
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.10"}
    return nb


def main() -> None:
    output_path = Path("notebooks/polymind_v2_colab.ipynb")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    output_path.write_text(nbf.writes(notebook), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
