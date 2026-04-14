from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .chunking import inspect_chunks as build_chunk_inspection
from .chunking import semantic_chunk_book
from .config import (
    BookSection,
    ChunkInspectionRow,
    ChunkRecord,
    PolymindConfig,
    QAResponse,
    RetrievalInspection,
    RewardBreakdown,
    RewardInspection,
    StageProfile,
    configure_logging,
    dataclass_to_dict,
)
from .embeddings import embed_chunks, load_embedding_model
from .generation import generate_answer_candidates, generate_level_summary, load_generator
from .ingestion import ingest_book
from .memory import ConversationMemory, rewrite_followup_query
from .retrieval import build_index, load_index, retrieve_chunks, retrieve_for_concepts, save_index
from .rl import EpsilonGreedyBanditPolicy, select_best_answer_with_policy

LOGGER = logging.getLogger("polymind.system")


def _stage_profile(stage: str, start: float, notes: str = "", device: str = "cpu") -> StageProfile:
    return StageProfile(
        stage=stage,
        runtime_s=round(time.perf_counter() - start, 4),
        device=device,
        peak_memory_mb=0.0,
        notes=notes,
    )


class PolymindSystem:
    def __init__(
        self,
        config: Optional[PolymindConfig] = None,
        embed_model=None,
        generator=None,
        reward_model=None,
    ) -> None:
        self.config = (config or PolymindConfig()).resolve_paths()
        self.config.ensure_directories()
        self.logger = configure_logging(self.config.log_level)
        self.embed_model = embed_model
        self.reward_model = reward_model
        self.generator = generator
        self.memory = ConversationMemory(config=self.config, generator=generator)
        self.policy = (
            EpsilonGreedyBanditPolicy.from_json(self.config.rl_stats_path(), self.config)
            if self.config.rl_stats_path().exists()
            else EpsilonGreedyBanditPolicy(self.config)
        )
        self.sections: List[BookSection] = []
        self.chunks: List[ChunkRecord] = []
        self.chunk_embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.index = None
        self.metadata: Dict[str, object] = {}
        self.last_chunk_inspection: List[ChunkInspectionRow] = []
        self.last_retrieval_inspection: Optional[RetrievalInspection] = None
        self.last_reward_inspection: Optional[RewardInspection] = None
        self.last_profiles: List[StageProfile] = []

    def _ensure_embed_model(self):
        if self.embed_model is None:
            self.embed_model = load_embedding_model(self.config.embedding_model_name)
        if self.reward_model is None:
            self.reward_model = self.embed_model
        return self.embed_model

    def _ensure_generator(self):
        if self.generator is None:
            self.generator = load_generator(self.config)
            self.memory.generator = self.generator
        return self.generator

    def _artifacts_ready(self) -> bool:
        return all(
            path.exists()
            for path in [
                self.config.chunks_path(),
                self.config.embeddings_path(),
                self.config.metadata_path(),
                self.config.index_path(),
            ]
        )

    def _save_chunks(self) -> None:
        payload = [asdict(chunk) for chunk in self.chunks]
        self.config.chunks_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_chunks(self) -> List[ChunkRecord]:
        payload = json.loads(self.config.chunks_path().read_text(encoding="utf-8"))
        return [ChunkRecord(**item) for item in payload]

    def _save_metadata(self) -> None:
        payload = {"config": self.config.to_dict(), "metadata": self.metadata}
        self.config.metadata_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_metadata(self) -> Dict[str, object]:
        payload = json.loads(self.config.metadata_path().read_text(encoding="utf-8"))
        return dict(payload.get("metadata", {}))

    def load_from_artifacts(self) -> bool:
        if not self._artifacts_ready():
            self.logger.info("Artifacts not found; load_from_artifacts skipped.")
            return False
        self.logger.info("Loading persisted artifacts from %s", self.config.artifact_dir)
        self.chunks = self._load_chunks()
        self.chunk_embeddings = np.load(self.config.embeddings_path())
        self.index = load_index(self.config.index_path())
        self.metadata = self._load_metadata()
        if self.config.rl_stats_path().exists():
            self.policy = EpsilonGreedyBanditPolicy.from_json(self.config.rl_stats_path(), self.config)
        self.last_chunk_inspection = self.inspect_chunks(limit=10)
        return True

    def load_book(self) -> bool:
        return self.load_from_artifacts()

    def index_book(self, book_path: str | Path, force_recompute: bool = False) -> Dict[str, object]:
        if not force_recompute and self.load_from_artifacts():
            return {"status": "loaded", "metadata": self.metadata}

        profiles: List[StageProfile] = []

        ingest_start = time.perf_counter()
        ingestion = ingest_book(book_path=book_path, config=self.config)
        self.sections = list(ingestion["sections"])
        self.metadata = dict(ingestion["metadata"])
        profiles.append(_stage_profile("ingest_book", ingest_start, notes=str(book_path)))

        chunk_start = time.perf_counter()
        self.chunks = semantic_chunk_book(self.sections, self.config)
        self.last_chunk_inspection = build_chunk_inspection(self.chunks, self.config, limit=12)
        profiles.append(_stage_profile("semantic_chunk_book", chunk_start, notes=f"chunks={len(self.chunks)}"))

        model = self._ensure_embed_model()
        embedding_start = time.perf_counter()
        self.chunk_embeddings, embedding_profile = embed_chunks(self.chunks, model, self.config)
        profiles.append(embedding_profile)
        profiles[-1].notes = f"chunks={len(self.chunks)}"

        index_start = time.perf_counter()
        self.index = build_index(self.chunk_embeddings)
        profiles.append(_stage_profile("build_index", index_start, notes=f"vectors={len(self.chunk_embeddings)}"))

        self._save_chunks()
        np.save(self.config.embeddings_path(), self.chunk_embeddings)
        save_index(self.index, self.config.index_path())
        self.policy.to_json(self.config.rl_stats_path())
        self.metadata["profiles"] = [dataclass_to_dict(profile) for profile in profiles]
        self._save_metadata()
        self.last_profiles = profiles
        return {"status": "indexed", "metadata": self.metadata, "profiles": dataclass_to_dict(profiles)}

    def inspect_chunks(self, limit: int = 10) -> List[ChunkInspectionRow]:
        self.last_chunk_inspection = build_chunk_inspection(self.chunks, self.config, limit=limit)
        return self.last_chunk_inspection

    def inspect_retrieval(self, question: Optional[str] = None, top_k: Optional[int] = None):
        if question is None:
            return self.last_retrieval_inspection
        self._ensure_embed_model()
        rewritten = rewrite_followup_query(question, self.memory, generator=self.generator)
        results, inspection, profile = retrieve_chunks(
            query=question,
            rewritten_query=rewritten,
            chunks=self.chunks,
            chunk_embeddings=self.chunk_embeddings,
            index=self.index,
            embed_model=self.embed_model,
            config=self.config,
            generator=self.generator,
            top_k=top_k,
        )
        self.last_retrieval_inspection = inspection
        self.last_profiles = [profile]
        return {"inspection": inspection, "results": results, "profile": profile}

    def inspect_rewards(self):
        return self.last_reward_inspection

    def inspect_memory(self):
        return self.memory.inspect_memory()

    def ask(
        self,
        question: str,
        use_rl: bool = True,
        top_k: Optional[int] = None,
    ) -> Dict[str, object]:
        if not self.chunks or self.index is None:
            raise RuntimeError("The book is not indexed yet. Call index_book() or load_from_artifacts() first.")

        self._ensure_embed_model()
        self._ensure_generator()

        profiles: List[StageProfile] = []

        rewrite_start = time.perf_counter()
        rewritten_question = rewrite_followup_query(question, self.memory, generator=self.generator)
        profiles.append(_stage_profile("rewrite_followup_query", rewrite_start, notes=rewritten_question))

        results, inspection, retrieval_profile = retrieve_chunks(
            query=question,
            rewritten_query=rewritten_question,
            chunks=self.chunks,
            chunk_embeddings=self.chunk_embeddings,
            index=self.index,
            embed_model=self.embed_model,
            config=self.config,
            generator=self.generator,
            top_k=top_k,
        )
        self.last_retrieval_inspection = inspection
        profiles.append(retrieval_profile)

        candidate_start = time.perf_counter()
        memory_context = self.memory.build_memory_summary()
        candidates = generate_answer_candidates(
            question=rewritten_question,
            retrieved_results=results,
            generator=self.generator,
            config=self.config,
            task="qa",
            memory_context=memory_context,
        )
        profiles.append(_stage_profile("generate_answer_candidates", candidate_start, notes=f"arms={len(candidates)}"))

        raw_chunk_ids = [row["chunk_id"] for row in inspection.raw_top_k]
        selected_ids = {result.chunk_id for result in results}
        held_out_text = ""
        chunk_lookup = {chunk.chunk_id: chunk for chunk in self.chunks}
        for chunk_id in raw_chunk_ids:
            if chunk_id not in selected_ids and chunk_id in chunk_lookup:
                held_out_text = chunk_lookup[chunk_id].text
                break

        if use_rl:
            selected, reward_breakdowns, reward_inspection = select_best_answer_with_policy(
                candidates=candidates,
                retrieved_results=results,
                model=self.reward_model or self.embed_model,
                policy=self.policy,
                config=self.config,
                held_out_text=held_out_text,
            )
        else:
            selected = candidates[0]
            reward_breakdowns = []
            reward_inspection = None

        self.last_reward_inspection = reward_inspection
        self.policy.to_json(self.config.rl_stats_path())
        selected_reward = 0.0
        for breakdown in reward_breakdowns:
            if breakdown.arm_name == selected.arm_name:
                selected_reward = breakdown.total_reward
                break

        memory_turn = self._remember_answer(
            question=question,
            rewritten_question=rewritten_question,
            answer=selected.answer,
            results=results,
            selected_arm=selected.arm_name,
            reward=selected_reward,
            citations=selected.citations,
        )

        response = QAResponse(
            question=question,
            rewritten_question=rewritten_question,
            answer=selected.answer,
            selected_arm=selected.arm_name,
            verified_citations=selected.citations,
            retrieved_chunks=results,
            candidate_rewards=reward_breakdowns,
            profile=profiles,
            metadata={
                "memory_turn": dataclass_to_dict(memory_turn),
                "retrieval_inspection": dataclass_to_dict(inspection),
                "reward_inspection": dataclass_to_dict(reward_inspection) if reward_inspection else {},
            },
        )
        self.last_profiles = profiles
        return dataclass_to_dict(response)

    def _remember_answer(
        self,
        question: str,
        rewritten_question: str,
        answer: str,
        results: Sequence[ChunkRecord | object],
        selected_arm: str,
        reward: float,
        citations: List[str],
    ):
        from .config import MemoryTurn

        turn = MemoryTurn(
            question=question,
            rewritten_question=rewritten_question,
            answer=answer,
            chunk_ids=[int(getattr(result, "chunk_id")) for result in results],
            selected_arm=selected_arm,
            reward=reward,
            citations=citations,
        )
        self.memory.add_turn(turn)
        return turn

    def summarize(self, query: str, level: str = "tldr") -> Dict[str, object]:
        if not self.chunks or self.index is None:
            raise RuntimeError("The book is not indexed yet.")
        self._ensure_embed_model()
        self._ensure_generator()
        top_k = self.config.top_k if level == "tldr" else self.config.top_k + 2
        rewritten = rewrite_followup_query(query, self.memory, generator=self.generator)
        results, inspection, profile = retrieve_chunks(
            query=query,
            rewritten_query=rewritten,
            chunks=self.chunks,
            chunk_embeddings=self.chunk_embeddings,
            index=self.index,
            embed_model=self.embed_model,
            config=self.config,
            generator=self.generator,
            top_k=top_k,
        )
        summary = generate_level_summary(
            query=rewritten,
            retrieved_results=results,
            generator=self.generator,
            config=self.config,
            level=level,
            memory_context=self.memory.build_memory_summary(),
        )
        self.last_retrieval_inspection = inspection
        return {
            "query": query,
            "rewritten_query": rewritten,
            "level": level,
            "summary": summary,
            "retrieved_chunks": dataclass_to_dict(results),
            "profile": dataclass_to_dict(profile),
        }

    def link_concepts(self, concept_a: str, concept_b: str) -> Dict[str, object]:
        if not self.chunks or self.index is None:
            raise RuntimeError("The book is not indexed yet.")
        self._ensure_embed_model()
        self._ensure_generator()
        results, inspections = retrieve_for_concepts(
            concept_a=concept_a,
            concept_b=concept_b,
            chunks=self.chunks,
            chunk_embeddings=self.chunk_embeddings,
            index=self.index,
            embed_model=self.embed_model,
            config=self.config,
            generator=self.generator,
        )
        question = f"Connect these ideas in the book: {concept_a} and {concept_b}"
        candidates = generate_answer_candidates(
            question=question,
            retrieved_results=results,
            generator=self.generator,
            config=self.config,
            task="link_concepts",
            memory_context=self.memory.build_memory_summary(),
        )
        selected, reward_breakdowns, reward_inspection = select_best_answer_with_policy(
            candidates=candidates,
            retrieved_results=results,
            model=self.reward_model or self._ensure_embed_model(),
            policy=self.policy,
            config=self.config,
        )
        self.last_reward_inspection = reward_inspection
        return {
            "concept_a": concept_a,
            "concept_b": concept_b,
            "answer": selected.answer,
            "selected_arm": selected.arm_name,
            "verified_citations": selected.citations,
            "retrieved_chunks": dataclass_to_dict(results),
            "inspections": dataclass_to_dict(inspections),
            "candidate_rewards": dataclass_to_dict(reward_breakdowns),
        }
