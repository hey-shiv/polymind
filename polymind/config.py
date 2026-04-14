from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def default_answer_arms() -> Dict[str, Dict[str, Any]]:
    return {
        "grounded_concise": {
            "temperature": 0.2,
            "instruction": "Answer briefly and directly. Stay close to the retrieved evidence.",
        },
        "teacher_explainer": {
            "temperature": 0.4,
            "instruction": "Answer like a patient teacher. Explain the idea step by step.",
        },
        "synthesizer": {
            "temperature": 0.7,
            "instruction": "Connect themes across the evidence while staying grounded.",
        },
    }


def default_task_token_budgets() -> Dict[str, int]:
    return {
        "tldr": 120,
        "qa": 220,
        "chapter": 220,
        "deep": 400,
        "link_concepts": 400,
        "rewrite": 96,
        "memory_summary": 160,
        "hyde": 160,
    }


def default_experiment_grid() -> Dict[str, List[Any]]:
    return {
        "top_k": [4, 6, 8, 10],
        "chunk_target_words": [280, 360, 420, 520],
        "temperatures": [0.2, 0.4, 0.7],
    }


def _path_string(path: Path | str) -> str:
    return str(path) if isinstance(path, Path) else path


@dataclass
class PolymindConfig:
    project_name: str = "polymind_v2"
    project_root: Path = Path("/content/drive/MyDrive/polymind_v2")
    data_dir_name: str = "data"
    artifact_dir_name: str = "artifacts"
    notebook_dir_name: str = "notebooks"
    log_level: int = logging.INFO
    random_seed: int = 42
    chunk_target_words: int = 420
    chunk_overlap_words: int = 80
    min_chunk_words: int = 240
    max_chunk_words: int = 540
    embedding_batch_size: int = 64
    top_k: int = 8
    initial_retrieval_multiplier: int = 3
    chapter_diversity_limit: int = 2
    retrieval_dedup_threshold: float = 0.92
    tie_margin: float = 0.03
    reward_grounding_weight: float = 0.40
    reward_overlap_weight: float = 0.25
    reward_citation_weight: float = 0.25
    reward_divergence_weight: float = 0.10
    min_epsilon: float = 0.05
    max_epsilon: float = 0.30
    epsilon_decay_base: float = 0.95
    epsilon_decay_window: int = 10
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    generator_model_name: str = "google/flan-t5-xl"
    generator_fallback_model_name: str = "google/flan-t5-base"
    use_8bit_quantization: bool = True
    answer_arms: Dict[str, Dict[str, Any]] = field(default_factory=default_answer_arms)
    task_token_budgets: Dict[str, int] = field(default_factory=default_task_token_budgets)
    experiment_grid: Dict[str, List[Any]] = field(default_factory=default_experiment_grid)
    enable_stage_profiling: bool = True
    enable_memory_compression: bool = True
    enable_hyde: bool = True
    enable_citation_verification: bool = True
    enable_logging: bool = True
    chunks_filename: str = "chunks.json"
    embeddings_filename: str = "embeddings.npy"
    metadata_filename: str = "metadata.json"
    index_filename: str = "faiss.index"
    rl_stats_filename: str = "rl_bandit_stats.json"

    @property
    def data_dir(self) -> Path:
        return self.project_root / self.data_dir_name

    @property
    def artifact_dir(self) -> Path:
        return self.project_root / self.artifact_dir_name

    def resolve_paths(self, root_override: Path | str | None = None) -> "PolymindConfig":
        if root_override is not None:
            self.project_root = Path(root_override)
        self.project_root = Path(self.project_root)
        return self

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def artifact_path(self, name: str) -> Path:
        return self.artifact_dir / name

    def chunks_path(self) -> Path:
        return self.artifact_path(self.chunks_filename)

    def embeddings_path(self) -> Path:
        return self.artifact_path(self.embeddings_filename)

    def metadata_path(self) -> Path:
        return self.artifact_path(self.metadata_filename)

    def index_path(self) -> Path:
        return self.artifact_path(self.index_filename)

    def rl_stats_path(self) -> Path:
        return self.artifact_path(self.rl_stats_filename)

    def token_budget(self, task: str) -> int:
        return self.task_token_budgets.get(task, self.task_token_budgets["qa"])

    def epsilon(self, total_interactions: int) -> float:
        decayed = self.max_epsilon * (
            self.epsilon_decay_base ** (total_interactions // self.epsilon_decay_window)
        )
        return max(self.min_epsilon, decayed)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["project_root"] = _path_string(self.project_root)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PolymindConfig":
        payload = dict(payload)
        if "project_root" in payload:
            payload["project_root"] = Path(payload["project_root"])
        return cls(**payload)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


@dataclass
class BookSection:
    chapter_id: int
    chapter_title: str
    text: str
    start_char: int
    end_char: int
    source_path: str = ""


@dataclass
class ChunkRecord:
    chunk_id: int
    chapter_id: int
    chapter_title: str
    text: str
    start_char: int
    end_char: int
    sentence_count: int
    word_count: int
    source_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunk_id: int
    chapter_id: int
    chapter_title: str
    text: str
    score: float
    rank: int
    source_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerCandidate:
    arm_name: str
    prompt: str
    answer: str
    temperature: float
    max_new_tokens: int
    citations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardBreakdown:
    arm_name: str
    grounding_similarity: float
    normalized_keyword_overlap: float
    citation_validity: float
    divergence_penalty: float
    policy_bonus: float
    total_reward: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BanditArmStats:
    arm_name: str
    pulls: int = 0
    wins: int = 0
    total_reward: float = 0.0
    recent_rewards: List[float] = field(default_factory=list)

    @property
    def average_reward(self) -> float:
        if self.pulls == 0:
            return 0.0
        return self.total_reward / self.pulls


@dataclass
class MemoryTurn:
    question: str
    rewritten_question: str
    answer: str
    chunk_ids: List[int]
    selected_arm: str
    reward: float
    citations: List[str] = field(default_factory=list)


@dataclass
class ChunkInspectionRow:
    chunk_id: int
    chapter_id: int
    chapter_title: str
    word_count: int
    sentence_count: int
    start_char: int
    end_char: int
    overlap_ratio: float
    warnings: List[str] = field(default_factory=list)
    preview: str = ""


@dataclass
class RetrievalInspection:
    query: str
    rewritten_query: str
    hyde_query: str
    raw_top_k: List[Dict[str, Any]]
    selected_results: List[Dict[str, Any]]
    chapter_counts: Dict[int, int]
    dedup_events: List[str]
    diversity_events: List[str]


@dataclass
class RewardInspection:
    preferred_arm: str
    selected_arm: str
    epsilon: float
    tie_margin: float
    reward_breakdowns: List[RewardBreakdown]


@dataclass
class MemoryInspection:
    summary: str
    recent_turns: List[MemoryTurn]
    compression_events: int
    rewrite_context: str


@dataclass
class StageProfile:
    stage: str
    runtime_s: float
    device: str
    peak_memory_mb: float
    notes: str = ""


@dataclass
class QAResponse:
    question: str
    rewritten_question: str
    answer: str
    selected_arm: str
    verified_citations: List[str]
    retrieved_chunks: List[RetrievalResult]
    candidate_rewards: List[RewardBreakdown]
    profile: List[StageProfile]
    metadata: Dict[str, Any] = field(default_factory=dict)


def dataclass_to_dict(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: dataclass_to_dict(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {key: dataclass_to_dict(val) for key, val in value.items()}
    if isinstance(value, list):
        return [dataclass_to_dict(item) for item in value]
    return value


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("polymind")
    logger.setLevel(level)
    return logger
