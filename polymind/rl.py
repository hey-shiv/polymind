from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import (
    AnswerCandidate,
    BanditArmStats,
    PolymindConfig,
    RetrievalResult,
    RewardBreakdown,
    RewardInspection,
)

LOGGER = logging.getLogger("polymind.rl")

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "is",
    "it",
    "that",
    "this",
    "for",
    "on",
    "with",
    "as",
    "by",
    "be",
    "are",
    "from",
    "at",
    "into",
}


def _tokenize(text: str) -> List[str]:
    tokens = [
        token.strip(".,!?;:\"'()[]{}").lower()
        for token in text.split()
    ]
    return [token for token in tokens if token and token not in STOPWORDS]


def _top_keywords(texts: Sequence[str], top_n: int = 20) -> List[str]:
    counts: Dict[str, int] = {}
    for text in texts:
        for token in _tokenize(text):
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [word for word, _count in ranked[:top_n]]


class EpsilonGreedyBanditPolicy:
    def __init__(self, config: PolymindConfig, stats: Optional[Dict[str, BanditArmStats]] = None):
        self.config = config
        self.random = random.Random(config.random_seed)
        self.stats = stats or {
            arm_name: BanditArmStats(arm_name=arm_name) for arm_name in config.answer_arms
        }

    @property
    def total_interactions(self) -> int:
        return sum(arm.pulls for arm in self.stats.values())

    @property
    def epsilon(self) -> float:
        return self.config.epsilon(self.total_interactions)

    def choose_preferred_arm(self) -> str:
        if self.random.random() < self.epsilon:
            return self.random.choice(list(self.stats.keys()))
        ranked = sorted(
            self.stats.values(),
            key=lambda item: (item.average_reward, item.wins, -item.pulls),
            reverse=True,
        )
        return ranked[0].arm_name

    def record_outcome(self, arm_name: str, reward: float, won: bool) -> None:
        stats = self.stats.setdefault(arm_name, BanditArmStats(arm_name=arm_name))
        stats.pulls += 1
        stats.total_reward += reward
        if won:
            stats.wins += 1
        stats.recent_rewards.append(round(reward, 4))
        stats.recent_rewards = stats.recent_rewards[-50:]

    def learning_curve(self) -> List[Dict[str, float]]:
        rows = []
        for arm_name, stats in self.stats.items():
            rows.append(
                {
                    "arm_name": arm_name,
                    "pulls": float(stats.pulls),
                    "wins": float(stats.wins),
                    "average_reward": float(stats.average_reward),
                }
            )
        return rows

    def to_json(self, path: str | Path) -> None:
        payload = {
            "arms": {
                arm_name: {
                    "arm_name": stats.arm_name,
                    "pulls": stats.pulls,
                    "wins": stats.wins,
                    "total_reward": stats.total_reward,
                    "recent_rewards": stats.recent_rewards,
                }
                for arm_name, stats in self.stats.items()
            }
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path, config: PolymindConfig) -> "EpsilonGreedyBanditPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        saved_arms = payload.get("arms", {})
        stats: Dict[str, BanditArmStats] = {}
        for arm_name in config.answer_arms:
            arm_payload = saved_arms.get(arm_name, {})
            stats[arm_name] = BanditArmStats(
                arm_name=arm_name,
                pulls=int(arm_payload.get("pulls", 0)),
                wins=int(arm_payload.get("wins", 0)),
                total_reward=float(arm_payload.get("total_reward", 0.0)),
                recent_rewards=list(arm_payload.get("recent_rewards", [])),
            )
        return cls(config=config, stats=stats)


def compute_answer_reward(
    candidate: AnswerCandidate,
    retrieved_results: Sequence[RetrievalResult],
    model,
    config: PolymindConfig,
    held_out_text: str = "",
) -> RewardBreakdown:
    context_text = "\n".join(result.text for result in retrieved_results)
    answer_embedding = model.encode([candidate.answer], convert_to_numpy=True, normalize_embeddings=True)[0]
    context_embeddings = model.encode(
        [result.text for result in retrieved_results],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    mean_context_embedding = np.mean(context_embeddings, axis=0)
    grounding_similarity = float(np.dot(answer_embedding, mean_context_embedding))

    retrieved_keywords = _top_keywords([context_text], top_n=24)
    answer_tokens = _tokenize(candidate.answer)
    keyword_hits = sum(1 for token in answer_tokens if token in set(retrieved_keywords))
    length_baseline = max(1.0, math.sqrt(max(1, len(answer_tokens))), float(len(retrieved_keywords)))
    normalized_keyword_overlap = min(1.0, keyword_hits / length_baseline)

    if candidate.citations:
        citation_validity = 1.0
    else:
        citation_validity = 0.4

    divergence_penalty = 0.0
    if held_out_text:
        held_out_embedding = model.encode(
            [held_out_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        held_out_similarity = float(np.dot(answer_embedding, held_out_embedding))
        divergence_penalty = max(0.0, held_out_similarity - grounding_similarity)
    else:
        held_out_similarity = 0.0

    total_reward = (
        config.reward_grounding_weight * grounding_similarity
        + config.reward_overlap_weight * normalized_keyword_overlap
        + config.reward_citation_weight * citation_validity
        - config.reward_divergence_weight * divergence_penalty
    )
    return RewardBreakdown(
        arm_name=candidate.arm_name,
        grounding_similarity=round(grounding_similarity, 4),
        normalized_keyword_overlap=round(normalized_keyword_overlap, 4),
        citation_validity=round(citation_validity, 4),
        divergence_penalty=round(divergence_penalty, 4),
        policy_bonus=0.0,
        total_reward=round(total_reward, 4),
        details={
            "answer_length": len(answer_tokens),
            "keyword_hits": keyword_hits,
            "held_out_similarity": round(held_out_similarity, 4),
        },
    )


def select_best_answer_with_policy(
    candidates: Sequence[AnswerCandidate],
    retrieved_results: Sequence[RetrievalResult],
    model,
    policy: EpsilonGreedyBanditPolicy,
    config: PolymindConfig,
    held_out_text: str = "",
) -> Tuple[AnswerCandidate, List[RewardBreakdown], RewardInspection]:
    preferred_arm = policy.choose_preferred_arm()
    reward_map: Dict[str, RewardBreakdown] = {}
    for candidate in candidates:
        reward_map[candidate.arm_name] = compute_answer_reward(
            candidate=candidate,
            retrieved_results=retrieved_results,
            model=model,
            config=config,
            held_out_text=held_out_text,
        )

    ranked_candidates = sorted(
        candidates,
        key=lambda item: reward_map[item.arm_name].total_reward,
        reverse=True,
    )
    selected = ranked_candidates[0]
    if len(ranked_candidates) > 1:
        leader = reward_map[ranked_candidates[0].arm_name]
        runner_up = reward_map[ranked_candidates[1].arm_name]
        if abs(leader.total_reward - runner_up.total_reward) <= config.tie_margin:
            if preferred_arm in {ranked_candidates[0].arm_name, ranked_candidates[1].arm_name}:
                selected = next(item for item in ranked_candidates[:2] if item.arm_name == preferred_arm)
                reward_map[selected.arm_name].policy_bonus = 0.01
                reward_map[selected.arm_name].total_reward = round(
                    reward_map[selected.arm_name].total_reward + 0.01,
                    4,
                )

    for candidate in candidates:
        reward = reward_map[candidate.arm_name].total_reward
        policy.record_outcome(candidate.arm_name, reward, won=(candidate.arm_name == selected.arm_name))

    inspection = RewardInspection(
        preferred_arm=preferred_arm,
        selected_arm=selected.arm_name,
        epsilon=round(policy.epsilon, 4),
        tie_margin=config.tie_margin,
        reward_breakdowns=[reward_map[candidate.arm_name] for candidate in candidates],
    )
    return selected, inspection.reward_breakdowns, inspection
