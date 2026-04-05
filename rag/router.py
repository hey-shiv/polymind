from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouteDecision:
    model_name: str
    reason: str


def extract_query_features(query: str) -> dict[str, bool | int]:
    lowered = query.lower()
    return {
        "word_count": len(query.split()),
        "asks_for_steps": "step" in lowered or "how" in lowered,
        "asks_for_comparison": "compare" in lowered or "difference" in lowered,
        "asks_for_shortness": "brief" in lowered or "short" in lowered or "concise" in lowered,
        "asks_for_depth": "deep" in lowered or "detailed" in lowered or "explain" in lowered,
    }


class ModelRouter:
    def __init__(self, available_models: set[str]):
        if not available_models:
            raise ValueError("ModelRouter requires at least one available model.")
        self.available_models = set(available_models)

    def choose_generation_model(self, personality_id: str, query: str) -> RouteDecision:
        features = extract_query_features(query)
        preferred_by_persona = {
            "elon_musk": "mistral",
            "robert_greene": "secondary",
            "steve_jobs": "secondary",
        }

        preferred = preferred_by_persona.get(personality_id, "mistral")
        reasons = [f"default route for {personality_id}"]

        if bool(features["asks_for_depth"]) or int(features["word_count"]) > 20:
            preferred = "mistral"
            reasons.append("query asks for more depth")
        elif bool(features["asks_for_shortness"]):
            preferred = "secondary"
            reasons.append("query asks for brevity")

        if preferred not in self.available_models:
            fallback_order = ["secondary", "mistral", "bootstrap"]
            preferred = next(
                (
                    candidate
                    for candidate in fallback_order
                    if candidate in self.available_models
                ),
                next(iter(self.available_models)),
            )
            reasons.append("preferred model unavailable, using fallback")

        return RouteDecision(model_name=preferred, reason="; ".join(reasons))

    def choose_reviewer_model(self) -> str:
        if "secondary" in self.available_models:
            return "secondary"
        if "mistral" in self.available_models:
            return "mistral"
        return next(iter(self.available_models))
