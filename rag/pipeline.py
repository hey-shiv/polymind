from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from config import (
    PERSONA_GENERATION_DEFAULTS,
    SETTINGS,
    Settings,
    ensure_local_layout,
    load_personality_cards,
)
from generate import generate_text, load_model
from rag.aggregator import ResponseAggregator
from rag.prompt_builder import PromptBuilder
from rag.retriever import PersonaRAGService, build_personality_retrievers
from rag.router import ModelRouter
from rag.types import QueryRequest
from src.device import get_default_device

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional dependency.
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None


EXPANSION_STOPWORDS = {
    "about",
    "after",
    "against",
    "because",
    "between",
    "could",
    "every",
    "their",
    "there",
    "these",
    "those",
    "which",
    "would",
}


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _first_content_words(text: str, limit: int = 6) -> list[str]:
    words: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z'-]+", text.lower()):
        if len(token) < 4 or token in EXPANSION_STOPWORDS:
            continue
        if token not in words:
            words.append(token)
        if len(words) >= limit:
            break
    return words


class MiniLLMService:
    model_name = "mini"

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.device = get_default_device()
        self._model = None
        self._tokenizer = None

    @property
    def is_available(self) -> bool:
        return self.model_path.exists()

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        self._model, self._tokenizer = load_model(self.model_path, self.device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
    ) -> str:
        self._ensure_loaded()
        return generate_text(
            model=self._model,
            tokenizer=self._tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=self.device,
        )


class TransformersLocalModel:
    def __init__(
        self,
        model_dir: Path,
        model_name: str,
        device: str | None = None,
        load_in_4bit: bool | None = None,
    ):
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError(
                "Transformers-based local generation requires the `transformers` package."
            )

        if not self.is_available(model_dir):
            raise FileNotFoundError(f"No local model directory found at {model_dir}")

        self.model_dir = model_dir
        self.model_name = model_name
        self.device = torch.device(device or get_default_device())
        self.load_in_4bit = bool(load_in_4bit if load_in_4bit is not None else self.device.type == "cuda")
        dtype = torch.float16 if self.device.type != "cpu" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=True,
        )

        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        if self.device.type == "cuda":
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = dtype

        if (
            self.load_in_4bit
            and self.device.type == "cuda"
            and BitsAndBytesConfig is not None
        ):
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif "torch_dtype" not in model_kwargs:
            model_kwargs["torch_dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            **model_kwargs,
        )

        if "device_map" not in model_kwargs:
            self.model.to(self.device)
        self.model.eval()
        self.input_device = self._infer_input_device()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def is_available(model_dir: Path) -> bool:
        return model_dir.exists() and (model_dir / "config.json").exists()

    def _infer_input_device(self) -> torch.device:
        model_device = getattr(self.model, "device", None)
        if isinstance(model_device, torch.device):
            return model_device

        if hasattr(self.model, "hf_device_map"):
            for mapped_device in self.model.hf_device_map.values():
                if isinstance(mapped_device, str) and mapped_device not in {"cpu", "disk"}:
                    return torch.device(mapped_device)

        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 220,
        temperature: float = 0.7,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {key: value.to(self.input_device) for key, value in inputs.items()}

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if text.startswith(prompt):
            stripped = text[len(prompt) :].strip()
            return stripped or text.strip()
        return text.strip()


class BootstrapPersonaModel:
    model_name = "bootstrap"

    PERSONA_STYLES = {
        "elon musk": {
            "opening": "Start by asking whether the problem is real, important, and constrained by something objective.",
            "reflection": "The useful move is to strip away ceremony, find the bottleneck, and iterate until the signal gets better.",
            "takeaway": "Pick one meaningful constraint, work on it directly, and let iteration do the compounding.",
        },
        "robert greene": {
            "opening": "Most people confuse motion with progress. Real success comes from strategic patience and emotional control.",
            "reflection": "What matters is not only effort, but timing, leverage, and the ability to read hidden incentives without wasting energy.",
            "takeaway": "Slow down, study the pattern, and move only where patience creates leverage.",
        },
        "steve jobs": {
            "opening": "Success usually looks simple from the outside because the hard work was choosing what to ignore.",
            "reflection": "Focus, taste, and care for the end experience create the kind of work people remember and trust.",
            "takeaway": "Say no to the noise, obsess over the essential thing, and raise the quality bar.",
        },
    }

    @staticmethod
    def _extract_section(prompt: str, label: str, next_label: str | None = None) -> str:
        if next_label:
            pattern = rf"{label}:\n(.*?)(?:\n\n{next_label}:|\Z)"
        else:
            pattern = rf"{label}:\n(.*)\Z"
        match = re.search(pattern, prompt, flags=re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _compress_excerpt(text: str, word_limit: int = 24) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split()
        if len(words) <= word_limit:
            return text
        return " ".join(words[:word_limit]).rstrip(" ,.;:") + "..."

    def _extract_context_excerpts(self, context_text: str, limit: int = 3) -> list[str]:
        blocks = re.findall(
            r"\[Source \d+\]\s+[^\n]+\n(.*?)(?=\n\n\[Source |\Z)",
            context_text,
            flags=re.DOTALL,
        )
        excerpts = []
        for block in blocks[:limit]:
            sentence = re.split(r"(?<=[.!?])\s+", block.strip())[0]
            excerpts.append(self._compress_excerpt(sentence))
        return [excerpt for excerpt in excerpts if excerpt]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 220,
        temperature: float = 0.7,
    ) -> str:
        del max_new_tokens, temperature

        persona_text = self._extract_section(prompt, "PERSONA", "USER QUESTION")
        query = self._extract_section(prompt, "USER QUESTION", "EXPANDED RETRIEVAL HINTS")
        context_text = self._extract_section(prompt, "RETRIEVED CONTEXT", "INSTRUCTIONS")

        persona_name_match = re.search(r"Name:\s*(.+)", persona_text)
        persona_name = persona_name_match.group(1).strip() if persona_name_match else "Local Persona"
        style = self.PERSONA_STYLES.get(persona_name.lower(), self.PERSONA_STYLES["steve jobs"])
        excerpts = self._extract_context_excerpts(context_text)

        if excerpts:
            evidence_line = "The retrieved material keeps pointing to " + "; ".join(excerpts) + "."
        else:
            evidence_line = (
                "This bootstrap run is using lightweight persona notes, so the answer stays "
                "conservative and evidence-first rather than pretending to know more than it does."
            )

        focus_terms = _first_content_words(query, limit=3)
        focus_line = (
            f'For "{query}", the strongest signal is to keep your attention on '
            f"{', '.join(focus_terms)}."
            if focus_terms
            else f'For "{query}", the strongest signal is to stay close to the core constraint.'
        )

        return (
            f"{style['opening']}\n\n"
            f"{focus_line}\n\n"
            f"{evidence_line}\n\n"
            f"{style['reflection']}\n\n"
            f"Takeaway: {style['takeaway']}"
        ).strip()


class ModelRegistry:
    def __init__(self, models: dict[str, Any]):
        self.models = dict(models)

    def get(self, model_name: str) -> Any:
        return self.models[model_name]

    def available_generation_models(self) -> set[str]:
        return {name for name in self.models if name != "mini"}


def expand_query_with_mini_llm(mini_service: MiniLLMService, query: str) -> str:
    prompt = f"Topic: {query}\nRelated themes:\n- "
    raw = mini_service.generate(prompt, max_new_tokens=48, temperature=0.85)
    cleaned = raw.replace(prompt, " ")
    keywords = _first_content_words(cleaned, limit=8)
    return ", ".join(keywords)


def safe_expand_query(mini_service: MiniLLMService | None, query: str) -> str:
    expansions: list[str] = []

    if mini_service is not None and mini_service.is_available:
        try:
            expanded = expand_query_with_mini_llm(mini_service, query)
            if len(expanded.split()) >= 3:
                expansions.append(expanded)
        except Exception:
            pass

    heuristic_terms = _first_content_words(query, limit=5)
    if heuristic_terms:
        expansions.append(", ".join(heuristic_terms))

    merged_terms: list[str] = []
    for expansion in expansions:
        for token in re.split(r"[,;\n]+", expansion):
            candidate = token.strip().lower()
            if not candidate or candidate in merged_terms:
                continue
            merged_terms.append(candidate)

    if not merged_terms:
        return query

    return f"{query}. Related ideas: {', '.join(merged_terms[:8])}"


class MultiPersonalityPipeline:
    def __init__(
        self,
        mini_service: MiniLLMService | None,
        rag_service: PersonaRAGService,
        prompt_builder: PromptBuilder,
        model_router: ModelRouter,
        model_registry: ModelRegistry,
        aggregator: ResponseAggregator,
        personality_cards: dict[str, dict[str, Any]],
    ):
        self.mini_service = mini_service
        self.rag_service = rag_service
        self.prompt_builder = prompt_builder
        self.model_router = model_router
        self.model_registry = model_registry
        self.aggregator = aggregator
        self.personality_cards = personality_cards

    def _write_run_log(self, result: dict[str, Any]) -> None:
        ensure_local_layout()
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        query_slug = _slugify(result["query"])[:40] or "run"
        path = SETTINGS.runs_dir / f"{timestamp}-{query_slug}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(self, request: QueryRequest | str) -> dict[str, Any]:
        if isinstance(request, str):
            request = QueryRequest(query=request)

        query = request.query.strip()
        if not query:
            raise ValueError("query must not be empty.")

        top_k = max(1, int(request.top_k))
        style_strength = max(0.0, min(1.0, float(request.style_strength)))
        expanded_query = safe_expand_query(self.mini_service, query)

        bundles = self.rag_service.build_all(
            query=query,
            expanded_query=expanded_query,
            top_k=top_k,
            personality_ids=SETTINGS.persona_ids,
        )

        raw_responses: list[dict[str, Any]] = []
        trace: dict[str, Any] = {
            "query": query,
            "expanded_query": expanded_query,
            "mode_requested": "debate" if request.debate_mode else "standard",
            "mode_executed": "standard",
            "routing": {},
            "retrieval": {},
            "available_models": sorted(self.model_registry.available_generation_models()),
        }

        if request.debate_mode:
            trace["note"] = "Debate mode is scaffolded in the UI, but this build still executes standard comparison generation."

        for personality_id in SETTINGS.persona_ids:
            bundle = bundles[personality_id]
            prompt = self.prompt_builder.build(
                query=query,
                bundle=bundle,
                style_strength=style_strength,
            )

            route = self.model_router.choose_generation_model(personality_id, query)
            generation_settings = PERSONA_GENERATION_DEFAULTS.get(
                personality_id,
                {"temperature": 0.7, "max_new_tokens": 220},
            )
            model = self.model_registry.get(route.model_name)
            answer = model.generate(prompt, **generation_settings)

            trace["routing"][personality_id] = {
                "model": route.model_name,
                "reason": route.reason,
            }
            trace["retrieval"][personality_id] = {
                "backend": bundle.retrieval_backend,
                "sources": [chunk.source_label for chunk in bundle.chunks],
            }

            raw_responses.append(
                {
                    "personality_id": personality_id,
                    "model_name": route.model_name,
                    "answer": answer,
                    "chunks": bundle.chunks,
                }
            )

        result = self.aggregator.aggregate(
            query=query,
            expanded_query=expanded_query,
            raw_responses=raw_responses,
            trace=trace,
        )
        self._write_run_log(result)
        return result


def build_model_registry(settings: Settings = SETTINGS) -> ModelRegistry:
    models: dict[str, Any] = {"bootstrap": BootstrapPersonaModel()}
    use_4bit = str(os.getenv("POLYMIND_USE_4BIT", str(settings.default_use_4bit))).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if settings.mistral_model_dir.exists():
        try:
            models["mistral"] = TransformersLocalModel(
                model_dir=settings.mistral_model_dir,
                model_name="mistral",
                load_in_4bit=use_4bit,
            )
        except Exception:
            pass

    secondary_dir = (
        settings.secondary_model_dir
        if settings.secondary_model_dir.exists()
        else settings.legacy_secondary_model_dir
    )
    if secondary_dir.exists():
        try:
            models["secondary"] = TransformersLocalModel(
                model_dir=secondary_dir,
                model_name="secondary",
                load_in_4bit=use_4bit,
            )
        except Exception:
            pass

    return ModelRegistry(models=models)


def build_pipeline(settings: Settings = SETTINGS) -> MultiPersonalityPipeline:
    ensure_local_layout()
    personality_cards = load_personality_cards(settings.personality_cards_path)

    mini_service: MiniLLMService | None = None
    if settings.mini_checkpoint.exists():
        mini_service = MiniLLMService(settings.mini_checkpoint)

    retrievers = build_personality_retrievers(settings)
    rag_service = PersonaRAGService(retrievers)
    prompt_builder = PromptBuilder(settings)
    model_registry = build_model_registry(settings)
    model_router = ModelRouter(model_registry.available_generation_models())
    aggregator = ResponseAggregator(personality_cards)

    return MultiPersonalityPipeline(
        mini_service=mini_service,
        rag_service=rag_service,
        prompt_builder=prompt_builder,
        model_router=model_router,
        model_registry=model_registry,
        aggregator=aggregator,
        personality_cards=personality_cards,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-personality PolyMind pipeline.")
    parser.add_argument(
        "query",
        nargs="?",
        default="What does success require in the long term?",
        help="Question to run through the pipeline.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=SETTINGS.default_top_k,
        help="Number of chunks to retrieve per personality.",
    )
    parser.add_argument(
        "--style-strength",
        type=float,
        default=SETTINGS.default_style_strength,
        help="How strongly to emphasize each persona style.",
    )
    parser.add_argument(
        "--debate-mode",
        action="store_true",
        help="Record debate mode intent in the trace. Standard comparison remains active in this build.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = build_pipeline()
    result = pipeline.run(
        QueryRequest(
            query=args.query,
            top_k=args.top_k,
            style_strength=args.style_strength,
            debate_mode=args.debate_mode,
        )
    )

    for card in result["cards"]:
        print("=" * 80)
        print(f"{card['display_name']} | model: {card['model_name']}")
        print(card["answer"])
        print("Sources:")
        for citation in card["citations"]:
            print(f"- {citation['source_label']}")


if __name__ == "__main__":
    main()
