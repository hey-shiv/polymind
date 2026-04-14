from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .config import AnswerCandidate, PolymindConfig, RetrievalResult
from .embeddings import detect_device
from .retrieval import format_retrieval_context

LOGGER = logging.getLogger("polymind.generation")

CITATION_PATTERN = re.compile(r"\[Chapter\s+(\d+)\s+\|\s+Chunk\s+(\d+)\]")


@dataclass
class GeneratorBundle:
    model: object
    tokenizer: object
    model_name: str
    device: str

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 0.2,
    ) -> str:
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise ImportError("torch is required for generator inference.") from exc

        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        if self.device == "cuda":
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
        outputs = self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-4),
            top_p=0.95,
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.strip()


def cleanup_cuda_memory() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def load_generator(config: Optional[PolymindConfig] = None) -> GeneratorBundle:
    config = config or PolymindConfig()
    device = detect_device()
    try:
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise ImportError("transformers and torch are required for generation.") from exc

    use_8bit = bool(config.use_8bit_quantization and device == "cuda")
    model_name = config.generator_model_name if device == "cuda" else config.generator_fallback_model_name
    LOGGER.info("Loading generator %s on %s", model_name, device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_kwargs = {"device_map": "auto" if device == "cuda" else None}
    if use_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
    if device != "cuda":
        model.to("cpu")
    return GeneratorBundle(model=model, tokenizer=tokenizer, model_name=model_name, device=device)


def build_grounded_prompt(
    question: str,
    retrieved_results: Sequence[RetrievalResult],
    instruction: str,
    memory_context: str = "",
) -> str:
    context = format_retrieval_context(retrieved_results)
    memory_block = f"Conversation memory:\n{memory_context}\n\n" if memory_context else ""
    return (
        "You are Polymind, a grounded deep reading assistant.\n"
        f"{instruction}\n"
        "Use only the retrieved context. When evidence supports it, cite chunks as "
        "[Chapter 03 | Chunk 014].\n\n"
        f"{memory_block}"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer:"
    )


def _call_generator(generator, prompt: str, max_new_tokens: int, temperature: float) -> str:
    if generator is None:
        return ""
    if isinstance(generator, GeneratorBundle):
        return generator.generate_text(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    if hasattr(generator, "generate_text"):
        return generator.generate_text(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    if callable(generator):
        return generator(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    raise TypeError("Unsupported generator interface.")


def verify_citations(
    answer: str,
    retrieved_results: Sequence[RetrievalResult],
) -> tuple[str, List[str]]:
    valid_citations = {
        f"[Chapter {result.chapter_id:02d} | Chunk {result.chunk_id:03d}]"
        for result in retrieved_results
    }
    citations = CITATION_PATTERN.findall(answer)
    verified: List[str] = []
    for chapter_id, chunk_id in citations:
        formatted = f"[Chapter {int(chapter_id):02d} | Chunk {int(chunk_id):03d}]"
        if formatted in valid_citations:
            verified.append(formatted)
        else:
            answer = answer.replace(formatted, "").replace("  ", " ")
    return answer.strip(), verified


def generate_answer_candidates(
    question: str,
    retrieved_results: Sequence[RetrievalResult],
    generator,
    config: PolymindConfig,
    task: str = "qa",
    memory_context: str = "",
) -> List[AnswerCandidate]:
    candidates: List[AnswerCandidate] = []
    max_new_tokens = config.token_budget(task)
    for arm_name, arm in config.answer_arms.items():
        temperature = float(arm["temperature"])
        instruction = str(arm["instruction"])
        prompt = build_grounded_prompt(
            question=question,
            retrieved_results=retrieved_results,
            instruction=instruction,
            memory_context=memory_context,
        )
        answer = _call_generator(
            generator,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        answer, citations = verify_citations(answer, retrieved_results)
        candidates.append(
            AnswerCandidate(
                arm_name=arm_name,
                prompt=prompt,
                answer=answer,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                citations=citations,
                metadata={"instruction": instruction, "task": task},
            )
        )
        cleanup_cuda_memory()
    return candidates


def generate_level_summary(
    query: str,
    retrieved_results: Sequence[RetrievalResult],
    generator,
    config: PolymindConfig,
    level: str = "tldr",
    memory_context: str = "",
) -> str:
    instruction_map: Dict[str, str] = {
        "tldr": "Write a compact summary in 3-5 lines.",
        "chapter": "Write a chapter-level summary with coverage across the retrieved evidence.",
        "deep": "Write a deep explanatory summary that teaches the concept carefully.",
    }
    prompt = build_grounded_prompt(
        question=query,
        retrieved_results=retrieved_results,
        instruction=instruction_map.get(level, instruction_map["tldr"]),
        memory_context=memory_context,
    )
    answer = _call_generator(
        generator,
        prompt=prompt,
        max_new_tokens=config.token_budget(level),
        temperature=0.2,
    )
    verified_answer, _ = verify_citations(answer, retrieved_results)
    cleanup_cuda_memory()
    return verified_answer
