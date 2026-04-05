from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from config import SETTINGS, ensure_local_layout, is_colab_runtime
from pipeline.epub_to_text import configure_logger, process_books_in_folder
from rag.build_embeddings import build_all_indices
from rag.ingest_books import ingest_books


def _get_hf_token(token: str | None = None) -> str | None:
    return token or os.getenv(SETTINGS.hf_token_env_var)


def _require_huggingface_hub():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Model download helpers require `huggingface_hub`. "
            "Install `requirements-colab.txt` first."
        ) from exc
    return snapshot_download


def inspect_book_folders() -> dict[str, Any]:
    ensure_local_layout()
    summary: dict[str, Any] = {}

    for personality_id in SETTINGS.persona_ids:
        persona_dir = SETTINGS.raw_books_dir / personality_id
        txt_files = sorted(path.name for path in persona_dir.rglob("*.txt"))
        epub_files = sorted(path.name for path in persona_dir.rglob("*.epub"))
        pdf_files = sorted(path.name for path in persona_dir.rglob("*.pdf"))
        summary[personality_id] = {
            "directory": str(persona_dir),
            "txt_files": txt_files,
            "epub_files": epub_files,
            "pdf_files": pdf_files,
            "count": len(txt_files),
        }

    return summary


def prepare_book_text_sources() -> dict[str, Any]:
    ensure_local_layout()
    logger = configure_logger()
    conversions: dict[str, Any] = {}

    for personality_id in SETTINGS.persona_ids:
        persona_dir = SETTINGS.raw_books_dir / personality_id
        results = process_books_in_folder(
            persona_dir,
            save_txt=True,
            output_dir=persona_dir,
            recursive=True,
            logger=logger,
        )
        conversions[personality_id] = {
            "converted_files": [str(result.output_path) for result in results if result.output_path],
            "converted_count": len([result for result in results if result.output_path]),
        }

    return conversions


def assert_books_present() -> dict[str, Any]:
    summary = inspect_book_folders()
    missing = [persona_id for persona_id, item in summary.items() if item["count"] == 0]
    if missing:
        raise ValueError(
            "Missing persona book files for: "
            + ", ".join(missing)
            + ". Put `.txt`, `.epub`, or `.pdf` files into the corresponding `data/raw/books/<persona>/` folders first."
        )
    return summary


def download_model_snapshot(
    repo_id: str,
    target_dir: Path,
    token: str | None = None,
    revision: str | None = None,
) -> Path:
    ensure_local_layout()
    snapshot_download = _require_huggingface_hub()
    target_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        token=_get_hf_token(token),
        revision=revision,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return target_dir


def prepare_embedding_model(
    model_id: str = SETTINGS.default_embedding_model_id,
    token: str | None = None,
) -> Path:
    return download_model_snapshot(
        repo_id=model_id,
        target_dir=SETTINGS.embedding_model_dir,
        token=token,
    )


def prepare_generation_models(
    mistral_model_id: str = SETTINGS.default_mistral_model_id,
    secondary_model_id: str | None = SETTINGS.default_secondary_model_id,
    token: str | None = None,
) -> dict[str, str]:
    prepared = {
        "mistral": str(
            download_model_snapshot(
                repo_id=mistral_model_id,
                target_dir=SETTINGS.mistral_model_dir,
                token=token,
            )
        )
    }

    if secondary_model_id:
        prepared["secondary"] = str(
            download_model_snapshot(
                repo_id=secondary_model_id,
                target_dir=SETTINGS.secondary_model_dir,
                token=token,
            )
        )

    return prepared


def build_retrieval_assets() -> dict[str, Any]:
    chunks_written = ingest_books()
    index_stats = build_all_indices()
    return {
        "chunks_written": chunks_written,
        "indices": index_stats,
        "chunks_path": str(SETTINGS.chunks_path),
        "embeddings_dir": str(SETTINGS.embeddings_dir),
    }


def prepare_project_runtime(
    mistral_model_id: str = SETTINGS.default_mistral_model_id,
    secondary_model_id: str | None = SETTINGS.default_secondary_model_id,
    embedding_model_id: str = SETTINGS.default_embedding_model_id,
    token: str | None = None,
) -> dict[str, Any]:
    conversions = prepare_book_text_sources()
    books = assert_books_present()
    embedding_dir = prepare_embedding_model(model_id=embedding_model_id, token=token)
    models = prepare_generation_models(
        mistral_model_id=mistral_model_id,
        secondary_model_id=secondary_model_id,
        token=token,
    )
    retrieval = build_retrieval_assets()

    from ui.gradio_app import get_pipeline

    get_pipeline.cache_clear()

    summary = {
        "book_conversions": conversions,
        "books": books,
        "embedding_model_dir": str(embedding_dir),
        "models": models,
        "retrieval": retrieval,
    }

    summary_path = SETTINGS.outputs_dir / "colab_runtime_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_quickstart() -> None:
    ensure_local_layout()
    print("PolyMind Colab Quickstart")
    print("")
    print("1. Install dependencies:")
    print("   pip install -r requirements-colab.txt")
    print("")
    print("2. Put persona source files in (`.txt`, `.epub`, or `.pdf`):")
    for personality_id in SETTINGS.persona_ids:
        print(f"   {SETTINGS.raw_books_dir / personality_id}")
    print("")
    print("3. Download models and build retrieval in one step:")
    print("   from demo.colab_quickstart import prepare_project_runtime")
    print("   prepare_project_runtime(")
    print(f"       mistral_model_id='{SETTINGS.default_mistral_model_id}',")
    print(f"       secondary_model_id='{SETTINGS.default_secondary_model_id}',")
    print("   )")
    print("")
    print("4. Launch the app from a notebook cell:")
    print("   from demo.colab_quickstart import launch_app")
    print("   launch_app()")


def launch_app(share: bool | None = None, debug: bool = False, **kwargs):
    from ui.gradio_app import demo

    share = is_colab_runtime() if share is None else share
    return demo.launch(share=share, debug=debug, **kwargs)
