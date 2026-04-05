# PolyMind Colab Instruction Manual

## Purpose

This manual explains exactly what you need to do to run the multi-personality PolyMind project in Google Colab.

The current project supports:

- persona-based book folders
- automatic EPUB and PDF to TXT conversion
- chunking and retrieval setup
- FAISS index building
- local embedding model download
- local Mistral and LLaMA model download
- side-by-side Gradio comparison UI

This guide assumes:

- you will run the project in Google Colab
- your book files are already organized into persona folders
- you want Mistral and LLaMA loaded locally inside Colab

## Folder Structure You Must Have

Inside the project, keep your persona source files in these folders:

```text
data/raw/books/elon_musk/
data/raw/books/robert_greene/
data/raw/books/steve_jobs/
```

Accepted book formats:

- `.txt`
- `.epub`
- `.pdf`

If you place EPUB or PDF files in these folders, the project will automatically convert them into TXT files during setup.

## Recommended Colab Runtime

Open Google Colab and select:

- Runtime -> Change runtime type
- Hardware accelerator -> GPU

Using GPU is strongly recommended because Mistral and LLaMA generation will be slow on CPU.

## Step 1: Open the Project in Colab

Upload the repository to Colab, mount Google Drive, or clone the repo into Colab.

Then make sure your notebook is running from the project root directory.

## Step 2: Install Dependencies

Run this cell:

```python
!pip install -r requirements-colab.txt
```

Important note:

- do not replace Colab's preinstalled PyTorch unless you have a very specific reason
- `requirements-colab.txt` is designed to keep Colab's GPU Torch intact

## Step 3: Add Hugging Face Token if Needed

If your chosen LLaMA model is gated, set your Hugging Face token before downloading models.

Run this cell:

```python
import os
os.environ["HF_TOKEN"] = "your_huggingface_token_here"
```

If your selected model is public, this may not be necessary.

## Step 4: Make Sure the Books Are in the Correct Folders

Before running the setup, verify that your persona books are inside:

```text
data/raw/books/elon_musk/
data/raw/books/robert_greene/
data/raw/books/steve_jobs/
```

Example:

```text
data/raw/books/elon_musk/book1.pdf
data/raw/books/elon_musk/interview_notes.txt
data/raw/books/robert_greene/mastery.epub
data/raw/books/steve_jobs/speech_notes.txt
```

## Step 5: Run Full Project Preparation

Run this cell:

```python
from demo.colab_quickstart import prepare_project_runtime

prepare_project_runtime(
    mistral_model_id="mistralai/Mistral-7B-Instruct-v0.2",
    secondary_model_id="meta-llama/Llama-3.2-3B-Instruct",
)
```

What this setup does:

1. converts EPUB and PDF files into TXT files when needed
2. checks that each personality folder has usable text
3. downloads the embedding model
4. downloads the Mistral model into `models/mistral/merged/`
5. downloads the LLaMA model into `models/llama/merged/`
6. ingests and chunks all persona text
7. builds FAISS retrieval indices
8. refreshes the internal app pipeline cache

## Step 6: Launch the App

Run this cell:

```python
from demo.colab_quickstart import launch_app
launch_app()
```

This launches the Gradio UI.

In Colab, the app is configured to launch with sharing enabled automatically.

## Step 7: Use the Interface

Inside the Gradio UI:

- enter a question
- set style strength
- set top-k retrieval depth
- optionally show trace output

The app will return:

- Elon Musk response card
- Robert Greene response card
- Steve Jobs response card
- simple comparison metrics
- routing and retrieval trace if enabled

## Recommended First Test Questions

Use these first:

- `What is success?`
- `How do great builders stay focused?`
- `How should I respond to failure?`
- `What matters more: strategy or execution?`
- `How do I build something meaningful?`

## Important Output Locations

During setup and runtime, the project writes useful files here:

```text
data/processed/chunks.jsonl
embeddings/
outputs/runs/
outputs/colab_runtime_summary.json
```

What they mean:

- `data/processed/chunks.jsonl` = cleaned and chunked corpus
- `embeddings/` = FAISS indices and retrieval metadata
- `outputs/runs/` = saved query traces
- `outputs/colab_runtime_summary.json` = summary of setup actions

## If You Want Different Models

You can change the model IDs when calling `prepare_project_runtime(...)`.

Example:

```python
prepare_project_runtime(
    mistral_model_id="mistralai/Mistral-7B-Instruct-v0.2",
    secondary_model_id="meta-llama/Llama-3.1-8B-Instruct",
)
```

Only do this if your Colab runtime has enough GPU memory.

## Troubleshooting

### Problem: setup says books are missing

Check that each persona folder contains at least one supported file:

- `.txt`
- `.epub`
- `.pdf`

### Problem: LLaMA download fails

Possible causes:

- missing Hugging Face token
- token does not have access to the gated model
- incorrect model ID

### Problem: FAISS or sentence-transformers import error

Usually this means the dependencies were not installed correctly.

Run again:

```python
!pip install -r requirements-colab.txt
```

### Problem: generation is too slow

Make sure:

- Colab GPU runtime is enabled
- you are not running on CPU accidentally
- model choice is reasonable for the available memory

### Problem: CUDA out of memory

Try one of these:

- restart the runtime and run fewer extra cells
- use a smaller secondary model
- avoid unnecessarily large open notebooks and GPU allocations

## Minimum Practical Workflow

If you want the shortest working flow, do exactly this:

```python
!pip install -r requirements-colab.txt
```

```python
import os
os.environ["HF_TOKEN"] = "your_token_if_needed"
```

```python
from demo.colab_quickstart import prepare_project_runtime, launch_app

prepare_project_runtime(
    mistral_model_id="mistralai/Mistral-7B-Instruct-v0.2",
    secondary_model_id="meta-llama/Llama-3.2-3B-Instruct",
)
launch_app()
```

## Final Checklist

Before demoing, confirm all of these:

- Colab runtime is GPU
- books are inside the correct persona folders
- dependencies installed from `requirements-colab.txt`
- Hugging Face token set if required
- `prepare_project_runtime(...)` completed successfully
- Gradio app launches
- one test query works end to end

## Main Files Behind the Workflow

These are the key implementation files:

- `demo/colab_quickstart.py`
- `rag/ingest_books.py`
- `rag/build_embeddings.py`
- `rag/pipeline.py`
- `ui/gradio_app.py`

## One-Line Summary

Put the books in the persona folders, install `requirements-colab.txt`, run `prepare_project_runtime(...)`, then run `launch_app()`.
