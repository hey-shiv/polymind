# PolyMind

PolyMind is a local LLM project built with PyTorch. It combines a compact educational transformer with supporting experiments for retrieval, interfaces, and local model workflows so the codebase can grow into a broader multi-model system over time.

The project is designed to be easy to read while still using several ideas from modern transformer architectures:

- `RMSNorm` instead of standard LayerNorm
- `RoPE` (Rotary Positional Embeddings) instead of learned positional embeddings
- `Grouped Query Attention (GQA)` to reduce KV head cost
- `SwiGLU` as the feed-forward activation block

## Architecture

The core model lives in [`src/model.py`](src/model.py) and is implemented as a pre-norm transformer.

### Model Flow

1. BPE token ids are mapped to embeddings.
2. Tokens pass through a stack of transformer blocks.
3. Each block applies:
   - `RMSNorm`
   - `GroupedQueryAttention` with `RoPE`
   - residual connection
   - `RMSNorm`
   - `SwiGLU`
   - residual connection
4. A final `RMSNorm` is applied.
5. A tied output projection predicts the next token.

### Modern Components

- `RMSNorm`
  Normalizes activations using root-mean-square statistics with a learned scale parameter.
- `RoPE`
  Encodes token position by rotating query and key vectors, allowing attention to capture relative positions naturally.
- `Grouped Query Attention`
  Uses more query heads than key/value heads, which keeps attention expressive while reducing KV projection cost.
- `SwiGLU`
  Uses a SiLU-gated feed-forward pathway that is commonly used in modern LLMs.

### Default Training Configuration

The current training script uses:

- `d_model = 256`
- `n_layers = 4`
- `n_heads = 8`
- `n_kv_heads = 2`
- `ffn_hidden_dim = 680`
- `context_length = 256`
- `batch_size = 64`

## Dataset

The project trains on the TinyStories dataset from Hugging Face.

- Source: `roneneldan/TinyStories`
- Raw text file: `data/dataset.txt`
- Tokenization: byte-level BPE via `tokenizer.json`
- Split: 90% train / 10% validation over the token stream

The dataset is downloaded automatically the first time you run `download_dataset.py`, `train_tokenizer.py`, or `train.py`.

## Installation

Create a virtual environment if you want, then install dependencies:

```bash
pip install -r requirements.txt
```

For Google Colab, install only the missing libraries so you keep Colab's GPU-enabled PyTorch:

```bash
pip install tokenizers datasets numpy
```

## How To Train

Run the full pipeline:

```bash
python download_dataset.py
```

```bash
python train_tokenizer.py
```

```bash
python train.py
```

What the training script does:

- ensures `data/dataset.txt` exists
- loads `tokenizer.json`
- converts TinyStories into BPE token ids
- initializes `MiniLLM`
- trains with `AdamW`
- prints training loss every 100 steps
- saves a checkpoint to `model.pt`

The saved checkpoint includes:

- model weights
- model config
- tokenizer metadata

## Generated Files

Keep generated training artifacts such as `data/dataset.txt`, token caches, or `model.pt` as local working files instead of treating them as project source files.

## How To Generate Text

After training, generate text with:

```bash
python generate.py --prompt "Once upon a time"
```

You can also control sampling temperature and output length:

```bash
python generate.py --prompt "Once upon a time" --temperature 0.8 --max-new-tokens 300
```

Useful flags:

- `--prompt` starting text for generation
- `--temperature` sampling temperature, where `0` is greedy decoding
- `--max-new-tokens` number of tokens to generate
- `--model-path` optional path to a checkpoint
- `--seed` optional random seed for reproducible sampling

## Project Structure

```text
.
├── src/
│   ├── model.py
│   ├── tokenizer.py
│   └── dataset.py
├── notebooks/
│   └── building-modern-llm.ipynb
├── train.py
├── generate.py
└── requirements.txt
```

## Summary

PolyMind remains a compact, readable LLM training project at its core, and it is set up to support the wider local AI tooling you are building around it.
