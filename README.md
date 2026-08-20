# RAG From Scratch

[`langchain-ai/rag-from-scratch`](https://github.com/langchain-ai/rag-from-scratch)
の Jupyter notebooks を、`uv` で依存関係を管理する
[marimo](https://marimo.io/) notebooks に移行した学習用リポジトリです。

LLMs are trained on a large but fixed corpus of data, limiting their ability to
reason about private or recent information. Retrieval augmented generation
(RAG) expands an LLM's knowledge base by grounding generation in retrieved
documents.

These notebooks accompany a
[video playlist](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x)
that builds up an understanding of RAG from scratch, starting with indexing,
retrieval, and generation.

![rag_detail_v2](https://github.com/langchain-ai/rag-from-scratch/assets/122662504/54a2d76c-b07e-49e7-b4ce-fc45667360a1)

## Setup

Python 3.12 and `uv` are required.

```shell
cp .env.example .env
uv sync --locked
```

Set `OPENROUTER_API_KEY` in `.env`. Part 15 also uses `COHERE_API_KEY` for
reranking. When using `mise`, the model defaults in `mise.toml` and variables
from `.env` are loaded automatically.

## Run

Open a notebook in edit mode:

```shell
uv run marimo edit rag_from_scratch_1_to_4.py
```

The notebooks are split into the following files:

| Parts | File |
| --- | --- |
| 1-4 | `rag_from_scratch_1_to_4.py` |
| 5-9 | `rag_from_scratch_5_to_9.py` |
| 10-11 | `rag_from_scratch_10_and_11.py` |
| 12-14 | `rag_from_scratch_12_to_14.py` |
| 15-18 | `rag_from_scratch_15_to_18.py` |

The original Jupyter notebooks are retained in `old/` as migration references.

## Development

Install the Git hook after `uv sync`:

```shell
uv run pre-commit install
```

The hook runs Ruff, `marimo check --strict`, and Gitleaks. Run all checks
manually with:

```shell
uv run pre-commit run --all-files
```