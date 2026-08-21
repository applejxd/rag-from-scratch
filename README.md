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

Python 3.12 and [mise](https://mise.jdx.dev/) are required. SOPS uses age to
decrypt `.env.json`; place the private key corresponding to the recipient in
`.sops.yaml` at `~/.config/sops/age/keys.txt`. Never commit the private key.

```shell
mise trust
mise install
mise exec -- uv sync --locked
```

`mise` installs `uv` and SOPS, decrypts `.env.json`, and supplies its values as
environment variables without creating a plaintext file. Edit the encrypted
file with:

```shell
mise exec -- sops .env.json
```

`OPENROUTER_API_KEY` is required. Part 15 also uses `COHERE_API_KEY` for
reranking. SOPS opens the decrypted content in a temporary editor buffer and
encrypts it again when saving.

## Run

Open a notebook in edit mode:

```shell
mise exec -- uv run marimo edit rag_from_scratch_1_to_4.py
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
mise exec -- uv run pre-commit install
```

The hook verifies that `.env.json` is SOPS-encrypted and runs Ruff,
`marimo check --strict`, and Gitleaks. Run all checks manually with:

```shell
mise exec -- uv run pre-commit run --all-files
```