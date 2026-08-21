# RAG From Scratch

[`langchain-ai/rag-from-scratch`](https://github.com/langchain-ai/rag-from-scratch)
の Jupyter notebooks を、`uv` で依存関係を管理する
[marimo](https://marimo.io/) notebooks に移行した学習用リポジトリです。

LLMは大規模ではあるものの固定されたコーパスで学習されているため、非公開情報や最新情報についての推論を苦手とします。Retrieval Augmented Generation（RAG）は、検索した文書に生成内容を根拠付けることで、LLMの知識ベースを拡張する手法です。

これらのノートブックは、インデックス作成・検索・生成という基礎から段階的にRAGへの理解を深める
[動画プレイリスト](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x)
に沿った教材です。

![rag_detail_v2](https://github.com/langchain-ai/rag-from-scratch/assets/122662504/54a2d76c-b07e-49e7-b4ce-fc45667360a1)

## セットアップ

Python 3.12 と [mise](https://mise.jdx.dev/) が必要です。SOPSはageを使って
`.env.json` を復号します。`.sops.yaml` に記載されたrecipientに対応する秘密鍵を
`~/.config/sops/age/keys.txt` に配置してください。秘密鍵を絶対にコミットしないでください。

```shell
mise trust
mise install
mise exec -- uv sync --locked
```

`mise` が `uv` と SOPS をインストールし、`.env.json` を復号して平文ファイルを
作成せずに環境変数として値を渡します。暗号化されたファイルを編集するには
以下を実行してください。

```shell
mise exec -- sops .env.json
```

`OPENROUTER_API_KEY` が必須です。パート15では再ランキングのために
`COHERE_API_KEY` も使用します。SOPSは復号した内容を一時的なエディタバッファで
開き、保存時に再度暗号化します。

## 実行方法

ノートブックを編集モードで開きます。

```shell
mise exec -- uv run marimo edit rag_from_scratch_1_to_4.py
```

ノートブックは以下のファイルに分割されています。

| パート | ファイル |
| --- | --- |
| 1-4 | `rag_from_scratch_1_to_4.py` |
| 5-9 | `rag_from_scratch_5_to_9.py` |
| 10-11 | `rag_from_scratch_10_and_11.py` |
| 12-14 | `rag_from_scratch_12_to_14.py` |
| 15-18 | `rag_from_scratch_15_to_18.py` |

元のJupyterノートブックは、移行時の参考資料として `old/` に保持しています。

## 開発

`uv sync` の後にGit hookをインストールしてください。

```shell
mise exec -- uv run pre-commit install
```

このhookは `.env.json` がSOPSで暗号化されていることを確認したうえで、Ruff、
`marimo check --strict`、Gitleaksを実行します。すべてのチェックを手動で
実行するには以下を使用してください。

```shell
mise exec -- uv run pre-commit run --all-files
```
