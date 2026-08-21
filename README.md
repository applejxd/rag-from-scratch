# RAG From Scratch

[`langchain-ai/rag-from-scratch`](https://github.com/langchain-ai/rag-from-scratch)
の Jupyter notebooks を、`uv` で依存関係を管理する
[marimo](https://marimo.io/) notebooks に移行した学習用リポジトリです。

以下は元リポジトリのREADMEからの引用（日本語訳）です。

> LLMは大規模ではあるものの固定されたコーパスで学習されているため、非公開情報や最新情報についての推論を苦手とします。ファインチューニングはこれを緩和する一つの方法ですが、[事実の想起には不向きであることが多く](https://www.anyscale.com/blog/fine-tuning-is-for-form-not-facts)、[コストもかかります](https://www.glean.com/blog/how-to-build-an-ai-assistant-for-the-enterprise)。
>
> Retrieval Augmented Generation（RAG）は、外部データソースから検索した文書を用いてin-context learningを通じLLMの生成を根拠付けることで、LLMの知識ベースを拡張する、人気があり強力な手法として登場しました。
>
> これらのノートブックは、インデックス作成・検索・生成の基礎から始めてRAGをゼロから理解できるように構成された[動画プレイリスト](https://youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&feature=shared)に付随するものです。
>
> —— [langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch) の README より翻訳・引用

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

`OPENROUTER_API_KEY` が必須です。パート15の再ランキングは `sentence-transformers`
のCrossEncoderをローカルで実行するため、追加のAPIキーは不要です。SOPSは復号した
内容を一時的なエディタバッファで開き、保存時に再度暗号化します。

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

既知の問題や環境固有の注意点は `docs/` にまとめています。

- [`docs/known-issues.md`](docs/known-issues.md)：marimoとtransformers/IPythonの相互作用など
- [`docs/environment-notes.md`](docs/environment-notes.md)：GPU検出・herdr環境に関する注意点
- [`docs/secrets-management.md`](docs/secrets-management.md)：SOPSの運用コマンドと注意点
