# OpenRouter依存をローカルLLMへ置き換える検討（未実装）

このドキュメントは、`OPENROUTER_API_KEY` を使っている箇所を8GB VRAM環境で
動くローカルLLMへ置き換えられるかを検討した記録である。**まだ実装していない。**
実装する場合は、末尾の「決めるべきこと」を先にユーザーと確認すること。

## 結論（暫定）

技術的には同一シナリオを達成できる可能性が高いが、パート10・11の構造化出力
（Function Calling）だけ精度が落ちるリスクがある。

## 現状の依存箇所（2026-08-22時点で確認済み）

- `make_chat_model()` / `make_embeddings()`：全5ノートブック共通のヘルパー関数。
  `langchain_openai` の `ChatOpenAI` / `OpenAIEmbeddings` を、
  `OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"` へ向けて初期化している。
- パート10（`rag_from_scratch_10_and_11.py`）：`RouteQuery`
  （`Literal["python_docs", "js_docs", "golang_docs"]` の単一フィールド）への
  `with_structured_output()`。
- パート11（同ファイル）：`TutorialSearch`
  （7フィールド、`datetime.date | None` を含む複雑なスキーマ）への
  `with_structured_output()`。
- パート12〜14（`rag_from_scratch_12_to_14.py`）・パート15
  （`rag_from_scratch_15_to_18.py`）は、PyLate（ColBERT）とCrossEncoderで
  既にローカル実行のため対象外。

## 実現方式の候補

Ollama はOpenAI互換の `/v1` エンドポイントを持つため、`ChatOpenAI` /
`OpenAIEmbeddings` の `base_url` とモデル名を変えるだけで既存コード構造を
維持できる。`mise` のツールレジストリにも `ollama`
（`aqua:ollama/ollama` / `asdf:virtualstaticvoid/asdf-ollama`）があり、
`sops` / `uv` と同様にバージョン固定して `mise.toml` に追加できる。

代替として `llama-cpp-python` も検討したが、Ollamaより軽量な反面、
Python プロセス内へモデルを埋め込む形になり、`make_chat_model()` /
`make_embeddings()` の抽象化が崩れやすい。

## VRAM見積もり（8GB中、実測未実施）

- チャットモデル（Qwen2.5-7B-Instruct Q4量子化を想定）：約4.7GB
- 埋め込みモデル（nomic-embed-text等）：数百MB、CPU実行も可能
- 単一ノートブック内では十分な余裕がある見込み（他ノートブックのColBERT/
  CrossEncoderは独立プロセスのため通常は同時実行しない）

**注意：この数値はモデルカード等からの推定であり、`docs/environment-notes.md`
に記載した実測方式（`torch.cuda.memory_allocated()` 等）でこの環境において
実測していない。実装時は必ず実測して確認すること。**

## リスク・懸念事項

- パート10（単純な分類、Literalが3値）はローカル7〜8Bモデルでもほぼ問題ない見込み。
- パート11（複雑なスキーマ、日付フィールドを含む）は、7〜8Bクラスの量子化
  モデルだと `gpt-5-nano` / `gpt-4o-mini` より構造化出力の精度が下がる
  可能性がある（動作はするが結果の質が変わりうる）。
- パート1のトークン数カウント例（`tiktoken` の `cl100k_base`）はOpenAI固有の
  トークナイザ解説であり、生成モデルを変更した場合は「実際に生成へ使う
  モデルのトークナイザとは異なる」旨の注記が必要になる。

## 決めるべきこと（実装前にユーザーへ確認）

1. **モデルサイズ方針**：7〜8B量子化モデル（精度優先）か、3Bクラスの小型
   モデル（余裕優先、構造化出力の精度はさらに下がる）か。
2. **適用範囲**：OpenRouterを完全に置き換えてAPIキー不要にするか、
   環境変数で切り替えられる選択肢として追加するか（OpenRouterはデフォルトの
   まま残す）。
3. **パート11の精度リスクの許容**：構造化出力の精度低下リスクを受け入れて
   進めるか。
4. **ローカルLLM提供ツール**：Ollama（推奨、OpenAI互換エンドポイント、
   `mise`でバージョン固定可能）か、`llama-cpp-python`（より軽量、Pythonプロセス
   内埋め込み）か。

## 参考

- `docs/environment-notes.md`：GPUメモリの実測方法
- `docs/known-issues.md`：marimo環境での既知の相互作用
