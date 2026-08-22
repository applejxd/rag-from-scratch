# 元教材とのカバー状況

[`langchain-ai/rag-from-scratch`](https://github.com/langchain-ai/rag-from-scratch)
の全18パートに対して、本リポジトリが何をカバーしているかの監査記録。

元ノートは `old/` に未変更のまま保存してある。

## パート別の対応

| パート | 内容 | 元 | 現在 |
| --- | --- | --- | --- |
| 1 | 概要 | コード | 対応 |
| 2 | インデックス作成 | コード | 対応 |
| 3 | 検索 | コード | 対応 |
| 4 | 生成 | コード | 対応 |
| 5 | Multi Query | コード | 対応 |
| 6 | RAG-Fusion | コード | 対応 |
| 7 | 分解 | コード | 対応 |
| 8 | Step Back | コード | 対応 |
| 9 | HyDE | コード | 対応 |
| 10 | 論理・セマンティックルーティング | コード | 対応 |
| 11 | クエリ構造化 | コード | 対応 |
| 12 | 複数表現インデックス | コード | 対応 |
| 13 | RAPTOR | **参照リンクのみ** | 同じく参照リンクのみ |
| 14 | ColBERT | コード | 対応（ライブラリ変更あり） |
| 15 | 再ランキング | コード | 対応（ライブラリ変更あり） |
| 16 | CRAG | **参照リンクのみ** | 同じく参照リンクのみ |
| 17 | Self-RAG | **参照リンクのみ** | 同じく参照リンクのみ |
| 18 | 長いコンテキスト | **参照リンクのみ** | 同じく参照リンクのみ |

パート13、16〜18は元ノートにも実行可能なコードがなく、動画とノートブックへの
リンクだけが置かれている。この点は元と同等である。

## 監査の方法

元ノートの全コードセルから、次を除いた実質70セルを対象に、
識別子が現行実装へ引き継がれているかを機械的に照合した。

- `! pip install ...`
- `os.environ['OPENAI_API_KEY'] = <your-api-key>` などのキー代入
- `os.environ['LANGCHAIN_...']` の LangSmith 設定
- import だけのセル

照合で差分が出たセルは個別に中身を確認した。多くは変数名の変更だけだった。

| 元 | 現在 |
| --- | --- |
| `embd` | `embedding_model` |
| `llm = ChatOpenAI(...)` | `chat_model = make_chat_model()` |
| `prompt_hub_rag` | `reusable_rag_prompt` |
| `retrieve_and_rag` | `retrieve_and_answer_subquestions` |

## 監査で見つかったギャップと対応

どちらも「検索コンポーネントをLangChainチェーンの部品として使う」例が
失われていた。講座全体がRAGチェーンの構築を扱うため、趣旨の一部が欠けていた。

### 1. ColBERT をリトリーバーとして使う例

元ノートの最終セル:

```python
retriever = RAG.as_langchain_retriever(k=3)
retriever.invoke("What animation studio did Miyazaki found?")
```

RAGatouille をやめたことでこの一行が失われ、
`(位置, スコア)` のタプルを返すだけになっていた。

**対応**：`ColBERTRetriever(BaseRetriever)` を追加した。
`_get_relevant_documents()` だけを実装し、`Document` のリストを返す。
RAGチェーンへ差し込んで回答を生成するセルも置いた。

### 2. 再ランカーをチェーンの部品にする例

元ノート:

```python
compression_retriever = ContextualCompressionRetriever(
    base_compressor=CohereRerank(), base_retriever=retriever
)
```

**`ContextualCompressionRetriever` は LangChain 1.x で削除されている。**
`langchain.retrievers` モジュール自体が存在せず、
`langchain_community.retrievers` / `langchain_classic` / `langchain_core.retrievers`
のいずれにも移設されていないことを確認した。

**対応**：`RerankingRetriever(BaseRetriever)` を追加した。
`base_retriever` で候補を取り、`rerank_with_cross_encoder` で絞る。
元と同じく1回の `invoke()` で完了し、チェーンへ差し込める。

この2つは、単に元へ揃えるだけでなく、
**LangChain 1.x で消えたAPIを現代的にどう書き直すか**を示す教材にもなっている。

## 意図的に変えているもの

| 項目 | 元 | 現在 | 記録 |
| --- | --- | --- | --- |
| プロンプト取得 | `hub.pull("rlm/rag-prompt")` | ローカルへ同内容を写す | — |
| ColBERT | RAGatouille | ST v6 + fast-plaid | `docs/colbert-stack.md` |
| 再ランキング | Cohere Rerank API | ローカル CrossEncoder | — |
| トレース | LangSmith | `ConsoleCallbackHandler` | `docs/observability.md` |
| ローカルLLM | — | 未実装（検討のみ） | `docs/local-llm-consideration.md` |

## 最新Python環境としての健全性

- Python 3.12 で動作する
- 主要ライブラリの import 時に DeprecationWarning は出ない
- 元ノートが使っていた `get_relevant_documents()`（現在は非推奨）は
  すべて `invoke()` へ移行済み

## 結果の再現性について

ライブラリとモデルの両方が元と異なるため、**出力を元ノートと直接比較することは
できない**。特にパート14はモデル自体が別物である。
元の内容を忠実に確認したい場合は `old/` を参照する。
