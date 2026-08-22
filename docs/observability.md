# 可観測性（LangSmith相当）の検討

元ノートは5ファイルすべてで LangSmith のトレースを有効化していた。

```python
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = <your-api-key>
```

本リポジトリではこれを削除している。代替を導入すべきか検討した記録。

## 結論：常設の導入はしない

必要なときだけ LangChain 標準の `ConsoleCallbackHandler` を使う。
追加の依存もアカウントも不要で、すでに利用可能なことを確認済み。

## なぜ常設しないか

### 1. marimo のセル出力が主要な役割をすでに担っている

LangSmith の教材上の価値は「チェーンの途中で何が起きたか」を見ることだった。
本リポジトリはセルを処理単位へ分割し、中間値を直接表示している。

- 生成された検索クエリ
- RRF の融合スコア
- 蓄積される質問と回答の組
- ColBERT のトークン形状と MaxSim スコア
- 再ランキング前後の関連度スコア

元ノートがトレースへのリンクで補っていた部分は、ノートブック上で見えている。

### 2. 外部サービス依存を減らす方針と衝突する

このリポジトリは、LangChain Hub（プロンプト取得）と Cohere（再ランキング）への
実行時依存をすでに外している。トレース基盤を常設すると、その方針に逆行する。

- LangSmith：アカウントと API キーが必要。SOPS への鍵追加も要る
- Langfuse / Phoenix：セルフホストなら API キー不要だが、
  Docker などの常駐プロセスが前提になる

学習用の単一リポジトリに対して、設定負担が見合わない。

### 3. 公開サイトへ影響する

`scripts/build-site.sh` は全ノートブックを実行して HTML を生成する。
トレース基盤が常設されていると、ビルド時にも接続先が必要になり、
CI やクリーンな環境での再現性が下がる。

## 必要になったときの手段

### 軽量：LangChain 標準のコールバック（推奨）

依存追加なしで、実際に送られたプロンプトと返答を確認できる。

```python
from langchain_core.tracers import ConsoleCallbackHandler

rag_chain.invoke(
    "What is Task Decomposition?",
    config={"callbacks": [ConsoleCallbackHandler()]},
)
```

チェーン全体を一時的に詳細表示したい場合は次も使える。

```python
from langchain_core.globals import set_debug

set_debug(True)
```

marimo のセル出力にそのまま表示されるため、追加の画面を開く必要がない。

### 本格的に必要な場合：Langfuse

トレースの永続化、実行間の比較、コストやレイテンシの集計が要るなら
Langfuse（MIT、セルフホスト可、API キー不要）が候補になる。
LangChain のコールバックとして接続できる。

導入する場合は次を伴う。

- Docker などで Langfuse を起動する手順を README へ追加
- 各ノートブックの `make_chat_model()` へコールバックを注入
- `scripts/build-site.sh` 実行時に接続先が無くても失敗しないフォールバック

現時点ではこの負担に見合う要件がないため見送る。

## 判断を見直す条件

- 実行ごとの回答品質を比較・評価したくなったとき
- トークン使用量やコストを継続的に追いたくなったとき
- ノートブックを超えたアプリケーションへ発展させるとき
