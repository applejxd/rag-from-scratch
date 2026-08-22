# 既知の問題と対処

## marimo importと transformers の相互作用でIPythonが必須になる

### 症状

`sentence-transformers`（`CrossEncoder` など）や `transformers.Trainer` を使うコードを
marimoノートブック内でimportすると、以下のエラーで失敗する。

```text
ModuleNotFoundError: No module named 'IPython'
```

### 原因

`transformers/utils/import_utils.py` の `is_in_notebook()` が、
`"marimo" in sys.modules` を明示的にチェックしてNotebook環境と判定する。

```python
def is_in_notebook() -> bool:
    try:
        # Check if we are running inside Marimo
        if "marimo" in sys.modules:
            return True
        ...
```

`transformers.trainer` はモジュールロード時に `is_in_notebook()` が `True` なら
`from .utils.notebook import NotebookProgressCallback` を実行し、これが
`import IPython.display as disp` を無条件に呼び出す。marimoは元々IPythonに
依存しないため、`ipython` パッケージがインストールされていないと失敗する。

この問題は、marimoノートブック内で `import marimo` した**後に**
`sentence_transformers` や `transformers.Trainer` 系のコードをimportすると
必ず再現する。marimoはファイル冒頭で `import marimo` するため、
import順序を変えても回避できない。

### 対処

`ipython` を直接依存として追加する。コードから直接importしていなくても、
`transformers` の互換性のために必要になる。

```toml
# pyproject.toml
dependencies = [
    # ipython: not imported directly, but required so that `sentence-transformers`
    # (via `transformers`) can import cleanly under marimo. `transformers`
    # detects marimo as a notebook environment and unconditionally imports
    # `IPython.display` at that point.
    "ipython>=9.16.1",
    ...
]
```

依存整理の際に「コードからimportされていない」という理由だけで
`ipython` を削除しないよう注意する（一度削除して再度必要になった実績あり）。

### 参考

- `rag_from_scratch_15_to_18.py` の `rerank_with_cross_encoder()` がこの問題の実例

## Ruff の `--fix` が `import marimo as mo` を削除してしまう

### 症状

ノートブックの Markdown セルが実行時に次のエラーで失敗し、解説がすべて
エラー表示になる。`marimo check --strict` は**実行時エラーを検出しないため素通りする**。

```text
MarimoExceptionRaisedError: name 'mo' is not defined
```

### 原因

marimo のセルには2つの書き方があり、Ruff から見た `mo` の扱いが異なる。

```python
# 形式A: mo をグローバル参照する（Ruff が使用を検出できる）
with app.setup:
    import marimo as mo

@app.cell(hide_code=True)
def _():
    mo.md(r"""...""")


# 形式B: mo を引数で受け取る（Ruff からは import が未使用に見える）
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""...""")
```

形式Bのファイルに対して `ruff check --fix` を実行すると、モジュールレベルの
`import marimo as mo` が F401（unused-import）と判定されて**削除される**。
その結果、`mo` を定義するセルが存在しなくなり、全 Markdown セルが実行時に落ちる。

実際にコミット `2edf5de` の `ruff --fix` で `rag_from_scratch_1_to_4.py` から
`import marimo as mo` が削除され、13個の Markdown セルすべてが壊れた
（`marimo export html` を実行するまで気づけなかった）。

### 対処

セルの書き方を**形式Aに統一する**。`with app.setup:` で `import marimo as mo` し、
セルの署名は `def _():` にする。こうすると `mo` がセル本体からグローバル参照される
ため Ruff が使用を検出でき、`--fix` で削除される回帰も起きない。

```shell
# 形式Bが残っていないか確認する
grep -c 'def _(mo):' rag_from_scratch_*.py
```

### 検知方法

`marimo check --strict` では検出できない。`marimo export html` を実行して
標準エラーに `name 'mo' is not defined` が出ないことを確認するのが確実
（`scripts/build-site.sh` はこの検査を含む）。

## 生成クエリの空行が埋め込みAPIで400エラーになる

### 症状

Multi Query / RAG-Fusion / 質問分解のセルが、実行のたびに成功したり
失敗したりする（フレーク）。失敗時は次のエラーになる。

```text
Error code: 400 - {'error': {'message': '[{ ... "code": "too_small",
  "path": ["input", 0],
  "message": "Too small: expected string to have >=1 characters" }]'}}
```

### 原因

LLMに複数クエリを改行区切りで生成させ、`x.split("\n")` で分割している箇所があった。
モデルが箇条書きを**空行を挟んで**出力すると、分割結果に空文字列が混ざる。
その空文字列がそのまま検索クエリとしてリトリーバーへ渡り、埋め込みAPIが
空入力を拒否して400を返す。

モデル出力に依存するため、同じコードでも実行のたびに結果が変わる。実際、
`rag_from_scratch_15_to_18.py` は同じ脆弱性を持ちながら偶然成功していた。

### 対処

分割時に空行を除去する。共通ヘルパーを `with app.setup:` に置き、
`| (lambda x: x.split("\n"))` の代わりに使う。

```python
def split_queries(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]
```

`rag_from_scratch_5_to_9.py`（3箇所）と `rag_from_scratch_15_to_18.py`（1箇所）へ
適用済み。新しくクエリ生成チェーンを追加する場合も同じヘルパーを使うこと。

## セルを分割するときの marimo の変数ルール

marimo はセルを関数として扱い、依存関係をDAGで解決する。セルを細かく分割する際は
次の3点に注意する。

### 1. 変数はノートブック全体で一意でなければならない

同じ名前を複数のセルで代入すると `multiple-definitions` エラーになる。
同じ処理を別のパートで繰り返す場合は、`overview_chunks` のように用途を表す
接頭辞を付けて衝突を避ける。

```shell
# 衝突は静的検査で検出できる
mise exec -- uv run marimo check --strict rag_from_scratch_*.py
```

### 2. `_` 接頭辞の変数はセルを跨げない

アンダースコアで始まる変数はセルローカルであり、他のセルから参照できない。
1つのセル内で完結していた `_loader` のような変数を持つセルを分割するときは、
通常の名前へ変更して `return` する必要がある。

### 3. セルの最後の式が出力として表示される

中間値を見せたい場合は、代入だけで終わらせず最後に式を置く。
`len(document_chunks)` のように件数を出すだけでも、処理の途中経過を追いやすくなる。
`print()` を使うと複数行の文字列もそのまま確認できる。

```python
@app.cell
def _(blog_documents):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=50
    )
    document_chunks = text_splitter.split_documents(blog_documents)
    len(document_chunks)  # ← セルの出力として表示される
    return (document_chunks,)
```

### クラス定義

セル内で完結しないクラスは、`@app.class_definition` を付けてモジュール直下へ置く
（`RouteQuery` と `TutorialSearch` がこの形式）。

## PyLate と sentence-transformers v6 は共存できない

### 症状

`pylate` を入れたまま `sentence-transformers` を v6 へ上げようとすると、
依存解決に失敗する。

### 原因

PyLate はどのバージョンも `sentence-transformers` を**完全一致で固定**する。

| pylate | 固定先 |
| --- | --- |
| 1.4.0 | `sentence-transformers==5.1.1` |
| 1.6.0 | `sentence-transformers==5.3.0` |

一方 `sentence-transformers` 6.0.0 は `transformers>=5.0.0,<6.0.0` を要求する。
PyLate 側は `transformers<=5.3.0` などの上限も持つため、両立しない。

この固定は ColBERT を使わないパートにも影響する。パート15の `CrossEncoder` も
同じ `sentence-transformers` を使うため、PyLate を入れている限りそちらも
古いバージョンに縛られる。

### 対処

PyLate をやめ、`sentence-transformers` v6 の `MultiVectorEncoder` と
`fast-plaid` を直接使う。PyLate は元から `fast-plaid` のラッパーだったため、
インデックスの実装は変わらない。

判断の経緯と実測値は `docs/colbert-stack.md` に記録している。

### 確認方法

```shell
# 依存の固定先を確認する
mise exec -- uv run python -c "
import importlib.metadata as m
print(m.requires('pylate'))
"
```

## LangChain 1.x で削除されたAPI

元ノート（LangChain 0.1 系）が使っていたAPIのうち、現行の 1.x で
利用できなくなったもの。

### `ContextualCompressionRetriever`

```python
# 元ノート。現在は動かない
from langchain.retrievers import ContextualCompressionRetriever
```

`langchain.retrievers` モジュール自体が存在しない。
`langchain_community.retrievers` / `langchain_classic` /
`langchain_core.retrievers` のいずれにも移設されていないことを確認済み。

**対処**：`BaseRetriever` を継承して自分で書く。実装が必要なのは
`_get_relevant_documents()` の1メソッドだけ。
`rag_from_scratch_15_to_18.py` の `RerankingRetriever` がその例。

### `get_relevant_documents()`

非推奨。`invoke()` を使う。本リポジトリでは移行済み。

### 確認方法

```shell
mise exec -- uv run python -c "
from langchain_core.retrievers import BaseRetriever
print(BaseRetriever.__abstractmethods__)
"
```

### 補足：`BaseRetriever` は Pydantic モデル

モデルやインデックスのような任意のオブジェクトをフィールドに持つ場合は
`model_config = ConfigDict(arbitrary_types_allowed=True)` が必要。
素の dict で書くと Ruff の RUF012 に引っかかるため `ConfigDict` を使う。

## エディタ上では出るが実際には問題ない診断

### 症状

VS Code などで marimo notebook を開くと、実際には存在しない警告・エラーが出る。

| コード | 内容 | 件数の例 |
| --- | --- | --- |
| `F401` | `marimo` imported but unused | 25 |
| `F821` / `unresolved-reference` | 未定義の名前 | 各14 |
| `I001` | import が未整列 | 3 |

CLI では同じファイルが問題なく通る。

```shell
mise exec -- uvx ruff@0.16.3 check rag_from_scratch_*.py   # All checks passed!
mise exec -- uv run marimo check --strict rag_from_scratch_*.py  # 出力なし
```

### 原因

エディタは marimo notebook を `vscode-notebook-cell://` スキームで扱い、
**セルを1つずつ独立したファイルとして** linter へ渡す。

marimo のセルは実際にはファイル全体で1つの名前空間を共有し、
`with app.setup:` で定義した名前を各セルが参照する。
セル単体で見ると、その import も定義も見えないため、
「未使用の import」「未定義の名前」に見えてしまう。

### 対処

**エディタの表示ではなく CLI の結果を信頼する。** pre-commit も CLI を使う。

```shell
mise exec -- uv run pre-commit run --all-files
```

セルを跨いだ参照は marimo の正常な仕様であり、
`marimo check --strict` が本当の依存関係を検証している。
