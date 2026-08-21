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
