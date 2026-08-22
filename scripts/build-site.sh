#!/usr/bin/env bash
#
# Export the marimo notebooks to static HTML under site/ for GitHub Pages.
#
# `marimo export html` executes each notebook, so this requires
# OPENROUTER_API_KEY (via SOPS/mise) and incurs API charges. Run it through
# `mise exec --` so the decrypted environment variables are available.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

site_dir="site"
log_dir="$(mktemp -d)"
trap 'rm -rf "$log_dir"' EXIT

notebooks=(
    rag_from_scratch_1_to_4.py
    rag_from_scratch_5_to_9.py
    rag_from_scratch_10_and_11.py
    rag_from_scratch_12_to_14.py
    rag_from_scratch_15_to_18.py
)

titles=(
    "パート1〜4：概要・インデックス作成・検索・生成"
    "パート5〜9：クエリ変換"
    "パート10〜11：ルーティングとクエリ構築"
    "パート12〜14：インデックス作成の応用"
    "パート15〜18：検索と生成の応用"
)

mkdir -p "$site_dir"
# Disable Jekyll so GitHub Pages serves the exported assets verbatim.
touch "$site_dir/.nojekyll"

failed=0

for i in "${!notebooks[@]}"; do
    notebook="${notebooks[$i]}"
    output="$site_dir/${notebook%.py}.html"
    log="$log_dir/${notebook%.py}.log"

    echo "==> exporting $notebook"

    # Do not pass --sandbox: the notebooks no longer carry PEP 723 inline
    # dependencies, so an isolated environment cannot resolve them.
    if ! uv run marimo export html "$notebook" -o "$output" --force >"$log" 2>&1; then
        echo "    FAILED: marimo export returned a non-zero status" >&2
        sed 's/^/    /' "$log" >&2
        failed=1
        continue
    fi

    # `marimo export html` exits 0 even when cells raise, so inspect the log.
    if grep -q "some cells failed to execute" "$log"; then
        echo "    FAILED: some cells raised during execution" >&2
        grep -E "MarimoExceptionRaisedError|Error:" "$log" | sort -u | sed 's/^/    /' >&2
        failed=1
        continue
    fi

    echo "    wrote $output"
done

if [[ $failed -ne 0 ]]; then
    echo "Export failed. The generated HTML would contain error output." >&2
    exit 1
fi

# The exported HTML references images as relative `imgs/...` paths.
# site/imgs/ is gitignored; the Pages workflow recreates it during deploy.
rm -rf "${site_dir:?}/imgs"
cp -r imgs "$site_dir/imgs"

{
    cat <<'HEADER'
<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG From Scratch (marimo notebooks)</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 44rem; margin: 3rem auto; padding: 0 1rem; line-height: 1.7; }
li { margin-bottom: 0.5rem; }
</style>
</head>
<body>
<h1>RAG From Scratch</h1>
<p>
<a href="https://github.com/langchain-ai/rag-from-scratch">langchain-ai/rag-from-scratch</a>
の Jupyter notebooks を <a href="https://marimo.io/">marimo</a> へ移行し、日本語化した学習用教材です。
</p>
<ul>
HEADER

    for i in "${!notebooks[@]}"; do
        notebook="${notebooks[$i]}"
        printf '<li><a href="%s.html">%s</a></li>\n' "${notebook%.py}" "${titles[$i]}"
    done

    cat <<'FOOTER'
</ul>
<p><a href="https://github.com/applejxd/rag-from-scratch">リポジトリ</a></p>
</body>
</html>
FOOTER
} >"$site_dir/index.html"

echo "==> wrote $site_dir/index.html"
echo "Done. Preview with: python -m http.server --directory $site_dir"
