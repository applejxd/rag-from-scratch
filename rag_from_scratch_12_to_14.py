import marimo

__generated_with = "0.20.4"
app = marimo.App()

with app.setup:
    import os
    import uuid

    import marimo as mo
    import torch
    from fast_plaid.search import FastPlaid
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pydantic import ConfigDict
    from sentence_transformers import MultiVectorEncoder

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    COLBERT_INDEX_DIR = ".colbert-index"
    DEFAULT_CHAT_MODEL = os.environ.get("MODEL", "openai/gpt-5-nano")
    DEFAULT_EMBEDDING_MODEL = os.environ.get(
        "EMBEDDING_MODEL", "openai/text-embedding-3-small"
    )
    DEFAULT_CHAT_TIKTOKEN_MODEL = os.environ.get("TIKTOKEN_CHAT_MODEL", "gpt-4o-mini")
    DEFAULT_EMBEDDING_TIKTOKEN_MODEL = os.environ.get(
        "TIKTOKEN_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    def colbert_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

    def format_colbert_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def load_rag_prompt():
        """Local copy of the LangChain Hub prompt `rlm/rag-prompt`.

        The original notebooks pulled this at runtime with
        `hub.pull("rlm/rag-prompt")`. It is reproduced here so the notebook
        does not depend on the Hub being reachable, and so the exact wording
        is visible. See https://smith.langchain.com/hub/rlm/rag-prompt
        """
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an assistant for question-answering tasks. "
                        "Use the following pieces of retrieved context to "
                        "answer the question. If you don't know the answer, "
                        "just say that you don't know. Use three sentences "
                        "maximum and keep the answer concise."
                    ),
                ),
                ("human", "Question: {question}\nContext: {context}\nAnswer:"),
            ]
        )

    def make_chat_model():
        return ChatOpenAI(
            model_name=DEFAULT_CHAT_MODEL,
            openai_api_base=OPENROUTER_BASE_URL,
            openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
            temperature=0,
            tiktoken_model_name=DEFAULT_CHAT_TIKTOKEN_MODEL,
            max_retries=0,
        )

    def make_embeddings():
        return OpenAIEmbeddings(
            model=DEFAULT_EMBEDDING_MODEL,
            openai_api_base=OPENROUTER_BASE_URL,
            openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
            tiktoken_model_name=DEFAULT_EMBEDDING_TIKTOKEN_MODEL,
            check_embedding_ctx_length=False,
            tiktoken_enabled=False,
        )

    class SimpleMultiVectorRetriever:
        def __init__(self, vectorstore, id_key: str):
            self.vectorstore = vectorstore
            self.id_key = id_key
            self._docstore = {}

        def set_parent_documents(self, docs_by_id):
            self._docstore = dict(docs_by_id)

        def invoke(self, query: str, k: int = 4):
            child_docs = self.vectorstore.similarity_search(query, k=k)
            parent_docs = []
            seen_ids = set()
            for child_doc in child_docs:
                doc_id = child_doc.metadata.get(self.id_key)
                if doc_id in seen_ids:
                    continue
                parent_doc = self._docstore.get(doc_id)
                if parent_doc is None:
                    continue
                seen_ids.add(doc_id)
                parent_docs.append(parent_doc)
            return parent_docs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # RAGをゼロから学ぶ：インデックス作成

    ![インデックス作成の概要](./imgs/indexing_overview.png)

    ## はじめに：チャンキング

    このノートブックでは、ドキュメントのチャンキング／分割そのものは詳しく扱いません。

    ドキュメントのチャンキングを詳しく学ぶには、Greg Kamradtによる次の動画がおすすめです。

    https://www.youtube.com/watch?v=8OJC21T2SL4
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## このノートブックの共通部品

    冒頭の `with app.setup` ブロック（marimo上では折りたたまれています）で、以下の共通部品を定義しています。パート12では要約を使う複数表現の検索、パート14ではColBERTの検索を実装するため、通常のRAG用ヘルパーに加えてColBERT用の整形関数や自作リトリーバーも用意しています。

    - `make_chat_model()`：回答生成用のチャットモデル。OpenRouter経由で `openai/gpt-5-nano`（環境変数 `MODEL` で変更可）を呼びます。出力を安定させるため `temperature=0` を指定し、このノートでは失敗をすぐ確認できるよう `max_retries=0` にしています。
    - `make_embeddings()`：埋め込みモデル。既定は `openai/text-embedding-3-small` です。OpenRouterはOpenAI互換ですがトークナイザ情報を返さないため、クライアント側のトークン数チェックを無効化（`check_embedding_ctx_length=False` / `tiktoken_enabled=False`）しています。
    - `load_rag_prompt()`：LangChain Hubの `rlm/rag-prompt` と同じ内容のプロンプト。元ノートは実行時に `hub.pull()` で取得していましたが、外部サービスへの依存を避け、かつ文面をその場で読めるようローカルへ写しています。
    - `format_colbert_docs(docs)`：`Document` のリストを、本文を空行2つでつないだ1つの文字列へ変換します。ColBERT検索結果をプロンプトの `{context}` へ埋め込むために使います。
    - `colbert_device()`：CUDAが使える環境では `cuda`、使えない場合は `cpu` を返します。ColBERTモデルとPLAIDインデックスの実行デバイス指定に使います。
    - `COLBERT_INDEX_DIR = ".colbert-index"`：パート14で作るPLAIDインデックスの保存先ディレクトリです。ChromaのインメモリDBと違い、ディスク上に作成されます。

    `SimpleMultiVectorRetriever` はこのノート固有の簡易リトリーバーです。要約だけを検索対象にしつつ、回答生成には対応する元文書を返すために使います。

    - 入力：要約を入れたベクトルストアと、要約メタデータ内で元文書IDを表すキー名（ここでは `doc_id`）
    - 保持する状態：要約用ベクトルストアと、`doc_id` から元文書 `Document` を引くメモリ上の辞書 `_docstore`
    - 登録：`set_parent_documents()` で `doc_id` と元文書の対応表を `_docstore` に保存します
    - 検索：`invoke(query, k=4)` は要約を類似度検索し、ヒットした要約のメタデータから `doc_id` を取り出し、重複を除きながら対応する元文書を返します
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## パート12：複数表現によるインデックス作成

    検索に適した短い要約をベクトル化し、ヒットした要約から情報量の多い元文書を返します。検索用表現と回答用文書を分けることで、検索精度と回答に必要な文脈量を両立します。

    処理の流れ：

     ![複数表現によるインデックス作成の流れ](./imgs/multi-representation_indexing.png)

    ドキュメント：

    https://blog.langchain.dev/semi-structured-multi-modal-rag/

    https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector

    論文：

    https://arxiv.org/abs/2312.06648

    次のセルでは、Lilian Wengのブログ記事2本を読み込みます。`WebBaseLoader` に `bs_kwargs` を渡していないため、他パートのように `SoupStrainer` で本文部分だけに絞らず、ページ全体を読み込みます。

    - データ：[LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) と [Thinking about High-Quality Human Data](https://lilianweng.github.io/posts/2024-02-05-human-data-quality/)
    - 前処理：HTMLの絞り込みなし。`WebBaseLoader.load()` が返す各ページの `Document` をそのまま使います
    - 出力：`source_documents` = 読み込んだ `Document` リスト（表示される件数は2件）
    """)
    return


@app.cell
def _():
    agent_article_loader = WebBaseLoader(
        "https://lilianweng.github.io/posts/2023-06-23-agent/"
    )
    source_documents = agent_article_loader.load()

    data_quality_loader = WebBaseLoader(
        "https://lilianweng.github.io/posts/2024-02-05-human-data-quality/"
    )
    source_documents.extend(data_quality_loader.load())
    len(source_documents)
    return (source_documents,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    各文書の全文をLLMへ渡し、検索用の短い要約を作ります。`batch()` は複数の `Document` をまとめてチェーンへ流すための実行方法で、`max_concurrency=5` により同時リクエスト数を最大5件に制限しています。出力は元文書と同じ順序の要約文字列リストです。

    - 入力：`source_documents`（各 `Document` の `page_content` 全文）
    - 処理：`Summarize the following document:` というプロンプト、チャットモデル、文字列出力パーサーをLCELで接続
    - 出力：`document_summaries` = 各文書の要約文字列のリスト
    """)
    return


@app.cell
def _(source_documents):
    summarization_chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
        | make_chat_model()
        | StrOutputParser()
    )

    document_summaries = summarization_chain.batch(
        source_documents, {"max_concurrency": 5}
    )
    return (document_summaries,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    元文書と要約の文字数を比べます。検索対象を短い要約にすることで、埋め込みが記事全体の話題に薄められるのを防げます。
    """)
    return


@app.cell
def _(document_summaries, source_documents):
    [
        (len(doc.page_content), len(summary))
        for doc, summary in zip(source_documents, document_summaries)
    ]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 要約と元文書の対応付け

    各要約には元文書と共通の `doc_id` を付けます。文書1件ごとにUUIDを1つ生成し、要約側のメタデータと元文書側の対応表を結びつける鍵として使います。検索対象は要約ですが、ヒット後は `doc_id` を使って元文書を返すため、回答生成では省略前の内容を利用できます。

    - 入力：`source_documents`
    - 処理：`id_key = "doc_id"` をキー名にし、`uuid.uuid4()` で各文書のIDを生成
    - 出力：`source_document_ids` = 元文書と同じ順序のUUID文字列リスト
    """)
    return


@app.cell
def _(source_documents):
    id_key = "doc_id"
    source_document_ids = [str(uuid.uuid4()) for _ in source_documents]
    return id_key, source_document_ids


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    要約を `Document` へ包み、メタデータへ対応する `doc_id` を持たせます。
    """)
    return


@app.cell
def _(document_summaries, id_key, source_document_ids):
    summary_documents = [
        Document(
            page_content=summary,
            metadata={id_key: source_document_ids[index]},
        )
        for index, summary in enumerate(document_summaries)
    ]
    summary_documents[0].metadata
    return (summary_documents,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    要約をベクトルストアへ登録し、リトリーバーには `doc_id` から元文書を引くための対応表を渡します。要約だけをベクトル化して検索し、元文書は `SimpleMultiVectorRetriever` のメモリ上の辞書に置く二層構成です。

    - データ：`summary_documents`（2本の記事の要約。メタデータに `doc_id` を保持）
    - 前処理：追加の分割なし。要約文字列をそのまま埋め込み対象にします
    - DB：Chroma、`collection_name="summaries"`。`persist_directory` を指定していないため**インメモリ**です
    - 出力：`multi_vector_retriever` = 要約検索から元文書を返す `SimpleMultiVectorRetriever`
    """)
    return


@app.cell
def _(id_key, source_document_ids, source_documents, summary_documents):
    summary_vectorstore = Chroma(
        collection_name="summaries",
        embedding_function=make_embeddings(),
    )
    summary_vectorstore.add_documents(summary_documents)

    multi_vector_retriever = SimpleMultiVectorRetriever(
        vectorstore=summary_vectorstore, id_key=id_key
    )
    multi_vector_retriever.set_parent_documents(
        zip(source_document_ids, source_documents)
    )
    return multi_vector_retriever, summary_vectorstore


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    まずベクトルストアを直接検索し、ヒットした「要約」を確認します。
    """)
    return


@app.cell
def _(summary_vectorstore):
    summary_query = "Memory in agents"
    matching_summaries = summary_vectorstore.similarity_search(summary_query, k=1)
    matching_summaries[0]
    return (summary_query,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    同じ質問をリトリーバー経由で検索すると、要約ではなく対応する「元文書」が返ります。
    """)
    return


@app.cell
def _(multi_vector_retriever, summary_query):
    retrieved_parent_documents = multi_vector_retriever.invoke(summary_query)
    retrieved_parent_documents[0].page_content[0:500]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `SimpleMultiVectorRetriever` は、要約に付与した `doc_id` から元文書を復元します。関連する標準的な実装として、[親ドキュメントリトリーバー](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever)があります。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## パート13：RAPTOR

    RAPTORは文書を階層的にクラスタリング・要約し、細部のチャンクと上位の要約を同じ検索対象にする手法です。このノートブックでは実装を実行せず、概念図と参考資料のみを示します。

    処理の流れ：

    ![RAPTORの流れ](./imgs/RAPTOR.png)

    詳細解説動画：

    https://www.youtube.com/watch?v=jbGchdTL7d0

    論文：

    https://arxiv.org/pdf/2401.18059.pdf

    完全なコード：

    https://github.com/langchain-ai/langchain/blob/master/cookbook/RAPTOR.ipynb
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## パート14：ColBERT

    通常の密ベクトル検索が文書全体を一つのベクトルに圧縮するのに対し、ColBERTはクエリと文書の各トークンを別々のベクトルとして保持します。クエリの各トークンについて文書側との最大類似度（MaxSim）を求め、その合計で文書を順位付けする「Late Interaction」が特徴です。

    ### 使用ライブラリの変遷

    このパートの実装は次の経緯で変わっています。

    | 版 | エンコーダ | インデックス |
    | --- | --- | --- |
    | 元ノート | RAGatouille | RAGatouille |
    | 旧版 | PyLate `models.ColBERT` | PyLate `indexes.PLAID` |
    | 現在 | `sentence-transformers` の `MultiVectorEncoder` | `fast-plaid` の `FastPlaid` |

    PyLateは内部で `fast-plaid` を使っていたため、インデックスの実装は旧版と同じです。
    変更したのは、PyLateというラッパーを外して `fast-plaid` を直接呼ぶようにした点と、
    エンコーダを `sentence-transformers` v6 の `MultiVectorEncoder` へ移した点です。
    これによりトークン単位の埋め込みとMaxSimスコアを直接確認できるようになり、
    パート15の `CrossEncoder` と同じライブラリに揃いました。
    詳細は `docs/colbert-stack.md` を参照してください。

    使用モデルは元ノートの `colbert-ir/colbertv2.0` ではなく
    `lightonai/GTE-ModernColBERT-v1` です。ライブラリとモデルの両方が異なるため、
    検索結果を元ノートと直接比較することはできません。
    初回実行時はモデルのダウンロードとローカルインデックスの作成が必要です。次のセルでは `COLBERT_MODEL` 環境変数があればその値を使い、未指定なら `lightonai/GTE-ModernColBERT-v1` を使うモデル名として `colbert_model_name` に保存します。
    """)
    return


@app.cell
def _():
    colbert_model_name = os.environ.get(
        "COLBERT_MODEL", "lightonai/GTE-ModernColBERT-v1"
    )
    return (colbert_model_name,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    検索対象として、英語版Wikipediaの「Hayao_Miyazaki」記事を取得します。`get_wikipedia_page()` はWikipedia APIからページ本文をプレーンテキストで取り出す小さな自作関数で、次のセルでは取得した文字数を表示します。

    - 入力：Wikipediaページタイトル（ここでは `"Hayao_Miyazaki"`）
    - 処理：`https://en.wikipedia.org/w/api.php` に `action=query`、`prop=extracts`、`explaintext=True` を指定して問い合わせます。Wikipediaの作法に従い、`User-Agent` ヘッダも付けています
    - 出力：`miyazaki_article` = `page.get("extract")` の返り値。対象ページでは記事本文のプレーンテキスト文字列（表示される数値は文字数）ですが、ページが見つからない場合などは `None` になる可能性があります
    """)
    return


@app.cell
def _():
    import requests

    def get_wikipedia_page(title: str):
        """
        Retrieve the full text content of a Wikipedia page.

        :param title: str - Title of the Wikipedia page.
        :return: str - Full text content of the page as raw string.
        """
        # Wikipedia API endpoint
        URL = "https://en.wikipedia.org/w/api.php"

        # Parameters for the API request
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        }

        # Custom User-Agent header to comply with Wikipedia's best practices
        headers = {"User-Agent": "PyLate_tutorial/0.0.1"}

        response = requests.get(URL, params=params, headers=headers)
        data = response.json()

        # Extracting page content
        page = next(iter(data["query"]["pages"].values()))
        return page.get("extract")

    miyazaki_article = get_wikipedia_page("Hayao_Miyazaki")
    len(miyazaki_article)
    return (miyazaki_article,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    記事をパッセージへ分割します。元ノートは RAGatouille に
    `max_document_length=180, split_documents=True` を渡して内部で分割させていました。
    ここでは同じ粒度になるよう、トークン基準で180トークン・重なりなしで分割します。

    ColBERTはパッセージごとにトークン単位のベクトルを持つため、
    パッセージを短く保つとインデックスの肥大を抑えられます。
    なお `GTE-ModernColBERT-v1` は最大299トークンまでしか受け付けず、
    それを超える分は切り捨てられます。分割後の `colbert_passages` の件数が、そのままPLAIDインデックスへ登録するパッセージ数になります。

    - データ：`miyazaki_article`（英語版Wikipediaの「Hayao_Miyazaki」記事本文）
    - 前処理：`RecursiveCharacterTextSplitter.from_tiktoken_encoder()` / `chunk_size=180`, `chunk_overlap=0`（トークン基準）
    - 出力：`colbert_passages` = パッセージ文字列のリスト（表示される数値がパッセージ数）
    """)
    return


@app.cell
def _(miyazaki_article):
    # Matches the original notebook's RAGatouille setting
    # (max_document_length=180, split_documents=True).
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=180, chunk_overlap=0
    )
    colbert_passages = text_splitter.split_text(miyazaki_article)
    len(colbert_passages)
    return (colbert_passages,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ColBERTモデルでパッセージを埋め込みます。通常の埋め込みが1文書あたり1ベクトルなのに対し、ColBERTは**トークンごとに1ベクトル**を作ります。`encode_document()` はパッセージごとに `(トークン数, 次元)` のテンソルを返すため、長さの違うパッセージは行数の違うテンソルになります。次のセルは先頭5件のテンソル形状を表示します。
    """)
    return


@app.cell
def _(colbert_model_name, colbert_passages):
    colbert_model = MultiVectorEncoder(colbert_model_name, device=colbert_device())
    passage_embeddings = colbert_model.encode_document(colbert_passages)
    [tuple(embedding.shape) for embedding in passage_embeddings[:5]]
    return colbert_model, passage_embeddings


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    質問側は `encode_query()` で埋め込みます。入力は質問文字列 `colbert_query`、出力はクエリトークンごとの埋め込みテンソル `query_embedding` です。文書側と違い、モデルによってはクエリを固定長へ揃えるため、行数が一定になることがあります。次のセルは `query_embedding` の形状を表示します。
    """)
    return


@app.cell
def _(colbert_model):
    colbert_query = "What animation studio did Miyazaki found?"
    query_embedding = colbert_model.encode_query(colbert_query)
    tuple(query_embedding.shape)
    return colbert_query, query_embedding


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### MaxSimによる採点

    `similarity()` がLate Interactionの採点そのものです。クエリの各トークンについて、文書側のどのトークンと最も似ているかを求め、その最大値をクエリトークン全体で合計します。入力は `query_embedding.unsqueeze(0)` と `passage_embeddings`、出力の `exhaustive_scores` は各パッセージに対するスコアのテンソルです。パッセージ数が少ないうちは、この総当たり計算で十分です。
    """)
    return


@app.cell
def _(colbert_model, passage_embeddings, query_embedding):
    exhaustive_scores = colbert_model.similarity(
        query_embedding.unsqueeze(0), passage_embeddings
    )[0]
    exhaustive_top = exhaustive_scores.topk(3)
    [
        (int(i), round(float(s), 4))
        for s, i in zip(exhaustive_top.values, exhaustive_top.indices)
    ]
    return (exhaustive_scores,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### PLAIDインデックス

    パッセージ数が増えると総当たりでは追いつかなくなります。PLAIDは重心による枝刈りと量子化で、トークンごとのベクトルという多量のデータを実用的な容量と速度へ抑えます。`fast-plaid` はそのRust実装で、`encode_document()` が返したテンソルをそのまま受け取ります。

    - 入力：`passage_embeddings`（パッセージごとのトークン単位テンソル。各要素は `(トークン数, 次元)`）
    - DB：PLAIDインデックス。保存先は `COLBERT_INDEX_DIR` が指す `.colbert-index/` ディレクトリです
    - 永続性：他パートのインメモリChromaと違い、ディスク上に作られ、実行後も残ります。`.colbert-index/` は `.gitignore` 済みです
    - 出力：`colbert_index` = `FastPlaid` の検索用インデックス
    """)
    return


@app.cell
def _(passage_embeddings):
    colbert_index = FastPlaid(
        index=COLBERT_INDEX_DIR, device=colbert_device()
    )
    colbert_index.create(documents_embeddings=passage_embeddings)
    return (colbert_index,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    検索結果は `(パッセージの位置, スコア)` の組で返ります。PLAIDは近似検索なので、次のセルでは総当たりのMaxSim結果と並べて比較します。
    """)
    return


@app.cell
def _(colbert_index, query_embedding):
    colbert_results = colbert_index.search(
        queries_embeddings=query_embedding.unsqueeze(0), top_k=3
    )
    colbert_results[0]
    return (colbert_results,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    総当たりとPLAIDのスコアを並べて比較します。コーパスが大きくなると、PLAIDの近似により総当たりのMaxSimと差が出ることがあります。
    """)
    return


@app.cell
def _(colbert_results, exhaustive_scores):
    [
        {
            "passage": position,
            "plaid": round(float(score), 4),
            "exhaustive": round(float(exhaustive_scores[position]), 4),
        }
        for position, score in colbert_results[0]
    ]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    位置からパッセージ本文を取り出します。
    """)
    return


@app.cell
def _(colbert_passages, colbert_results):
    retrieved_passages = [
        {
            "position": position,
            "score": round(float(score), 4),
            "text": colbert_passages[position],
        }
        for position, score in colbert_results[0]
    ]
    retrieved_passages
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### LangChainのリトリーバーとして使う

    ここまでは検索結果を `(位置, スコア)` のタプルとして直接扱ってきました。
    ただしRAGチェーンへ差し込むには、LangChainが期待する
    「クエリ文字列を受け取り `Document` のリストを返す」形にする必要があります。

    元ノートは `RAG.as_langchain_retriever(k=3)` の一行でこれを行っていました。
    RAGatouilleを使わない現在は、`BaseRetriever` を継承して同じ役割のクラスを作ります。
    実装すべきは `_get_relevant_documents()` の1メソッドだけです。

    `BaseRetriever` はPydanticモデルなので、モデルやインデックスのような
    任意のオブジェクトを保持するには `arbitrary_types_allowed` が必要です。

    - 保持する属性：`encoder`（クエリを埋め込む `MultiVectorEncoder`）、`index`（検索する `FastPlaid`）、`passages`（位置から本文を引くパッセージ文字列リスト）、`k`（返す件数）
    - 入力：`_get_relevant_documents()` に渡されるクエリ文字列
    - 処理：クエリを `encode_query()` で埋め込み、PLAIDインデックスを `top_k=self.k` で検索し、返った位置からパッセージ本文を取り出します
    - 出力：`page_content` にパッセージ本文、`metadata` に `position` と `score` を持つ `Document` のリスト
    """)
    return


@app.class_definition
class ColBERTRetriever(BaseRetriever):
    """Expose a fast-plaid ColBERT index as a LangChain retriever.

    Equivalent to `RAG.as_langchain_retriever(k=...)` in the original
    notebook, which relied on RAGatouille.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    encoder: MultiVectorEncoder
    index: FastPlaid
    passages: list[str]
    k: int = 3

    def _get_relevant_documents(self, query, *, run_manager=None):
        query_embedding = self.encoder.encode_query(query)
        results = self.index.search(
            queries_embeddings=query_embedding.unsqueeze(0), top_k=self.k
        )
        return [
            Document(
                page_content=self.passages[position],
                metadata={"position": position, "score": float(score)},
            )
            for position, score in results[0]
        ]


@app.cell
def _(colbert_index, colbert_model, colbert_passages):
    colbert_retriever = ColBERTRetriever(
        encoder=colbert_model,
        index=colbert_index,
        passages=colbert_passages,
        k=3,
    )
    return (colbert_retriever,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    これで他のリトリーバーと同じ `invoke()` で呼び出せます。戻り値が `Document` に
    なったため、`format_colbert_docs` やLCELのパイプへそのまま渡せます。
    """)
    return


@app.cell
def _(colbert_query, colbert_retriever):
    colbert_documents = colbert_retriever.invoke(colbert_query)
    colbert_documents
    return (colbert_documents,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    実際にRAGチェーンへ差し込み、ColBERTで検索した内容から回答を生成します。
    ベクトル検索のリトリーバーと差し替えるだけで動く点が、
    LangChainの部品として揃える利点です。
    """)
    return


@app.cell
def _(colbert_query, colbert_retriever):
    colbert_rag_chain = (
        {
            "context": colbert_retriever | format_colbert_docs,
            "question": RunnablePassthrough(),
        }
        | load_rag_prompt()
        | make_chat_model()
        | StrOutputParser()
    )
    colbert_rag_chain.invoke(colbert_query)
    return


if __name__ == "__main__":
    app.run()
