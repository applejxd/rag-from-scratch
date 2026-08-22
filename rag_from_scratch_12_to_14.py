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
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    ## パート12：複数表現によるインデックス作成

    検索に適した短い要約をベクトル化し、ヒットした要約から情報量の多い元文書を返します。検索用表現と回答用文書を分けることで、検索精度と回答に必要な文脈量を両立します。

    処理の流れ：

     ![複数表現によるインデックス作成の流れ](./imgs/multi-representation_indexing.png)

    ドキュメント：

    https://blog.langchain.dev/semi-structured-multi-modal-rag/

    https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector

    論文：

    https://arxiv.org/abs/2312.06648
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
    各文書をLLMで要約します。`batch()` で並行実行するため、`max_concurrency` で同時リクエスト数を制限しています。
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

    各要約には元文書と共通の `doc_id` を付けます。検索対象は要約ですが、ヒット後は `doc_id` を使って元文書を返すため、回答生成では省略前の内容を利用できます。
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
    要約をベクトルストアへ登録し、リトリーバーには `doc_id` から元文書を引くための対応表を渡します。
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
    初回実行時はモデルのダウンロードとローカルインデックスの作成が必要です。
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
    検索対象として、宮崎駿のWikipedia記事を取得します。
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
    それを超える分は切り捨てられます。
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
    ColBERTモデルでパッセージを埋め込みます。通常の埋め込みが1文書あたり1ベクトルなのに対し、ColBERTは**トークンごとに1ベクトル**を作ります。`encode_document()` はパッセージごとに `(トークン数, 次元)` の行列を返すため、長さの違うパッセージは行数の違う行列になります。
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
    質問側は `encode_query()` で埋め込みます。文書側と違い、モデルによってはクエリを固定長へ揃えるため、行数が一定になることがあります。
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

    `similarity()` がLate Interactionの採点そのものです。クエリの各トークンについて、文書側のどのトークンと最も似ているかを求め、その最大値をクエリトークン全体で合計します。パッセージ数が少ないうちは、この総当たり計算で十分です。
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
    検索結果は `(パッセージの位置, スコア)` の組で返ります。PLAIDは近似検索ですが、この規模では総当たりのMaxSimと同じ順位・スコアになることを次のセルで確認できます。
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
    総当たりとPLAIDのスコアを並べて比較します。この規模では一致しますが、コーパスが大きくなるとPLAIDの近似により差が出ることがあります。
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


if __name__ == "__main__":
    app.run()
