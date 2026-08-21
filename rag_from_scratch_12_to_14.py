import marimo

__generated_with = "0.20.4"
app = marimo.App()

with app.setup:
    import os
    import uuid

    import marimo as mo
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_CHAT_MODEL = os.environ.get("MODEL", "openai/gpt-5-nano")
    DEFAULT_EMBEDDING_MODEL = os.environ.get(
        "EMBEDDING_MODEL", "openai/text-embedding-3-small"
    )
    DEFAULT_CHAT_TIKTOKEN_MODEL = os.environ.get("TIKTOKEN_CHAT_MODEL", "gpt-4o-mini")
    DEFAULT_EMBEDDING_TIKTOKEN_MODEL = os.environ.get(
        "TIKTOKEN_EMBEDDING_MODEL", "text-embedding-3-small"
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
    return (source_documents,)


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
    ### 要約と元文書の対応付け

    各要約には元文書と共通の `doc_id` を付けます。検索対象は要約ですが、ヒット後は `doc_id` を使って元文書を返すため、回答生成では省略前の内容を利用できます。
    """)
    return


@app.cell
def _(document_summaries, source_documents):
    summary_vectorstore = Chroma(
        collection_name="summaries",
        embedding_function=make_embeddings(),
    )

    id_key = "doc_id"

    multi_vector_retriever = SimpleMultiVectorRetriever(
        vectorstore=summary_vectorstore, id_key=id_key
    )
    source_document_ids = [str(uuid.uuid4()) for _ in source_documents]

    summary_documents = [
        Document(
            page_content=summary,
            metadata={id_key: source_document_ids[index]},
        )
        for index, summary in enumerate(document_summaries)
    ]

    summary_vectorstore.add_documents(summary_documents)
    multi_vector_retriever.set_parent_documents(
        zip(source_document_ids, source_documents)
    )
    return multi_vector_retriever, summary_vectorstore


@app.cell
def _(summary_vectorstore):
    summary_query = "Memory in agents"
    matching_summaries = summary_vectorstore.similarity_search(summary_query, k=1)
    matching_summaries[0]
    return (summary_query,)


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

    通常の密ベクトル検索が文書全体を一つのベクトルに圧縮するのに対し、ColBERTはクエリと文書の各トークンを別々のベクトルとして保持します。クエリの各トークンについて文書側との最大類似度を求め、その合計で文書を順位付けする「Late Interaction」が特徴です。

    元ノートのRAGatouilleによる例は、現在の実装ではPyLateを直接使う形へ置き換えています。`ColBERT` で埋め込みを生成し、`PLAID` インデックスへ保存して検索します。初回実行時はモデルのダウンロードとローカルインデックスの作成が必要です。
    """)
    return


@app.cell
def _():
    from pylate import indexes, models, retrieve

    colbert_model_name = os.environ.get(
        "PYLATE_MODEL", "lightonai/GTE-ModernColBERT-v1"
    )
    return colbert_model_name, indexes, models, retrieve


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
    return (miyazaki_article,)


@app.cell
def _(miyazaki_article):
    def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
        return chunks

    colbert_passages = chunk_text(miyazaki_article)
    colbert_passage_ids = [
        f"miyazaki-{index}" for index in range(len(colbert_passages))
    ]
    passages_by_id = dict(zip(colbert_passage_ids, colbert_passages))
    return colbert_passage_ids, colbert_passages, passages_by_id


@app.cell
def _(colbert_model_name, colbert_passage_ids, colbert_passages, indexes, models):
    colbert_model = models.ColBERT(model_name_or_path=colbert_model_name)
    passage_embeddings = colbert_model.encode(colbert_passages, is_query=False)
    colbert_index = indexes.PLAID(
        index_folder=".pylate-indexes",
        index_name="miyazaki-colbert",
        override=True,
    )
    colbert_index.add_documents(
        documents_ids=colbert_passage_ids,
        documents_embeddings=passage_embeddings,
    )
    return colbert_index, colbert_model


@app.cell
def _(colbert_index, colbert_model, retrieve):
    colbert_retriever = retrieve.ColBERT(index=colbert_index)
    colbert_query = "What animation studio did Miyazaki found?"
    query_embeddings = colbert_model.encode([colbert_query], is_query=True)
    colbert_results = colbert_retriever.retrieve(
        queries_embeddings=query_embeddings, k=3
    )
    colbert_results
    return (colbert_results,)


@app.cell
def _(colbert_results, passages_by_id):
    retrieved_passages = [
        {
            "id": result["id"],
            "score": result["score"],
            "text": passages_by_id[result["id"]],
        }
        for result in colbert_results[0]
    ]
    retrieved_passages
    return


if __name__ == "__main__":
    app.run()
