import marimo

__generated_with = "0.20.4"
app = marimo.App()

with app.setup:
    import os
    from operator import itemgetter

    import bs4
    import marimo as mo
    import torch
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.documents import Document
    from langchain_core.load import dumps, loads
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pydantic import ConfigDict
    from sentence_transformers import CrossEncoder

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

    def format_docs(docs):
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

    def split_queries(text: str) -> list[str]:
        """Split generated queries into lines, dropping blank ones.

        The model often separates queries with blank lines. Without this
        filtering the empty strings reach the embedding API, which rejects
        them with HTTP 400 (`expected string to have >=1 characters`).
        """
        return [line.strip() for line in text.split("\n") if line.strip()]

    def rerank_with_cross_encoder(question: str, docs, top_n: int = 3):
        if not docs:
            return []

        model_name = os.environ.get(
            "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cross_encoder = CrossEncoder(model_name, device=device)

        pairs = [(question, doc.page_content) for doc in docs]
        scores = cross_encoder.predict(pairs)

        ranked_docs = sorted(
            zip(docs, scores), key=lambda item: item[1], reverse=True
        )

        reranked_docs = []
        for source_doc, score in ranked_docs[:top_n]:
            reranked_docs.append(
                Document(
                    page_content=source_doc.page_content,
                    metadata={
                        **source_doc.metadata,
                        "relevance_score": float(score),
                    },
                )
            )
        return reranked_docs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # RAGをゼロから学ぶ：検索

    ![検索の概要](./imgs/retrieval_overview.png)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## パート15：再ランキング

    最初の検索では候補を広めに取得し、より精密なモデルで質問との関連度を再評価して上位だけを残します。RAG-Fusionが複数の検索順位を統合するのに対し、再ランキングは質問と各候補文書を直接比較します。

    ![再ランキングの流れ](./imgs/re-ranking.png)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    パート15で使うインデックスを作成します。ブログ記事を読み込み、トークン基準で分割します。
    """)
    return


@app.cell
def _():
    #### INDEXING ####
    blog_loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        },
    )
    blog_documents = blog_loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50,
    )
    document_chunks = text_splitter.split_documents(blog_documents)
    len(document_chunks)
    return (document_chunks,)


@app.cell
def _(document_chunks):
    vectorstore = Chroma.from_documents(
        documents=document_chunks,
        embedding=make_embeddings(),
    )
    retriever = vectorstore.as_retriever()
    return retriever, vectorstore


@app.cell
def _():
    # RAG-Fusion
    rag_fusion_template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    rag_fusion_prompt = ChatPromptTemplate.from_template(rag_fusion_template)
    return (rag_fusion_prompt,)


@app.cell
def _(rag_fusion_prompt):
    rag_fusion_query_generator = (
        rag_fusion_prompt
        | make_chat_model()
        | StrOutputParser()
        | split_queries
    )
    return (rag_fusion_query_generator,)


@app.cell
def _():
    reranking_question = "What is task decomposition for LLM agents?"
    return (reranking_question,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    比較のため、まずRAG-Fusion（順位の統合）で検索します。
    """)
    return


@app.cell
def _(rag_fusion_query_generator, retriever):
    def reciprocal_rank_fusion(results: list[list], k=60):
        """Combine ranked result lists using reciprocal rank fusion."""
        fused_scores = {}
        for documents in results:
            for rank, document in enumerate(documents):
                doc_str = dumps(document)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)
        return [
            (loads(doc), score)
            for doc, score in sorted(
                fused_scores.items(), key=lambda item: item[1], reverse=True
            )
        ]

    rag_fusion_retrieval_chain = (
        rag_fusion_query_generator | retriever.map() | reciprocal_rank_fusion
    )
    return (rag_fusion_retrieval_chain,)


@app.cell
def _(rag_fusion_retrieval_chain, reranking_question):
    fused_documents = rag_fusion_retrieval_chain.invoke(
        {"question": reranking_question}
    )
    len(fused_documents)
    return


@app.cell
def _(rag_fusion_retrieval_chain, reranking_question):
    answer_template = 'Answer the following question based on this context:\n\n{context}\n\nQuestion: {question}\n'
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    rag_fusion_answer_chain = (
        {
            "context": rag_fusion_retrieval_chain,
            "question": itemgetter("question"),
        }
        | answer_prompt
        | make_chat_model()
        | StrOutputParser()
    )
    rag_fusion_answer_chain.invoke({"question": reranking_question})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    再ランキングは、Cohereなどのマネージド型Rerank APIを使う方法もあります（[Cohere Re-Rank](https://python.langchain.com/docs/integrations/retrievers/cohere-reranker#doing-reranking-with-coherererank)、[解説記事](https://txt.cohere.com/rerank/)）。

    ここでは外部APIキーを必要としない方式として、`sentence-transformers` の[CrossEncoder](https://www.sbert.net/docs/cross_encoder/pretrained_models.html)をローカルで実行します。まず最初にベクトル検索で10件を取得し、質問と各文書のペアをCrossEncoderへ直接入力して関連度スコアを計算し、上位3件へ絞ります。GPUが使える環境では自動的にGPU上で実行され、関連度は各文書の `metadata["relevance_score"]` に保存されます。

    ![再ランキングの流れ（候補取得→スコアリング→上位選択）](./imgs/Cohere_Re-Rank.png)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    まず候補を広めに10件取得します。この時点の順位はベクトル類似度によるものです。
    """)
    return


@app.cell
def _(reranking_question, vectorstore):
    candidate_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    candidate_documents = candidate_retriever.invoke(reranking_question)
    len(candidate_documents)
    return candidate_documents, candidate_retriever


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    CrossEncoderで質問と各候補を直接比較し、上位3件へ絞ります。
    """)
    return


@app.cell
def _(candidate_documents, reranking_question):
    reranked_documents = rerank_with_cross_encoder(
        reranking_question, candidate_documents, top_n=3
    )
    reranked_documents
    return (reranked_documents,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    再ランキング後の関連度スコアを確認します。値が大きいほど質問との関連が強いと判断されています。
    """)
    return


@app.cell
def _(reranked_documents):
    [
        round(doc.metadata["relevance_score"], 4) for doc in reranked_documents
    ]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 再ランキングをチェーンの部品にする

    ここまでは候補の取得と再ランキングを手作業でつないでいました。
    元ノートは、この2段構えを1つのリトリーバーにまとめていました。

    ```python
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=CohereRerank(), base_retriever=retriever
    )
    ```

    ただし `ContextualCompressionRetriever` は **LangChain 1.x で削除**されており、
    `langchain.retrievers` モジュール自体が存在しません。
    現在は `BaseRetriever` を継承して同じ役割のクラスを自分で書きます。
    実装するのは `_get_relevant_documents()` の1メソッドだけです。
    """)
    return


@app.class_definition
class RerankingRetriever(BaseRetriever):
    """Retrieve with a base retriever, then rerank with a CrossEncoder.

    Fills the role of `ContextualCompressionRetriever`, which was removed in
    LangChain 1.x.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: BaseRetriever
    top_n: int = 3

    def _get_relevant_documents(self, query, *, run_manager=None):
        candidates = self.base_retriever.invoke(query)
        return rerank_with_cross_encoder(query, candidates, top_n=self.top_n)


@app.cell
def _(candidate_retriever):
    reranking_retriever = RerankingRetriever(
        base_retriever=candidate_retriever, top_n=3
    )
    return (reranking_retriever,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    1回の `invoke()` で「10件取得して3件へ絞る」までが完了します。
    """)
    return


@app.cell
def _(reranking_question, reranking_retriever):
    reranking_retriever.invoke(reranking_question)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    リトリーバーになったので、RAGチェーンへそのまま差し込めます。
    ベクトル検索のリトリーバーと差し替えるだけで、再ランキング付きの検索になります。
    """)
    return


@app.cell
def _(reranking_question, reranking_retriever):
    reranking_rag_chain = (
        {
            "context": reranking_retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | load_rag_prompt()
        | make_chat_model()
        | StrOutputParser()
    )
    reranking_rag_chain.invoke(reranking_question)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## パート16：検索（CRAG）

    CRAGは検索結果を評価し、関連性が低い場合に検索クエリの書き換えやWeb検索で補正する手法です。このノートブックでは実装を実行せず、詳細解説と実装例へのリンクのみを示します。

    `詳細解説`

    https://www.youtube.com/watch?v=E2shqsYwxck

    `ノートブック`

    https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb

    https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag_mistral.ipynb
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## パート17：自己評価を伴う生成（Self-RAG）

    Self-RAGは、モデルが検索の要否、取得文書の関連性、回答の根拠性を自己評価しながら生成する手法です。このノートブックでは実装を実行せず、LangGraphによる実装例を参照します。

    `ノートブック`

    https://github.com/langchain-ai/langgraph/tree/main/examples/rag

    https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag_mistral_nomic.ipynb
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## パート18：長いコンテキストの影響

    入力できるコンテキストが長くても、必要な情報の位置によって利用精度が下がる「Lost in the Middle」が起こり得ます。長い文書をすべて渡す方法と、検索で関連箇所を絞る方法は、精度・遅延・コストを含めて選択します。このノートブックでは参考動画とスライドのみを示します。

    `詳細解説`

    https://www.youtube.com/watch?v=SsHUNfhF32s

    `スライド`

    https://docs.google.com/presentation/d/1mJUiPBdtf58NfuSEQ7pVSEQ2Oqmek7F1i4gBwR6JDss/edit#slide=id.g26c0cb8dc66_0_0
    """)
    return


if __name__ == "__main__":
    app.run()
