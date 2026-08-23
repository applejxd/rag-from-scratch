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
    ## このノートブックの共通部品

    冒頭の `with app.setup` ブロック（marimo上では折りたたまれています）で、以下の共通部品を定義しています。各パートのコードセルはこれらを繰り返し使います。

    - `make_chat_model()`：回答生成用のチャットモデル。OpenRouter経由で `openai/gpt-5-nano`（環境変数 `MODEL` で変更可）を呼びます。出力を安定させるため `temperature=0` を指定し、このノートでは再試行しないよう `max_retries=0` も指定しています。
    - `make_embeddings()`：埋め込みモデル。既定は `openai/text-embedding-3-small` です。OpenRouterはOpenAI互換ですがトークナイザ情報を返さないため、クライアント側のトークン数チェックを無効化（`check_embedding_ctx_length=False` / `tiktoken_enabled=False`）しています。
    - `format_docs(docs)`：`Document` のリストを、本文を空行2つでつないだ1つの文字列へ変換します。検索結果をプロンプトの `{context}` へ埋め込むために使います。
    - `load_rag_prompt()`：LangChain Hubの `rlm/rag-prompt` と同じ内容のプロンプト。元ノートは実行時に `hub.pull()` で取得していましたが、外部サービスへの依存を避け、かつ文面をその場で読めるようローカルへ写しています。
    - `split_queries(text)`：LLMが生成した複数クエリの文字列を改行で分割し、空行を除いた `list[str]` にします。空文字列が埋め込みAPIへ渡らないようにするための前処理です。

    このノートの中心となる自作関数 `rerank_with_cross_encoder(question, docs, top_n=3)` は、ベクトル検索で得た候補をCrossEncoderで再評価します。

    - 入力：質問文字列 `question`、`Document` のリスト `docs`、残す件数 `top_n`
    - モデル：環境変数 `RERANK_MODEL` で指定し、既定は `cross-encoder/ms-marco-MiniLM-L-6-v2` です。`sentence-transformers` の `CrossEncoder` をローカルで実行するため、追加のAPIキーは不要です
    - 実行デバイス：`torch.cuda.is_available()` により、GPUがあれば `cuda`、なければ `cpu` を使います
    - 処理：`(質問, 文書本文)` のペアを作り、`predict()` で関連度スコアを算出して降順に並べます
    - 出力：上位 `top_n` 件を、`metadata["relevance_score"]` にスコアを持たせた新しい `Document` として返します。元の `Document` は変更しません
    - 注意：`docs` が空なら空リストを返します
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
    パート15で使うインデックスの元になるチャンクを作成します。ブログ記事を読み込み、HTMLの本文周辺だけを取り出してからトークン基準で分割し、分割後のチャンク数を出力します。

    - データ：Lilian Wengのブログ記事 [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
    - HTMLの絞り込み：`bs4.SoupStrainer` で `post-content` / `post-title` / `post-header` クラスの要素だけを解析対象にします
    - 前処理：`RecursiveCharacterTextSplitter.from_tiktoken_encoder()` / `chunk_size=300`, `chunk_overlap=50`（トークン基準）
    - 出力：`document_chunks` = 分割後の `Document` リスト（表示される数値がチャンク数）
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    分割したチャンクを埋め込みへ変換し、Chromaへ登録して検索用のリトリーバーを作成します。`retriever` と `vectorstore` の両方を返しているのは、後段で同じDBから `k=10` の別リトリーバー `candidate_retriever` を作るためです。

    - データ：`document_chunks`（Lilian Wengのブログ記事を300トークン単位で分割した `Document` リスト）
    - 前処理：前セルで作成済み（`chunk_size=300`, `chunk_overlap=50`、トークン基準）
    - DB：Chroma。`persist_directory` を指定していないため**インメモリ**です
    - 出力：`retriever` = 既定設定（類似度検索・`k=4`）のリトリーバー、`vectorstore` = Chromaのベクトルストア
    """)
    return


@app.cell
def _(document_chunks):
    vectorstore = Chroma.from_documents(
        documents=document_chunks,
        embedding=make_embeddings(),
    )
    retriever = vectorstore.as_retriever()
    return retriever, vectorstore


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    再ランキングとの比較用ベースラインとして、パート6のRAG-Fusionと同様の部品をここで再掲します。最初のセルで「1つの質問から4つの検索クエリを出力する」プロンプトを作り、次のセルでチャットモデル、文字列パーサー、`split_queries` をつないでクエリ生成チェーンにします。

    - 入力：`{question}` に入る質問文字列
    - 処理：LLMに関連する4つの検索クエリを生成させ、改行区切りの出力から空行を除きます
    - 出力：`rag_fusion_prompt` = プロンプト、`rag_fusion_query_generator` = 4つの検索クエリを返すチェーン
    """)
    return


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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    パート15全体で使い回す共通の質問を定義します。RAG-Fusionによる検索、RRFで融合した件数の確認、再ランキング、最終的なRAGチェーンの回答を同じ質問で比較します。
    """)
    return


@app.cell
def _():
    reranking_question = "What is task decomposition for LLM agents?"
    return (reranking_question,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    比較のため、まずRAG-Fusion（順位の統合）で検索します。`reciprocal_rank_fusion` はパート6（`rag_from_scratch_5_to_9.py`）と同一の実装を、このノートでも再掲しています。

    - 入力：複数クエリそれぞれの検索結果 `results`（`list[list]`）と定数 `k=60`
    - 処理：各検索結果での順位 `rank` だけを使い、文書ごとに `1 / (rank + k)` を合計します。文書は `dumps()` で文字列化して重複をまとめ、最後に `loads()` で `Document` に戻します
    - 出力：RRFスコアの降順に並んだ `(Document, score)` のタプルのリスト
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    共通の質問をRAG-Fusionの検索チェーンへ渡し、4つの検索クエリの結果をRRFで融合します。ここでは回答生成には進まず、融合後に何件の `(Document, score)` が残ったかだけを確認します。
    """)
    return


@app.cell
def _(rag_fusion_retrieval_chain, reranking_question):
    fused_documents = rag_fusion_retrieval_chain.invoke(
        {"question": reranking_question}
    )
    len(fused_documents)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    RAG-Fusionを使うベースライン回答を生成します。`context` にはRRFで融合した `(Document, score)` のリストを渡し、同じ `reranking_question` に対する回答を作ることで、後段の再ランキング付きRAGチェーンと比較できるようにします。
    """)
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

    ここでは外部APIキーを必要としない方式として、冒頭で定義した `rerank_with_cross_encoder` により `sentence-transformers` の[CrossEncoder](https://www.sbert.net/docs/cross_encoder/pretrained_models.html)をローカルで実行します。以降のセルでは、候補取得、CrossEncoderによる再評価、上位3件への絞り込みを順に確認します。

    ![再ランキングの流れ（候補取得→スコアリング→上位選択）](./imgs/Cohere_Re-Rank.png)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    まず候補を広めに10件取得します。通常の `retriever` は既定の `k=4` ですが、ここでは再ランキング前に候補を広く残すため `search_kwargs={"k": 10}` を指定します。この時点の順位はCrossEncoderではなくベクトル類似度によるものです。

    - 入力：共通質問 `reranking_question` と、インメモリChromaの `vectorstore`
    - 処理：`vectorstore.as_retriever(search_kwargs={"k": 10})` で候補取得用リトリーバーを作り、質問で検索します
    - 出力：`candidate_retriever` = `k=10` のリトリーバー、`candidate_documents` = ベクトル類似度順の候補 `Document` リスト（表示される数値が件数）
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

    - 属性：`base_retriever` は候補を取るリトリーバー、`top_n` は残す件数（既定3）です
    - `_get_relevant_documents()`：クエリ文字列を受け取り、`base_retriever` で候補を取得してから `rerank_with_cross_encoder` で上位 `top_n` 件の `Document` リストへ絞ります
    - `model_config = ConfigDict(arbitrary_types_allowed=True)`：`BaseRetriever` はPydanticモデルです。任意型のオブジェクトをフィールドに持つカスタムRetrieverでも検証エラーにしない設定を、このクラスでも明示しています
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
    リトリーバーになったので、RAGチェーンへそのまま差し込めます。チェーン全体の形は通常のRAGと同じで、`context` 側のリトリーバーを `reranking_retriever` に差し替えるだけです。これにより、検索結果を `format_docs` で整形する前に「10件取得して3件へ絞る」処理が組み込まれます。
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
