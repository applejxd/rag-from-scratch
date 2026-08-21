import marimo

__generated_with = "0.20.4"
app = marimo.App()

with app.setup:
    # Initialization code that runs before all other cells
    import os

    import bs4
    import numpy as np
    import tiktoken
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_CHAT_MODEL = os.environ.get("MODEL", "openai/gpt-5-nano")
    DEFAULT_EMBEDDING_MODEL = os.environ.get(
        "EMBEDDING_MODEL", "openai/text-embedding-3-small"
    )
    DEFAULT_CHAT_TIKTOKEN_MODEL = os.environ.get("TIKTOKEN_CHAT_MODEL", "gpt-4o-mini")
    DEFAULT_EMBEDDING_TIKTOKEN_MODEL = os.environ.get(
        "TIKTOKEN_EMBEDDING_MODEL", "text-embedding-3-small"
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

    def make_chat_model():
        return ChatOpenAI(
            model_name=DEFAULT_CHAT_MODEL,
            openai_api_base=OPENROUTER_BASE_URL,
            openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
            temperature=0,
            tiktoken_model_name=DEFAULT_CHAT_TIKTOKEN_MODEL,
        )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def load_rag_prompt():
        template = (
            "Answer the question based only on the following context:\n"
            "{context}\n\nQuestion: {question}\n"
        )
        return ChatPromptTemplate.from_template(template)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RAGをゼロから学ぶ：概要

    このノートブックでは、RAGアプリケーションをゼロから構築する流れを学びます。

    以下の図に示すRAGの全体像を、段階的に理解していきます。

    ![RAGの全体像](./imgs/overview.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## パート1：概要

    RAGは、質問に関連する文書を検索し、その内容をコンテキストとしてLLMへ渡す構成です。ここでは、文書の読み込み・分割・埋め込み・検索・回答生成という最小構成を一つのセルで確認します。

    [RAGクイックスタート](https://python.langchain.com/docs/use_cases/question_answering/quickstart)
    """)
    return


@app.cell
def _():
    #### INDEXING ####
    # Load Documents
    _loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        },
    )
    _documents = _loader.load()

    # Split
    _text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    _document_chunks = _text_splitter.split_documents(_documents)

    # Embed
    _vectorstore = Chroma.from_documents(
        documents=_document_chunks, embedding=make_embeddings()
    )

    _retriever = _vectorstore.as_retriever()

    #### RETRIEVAL and GENERATION ####

    # Prompt
    _prompt = load_rag_prompt()

    # LLM
    _chat_model = make_chat_model()

    # Chain
    _rag_chain = (
        {"context": _retriever | format_docs, "question": RunnablePassthrough()}
        | _prompt
        | _chat_model
        | StrOutputParser()
    )

    # Question
    _rag_chain.invoke("What is Task Decomposition?")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## パート2：インデックス作成

    検索対象の文書を小さなチャンクに分割し、各チャンクを埋め込みベクトルへ変換してベクトルストアへ保存します。以降のセルでは、この3段階を個別に確認します。

    ![インデックス作成の流れ](./imgs/indexing.png)
    """)
    return


@app.cell
def _():
    # Documents
    question = "What kinds of pets do I like?"
    document = "My favorite pet is a cat."
    return document, question


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### トークン

    モデルの入力上限やチャンクサイズは文字数ではなくトークン数で決まります。[トークン数を数える例](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)では、英語の場合は[1トークン約4文字](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)が大まかな目安です。
    """)
    return


@app.cell
def _(question):
    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    num_tokens_from_string(question, "cl100k_base")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 埋め込み

    [テキスト埋め込みモデル](https://python.langchain.com/docs/integrations/text_embedding/openai)は、文書と質問を意味を反映した数値ベクトルへ変換します。同じ次元のベクトル同士を比較することで、語句が完全一致しない場合でも意味の近い文書を探せます。
    """)
    return


@app.cell
def _(document, question):
    embedding_model = make_embeddings()

    query_embedding = embedding_model.embed_query(question)
    document_embedding = embedding_model.embed_query(document)
    len(query_embedding)
    return document_embedding, query_embedding


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    OpenAIの埋め込みでは[コサイン類似度](https://platform.openai.com/docs/guides/embeddings/frequently-asked-questions)が推奨されています。ベクトルの向きが近いほど値が1に近づき、意味的な類似度が高いと判断できます。
    """)
    return


@app.cell
def _(document_embedding, query_embedding):
    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    similarity = cosine_similarity(query_embedding, document_embedding)
    print("Cosine Similarity:", similarity)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 読み込み

    [ドキュメントローダー](https://python.langchain.com/docs/integrations/document_loaders/)は、Webページなどの外部データを、本文とメタデータを持つ共通の `Document` 形式へ変換します。
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
    # Load blog
    blog_documents = blog_loader.load()
    return (blog_documents,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 分割

    [テキスト分割](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter)では、検索とモデル入力に適した大きさへ文書を分けます。

    > このテキストスプリッターは、一般的なテキストに推奨されています。文字のリストを受け取り、チャンクが十分に小さくなるまで、その文字を順番に使って分割を試みます。デフォルトのリストは `['\n\n', '\n', ' ', '"']` です。これにより、意味的な関連が強いと考えられる段落、文、単語を、可能な限りまとめたまま分割できます。
    """)
    return


@app.cell
def _(blog_documents):
    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=50
    )
    # Make splits
    document_chunks = text_splitter.split_documents(blog_documents)
    return (document_chunks,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 保存

    [ベクトルストア](https://python.langchain.com/docs/integrations/vectorstores/)は、埋め込みベクトルと元のチャンクを対応付けて保存し、質問に近いチャンクを検索できるようにします。
    """)
    return


@app.cell
def _(document_chunks):
    # Index
    vectorstore = Chroma.from_documents(
        documents=document_chunks, embedding=make_embeddings()
    )
    return (vectorstore,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## パート3：検索

    リトリーバーは質問を埋め込み、ベクトルストアから類似度の高いチャンクを返します。ここでは `k=1` とし、最も近い1件を取得して内容を確認します。
    """)
    return


@app.cell
def _(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    return (retriever,)


@app.cell
def _(retriever):
    retrieved_documents = retriever.invoke("What is Task Decomposition?")
    return (retrieved_documents,)


@app.cell
def _(retrieved_documents):
    len(retrieved_documents)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## パート4：生成

    検索したチャンクと質問をプロンプトへ埋め込み、チャットモデルで回答を生成します。まず各要素を手動で接続し、その後に検索から出力解析までを一つのRAGチェーンへまとめます。

    ![回答生成の流れ](./imgs/generation.png)
    """)
    return


@app.cell
def _():
    generation_template = "Answer the question based only on the following context:\n{context}\n\nQuestion: {question}\n"
    generation_prompt = ChatPromptTemplate.from_template(generation_template)
    # Prompt
    generation_prompt
    return (generation_prompt,)


@app.cell
def _():
    # LLM
    chat_model = make_chat_model()
    return (chat_model,)


@app.cell
def _(chat_model, generation_prompt):
    # Chain
    generation_chain = generation_prompt | chat_model
    return (generation_chain,)


@app.cell
def _(generation_chain, retrieved_documents):
    # Run
    generation_chain.invoke(
        {
            "context": format_docs(retrieved_documents),
            "question": "What is Task Decomposition?",
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 再利用可能なRAGプロンプト

    元ノートではLangChain Hubからプロンプトを取得していましたが、このリポジトリでは外部Hubへ依存せず再現できるよう、同じ役割のプロンプトを `load_rag_prompt()` でローカルに定義しています。
    """)
    return


@app.cell
def _():
    reusable_rag_prompt = load_rag_prompt()
    return (reusable_rag_prompt,)


@app.cell
def _(reusable_rag_prompt):
    reusable_rag_prompt
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 検索と生成の統合

    [RAGチェーン](https://python.langchain.com/docs/expression_language/get_started#rag-search-example)では、質問をリトリーバーとプロンプトの両方へ渡します。検索結果は `format_docs` でコンテキストへ変換し、最後に `StrOutputParser` でモデルの応答から文字列だけを取り出します。
    """)
    return


@app.cell
def _(chat_model, reusable_rag_prompt, retriever):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | reusable_rag_prompt
        | chat_model
        | StrOutputParser()
    )
    rag_chain.invoke("What is Task Decomposition?")
    return


if __name__ == "__main__":
    app.run()
