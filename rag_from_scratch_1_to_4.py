import marimo

__generated_with = "0.20.4"
app = marimo.App()

with app.setup:
    # Initialization code that runs before all other cells
    import os

    import bs4
    import marimo as mo
    import numpy as np
    import tiktoken
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.tracers import ConsoleCallbackHandler
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # RAGをゼロから学ぶ：概要

    このノートブックでは、RAGアプリケーションをゼロから構築する流れを学びます。

    以下の図に示すRAGの全体像を、段階的に理解していきます。

    ![RAGの全体像](./imgs/overview.png)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## このノートブックの共通部品

    冒頭の `with app.setup` ブロック（marimo上では折りたたまれています）で、以下の共通部品を定義しています。各パートのコードセルはこれらを繰り返し使います。

    - `make_chat_model()`：回答生成用のチャットモデル。OpenRouter経由で `openai/gpt-5-nano`（環境変数 `MODEL` で変更可）を呼びます。出力を安定させるため `temperature=0` を指定しています。
    - `make_embeddings()`：埋め込みモデル。既定は `openai/text-embedding-3-small` です。OpenRouterはOpenAI互換ですがトークナイザ情報を返さないため、クライアント側のトークン数チェックを無効化（`check_embedding_ctx_length=False` / `tiktoken_enabled=False`）しています。
    - `format_docs(docs)`：`Document` のリストを、本文を空行2つでつないだ1つの文字列へ変換します。検索結果をプロンプトの `{context}` へ埋め込むために使います。
    - `load_rag_prompt()`：LangChain Hubの `rlm/rag-prompt` と同じ内容のプロンプト。元ノートは実行時に `hub.pull()` で取得していましたが、外部サービスへの依存を避け、かつ文面をその場で読めるようローカルへ写しています。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## パート1：概要

    RAGは、質問に関連する文書を検索し、その内容をコンテキストとしてLLMへ渡す構成です。ここではまず全体の流れを、インデックス作成・チェーン構築・実行の3段階に分けて確認します。パート2以降で各段階を詳しく見ていきます。

    [RAGクイックスタート](https://python.langchain.com/docs/use_cases/question_answering/quickstart)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### インデックス作成

    Webページを読み込み、チャンクへ分割し、埋め込みベクトルとしてベクトルストアへ保存します。次のセルは読み込みと分割までを行い、分割後のチャンク数を出力します。

    - データ：Lilian Wengのブログ記事 [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
    - 前処理：`bs4.SoupStrainer` でHTMLのうち `post-content` / `post-title` / `post-header` クラスの要素だけを解析対象にし、ナビゲーションやフッターを除外
    - 分割：`RecursiveCharacterTextSplitter`（**文字数**基準）で `chunk_size=1000`、`chunk_overlap=200`
    - 出力：`overview_chunks` = 分割後の `Document` リスト（表示される数値がチャンク数）
    """)
    return


@app.cell
def _():
    #### INDEXING ####
    overview_loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        },
    )
    overview_documents = overview_loader.load()

    overview_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    overview_chunks = overview_splitter.split_documents(overview_documents)
    len(overview_chunks)
    return (overview_chunks,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    分割したチャンクを埋め込みベクトルへ変換し、ベクトルストアへ登録します。

    - 入力：`overview_chunks`（前のセルで分割した `Document` リスト）
    - DB：Chroma。`persist_directory` を指定していないため**インメモリ**で、ノートブックのプロセスが終わると消えます
    - 出力：`overview_retriever` = 既定設定（類似度検索・`k=4`）のリトリーバー
    """)
    return


@app.cell
def _(overview_chunks):
    overview_vectorstore = Chroma.from_documents(
        documents=overview_chunks, embedding=make_embeddings()
    )
    overview_retriever = overview_vectorstore.as_retriever()
    return (overview_retriever,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### チェーン構築と実行

    検索・プロンプト・LLM・出力解析をLCELのパイプでつなぎます。`RunnablePassthrough()` は入力の質問をそのまま `question` へ渡す役割です。
    """)
    return


@app.cell
def _(overview_retriever):
    #### RETRIEVAL and GENERATION ####
    overview_chain = (
        {
            "context": overview_retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | load_rag_prompt()
        | make_chat_model()
        | StrOutputParser()
    )
    return (overview_chain,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    組み上げたチェーンを実行します。質問を渡すだけで、検索・プロンプト構築・生成・文字列抽出までが一度に走ります。パート2以降では、この一連の処理を分解して見ていきます。
    """)
    return


@app.cell
def _(overview_chain):
    overview_chain.invoke("What is Task Decomposition?")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## パート2：インデックス作成

    検索対象の文書を小さなチャンクに分割し、各チャンクを埋め込みベクトルへ変換してベクトルストアへ保存します。以降のセルでは、この3段階を個別に確認します。

    ![インデックス作成の流れ](./imgs/indexing.png)

    まず仕組みを最小の例で確かめるため、次のセルで質問 `question` と文書 `document` の2つの短い文字列を用意します。これらはトークン数の計算と埋め込みの比較にだけ使う、記事とは無関係なサンプルです。
    """)
    return


@app.cell
def _():
    # Documents
    question = "What kinds of pets do I like?"
    document = "My favorite pet is a cat."
    return document, question


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### トークン

    モデルの入力上限やチャンクサイズは文字数ではなくトークン数で決まります。[トークン数を数える例](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)では、英語の場合は[1トークン約4文字](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)が大まかな目安です。

    次のセルで定義する `num_tokens_from_string()` は、文字列とエンコーディング名を受け取り、そのエンコーディングでのトークン数（整数）を返します。`cl100k_base` はGPT-3.5／GPT-4系で使われるエンコーディングです。ここでは冒頭で定義した `question` のトークン数を数えます。
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
def _():
    mo.md(r"""
    ### 埋め込み

    [テキスト埋め込みモデル](https://python.langchain.com/docs/integrations/text_embedding/openai)は、文書と質問を意味を反映した数値ベクトルへ変換します。同じ次元のベクトル同士を比較することで、語句が完全一致しない場合でも意味の近い文書を探せます。

    次のセルでは、質問 `question` と文書 `document` の両方を `embed_query()` で埋め込み、ベクトルの次元数を表示します。既定の `text-embedding-3-small` のままなら通常1536次元ですが、環境変数 `EMBEDDING_MODEL` でモデルを変えると表示される次元数も変わります。
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
def _():
    mo.md(r"""
    OpenAIの埋め込みでは[コサイン類似度](https://platform.openai.com/docs/guides/embeddings/frequently-asked-questions)が推奨されています。ベクトルの向きが近いほど値が1に近づき、意味的な類似度が高いと判断できます。

    次のセルの `cosine_similarity(vec1, vec2)` は、2つのベクトルを受け取り、内積をそれぞれのノルムの積で割った値（-1〜1の実数）を返します。長さの影響を除いて「向き」だけを比べる指標です。
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
def _():
    mo.md(r"""
    ### 読み込み

    [ドキュメントローダー](https://python.langchain.com/docs/integrations/document_loaders/)は、Webページなどの外部データを、本文とメタデータを持つ共通の `Document` 形式へ変換します。

    ここからのパート2〜4では、パート1と同じLilian Wengのブログ記事を、同じ `SoupStrainer` の絞り込み（`post-content` / `post-title` / `post-header`）で読み込みます。出力は1件目の `Document` のメタデータで、取得元URLとページタイトルが入っています。
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
    blog_documents[0].metadata
    return (blog_documents,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 分割

    [テキスト分割](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter)では、検索とモデル入力に適した大きさへ文書を分けます。

    > このテキストスプリッターは、一般的なテキストに推奨されています。区切り文字のリストを受け取り、チャンクが十分に小さくなるまで、その区切り文字を順番に使って分割を試みます。デフォルトのリストは `['\n\n', '\n', ' ', '']` です。これにより、意味的な関連が強いと考えられる段落、文、単語を、可能な限りまとめたまま分割できます。

    次のセルはパート1と違い `from_tiktoken_encoder()` を使うため、`chunk_size` / `chunk_overlap` は**文字数ではなくトークン数**で解釈されます。ここでは300トークンごと、50トークンの重なりを持たせて分割し、チャンク数を出力します。パート1の1000文字／200文字より細かい単位です。
    """)
    return


@app.cell
def _(blog_documents):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=50
    )
    document_chunks = text_splitter.split_documents(blog_documents)
    len(document_chunks)
    return (document_chunks,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 保存

    [ベクトルストア](https://python.langchain.com/docs/integrations/vectorstores/)は、埋め込みベクトルと元のチャンクを対応付けて保存し、質問に近いチャンクを検索できるようにします。

    - 入力：`document_chunks`（300トークン単位で分割した `Document` リスト）
    - DB：Chroma。`persist_directory` を指定していないため**インメモリ**です
    - 出力：`vectorstore`。パート3の検索とパート4の生成は、このインデックスを共有します
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
def _():
    mo.md(r"""
    ## パート3：検索

    リトリーバーは質問を埋め込み、ベクトルストアから類似度の高いチャンクを返します。ここでは `k=1` とし、最も近い1件を取得して内容を確認します。
    """)
    return


@app.cell
def _(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    return (retriever,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    質問をリトリーバーへ渡して検索します。次の2セルは、検索の実行と、返ってきた `Document` の件数の確認です。`k=1` を指定しているので件数は1になります。
    """)
    return


@app.cell
def _(retriever):
    retrieved_documents = retriever.invoke("What is Task Decomposition?")
    return (retrieved_documents,)


@app.cell
def _(retrieved_documents):
    len(retrieved_documents)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    取得したチャンクの中身を確認します。質問と語句が一致していなくても、意味的に近い箇所が返っていることがわかります。
    """)
    return


@app.cell
def _(retrieved_documents):
    retrieved_documents[0].page_content
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## パート4：生成

    検索したチャンクと質問をプロンプトへ埋め込み、チャットモデルで回答を生成します。まず各要素を手動で接続し、その後に検索から出力解析までを一つのRAGチェーンへまとめます。

    ![回答生成の流れ](./imgs/generation.png)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    まずプロンプトのテンプレートを作ります。`{context}` に検索したチャンク、`{question}` に質問が埋め込まれます。「以下のコンテキストのみに基づいて答える」と指示することで、モデルの事前知識ではなく検索結果を根拠にさせます。セルの出力はテンプレートそのものの表示です。
    """)
    return


@app.cell
def _():
    generation_template = "Answer the question based only on the following context:\n{context}\n\nQuestion: {question}\n"
    generation_prompt = ChatPromptTemplate.from_template(generation_template)
    # Prompt
    generation_prompt
    return (generation_prompt,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    続く3つのセルで、生成を手動で組み立てます。

    1. `chat_model`：`make_chat_model()` でチャットモデルを用意する
    2. `generation_chain`：プロンプトとモデルを `|` でつなぐ
    3. 実行：パート3で取得した `retrieved_documents` を `format_docs` で1つの文字列にし、`context` として渡す

    戻り値は `StrOutputParser` を通していないため、文字列ではなく `AIMessage` オブジェクトです。
    """)
    return


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
def _():
    mo.md(r"""
    ### 再利用可能なRAGプロンプト

    元ノートは `hub.pull("rlm/rag-prompt")` でLangChain Hubから取得していました。このリポジトリでは外部Hubへ依存せず、かつ文面が読めるように、同じ内容を `load_rag_prompt()` としてローカルへ写しています。

    このプロンプトは、直前のパート4で手書きしたものより指示が具体的です。system側で「文脈から分からなければ分からないと答える」「3文以内で簡潔に」と制約しており、回答の長さや断り方が変わります。

    次の2セルは、プロンプトを読み込む処理と、その中身を表示する処理です。
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
def _():
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
    return (rag_chain,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### チェーンの中身を確認する

    元ノートはLangSmithでチェーンの実行内容を追跡していました。このリポジトリでは
    外部サービスを使わず、LangChain標準の `ConsoleCallbackHandler` で
    実際に送られたプロンプトと応答をその場で確認します。

    下のセルの出力に、テンプレートへ検索結果が埋め込まれた**実際の文面**が現れます。
    チェーン全体を常に詳細表示したい場合は `set_debug(True)` も使えます。
    採用理由と、Langfuse等を導入する場合の検討は `docs/observability.md` にあります。
    """)
    return


@app.cell
def _(rag_chain):
    rag_chain.invoke(
        "What is Task Decomposition?",
        config={"callbacks": [ConsoleCallbackHandler()]},
    )
    return


if __name__ == "__main__":
    app.run()
