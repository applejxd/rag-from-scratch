import marimo

__generated_with = "0.20.4"
app = marimo.App()

with app.setup:
    import os
    from operator import itemgetter

    import bs4
    import marimo as mo
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.load import dumps, loads
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import (
        ChatPromptTemplate,
        FewShotChatMessagePromptTemplate,
    )
    from langchain_core.runnables import RunnableLambda
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

    def split_queries(text: str) -> list[str]:
        """Split generated queries into lines, dropping blank ones.

        The model often separates queries with blank lines. Without this
        filtering the empty strings reach the embedding API, which rejects
        them with HTTP 400 (`expected string to have >=1 characters`).
        """
        return [line.strip() for line in text.split("\n") if line.strip()]

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
    # RAGをゼロから学ぶ：クエリ変換

    クエリ変換とは、検索に適した形へ質問を書き換えたり修正したりする一連の手法です。

    ![クエリ変換の概要](./imgs/query_overview.png)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## このノートブックの共通部品

    冒頭の `with app.setup` ブロック（marimo上では折りたたまれています）で、以下の共通部品を定義しています。パート5〜9のコードセルはこれらを繰り返し使います。

    - `make_chat_model()`：回答生成やクエリ変換に使うチャットモデル。OpenRouter経由で `openai/gpt-5-nano`（環境変数 `MODEL` で変更可）を呼びます。出力を安定させるため `temperature=0` を指定しています。
    - `make_embeddings()`：埋め込みモデル。既定は `openai/text-embedding-3-small` です。OpenRouterはOpenAI互換ですがトークナイザ情報を返さないため、クライアント側のトークン数チェックを無効化（`check_embedding_ctx_length=False` / `tiktoken_enabled=False`）しています。
    - `format_docs(docs)`：`Document` のリストを、本文を空行2つでつないだ1つの文字列へ変換します。検索結果をプロンプトの `{context}` へ埋め込むために使います。
    - `split_queries(text)`：LLMが生成した複数クエリの文字列を改行で分割し、前後の空白と空行を除いた `list[str]` にします。空行を除かないと空文字列が埋め込みAPIへ渡り、HTTP 400（`expected string to have >=1 characters`）になるためです。
    - `load_rag_prompt()`：LangChain Hubの `rlm/rag-prompt` と同じ内容のプロンプト。元ノートは実行時に `hub.pull()` で取得していましたが、外部サービスへの依存を避け、かつ文面をその場で読めるようローカルへ写しています。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## パート5：Multi Query

    一つの質問を異なる観点の複数クエリへ言い換え、それぞれの検索結果を重複排除して統合します。単一の表現だけでは取得できない関連文書を補うことが目的です。

    処理の流れ：

    ![Multi Queryの流れ](./imgs/multi_query.png)

    ドキュメント：

    * https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever

    ### インデックス
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ここから3つのコードセルで、パート5〜9が共有する単一のインデックスを作ります。まず検索対象の記事を読み込み、HTMLのうち本文・タイトル・ヘッダーに関係する要素だけを抽出します。

    - データ：Lilian Wengのブログ記事 [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
    - 前処理：`bs4.SoupStrainer` で `post-content` / `post-title` / `post-header` クラスのみを解析対象にします
    - DB：このセルではまだ作成せず、後続のコードセルで分割してChromaへ登録します
    - 出力：`blog_documents` = 読み込んだ `Document` リスト（表示される数値は1件目の本文文字数）
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
    len(blog_documents[0].page_content)
    return (blog_documents,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    読み込んだ記事を検索しやすいサイズのチャンクへ分割します。`from_tiktoken_encoder()` を使うため、`chunk_size` と `chunk_overlap` は文字数ではなくトークン数で解釈されます。

    - データ：`blog_documents`（上のセルで読み込んだLilian Wengの記事）
    - 前処理：`RecursiveCharacterTextSplitter.from_tiktoken_encoder()` / `chunk_size=300`, `chunk_overlap=50`（トークン基準）
    - DB：このセルではまだ作成せず、分割後のチャンクを次のコードセルでChromaへ登録します
    - 出力：`document_chunks` = 分割後の `Document` リスト（表示される数値はチャンク数）
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
    分割済みチャンクを埋め込みへ変換し、Chromaのベクトルストアへ登録してリトリーバーを作ります。`persist_directory` を指定していないため、このChromaはノートブック実行中だけ使うインメモリDBです。

    - データ：`document_chunks`（300トークン基準で分割した記事チャンク）
    - 前処理：`make_embeddings()` で各チャンクを埋め込みベクトルへ変換します
    - DB：Chroma。`persist_directory` なしの**インメモリ**で、パート5〜9すべてがこの単一の `retriever` を共有します
    - 出力：`retriever` = 既定設定（類似度検索・`k=4`）のリトリーバー
    """)
    return


@app.cell
def _(document_chunks):
    vectorstore = Chroma.from_documents(
        documents=document_chunks, embedding=make_embeddings()
    )
    retriever = vectorstore.as_retriever()
    return (retriever,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### プロンプト

    LLMに5種類の検索クエリを改行区切りで生成するよう指示します。このセルではプロンプトとクエリ生成チェーンを作り、`StrOutputParser` で文字列化した後、`split_queries` で `list[str]` に変換します。
    """)
    return


@app.cell
def _():
    # Multi Query: Different Perspectives
    multi_query_template = """You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by newlines. Original question: {question}"""
    multi_query_prompt = ChatPromptTemplate.from_template(multi_query_template)

    multi_query_generator = (
        multi_query_prompt
        | make_chat_model()
        | StrOutputParser()
        | split_queries
    )
    return (multi_query_generator,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    パート5とパート6で使い回す共通の評価質問を用意します。質問は `What is task decomposition for LLM agents?` で、LLMエージェントにおけるタスク分解を検索・回答できるかを見ます。
    """)
    return


@app.cell
def _():
    retrieval_question = "What is task decomposition for LLM agents?"
    return (retrieval_question,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    実際に生成されたクエリを確認します。元の質問が異なる言い回しへ展開されていることがわかります。
    """)
    return


@app.cell
def _(multi_query_generator, retrieval_question):
    generated_multi_queries = multi_query_generator.invoke(
        {"question": retrieval_question}
    )
    generated_multi_queries
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    各クエリで検索した結果を統合するチェーンを作ります。クエリ間で同じ文書が重複して取得されるため、自作関数 `get_unique_union` で重複を除きます。

    - 入力：`documents` = 複数クエリそれぞれの検索結果を持つ `list[list]`
    - 処理：各 `Document` を `dumps` で文字列化し、`set` で重複を取り除いてから `loads` で `Document` へ戻します。`Document` は直接 `set` に入れて重複判定できないため、文字列化しています。
    - 出力：重複を除いた `Document` のリスト。`multi_query_retrieval_chain` は、5クエリの生成指示、`retriever.map()` による各クエリでの検索、重複排除までをまとめたチェーンです。
    """)
    return


@app.cell
def _(multi_query_generator, retriever):
    def get_unique_union(documents: list[list]):
        """Return unique documents from multiple retrieval result lists."""
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs = list(set(flattened_docs))
        return [loads(doc) for doc in unique_docs]

    multi_query_retrieval_chain = (
        multi_query_generator | retriever.map() | get_unique_union
    )
    return (multi_query_retrieval_chain,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    共通の評価質問を `multi_query_retrieval_chain` に渡し、5クエリ分の検索結果を重複排除した後の件数を確認します。ここで表示される数値は、最終回答に渡す候補文書の数です。
    """)
    return


@app.cell
def _(multi_query_retrieval_chain, retrieval_question):
    multi_query_documents = multi_query_retrieval_chain.invoke(
        {"question": retrieval_question}
    )
    len(multi_query_documents)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    統合した文書をコンテキストとして最終的な回答を生成します。
    """)
    return


@app.cell
def _(multi_query_retrieval_chain, retrieval_question):
    answer_prompt = load_rag_prompt()
    multi_query_answer_chain = (
        {
            "context": multi_query_retrieval_chain,
            "question": itemgetter("question"),
        }
        | answer_prompt
        | make_chat_model()
        | StrOutputParser()
    )
    multi_query_answer_chain.invoke({"question": retrieval_question})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## パート6：RAG-Fusion

    RAG-Fusionも複数クエリで検索しますが、結果を単純に結合せず、各ランキングで上位に現れる文書へ高いスコアを与えるReciprocal Rank Fusion（RRF）で再順位付けします。検索システムごとの生スコアを比較せず、順位だけで統合できる点が特徴です。

    処理の流れ：

    ![RAG-Fusionの流れ](./imgs/rag_fusion.png)

    ドキュメント：

    * https://github.com/langchain-ai/langchain/blob/master/cookbook/rag_fusion.ipynb?ref=blog.langchain.dev

    ブログ／リポジトリ：

    * https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1

    ### プロンプト
    """)
    return


@app.cell
def _():
    rag_fusion_template = 'You are a helpful assistant that generates multiple search queries based on a single input query. \n\nGenerate multiple search queries related to: {question} \n\nOutput (4 queries):'
    # RAG-Fusion: Related
    rag_fusion_prompt = ChatPromptTemplate.from_template(rag_fusion_template)
    return (rag_fusion_prompt,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    RAG-Fusion用のクエリ生成チェーンを組み立てます。パート5のMulti Queryは5種類の別表現を作るプロンプトでしたが、ここでは関連する検索クエリを4件生成するよう指示しています。出力は改行区切りの文字列なので、最後に `split_queries` で空行を除いたリストへ変換します。
    """)
    return


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
    Reciprocal Rank Fusion（RRF）は、各検索結果での0始まりの順位 `rank` だけを使い、`1 / (rank + k)` を文書ごとに合計します。検索システムごとにスケールが異なる生スコアを比較せずに済むのが利点です。

    - 入力：`results` = 複数クエリそれぞれの検索結果を持つ `list[list]`
    - 処理：`dumps` で `Document` を文字列化して文書ごとの合計スコアを集計します。定数は `k=60` で、最上位文書の加点は `1 / (0 + 60)` です
    - 出力：`(Document, score)` のタプルのリスト。通常のリトリーバーが返す `Document` のリストとは型が違う点に注意します。
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
    共通の評価質問をRAG-Fusionの検索チェーンに渡し、RRFで融合・再順位付けした後の件数を確認します。出力の `fused_documents` は、文書と融合スコアのタプルをスコア降順に並べたリストです。
    """)
    return


@app.cell
def _(rag_fusion_retrieval_chain, retrieval_question):
    fused_documents = rag_fusion_retrieval_chain.invoke(
        {"question": retrieval_question}
    )
    len(fused_documents)
    return (fused_documents,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    上位の融合スコアを確認します。複数のクエリで繰り返し上位に現れた文書ほど高い値になります。
    """)
    return


@app.cell
def _(fused_documents):
    [round(score, 4) for _, score in fused_documents[:5]]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    RRFで再順位付けした結果をコンテキストとして回答を生成します。このチェーンでは `rag_fusion_retrieval_chain` が返す `(Document, score)` のタプル列をそのまま `context` へ渡しており、`format_docs` は通していません。最後に `StrOutputParser` を通すため、戻り値は文字列です。
    """)
    return


@app.cell
def _(rag_fusion_retrieval_chain, retrieval_question):
    rag_fusion_answer_chain = (
        {
            "context": rag_fusion_retrieval_chain,
            "question": itemgetter("question"),
        }
        | load_rag_prompt()
        | make_chat_model()
        | StrOutputParser()
    )
    rag_fusion_answer_chain.invoke({"question": retrieval_question})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    トレース：

    https://smith.langchain.com/public/071202c9-9f4d-41b1-bf9d-86b7c5a7525b/r
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## パート7：分解

    複雑な質問を、単独で検索・回答できる複数のサブ質問へ分解します。ここでは、前の回答を次の質問の背景として順に渡す方法と、各サブ質問を独立に回答して最後に統合する方法を比較します。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    次の3つのコードセルで、サブ質問を生成するプロンプトとチェーンを作り、パート7で扱う元の質問を用意します。`subquestion_generator` は入力された質問を3つのサブ質問へ分解するよう指示し、`StrOutputParser` で文字列化した後、`split_queries` で空行を除いたリストにします。ここでの元質問は、LLM自律エージェントシステムの主要コンポーネントを問う内容です。
    """)
    return


@app.cell
def _():
    subquestion_template = 'You are a helpful assistant that generates multiple sub-questions related to an input question. \n\nThe goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n\nGenerate multiple search queries related to: {question} \n\nOutput (3 queries):'
    # Decomposition
    subquestion_prompt = ChatPromptTemplate.from_template(subquestion_template)
    return (subquestion_prompt,)


@app.cell
def _(subquestion_prompt):
    subquestion_generator = (
        subquestion_prompt
        | make_chat_model()
        | StrOutputParser()
        | split_queries
    )
    return (subquestion_generator,)


@app.cell
def _():
    decomposition_question = (
        "What are the main components of an LLM-powered autonomous agent system?"
    )
    return (decomposition_question,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    生成されたサブ質問を確認します。元の質問が、単独で検索・回答できる粒度へ分割されます。
    """)
    return


@app.cell
def _(decomposition_question, subquestion_generator):
    subquestions = subquestion_generator.invoke(
        {"question": decomposition_question}
    )
    subquestions
    return (subquestions,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 再帰的に回答する

    サブ質問を順番に処理し、それまでに得た質問と回答の組を次のプロンプトへ追加します。後続の回答が先行する回答を参照できる一方、処理は直列になります。

    ![再帰的に回答する流れ](./imgs/answer_recursively.png)

    論文：

    * https://arxiv.org/pdf/2205.10625.pdf
    * https://arxiv.org/abs/2212.10509.pdf
    """)
    return


@app.cell
def _():
    recursive_answer_template = 'Here is the question you need to answer:\n\n\n --- \n {question} \n --- \n\n\nHere is any available background question + answer pairs:\n\n\n --- \n {q_a_pairs} \n --- \n\n\nHere is additional context relevant to the question: \n\n\n --- \n {context} \n --- \n\n\nUse the above context and any background question + answer pairs to answer the question: \n {question}\n'
    recursive_answer_prompt = ChatPromptTemplate.from_template(
        recursive_answer_template
    )
    return (recursive_answer_prompt,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    サブ質問に答えるチェーンを組み立てます。`q_a_pairs` に、それまでに得た質問と回答の組を渡す点が通常のRAGチェーンとの違いです。最後に `StrOutputParser` を通すため、各回答は文字列になります。
    """)
    return


@app.cell
def _(recursive_answer_prompt, retriever):
    decomposition_model = make_chat_model()
    recursive_rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "q_a_pairs": itemgetter("q_a_pairs"),
        }
        | recursive_answer_prompt
        | decomposition_model
        | StrOutputParser()
    )
    return decomposition_model, recursive_rag_chain


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    サブ質問を順に処理し、回答を `accumulated_qa` へ積み上げます。`accumulated_qa` は反復のたびに成長し、各回答が次の質問の背景として渡るため、処理は直列になります。

    - 入力：`subquestions` と、直前で作った `recursive_rag_chain`
    - 処理：`format_qa_pair(question, answer)` で1組の質問と回答を整形し、各サブ質問の回答後に `accumulated_qa` へ追記します
    - 出力：`accumulated_qa` = すべての質問・回答ペアを連結した文字列、`recursive_answer` = 最後のサブ質問への回答。この方式では最後のサブ質問への回答がそのまま最終出力になります。
    """)
    return


@app.cell
def _(recursive_rag_chain, subquestions):
    def format_qa_pair(question, answer):
        """Format Q and A pair"""
        return f"Question: {question}\nAnswer: {answer}".strip()

    accumulated_qa = ""
    recursive_answer = ""
    for subquestion in subquestions:
        recursive_answer = recursive_rag_chain.invoke(
            {"question": subquestion, "q_a_pairs": accumulated_qa}
        )
        accumulated_qa += (
            "\n---\n" + format_qa_pair(subquestion, recursive_answer)
        )
    return accumulated_qa, recursive_answer


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    次の2セルは、再帰的に作った結果の確認です。まず `print(accumulated_qa)` で整形済みのQ&Aコンテキスト全体を表示し、その後 `recursive_answer` として最後のサブ質問への回答を表示します。
    """)
    return


@app.cell
def _(accumulated_qa):
    print(accumulated_qa)
    return


@app.cell
def _(recursive_answer):
    recursive_answer
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    トレース：

    質問1：https://smith.langchain.com/public/faefde73-0ecb-4328-8fee-a237904115c0/r

    質問2：https://smith.langchain.com/public/6142cad3-b314-454e-b2c9-15146cfcce78/r

    質問3：https://smith.langchain.com/public/84bdca0f-0fa4-46d4-9f89-a7f25bd857fe/r
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 個別に回答する

    各サブ質問を互いに独立して検索・回答するため、並列化しやすい方法です。最後に質問と回答の組を一つのコンテキストへまとめ、元の質問に対する回答を合成します。

    ![個別に回答する流れ](./imgs/answer_individualy.png)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `retrieve_and_answer_subquestions` は、分解したサブ質問を個別に検索・回答する自作関数です。再帰的な方式と違い、各サブ質問の回答時に他の回答を背景として渡しません。

    - 入力：`question` = 分解したい元の質問文字列
    - 処理：`subquestion_generator` でサブ質問を作り、各サブ質問について共有リトリーバーで検索し、`load_rag_prompt()` と `decomposition_model` で回答します
    - 出力：`(answers, generated_subquestions)` = 回答リストとサブ質問リストのタプル。セルではパート7の質問に対して実行し、回答件数を確認します。
    """)
    return


@app.cell
def _(decomposition_model, decomposition_question, retriever, subquestion_generator):
    def retrieve_and_answer_subquestions(question):
        generated_subquestions = subquestion_generator.invoke({"question": question})
        answers = []
        for subquestion in generated_subquestions:
            retrieved_documents = retriever.invoke(subquestion)
            answer = (
                load_rag_prompt() | decomposition_model | StrOutputParser()
            ).invoke(
                {
                    "context": format_docs(retrieved_documents),
                    "question": subquestion,
                }
            )
            answers.append(answer)
        return answers, generated_subquestions

    individual_answers, individual_subquestions = (
        retrieve_and_answer_subquestions(decomposition_question)
    )
    len(individual_answers)
    return individual_answers, individual_subquestions


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    サブ質問と回答を1つのコンテキストへ整形します。再帰的な方式と違い、各回答は互いに独立して得られたものです。

    - 入力：`questions` = サブ質問リスト、`answers` = 対応する回答リスト
    - 処理：`format_qa_pairs` が番号付きの `Question i` / `Answer i` 形式へ連結します
    - 出力：`qa_context` = 元の質問への回答合成に渡すQ&Aコンテキスト。次のセルの `print(qa_context)` で、LLMへ渡す前の整形結果を確認します。
    """)
    return


@app.cell
def _(individual_answers, individual_subquestions):
    def format_qa_pairs(questions, answers):
        """Format Q and A pairs"""
        formatted_string = ""
        for i, (question, answer) in enumerate(zip(questions, answers), start=1):
            formatted_string += (
                f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
            )
        return formatted_string.strip()

    qa_context = format_qa_pairs(individual_subquestions, individual_answers)
    return (qa_context,)


@app.cell
def _(qa_context):
    print(qa_context)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    整形したコンテキストをもとに、元の質問への回答を合成します。
    """)
    return


@app.cell
def _(decomposition_model, decomposition_question, qa_context):
    synthesis_template = 'Here is a set of Q+A pairs:\n\n{context}\n\nUse these to synthesize an answer to the question: {question}\n'
    synthesis_prompt = ChatPromptTemplate.from_template(synthesis_template)
    synthesis_chain = synthesis_prompt | decomposition_model | StrOutputParser()
    synthesis_chain.invoke(
        {"context": qa_context, "question": decomposition_question}
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    トレース：

    https://smith.langchain.com/public/d8f26f75-3fb8-498a-a3a2-6532aa77f56b/r
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## パート8：Step Back

    元の質問を、より一般的で答えやすい「一歩引いた質問」へ言い換えます。元の質問と一般化した質問の両方で検索し、具体的な文脈と背景知識を合わせて回答します。

    ![Step Backの流れ](./imgs/step_back.png)

    論文：

    * https://arxiv.org/pdf/2310.06117.pdf
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    2つのFew-shot例を使い、具体的な質問をより一般的な「一歩引いた質問」へ言い換える方法をLLMへ示します。`Could the members of The Police perform lawful arrests?` を能力一般の問いへ、`Jan Sindel’s was born in what country?` を人物史の問いへ変換する例として渡します。
    """)
    return


@app.cell
def _():
    examples = [
        {
            "input": "Could the members of The Police perform lawful arrests?",
            "output": "what can the members of The Police do?",
        },
        {
            "input": "Jan Sindel’s was born in what country?",
            "output": "what is Jan Sindel’s personal history?",
        },
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt, examples=examples
    )
    step_back_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:",
            ),
            few_shot_prompt,
            ("user", "{question}"),
        ]
    )
    return (step_back_prompt,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Few-shot付きプロンプトをチャットモデルにつなぎ、一歩引いた質問を生成するチェーンを作ります。出力は検索に渡す一般化された質問文字列です。
    """)
    return


@app.cell
def _(step_back_prompt):
    step_back_query_generator = (
        step_back_prompt | make_chat_model() | StrOutputParser()
    )
    return (step_back_query_generator,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    パート8で使う質問を用意します。内容はパート5・6と同じ `What is task decomposition for LLM agents?` で、通常の検索と一歩引いた質問による検索を比較しやすくしています。
    """)
    return


@app.cell
def _():
    step_back_question = "What is task decomposition for LLM agents?"
    return (step_back_question,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    実際に生成された一歩引いた質問を確認します。元の質問より一般的な問いへ言い換えられます。
    """)
    return


@app.cell
def _(step_back_query_generator, step_back_question):
    step_back_query_generator.invoke({"question": step_back_question})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    元の質問と一歩引いた質問の両方で検索し、`normal_context` と `step_back_context` の2系統をプロンプトへ渡します。
    """)
    return


@app.cell
def _(retriever, step_back_query_generator, step_back_question):
    response_prompt_template = 'You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.\n\n# {normal_context}\n# {step_back_context}\n\n# Original Question: {question}\n# Answer:'
    response_prompt = ChatPromptTemplate.from_template(response_prompt_template)
    step_back_answer_chain = (
        {
            "normal_context": RunnableLambda(lambda x: x["question"])
            | retriever
            | format_docs,
            "step_back_context": step_back_query_generator
            | retriever
            | format_docs,
            "question": lambda x: x["question"],
        }
        | response_prompt
        | make_chat_model()
        | StrOutputParser()
    )
    step_back_answer_chain.invoke({"question": step_back_question})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## パート9：HyDE

    HyDEは、質問へ直接回答する仮想文書を先にLLMで生成し、その文書の埋め込みを検索クエリとして使います。短い質問よりも実文書に近い表現で検索できるため、質問と文書の表現差を埋められる場合があります。

    ![HyDEの流れ](./imgs/HyDE.png)

    ドキュメント：

    * https://github.com/langchain-ai/langchain/blob/master/cookbook/hypothetical_document_embeddings.ipynb

    論文：

    * https://arxiv.org/abs/2212.10496
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    HyDEの仮想文書生成チェーンを作ります。プロンプトでは、質問そのものを検索するのではなく「質問に答える科学論文の一節を書け」と指示し、その生成文を後続の検索クエリとして使います。
    """)
    return


@app.cell
def _():
    hyde_template = 'Please write a scientific paper passage to answer the question\nQuestion: {question}\nPassage:'
    hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
    hypothetical_document_generator = (
        hyde_prompt | make_chat_model() | StrOutputParser()
    )
    return (hypothetical_document_generator,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    パート9で使う質問を用意します。質問は `What is task decomposition for LLM agents?` で、タスク分解に関する仮想文書を生成してから検索する流れを確認します。
    """)
    return


@app.cell
def _():
    hyde_question = "What is task decomposition for LLM agents?"
    return (hyde_question,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    生成された仮想文書を確認します。質問そのものではなく、論文の一節のような文章が生成されます。この文章の埋め込みを検索クエリとして使います。
    """)
    return


@app.cell
def _(hyde_question, hypothetical_document_generator):
    hypothetical_document_generator.invoke({"question": hyde_question})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    仮想文書生成チェーンと共有リトリーバーを直列につなぎます。`hyde_retrieval_chain` は、LLMが生成した仮想文書の文字列をそのまま検索クエリとして `retriever` へ渡し、関連する `Document` を取得します。
    """)
    return


@app.cell
def _(hyde_question, hypothetical_document_generator, retriever):
    hyde_retrieval_chain = hypothetical_document_generator | retriever
    hyde_documents = hyde_retrieval_chain.invoke({"question": hyde_question})
    hyde_documents
    return (hyde_documents,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    HyDEで取得した文書を使って回答を生成します。検索用の `hyde_retrieval_chain` と回答生成用の `hyde_answer_chain` は別チェーンで、検索結果 `hyde_documents` を `format_docs` で整形して `context` へ手動で渡しています。
    """)
    return


@app.cell
def _(hyde_documents, hyde_question):
    hyde_answer_chain = load_rag_prompt() | make_chat_model() | StrOutputParser()
    hyde_answer_chain.invoke(
        {"context": format_docs(hyde_documents), "question": hyde_question}
    )
    return


if __name__ == "__main__":
    app.run()
