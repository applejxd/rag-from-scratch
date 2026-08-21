import marimo

__generated_with = "0.20.4"
app = marimo.App()

with app.setup:
    import datetime
    import os
    from typing import Literal

    import marimo as mo
    import numpy as np
    from langchain_community.document_loaders import YoutubeLoader
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from pydantic import BaseModel, Field

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

    def cosine_similarity(vec1, vec2):
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # RAGをゼロから学ぶ：ルーティング

    ![ルーティングの概要](./imgs/routing_overview.png)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## パート10：論理ルーティングとセマンティックルーティング

    ルーティングは、質問に応じて使用するデータソースやプロンプトを切り替える処理です。論理ルーティングではLLMの構造化出力で離散的な行き先を選び、セマンティックルーティングでは埋め込みの類似度で最も近い行き先を選びます。

    まず、Function Callingを使って質問をプログラミング言語別のデータソースへ分類します。

    処理の流れ：

    ![論理ルーティングとセマンティックルーティング](./imgs/logical_and_semantic_routing.png)

    ドキュメント：

    https://python.langchain.com/docs/use_cases/query_analysis/techniques/routing#routing-to-multiple-indexes
    """)
    return


@app.cell
def _():
    # Data model
    class RouteQuery(BaseModel):
        """Route a user query to the most relevant datasource."""

        datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
            ...,
            description="Given a user question choose which datasource would be most relevant for answering their question",
        )

    # LLM with function call 
    llm = make_chat_model()
    structured_llm = llm.with_structured_output(RouteQuery)

    # Prompt 
    system = """You are an expert at routing a user question to the appropriate data source.

    Based on the programming language the question is referring to, route it to the relevant data source."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    # Define router 
    router = prompt | structured_llm
    return (router,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `with_structured_output(RouteQuery)` により、自由文ではなく `datasource` フィールドを持つ値として結果を受け取ります。候補を型で限定すると、後続の分岐を決定的に記述できます。

    ![LLMによる構造化出力](./imgs/llm.with_structured_output.png)
    """)
    return


@app.cell
def _(router):
    routing_question = """Why doesn't the following code work:

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
    prompt.invoke("french")
    """

    route_result = router.invoke({"question": routing_question})
    return route_result, routing_question


@app.cell
def _(route_result):
    route_result
    return


@app.cell
def _(route_result):
    route_result.datasource
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    選択された `route_result.datasource` を使い、対応する検索チェーンへ処理を振り分けます。この例では文字列を返していますが、実際には各データソース用のRAGチェーンを接続します。

    https://python.langchain.com/docs/expression_language/how_to/routing
    """)
    return


@app.cell
def _(router):
    def choose_route(route):
        if "python_docs" in route.datasource.lower():
            return "chain for python_docs"
        elif "js_docs" in route.datasource.lower():
            return "chain for js_docs"
        return "chain for golang_docs"

    logical_routing_chain = router | RunnableLambda(choose_route)
    return (logical_routing_chain,)


@app.cell
def _(logical_routing_chain, routing_question):
    logical_routing_chain.invoke({"question": routing_question})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    トレース：

    https://smith.langchain.com/public/c2ca61b4-3810-45d0-a156-3d6a73e9ee2a/r
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### セマンティックルーティング

    候補プロンプトを事前に埋め込み、質問の埋め込みとのコサイン類似度が最も高いプロンプトを選びます。固定ラベルへ分類するより柔軟ですが、候補間の意味が近い場合は境界が曖昧になります。

    処理の流れ：

    ![セマンティックルーティングの流れ](./imgs/semantic_routing.png)

    ドキュメント：

    https://python.langchain.com/docs/expression_language/cookbook/embedding_router
    """)
    return


@app.cell
def _():
    physics_template = "You are a very smart physics professor. You are great at answering questions about physics in a concise and easy to understand manner. When you don't know the answer to a question you admit that you don't know.\n\nHere is a question:\n{query}"
    # Two prompts
    math_template = 'You are a very good mathematician. You are great at answering math questions. You are so good because you are able to break down hard problems into their component parts, answer the component parts, and then put them together to answer the broader question.\n\nHere is a question:\n{query}'
    embeddings = make_embeddings()
    prompt_templates = [physics_template, math_template]
    prompt_embeddings = embeddings.embed_documents(prompt_templates)

    def prompt_router(inputs):
        query_embedding = embeddings.embed_query(inputs["query"])
        similarities = [
            cosine_similarity(query_embedding, prompt_embedding)
            for prompt_embedding in prompt_embeddings
        ]
        most_similar = prompt_templates[int(np.argmax(similarities))]
        print("Using MATH" if most_similar == math_template else "Using PHYSICS")
        return PromptTemplate.from_template(most_similar)

    semantic_routing_chain = (
        {"query": RunnablePassthrough()}
        | RunnableLambda(prompt_router)
        | make_chat_model()
        | StrOutputParser()
    )
    print(semantic_routing_chain.invoke("What's a black hole"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    トレース：

    https://smith.langchain.com/public/98c25405-2631-4de8-b12a-1891aded3359/r
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # RAGをゼロから学ぶ：クエリ構築

    ![クエリ構築の概要](./imgs/query_construction.png)

    クエリ構築では、自然言語の質問を検索システムが解釈できる構造へ変換します。ここではメタデータフィルターを扱い、グラフやSQLについては以下の参考資料を参照します。

    https://blog.langchain.dev/query-construction/

    https://blog.langchain.dev/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## パート11：メタデータフィルター向けのクエリ構造化

    処理の流れ：

    ![クエリ構造化の流れ](./imgs/query_structuring.png)

    ベクトル類似度だけで検索すると、公開日や動画の長さといった条件を正確に扱えません。LLMで質問を意味検索用テキストとメタデータ条件へ分離し、ベクトル検索とフィルターを組み合わせます。まず、YouTubeの文字起こしに付随するメタデータを確認します。

    ドキュメント：

    https://python.langchain.com/docs/use_cases/query_analysis/techniques/structuring
    """)
    return


@app.cell
def _():
    transcript_documents = YoutubeLoader.from_youtube_url(
        "https://www.youtube.com/watch?v=pbAd8O1Lvm4", add_video_info=False
    ).load()

    transcript_documents[0].metadata
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    次の機能を持つインデックスを構築したと仮定します。

    1. 各ドキュメントの `contents` と `title` を対象に非構造化検索ができる
    2. `view count`、`publication date`、`length` を範囲指定で絞り込める

    `TutorialSearch` スキーマは、意味検索に使う文字列と範囲フィルターを分離します。LLMは質問に明示された条件だけを各フィールドへ設定します。
    """)
    return


@app.class_definition
class TutorialSearch(BaseModel):
    """Search over a database of tutorial videos about a software library."""
    content_search: str = Field(..., description='Similarity search query applied to video transcripts.')
    title_search: str = Field(..., description='Alternate version of the content search query to apply to video titles. Should be succinct and only include key words that could be in a video title.')
    min_view_count: int | None = Field(None, description='Minimum view count filter, inclusive. Only use if explicitly specified.')
    max_view_count: int | None = Field(None, description='Maximum view count filter, exclusive. Only use if explicitly specified.')
    earliest_publish_date: datetime.date | None = Field(None, description='Earliest publish date filter, inclusive. Only use if explicitly specified.')
    latest_publish_date: datetime.date | None = Field(None, description='Latest publish date filter, exclusive. Only use if explicitly specified.')
    min_length_sec: int | None = Field(None, description='Minimum video length in seconds, inclusive. Only use if explicitly specified.')
    max_length_sec: int | None = Field(None, description='Maximum video length in seconds, exclusive. Only use if explicitly specified.')

    def pretty_print(self) -> None:
        for field_name, field_info in type(self).model_fields.items():
            value = getattr(self, field_name)
            if value is not None and value != field_info.default:
                print(f'{field_name}: {value}')


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    次に、`TutorialSearch` を構造化出力として指定し、自然言語から検索条件を生成します。以下の例では、質問が具体的になるにつれて公開年や動画時間のフィールドが追加されます。
    """)
    return


@app.cell
def _():
    query_system_prompt = 'You are an expert at converting user questions into database queries. You have access to a database of tutorial videos about a software library for building LLM-powered applications. Given a question, return a database query optimized to retrieve the most relevant results.\n\nIf there are acronyms or words you are not familiar with, do not try to rephrase them.'
    query_prompt = ChatPromptTemplate.from_messages(
        [("system", query_system_prompt), ("human", "{question}")]
    )
    structured_query_model = make_chat_model().with_structured_output(TutorialSearch)
    query_analyzer = query_prompt | structured_query_model
    return (query_analyzer,)


@app.cell
def _(query_analyzer):
    query_analyzer.invoke({"question": "rag from scratch"}).pretty_print()
    return


@app.cell
def _(query_analyzer):
    query_analyzer.invoke(
        {"question": "videos on chat langchain published in 2023"}
    ).pretty_print()
    return


@app.cell
def _(query_analyzer):
    query_analyzer.invoke(
        {"question": "videos that are focused on the topic of chat langchain that are published before 2024"}
    ).pretty_print()
    return


@app.cell
def _(query_analyzer):
    query_analyzer.invoke(
        {
            "question": "how to use multi-modal models in an agent, only videos under 5 minutes"
        }
    ).pretty_print()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    これを各種ベクトルストアへ接続する方法は、[こちら](https://python.langchain.com/docs/modules/data_connection/retrievers/self_query#constructing-from-scratch-with-lcel)を参照してください。
    """)
    return


if __name__ == "__main__":
    app.run()
