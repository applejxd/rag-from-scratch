# /// script
# dependencies = ["langchain", "langchain-community", "langchain-openai", "langsmith", "numpy", "pytube", "tiktoken", "youtube-transcript-api"]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App()

with app.setup:
    import datetime
    import os
    from typing import Literal, Optional

    import marimo as mo
    import numpy as np
    from pydantic import BaseModel, Field

    from langchain_community.document_loaders import YoutubeLoader
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
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
    # Rag From Scratch: Routing

    ![image.png](./imgs/routing_overview.png)

    ## Environment

    `(1) Packages`
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: langchain_community tiktoken langchain-openai langsmith numpy langchain youtube-transcript-api pytube !pip install langchain_community tiktoken langchain-openai langsmith numpy langchain youtube-transcript-api pytube
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `(2) LangSmith`

    https://docs.smith.langchain.com/
    """)
    return


@app.cell
def _():
    if os.environ.get("LANGCHAIN_API_KEY"):
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part 10: Logical and Semantic routing

    Use function-calling for classification.

    Flow:

    ![Screenshot 2024-03-15 at 3.29.30 PM.png](./imgs/logical_and_semantic_routing.png)

    Docs:

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
    Note: we used function calling to produce structured output.

    ![Screenshot 2024-03-16 at 12.38.23 PM.png](./imgs/llm.with_structured_output.png)
    """)
    return


@app.cell
def _(router):
    question = """Why doesn't the following code work:

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
    prompt.invoke("french")
    """

    result = router.invoke({"question": question})
    return question, result


@app.cell
def _(result):
    result
    return


@app.cell
def _(result):
    result.datasource
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Once we have this, it is trivial to define a branch that uses `result.datasource`

    https://python.langchain.com/docs/expression_language/how_to/routing
    """)
    return


@app.cell
def _(router):
    def choose_route(result):
        if "python_docs" in result.datasource.lower():
            ### Logic here 
            return "chain for python_docs"
        elif "js_docs" in result.datasource.lower():
            ### Logic here 
            return "chain for js_docs"
        else:
            ### Logic here 
            return "golang_docs"

    full_chain = router | RunnableLambda(choose_route)
    return (full_chain,)


@app.cell
def _(full_chain, question):
    full_chain.invoke({"question": question})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Trace:

    https://smith.langchain.com/public/c2ca61b4-3810-45d0-a156-3d6a73e9ee2a/r
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Semantic routing

    Flow:

    ![Screenshot 2024-03-15 at 3.30.08 PM.png](./imgs/semantic_routing.png)

    Docs:

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

    def prompt_router(input):
        query_embedding = embeddings.embed_query(input['query'])
        similarities = [
            cosine_similarity(query_embedding, prompt_embedding)
            for prompt_embedding in prompt_embeddings
        ]
        most_similar = prompt_templates[int(np.argmax(similarities))]
        print('Using MATH' if most_similar == math_template else 'Using PHYSICS')
        return PromptTemplate.from_template(most_similar)
    chain = {'query': RunnablePassthrough()} | RunnableLambda(prompt_router) | make_chat_model() | StrOutputParser()
    # Embed prompts
    # Route question to prompt 
    print(chain.invoke("What's a black hole"))  # Embed question  # Compute similarity  # Chosen prompt 
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Trace:

    https://smith.langchain.com/public/98c25405-2631-4de8-b12a-1891aded3359/r
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Rag From Scratch: Query Construction

    ![Screenshot 2024-03-25 at 8.20.28 PM.png](./imgs/query_construction.png)

    For graph and SQL, see helpful resources:

    https://blog.langchain.dev/query-construction/

    https://blog.langchain.dev/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part 11: Query structuring for metadata filters

    Flow:

    ![Screenshot 2024-03-16 at 1.12.10 PM.png](./imgs/query_structuring.png)

    Many vectorstores contain metadata fields.

    This makes it possible to filter for specific chunks based on metadata.

    Let's look at some example metadata we might see in a database of YouTube transcripts.

    Docs:

    https://python.langchain.com/docs/use_cases/query_analysis/techniques/structuring
    """)
    return


@app.cell
def _():
    docs = YoutubeLoader.from_youtube_url(
        "https://www.youtube.com/watch?v=pbAd8O1Lvm4", add_video_info=True
    ).load()

    docs[0].metadata
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let’s assume we’ve built an index that:

    1. Allows us to perform unstructured search over the `contents` and `title` of each document
    2. And to use range filtering on `view count`, `publication date`, and `length`.

    We want to convert natural language into structured search queries.

    We can define a schema for structured search queries.
    """)
    return


@app.class_definition
class TutorialSearch(BaseModel):
    """Search over a database of tutorial videos about a software library."""
    content_search: str = Field(..., description='Similarity search query applied to video transcripts.')
    title_search: str = Field(..., description='Alternate version of the content search query to apply to video titles. Should be succinct and only include key words that could be in a video title.')
    min_view_count: Optional[int] = Field(None, description='Minimum view count filter, inclusive. Only use if explicitly specified.')
    max_view_count: Optional[int] = Field(None, description='Maximum view count filter, exclusive. Only use if explicitly specified.')
    earliest_publish_date: Optional[datetime.date] = Field(None, description='Earliest publish date filter, inclusive. Only use if explicitly specified.')
    latest_publish_date: Optional[datetime.date] = Field(None, description='Latest publish date filter, exclusive. Only use if explicitly specified.')
    min_length_sec: Optional[int] = Field(None, description='Minimum video length in seconds, inclusive. Only use if explicitly specified.')
    max_length_sec: Optional[int] = Field(None, description='Maximum video length in seconds, exclusive. Only use if explicitly specified.')

    def pretty_print(self) -> None:
        for field_name, field_info in type(self).model_fields.items():
            value = getattr(self, field_name)
            if value is not None and value != field_info.default:
                print(f'{field_name}: {value}')


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now, we prompt the LLM to produce queries.
    """)
    return


@app.cell
def _():
    system_1 = 'You are an expert at converting user questions into database queries. You have access to a database of tutorial videos about a software library for building LLM-powered applications. Given a question, return a database query optimized to retrieve the most relevant results.\n\nIf there are acronyms or words you are not familiar with, do not try to rephrase them.'
    prompt_1 = ChatPromptTemplate.from_messages([('system', system_1), ('human', '{question}')])
    llm_1 = make_chat_model()
    structured_llm_1 = llm_1.with_structured_output(TutorialSearch)
    query_analyzer = prompt_1 | structured_llm_1
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
    To then connect this to various vectorstores, you can follow [here](https://python.langchain.com/docs/modules/data_connection/retrievers/self_query#constructing-from-scratch-with-lcel).
    """)
    return


if __name__ == "__main__":
    app.run()
