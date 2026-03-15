# /// script
# dependencies = [
#     "chromadb",
#     "langchain",
#     "langchain-community",
#     "langchain-openai",
#     "marimo>=0.20.2",
#     "pyzmq>=27.1.0",
#     "tiktoken",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App()

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import os
    import bs4

    from langsmith import Client

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_chroma import Chroma
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    import tiktoken
    import numpy as np

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
        if os.environ.get("LANGCHAIN_API_KEY"):
            return Client().pull_prompt("rlm/rag-prompt")

        template = (
            "Answer the question based only on the following context:\n"
            "{context}\n\nQuestion: {question}\n"
        )
        return ChatPromptTemplate.from_template(template)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Rag From Scratch: Overview

    These notebooks walk through the process of building RAG app(s) from scratch.

    They will build towards a broader understanding of the RAG langscape, as shown here:

    ![Screenshot 2024-03-25 at 8.30.33 PM.png](./imgs/overview.png)

    ## Environment

    `(1) Packages`
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: langchain_community tiktoken langchain-openai chromadb langchain !pip install langchain_community tiktoken langchain-openai chromadb langchain
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
    ## Part 1: Overview

    [RAG quickstart](https://python.langchain.com/docs/use_cases/question_answering/quickstart)
    """)
    return


@app.cell
def _():
    #### INDEXING ####
    # Load Documents
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embed
    vectorstore = Chroma.from_documents(documents=splits, embedding=make_embeddings())

    retriever = vectorstore.as_retriever()

    #### RETRIEVAL and GENERATION ####

    # Prompt
    prompt = load_rag_prompt()

    # LLM
    llm = make_chat_model()

    # Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Question
    rag_chain.invoke("What is Task Decomposition?")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part 2: Indexing

    ![Screenshot 2024-02-12 at 1.36.56 PM.png](./imgs/indexing.png)
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
    [Count tokens](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb) considering [~4 char / token](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)
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
    [Text embedding models](https://python.langchain.com/docs/integrations/text_embedding/openai)
    """)
    return


@app.cell
def _(document, question):
    embd = make_embeddings()

    query_result = embd.embed_query(question)
    document_result = embd.embed_query(document)
    len(query_result)
    return document_result, query_result


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [Cosine similarity](https://platform.openai.com/docs/guides/embeddings/frequently-asked-questions) is recommended (1 indicates identical) for OpenAI embeddings.
    """)
    return


@app.cell
def _(document_result, query_result):
    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    similarity = cosine_similarity(query_result, document_result)
    print("Cosine Similarity:", similarity)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/)
    """)
    return


@app.cell
def _():
    #### INDEXING ####
    loader_1 = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    # Load blog
    blog_docs = loader_1.load()
    return (blog_docs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [Splitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter)

    > This text splitter is the recommended one for generic text. It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", "\"]. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.
    """)
    return


@app.cell
def _(blog_docs):
    # Split
    text_splitter_1 = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=50
    )
    # Make splits
    splits_1 = text_splitter_1.split_documents(blog_docs)
    return (splits_1,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [Vectorstores](https://python.langchain.com/docs/integrations/vectorstores/)
    """)
    return


@app.cell
def _(splits_1):
    # Index
    vectorstore_1 = Chroma.from_documents(
        documents=splits_1, embedding=make_embeddings()
    )
    retriever_1 = vectorstore_1.as_retriever()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part 3: Retrieval
    """)
    return


@app.cell
def _(splits_1):
    # Index
    vectorstore_2 = Chroma.from_documents(
        documents=splits_1, embedding=make_embeddings()
    )
    retriever_2 = vectorstore_2.as_retriever(search_kwargs={"k": 1})
    return (retriever_2,)


@app.cell
def _(retriever_2):
    docs_1 = retriever_2.invoke("What is Task Decomposition?")
    return (docs_1,)


@app.cell
def _(docs_1):
    len(docs_1)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part 4: Generation

    ![Screenshot 2024-02-12 at 1.37.38 PM.png](./imgs/generation.png)
    """)
    return


@app.cell
def _():
    template = "Answer the question based only on the following context:\n{context}\n\nQuestion: {question}\n"
    prompt_1 = ChatPromptTemplate.from_template(template)
    # Prompt
    prompt_1
    return (prompt_1,)


@app.cell
def _():
    # LLM
    llm_1 = make_chat_model()
    return (llm_1,)


@app.cell
def _(llm_1, prompt_1):
    # Chain
    chain = prompt_1 | llm_1
    return (chain,)


@app.cell
def _(chain, docs_1):
    # Run
    chain.invoke(
        {"context": format_docs(docs_1), "question": "What is Task Decomposition?"}
    )
    return


@app.cell
def _():
    prompt_hub_rag = load_rag_prompt()
    return (prompt_hub_rag,)


@app.cell
def _(prompt_hub_rag):
    prompt_hub_rag
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [RAG chains](https://python.langchain.com/docs/expression_language/get_started#rag-search-example)
    """)
    return


@app.cell
def _(llm_1, prompt_1, retriever_2):
    rag_chain_1 = (
        {"context": retriever_2 | format_docs, "question": RunnablePassthrough()}
        | prompt_1
        | llm_1
        | StrOutputParser()
    )
    rag_chain_1.invoke("What is Task Decomposition?")
    return


if __name__ == "__main__":
    app.run()
