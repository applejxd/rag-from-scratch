# /// script
# dependencies = ["chromadb", "langchain", "langchain-chroma", "langchain-community", "langchain-openai", "langchain-text-splitters", "langsmith", "pytube", "ragatouille", "requests", "tiktoken", "youtube-transcript-api"]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App()

with app.setup:
    import os
    import uuid

    import marimo as mo

    from langchain_core.stores import InMemoryByteStore
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.document_loaders import WebBaseLoader
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Rag From Scratch: Indexing

    ![Screenshot 2024-03-25 at 8.23.02 PM.png](./imgs/indexing_overview.png)

    ## Preface: Chunking

    We don't explicitly cover document chunking / splitting.

    For an excellent review of document chunking, see this video from Greg Kamradt:

    https://www.youtube.com/watch?v=8OJC21T2SL4

    ## Environment

    `(1) Packages`
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: langchain_community tiktoken langchain-openai langchain-chroma langchain-text-splitters langsmith chromadb langchain youtube-transcript-api pytube requests !pip install langchain_community tiktoken langchain-openai langchain-chroma langchain-text-splitters langsmith chromadb langchain youtube-transcript-api pytube requests
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
    ## Part 12: Multi-representation Indexing

    Flow:

     ![Screenshot 2024-03-16 at 5.54.55 PM.png](./imgs/multi-representation_indexing.png)

    Docs:

    https://blog.langchain.dev/semi-structured-multi-modal-rag/

    https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector

    Paper:

    https://arxiv.org/abs/2312.06648
    """)
    return


@app.cell
def _():
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    docs = loader.load()

    loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
    docs.extend(loader.load())
    return (docs,)


@app.cell
def _(docs):
    chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
        | make_chat_model()
        | StrOutputParser()
    )

    summaries = chain.batch(docs, {"max_concurrency": 5})
    return (summaries,)


@app.cell
def _(docs, summaries):
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(collection_name="summaries",
                         embedding_function=make_embeddings())

    # The storage layer for the parent documents
    store = InMemoryByteStore()
    id_key = "doc_id"

    # The retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )
    doc_ids = [str(uuid.uuid4()) for _ in docs]

    # Docs linked to summaries
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]

    # Add
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))
    return retriever, vectorstore


@app.cell
def _(vectorstore):
    query = "Memory in agents"
    sub_docs = vectorstore.similarity_search(query,k=1)
    sub_docs[0]
    return (query,)


@app.cell
def _(query, retriever):
    retrieved_docs = retriever.invoke(query)
    retrieved_docs[0].page_content[0:500]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Related idea is the [parent document retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part 13: RAPTOR

    Flow:

    ![Screenshot 2024-03-16 at 6.16.21 PM.png](./imgs/RAPTOR.png)

    Deep dive video:

    https://www.youtube.com/watch?v=jbGchdTL7d0

    Paper:

    https://arxiv.org/pdf/2401.18059.pdf

    Full code:

    https://github.com/langchain-ai/langchain/blob/master/cookbook/RAPTOR.ipynb
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part 14: ColBERT

    RAGatouille makes it as simple to use ColBERT.

    ColBERT generates a contextually influenced vector for each token in the passages.

    ColBERT similarly generates vectors for each token in the query.

    Then, the score of each document is the sum of the maximum similarity of each query embedding to any of the document embeddings:

    See [here](https://hackernoon.com/how-colbert-helps-developers-overcome-the-limits-of-rag) and [here](https://python.langchain.com/docs/integrations/retrievers/ragatouille) and [here](https://til.simonwillison.net/llms/colbert-ragatouille).
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: ragatouille !pip install -U ragatouille
    return


@app.cell
def _():
    from ragatouille import RAGPretrainedModel
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    return (RAG,)


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
        headers = {"User-Agent": "RAGatouille_tutorial/0.0.1 (ben@clavie.eu)"}

        response = requests.get(URL, params=params, headers=headers)
        data = response.json()

        # Extracting page content
        page = next(iter(data["query"]["pages"].values()))
        return page["extract"] if "extract" in page else None

    full_document = get_wikipedia_page("Hayao_Miyazaki")
    return (full_document,)


@app.cell
def _(RAG, full_document):
    RAG.index(
        collection=[full_document],
        index_name="Miyazaki-123",
        max_document_length=180,
        split_documents=True,
    )
    return


@app.cell
def _(RAG):
    results = RAG.search(query="What animation studio did Miyazaki found?", k=3)
    results
    return


@app.cell
def _(RAG):
    retriever_1 = RAG.as_langchain_retriever(k=3)
    retriever_1.invoke('What animation studio did Miyazaki found?')
    return


if __name__ == "__main__":
    app.run()
