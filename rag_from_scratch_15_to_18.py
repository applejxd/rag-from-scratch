# /// script
# dependencies = ["chromadb", "cohere", "langchain", "langchain-chroma", "langchain-classic", "langchain-community", "langchain-openai", "langchain-text-splitters", "langsmith", "tiktoken"]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App()

with app.setup:
    import os
    import bs4

    import marimo as mo

    from langchain_chroma import Chroma
    from langchain_core.load import dumps, loads
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

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Rag From Scratch: Retrieval

    ![Screenshot 2024-03-25 at 8.23.58 PM.png](./imgs/retrieval_overview.png)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Environment

    `(1) Packages`
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: langchain_community tiktoken langchain-openai langchain-chroma langchain-classic langchain-text-splitters langsmith chromadb langchain cohere !pip install langchain_community tiktoken langchain-openai langchain-chroma langchain-classic langchain-text-splitters langsmith chromadb langchain cohere
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
    ## Part 15: Re-ranking

    We showed this previously with RAG-fusion.

    ![Screenshot 2024-03-25 at 2.59.21 PM.png](./imgs/re-ranking.png)
    """)
    return


@app.cell
def _():
    #### INDEXING ####

    # Load blog
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    blog_docs = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50,
    )

    # Make splits
    splits = text_splitter.split_documents(blog_docs)

    # Index
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=make_embeddings(),
    )


    retriever = vectorstore.as_retriever()
    return retriever, vectorstore


@app.cell
def _():
    # RAG-Fusion
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    return (prompt_rag_fusion,)


@app.cell
def _(prompt_rag_fusion):
    generate_queries = (
        prompt_rag_fusion
        | make_chat_model()
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )
    return (generate_queries,)


@app.cell
def _(generate_queries, retriever):
    def reciprocal_rank_fusion(results: list[list], k=60):
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
            and an optional parameter k used in the RRF formula """
        fused_scores = {}
        for docs in results:  # Initialize a dictionary to hold fused scores for each unique document
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:  # Iterate through each list of ranked documents
                    fused_scores[doc_str] = 0
                previous_score = fused_scores[doc_str]  # Iterate through each document in the list, with its rank (position in the list)
                fused_scores[doc_str] = fused_scores[doc_str] + 1 / (rank + k)
        reranked_results = [(loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]  # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
        return reranked_results
    question = 'What is task decomposition for LLM agents?'  # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
    docs = retrieval_chain_rag_fusion.invoke({'question': question})
    len(docs)  # Retrieve the current score of the document, if any  # Update the score of the document using the RRF formula: 1 / (rank + k)  # Sort the documents based on their fused scores in descending order to get the final reranked results  # Return the reranked results as a list of tuples, each containing the document and its fused score
    return question, retrieval_chain_rag_fusion


@app.cell
def _(question, retrieval_chain_rag_fusion):
    from operator import itemgetter
    template_1 = 'Answer the following question based on this context:\n\n{context}\n\nQuestion: {question}\n'
    # RAG
    prompt = ChatPromptTemplate.from_template(template_1)
    llm = make_chat_model()
    final_rag_chain = {'context': retrieval_chain_rag_fusion, 'question': itemgetter('question')} | prompt | llm | StrOutputParser()
    final_rag_chain.invoke({'question': question})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can also use [Cohere Re-Rank](https://python.langchain.com/docs/integrations/retrievers/cohere-reranker#doing-reranking-with-coherererank).

    See [here](https://txt.cohere.com/rerank/):

    ![data-src-image-387e0861-93de-4823-84e0-7ae04f2be893.png](./imgs/Cohere_Re-Rank.png)
    """)
    return


@app.cell
def _():
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import CohereRerank

    return CohereRerank, ContextualCompressionRetriever


@app.cell
def _(CohereRerank, ContextualCompressionRetriever, question, vectorstore):
    retriever_1 = vectorstore.as_retriever(search_kwargs={'k': 10})
    compressor = CohereRerank()
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever_1)
    # Re-rank
    compressed_docs = compression_retriever.invoke(question)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 16 - Retrieval (CRAG)

    `Deep Dive`

    https://www.youtube.com/watch?v=E2shqsYwxck

    `Notebooks`

    https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb

    https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag_mistral.ipynb
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Generation



    ## 17 - Retrieval (Self-RAG)

    `Notebooks`

    https://github.com/langchain-ai/langgraph/tree/main/examples/rag

    https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag_mistral_nomic.ipynb
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 18 - Impact of long context

    `Deep dive`

    https://www.youtube.com/watch?v=SsHUNfhF32s

    `Slides`

    https://docs.google.com/presentation/d/1mJUiPBdtf58NfuSEQ7pVSEQ2Oqmek7F1i4gBwR6JDss/edit#slide=id.g26c0cb8dc66_0_0
    """)
    return


if __name__ == "__main__":
    app.run()
