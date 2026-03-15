# /// script
# dependencies = ["chromadb", "langchain", "langchain-chroma", "langchain-community", "langchain-openai", "langchain-text-splitters", "langsmith", "tiktoken"]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App()

with app.setup:
    import marimo as mo
    import os
    import bs4

    from langsmith import Client

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_chroma import Chroma
    from langchain_core.load import dumps, loads
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import (
        ChatPromptTemplate,
        FewShotChatMessagePromptTemplate,
    )
    from langchain_core.runnables import RunnableLambda
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
    # Rag From Scratch: Query Transformations

    Query transformations are a set of approaches focused on re-writing and / or modifying questions for retrieval.

    ![Screenshot 2024-03-25 at 8.08.30 PM.png](./imgs/query_overview.png)

    ## Environment

    `(1) Packages`
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: langchain_community tiktoken langchain-openai chromadb langchain langchain-chroma langchain-text-splitters langsmith !pip install langchain_community tiktoken langchain-openai chromadb langchain langchain-chroma langchain-text-splitters langsmith
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
    ## Part 5: Multi Query

    Flow:

    ![Screenshot 2024-02-12 at 12.39.59 PM.png](./imgs/multi_query.png)

    Docs:

    * https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever

    ### Index
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
        chunk_size=300, chunk_overlap=50
    )

    # Make splits
    splits = text_splitter.split_documents(blog_docs)

    # Index
    vectorstore = Chroma.from_documents(documents=splits, embedding=make_embeddings())

    retriever = vectorstore.as_retriever()
    return (retriever,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Prompt
    """)
    return


@app.cell
def _():
    # Multi Query: Different Perspectives
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_perspectives
        | make_chat_model()
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )
    return (generate_queries,)


@app.cell
def _(generate_queries, retriever):
    def get_unique_union(documents: list[list]):
        """ Unique union of retrieved docs """
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Get unique documents
        unique_docs = list(set(flattened_docs))
        # Return
        return [loads(doc) for doc in unique_docs]

    # Retrieve
    question = "What is task decomposition for LLM agents?"
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    docs = retrieval_chain.invoke({"question":question})
    len(docs)
    return question, retrieval_chain


@app.cell
def _(question, retrieval_chain):
    from operator import itemgetter
    template_1 = 'Answer the following question based on this context:\n\n{context}\n\nQuestion: {question}\n'
    prompt = ChatPromptTemplate.from_template(template_1)
    # RAG
    llm = make_chat_model()
    final_rag_chain = {'context': retrieval_chain, 'question': itemgetter('question')} | prompt | llm | StrOutputParser()
    final_rag_chain.invoke({'question': question})
    return itemgetter, llm


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part 6: RAG-Fusion

    Flow:

    ![Screenshot 2024-02-12 at 12.41.36 PM.png](./imgs/rag_fusion.png)

    Docs:

    * https://github.com/langchain-ai/langchain/blob/master/cookbook/rag_fusion.ipynb?ref=blog.langchain.dev

    Blog / repo:

    * https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1

    ### Prompt
    """)
    return


@app.cell
def _():
    template_2 = 'You are a helpful assistant that generates multiple search queries based on a single input query. \n\nGenerate multiple search queries related to: {question} \n\nOutput (4 queries):'
    # RAG-Fusion: Related
    prompt_rag_fusion = ChatPromptTemplate.from_template(template_2)
    return (prompt_rag_fusion,)


@app.cell
def _(prompt_rag_fusion):
    generate_queries_1 = prompt_rag_fusion | make_chat_model() | StrOutputParser() | (lambda x: x.split('\n'))
    return (generate_queries_1,)


@app.cell
def _(generate_queries_1, question, retriever):
    def reciprocal_rank_fusion(results: list[list], k=60):
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
            and an optional parameter k used in the RRF formula """
        fused_scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)  # Initialize a dictionary to hold fused scores for each unique document
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                previous_score = fused_scores[doc_str]  # Iterate through each list of ranked documents
                fused_scores[doc_str] = fused_scores[doc_str] + 1 / (rank + k)
        reranked_results = [(loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]  # Iterate through each document in the list, with its rank (position in the list)
        return reranked_results
    retrieval_chain_rag_fusion = generate_queries_1 | retriever.map() | reciprocal_rank_fusion  # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
    docs_1 = retrieval_chain_rag_fusion.invoke({'question': question})
    len(docs_1)  # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0  # Retrieve the current score of the document, if any  # Update the score of the document using the RRF formula: 1 / (rank + k)  # Sort the documents based on their fused scores in descending order to get the final reranked results  # Return the reranked results as a list of tuples, each containing the document and its fused score
    return (retrieval_chain_rag_fusion,)


@app.cell
def _(itemgetter, llm, question, retrieval_chain_rag_fusion):
    template_3 = 'Answer the following question based on this context:\n\n{context}\n\nQuestion: {question}\n'
    prompt_1 = ChatPromptTemplate.from_template(template_3)
    # RAG
    final_rag_chain_1 = {'context': retrieval_chain_rag_fusion, 'question': itemgetter('question')} | prompt_1 | llm | StrOutputParser()
    final_rag_chain_1.invoke({'question': question})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Trace:

    https://smith.langchain.com/public/071202c9-9f4d-41b1-bf9d-86b7c5a7525b/r
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part 7: Decomposition
    """)
    return


@app.cell
def _():
    template_4 = 'You are a helpful assistant that generates multiple sub-questions related to an input question. \n\nThe goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n\nGenerate multiple search queries related to: {question} \n\nOutput (3 queries):'
    # Decomposition
    prompt_decomposition = ChatPromptTemplate.from_template(template_4)
    return (prompt_decomposition,)


@app.cell
def _(prompt_decomposition):
    llm_1 = make_chat_model()
    generate_queries_decomposition = prompt_decomposition | llm_1 | StrOutputParser() | (lambda x: x.split('\n'))
    question_1 = 'What are the main components of an LLM-powered autonomous agent system?'
    # LLM
    # Chain
    # Run
    questions = generate_queries_decomposition.invoke({'question': question_1})
    return generate_queries_decomposition, question_1, questions


@app.cell
def _(questions):
    questions
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Answer recursively

    ![Screenshot 2024-02-18 at 1.55.32 PM.png](./imgs/answer_recursively.png)

    Papers:

    * https://arxiv.org/pdf/2205.10625.pdf
    * https://arxiv.org/abs/2212.10509.pdf
    """)
    return


@app.cell
def _():
    # Prompt
    template_5 = 'Here is the question you need to answer:\n\n\n --- \n {question} \n --- \n\n\nHere is any available background question + answer pairs:\n\n\n --- \n {q_a_pairs} \n --- \n\n\nHere is additional context relevant to the question: \n\n\n --- \n {context} \n --- \n\n\nUse the above context and any background question + answer pairs to answer the question: \n {question}\n'
    decomposition_prompt = ChatPromptTemplate.from_template(template_5)
    return (decomposition_prompt,)


@app.cell
def _(decomposition_prompt, itemgetter, questions, retriever):
    def format_qa_pair(question, answer):
        """Format Q and A pair"""
        formatted_string = ''
        formatted_string = formatted_string + f'Question: {question}\nAnswer: {answer}\n\n'
        return formatted_string.strip()
    llm_2 = make_chat_model()
    q_a_pairs = ''
    for q in questions:
        rag_chain = {'context': itemgetter('question') | retriever | format_docs, 'question': itemgetter('question'), 'q_a_pairs': itemgetter('q_a_pairs')} | decomposition_prompt | llm_2 | StrOutputParser()
        answer = rag_chain.invoke({'question': q, 'q_a_pairs': q_a_pairs})
    # llm
        q_a_pair = format_qa_pair(q, answer)
        q_a_pairs = q_a_pairs + '\n---\n' + q_a_pair
    return answer, llm_2


@app.cell
def _(answer):
    answer
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Trace:

    Question 1: https://smith.langchain.com/public/faefde73-0ecb-4328-8fee-a237904115c0/r

    Question 2: https://smith.langchain.com/public/6142cad3-b314-454e-b2c9-15146cfcce78/r

    Question 3: https://smith.langchain.com/public/84bdca0f-0fa4-46d4-9f89-a7f25bd857fe/r
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Answer individually

    ![Screenshot 2024-02-18 at 2.00.59 PM.png](./imgs/answer_individualy.png)
    """)
    return


@app.cell
def _(generate_queries_decomposition, llm_2, question_1, retriever):
    # Answer each sub-question individually 
    prompt_rag = load_rag_prompt()

    def retrieve_and_rag(question, prompt_rag, sub_question_generator_chain):
        """RAG on each sub-question"""
        sub_questions = sub_question_generator_chain.invoke({'question': question})
    # RAG prompt
        rag_results = []
        for sub_question in sub_questions:
            retrieved_docs = retriever.invoke(sub_question)
            answer = (prompt_rag | llm_2 | StrOutputParser()).invoke({'context': format_docs(retrieved_docs), 'question': sub_question})
            rag_results.append(answer)
        return (rag_results, sub_questions)  # Use our decomposition / 
    # Wrap the retrieval and RAG process in a RunnableLambda for integration into a chain
    answers, questions_1 = retrieve_and_rag(question_1, prompt_rag, generate_queries_decomposition)  # Initialize a list to hold RAG chain results  # Retrieve documents for each sub-question  # Use retrieved documents and sub-question in RAG chain
    return answers, questions_1


@app.cell
def _(answers, llm_2, question_1, questions_1):
    def format_qa_pairs(questions, answers):
        """Format Q and A pairs"""
        formatted_string = ''
        for i, (question, answer) in enumerate(zip(questions, answers), start=1):
            formatted_string = formatted_string + f'Question {i}: {question}\nAnswer {i}: {answer}\n\n'
        return formatted_string.strip()
    context = format_qa_pairs(questions_1, answers)
    template_6 = 'Here is a set of Q+A pairs:\n\n{context}\n\nUse these to synthesize an answer to the question: {question}\n'
    prompt_2 = ChatPromptTemplate.from_template(template_6)
    final_rag_chain_2 = prompt_2 | llm_2 | StrOutputParser()
    # Prompt
    final_rag_chain_2.invoke({'context': context, 'question': question_1})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Trace:

    https://smith.langchain.com/public/d8f26f75-3fb8-498a-a3a2-6532aa77f56b/r
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part 8: Step Back

    ![Screenshot 2024-02-12 at 1.14.43 PM.png](./imgs/step_back.png)

    Paper:

    * https://arxiv.org/pdf/2310.06117.pdf
    """)
    return


@app.cell
def _():
    # Few Shot Examples
    examples = [{'input': 'Could the members of The Police perform lawful arrests?', 'output': 'what can the members of The Police do?'}, {'input': 'Jan Sindel’s was born in what country?', 'output': 'what is Jan Sindel’s personal history?'}]
    example_prompt = ChatPromptTemplate.from_messages([('human', '{input}'), ('ai', '{output}')])
    few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples)
    # We now transform these to example messages
    prompt_3 = ChatPromptTemplate.from_messages([('system', 'You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:'), few_shot_prompt, ('user', '{question}')])  # Few shot examples  # New question
    return (prompt_3,)


@app.cell
def _(prompt_3):
    generate_queries_step_back = prompt_3 | make_chat_model() | StrOutputParser()
    question_2 = 'What is task decomposition for LLM agents?'
    generate_queries_step_back.invoke({'question': question_2})
    return generate_queries_step_back, question_2


@app.cell
def _(generate_queries_step_back, question_2, retriever):
    # Response prompt 
    response_prompt_template = 'You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.\n\n# {normal_context}\n# {step_back_context}\n\n# Original Question: {question}\n# Answer:'
    response_prompt = ChatPromptTemplate.from_template(response_prompt_template)
    chain = {'normal_context': RunnableLambda(lambda x: x['question']) | retriever | format_docs, 'step_back_context': generate_queries_step_back | retriever | format_docs, 'question': lambda x: x['question']} | response_prompt | make_chat_model() | StrOutputParser()
    chain.invoke({'question': question_2})  # Retrieve context using the normal question  # Retrieve context using the step-back question  # Pass on the question
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part 9: HyDE

    ![Screenshot 2024-02-12 at 1.12.45 PM.png](./imgs/HyDE.png)

    Docs:

    * https://github.com/langchain-ai/langchain/blob/master/cookbook/hypothetical_document_embeddings.ipynb

    Paper:

    * https://arxiv.org/abs/2212.10496
    """)
    return


@app.cell
def _():
    template_7 = 'Please write a scientific paper passage to answer the question\nQuestion: {question}\nPassage:'
    prompt_hyde = ChatPromptTemplate.from_template(template_7)
    # HyDE document generation
    generate_docs_for_retrieval = prompt_hyde | make_chat_model() | StrOutputParser()
    question_3 = 'What is task decomposition for LLM agents?'
    # Run
    generate_docs_for_retrieval.invoke({'question': question_3})
    return generate_docs_for_retrieval, question_3


@app.cell
def _(generate_docs_for_retrieval, question_3, retriever):
    # Retrieve
    retrieval_chain_1 = generate_docs_for_retrieval | retriever
    retrieved_docs = retrieval_chain_1.invoke({'question': question_3})
    retrieved_docs
    return (retrieved_docs,)


@app.cell
def _(llm_2, question_3, retrieved_docs):
    # RAG
    template_8 = 'Answer the following question based on this context:\n\n{context}\n\nQuestion: {question}\n'
    prompt_4 = ChatPromptTemplate.from_template(template_8)
    final_rag_chain_3 = prompt_4 | llm_2 | StrOutputParser()
    final_rag_chain_3.invoke({'context': format_docs(retrieved_docs), 'question': question_3})
    return


if __name__ == "__main__":
    app.run()
