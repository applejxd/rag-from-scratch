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
        """Local equivalent of the LangChain Hub prompt `rlm/rag-prompt`.

        See https://smith.langchain.com/hub/rlm/rag-prompt for the source
        prompt this template reproduces without depending on the Hub.
        """
        template = (
            "Answer the question based only on the following context:\n"
            "{context}\n\nQuestion: {question}\n"
        )
        return ChatPromptTemplate.from_template(template)


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
    ## パート5：Multi Query

    一つの質問を異なる観点の複数クエリへ言い換え、それぞれの検索結果を重複排除して統合します。単一の表現だけでは取得できない関連文書を補うことが目的です。

    処理の流れ：

    ![Multi Queryの流れ](./imgs/multi_query.png)

    ドキュメント：

    * https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever

    ### インデックス
    """)
    return


@app.cell
def _():
    #### INDEXING ####

    # Load blog
    blog_loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        },
    )
    blog_documents = blog_loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=50
    )

    # Make splits
    document_chunks = text_splitter.split_documents(blog_documents)

    # Index
    vectorstore = Chroma.from_documents(
        documents=document_chunks, embedding=make_embeddings()
    )

    retriever = vectorstore.as_retriever()
    return (retriever,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### プロンプト

    LLMに5種類の検索クエリを改行区切りで生成させます。生成した各クエリへ同じリトリーバーを適用するため、LCELの `retriever.map()` を使用します。
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


@app.cell
def _(multi_query_generator, retriever):
    def get_unique_union(documents: list[list]):
        """Return unique documents from multiple retrieval result lists."""
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs = list(set(flattened_docs))
        return [loads(doc) for doc in unique_docs]

    retrieval_question = "What is task decomposition for LLM agents?"
    multi_query_retrieval_chain = (
        multi_query_generator | retriever.map() | get_unique_union
    )
    multi_query_documents = multi_query_retrieval_chain.invoke(
        {"question": retrieval_question}
    )
    len(multi_query_documents)
    return multi_query_retrieval_chain, retrieval_question


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


@app.cell
def _(rag_fusion_prompt):
    rag_fusion_query_generator = (
        rag_fusion_prompt
        | make_chat_model()
        | StrOutputParser()
        | split_queries
    )
    return (rag_fusion_query_generator,)


@app.cell
def _(rag_fusion_query_generator, retrieval_question, retriever):
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
    fused_documents = rag_fusion_retrieval_chain.invoke(
        {"question": retrieval_question}
    )
    len(fused_documents)
    return (rag_fusion_retrieval_chain,)


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
    decomposition_question = (
        "What are the main components of an LLM-powered autonomous agent system?"
    )
    subquestions = subquestion_generator.invoke(
        {"question": decomposition_question}
    )
    return decomposition_question, subquestion_generator, subquestions


@app.cell
def _(subquestions):
    subquestions
    return


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


@app.cell
def _(recursive_answer_prompt, retriever, subquestions):
    def format_qa_pair(question, answer):
        """Format Q and A pair"""
        return f"Question: {question}\nAnswer: {answer}".strip()

    decomposition_model = make_chat_model()
    accumulated_qa = ""
    recursive_answer = ""
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
    for subquestion in subquestions:
        recursive_answer = recursive_rag_chain.invoke(
            {"question": subquestion, "q_a_pairs": accumulated_qa}
        )
        accumulated_qa += (
            "\n---\n" + format_qa_pair(subquestion, recursive_answer)
        )
    return decomposition_model, recursive_answer


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
    return individual_answers, individual_subquestions


@app.cell
def _(
    decomposition_model,
    decomposition_question,
    individual_answers,
    individual_subquestions,
):
    def format_qa_pairs(questions, answers):
        """Format Q and A pairs"""
        formatted_string = ""
        for i, (question, answer) in enumerate(zip(questions, answers), start=1):
            formatted_string += (
                f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
            )
        return formatted_string.strip()

    qa_context = format_qa_pairs(individual_subquestions, individual_answers)
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


@app.cell
def _(step_back_prompt):
    step_back_query_generator = (
        step_back_prompt | make_chat_model() | StrOutputParser()
    )
    step_back_question = "What is task decomposition for LLM agents?"
    step_back_query_generator.invoke({"question": step_back_question})
    return step_back_query_generator, step_back_question


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


@app.cell
def _():
    hyde_template = 'Please write a scientific paper passage to answer the question\nQuestion: {question}\nPassage:'
    hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
    hypothetical_document_generator = (
        hyde_prompt | make_chat_model() | StrOutputParser()
    )
    hyde_question = "What is task decomposition for LLM agents?"
    hypothetical_document_generator.invoke({"question": hyde_question})
    return hypothetical_document_generator, hyde_question


@app.cell
def _(hyde_question, hypothetical_document_generator, retriever):
    hyde_retrieval_chain = hypothetical_document_generator | retriever
    hyde_documents = hyde_retrieval_chain.invoke({"question": hyde_question})
    hyde_documents
    return (hyde_documents,)


@app.cell
def _(hyde_documents, hyde_question):
    hyde_answer_chain = load_rag_prompt() | make_chat_model() | StrOutputParser()
    hyde_answer_chain.invoke(
        {"context": format_docs(hyde_documents), "question": hyde_question}
    )
    return


if __name__ == "__main__":
    app.run()
