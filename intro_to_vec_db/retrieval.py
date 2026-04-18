import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureOpenAI, AzureChatOpenAI
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


load_dotenv()


def retrieve_data_withLCEL():
    print("Retrieving data with LCEL...")
    template = """Use the following piece of context to answer the question at the end.
    If you don't know the answer, just say "I don't know", don't make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of your answer.

    {context}

    Question: {question}

    Helpful Answer:
    """

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        deployment="text-embedding-ada-002",
        model="text-embedding-ada-002",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-05-15",
    )

    llm_azure_openai = AzureChatOpenAI(
        temperature=0,
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        model=os.environ["AZURE_OPENAI_MODEL"],
    )

    custom_rag_prompt = PromptTemplate.from_template(template)

    vectorstore = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings,
    )

    rag_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | custom_rag_prompt
        | llm_azure_openai
    )

    result = rag_chain.invoke("What is pinecone in machine learning?")
    print(result)


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def retrieve_data():
    print("Retrieving data...")
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        deployment="text-embedding-ada-002",
        model="text-embedding-ada-002",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-05-15",
    )

    llm_azure_openai = AzureChatOpenAI(
        temperature=0,
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        model=os.environ["AZURE_OPENAI_MODEL"],
    )

    query = "what is pinecode in machine learning?"
    # chain = PromptTemplate.from_template(template=query) | llm_azure_openai
    # result = chain.invoke(input={"query": query})
    # print(result.content)

    vectorstore = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings,
    )

    retrieval_qa_prompt_template = hub.pull("langchain-ai/retrieval-qa-chat")
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=create_stuff_documents_chain(
            llm_azure_openai, retrieval_qa_prompt_template
        ),
    )

    result = retrieval_chain.invoke(input={"input": query})
    print(result["answer"])


if __name__ == "__main__":
    # retrieve_data()
    retrieve_data_withLCEL()
