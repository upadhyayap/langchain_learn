import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS


load_dotenv()


def create_vectorstore():
    loader = PyPDFLoader(
        "/Users/anand.upadhayay/learning/python/langchain_learn/intro_to_vec_db/data/react_paper.pdf"
    )
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separator="\n"
    )
    splits = text_splitter.split_documents(docs)

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        deployment="text-embedding-ada-002",
        model="text-embedding-ada-002",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-05-15",
    )

    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local("faiss_index")
    print("Vectorstore created and saved")


if __name__ == "__main__":
    # create_vectorstore()
    # create_retrieval_chain()

    llm_azure_openai = AzureChatOpenAI(
        temperature=0,
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        model=os.environ["AZURE_OPENAI_MODEL"],
    )

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        deployment="text-embedding-ada-002",
        model="text-embedding-ada-002",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-05-15",
    )

    vectorstore = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever()
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        llm_azure_openai,
        retrieval_qa_chat_prompt,
    )

    retrieval_qa_chain = create_retrieval_chain(
        retriever,
        combine_docs_chain,
    )

    result = retrieval_qa_chain.invoke(
        {"input": "Give me the gist of ReAct in 3 sentences"}
    )

    print(result["answer"])
