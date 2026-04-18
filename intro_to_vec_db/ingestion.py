import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()


def ingest_data():
    print("Ingesting data...")
    loader = TextLoader("data/mediumblog1.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print(f"Broken down into {len(chunks)} chunks")

    print("Embedding and vectorizing...")
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        deployment="text-embedding-ada-002",
        model="text-embedding-ada-002",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-05-15",
    )

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=os.getenv("INDEX_NAME"),
    )

    print("Data ingested successfully!")


if __name__ == "__main__":
    ingest_data()
