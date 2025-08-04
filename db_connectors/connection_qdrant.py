import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from uuid import uuid4

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION"
)  # Optional: specify the collection name

def get_qdrant_vector_store():
    """Connects to Qdrant and returns a vector store instance."""
    if QDRANT_API_KEY:  # Hosted connection (with API key)
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:  # Local connection (no API key)
        qdrant = QdrantClient(url=QDRANT_URL)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY
    )

    # If collection does not exist it will be created automatically
    return QdrantVectorStore(
        client=qdrant,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
        # Optional: specify the number of dimensions for the embeddings
    )

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def search_qdrant_with_metadata(query):
    """Searches Qdrant for the given query"""
    vector_store = get_qdrant_vector_store()

    # Optional: specify the number of results to return
    results = vector_store.similarity_search_with_score(query, k=3)

    return results

def add_document_to_store(documents: list):
    """
    Adds a list of documents to the Qdrant vector store.
    """

    vector_store = get_qdrant_vector_store()

    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=uuids)