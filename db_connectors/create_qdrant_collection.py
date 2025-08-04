from qdrant_client.http.models import Distance, VectorParams
from connection_qdrant import get_qdrant_client
import os

new_collection = os.getenv("QDRANT_COLLECTION")
client = get_qdrant_client()

try:
    # First try to get the collection
    client.get_collection(new_collection)
    print(f"Collection '{new_collection}' already exists")
except Exception:
    # Only create if it doesn't exist
    client.create_collection(
        collection_name=new_collection,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )
    print(f"Created new collection '{new_collection}'")