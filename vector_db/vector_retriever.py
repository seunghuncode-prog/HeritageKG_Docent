import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction
)

DB_DIR = "vector_db/chroma_museum_db"

COLLECTION_NAME = "museum_relics"

embedding_function = (
    SentenceTransformerEmbeddingFunction(
        model_name="jhgan/ko-sroberta-multitask"
    )
)

class VectorRetriever:

    def __init__(self):

        client = chromadb.PersistentClient(
            path=DB_DIR
        )

        self.collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )

    def search(self, query):

        results = self.collection.query(
            query_texts=[query],
            n_results=3
        )

        return results