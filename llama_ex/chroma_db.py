import atexit
import threading

import chromadb
import posthog
from sentence_transformers import SentenceTransformer


class ChromaDb:
    """
    A class to represent a Chroma database connection.
    This class is a placeholder for the actual implementation.
    """

    def __init__(self, db_path: str, model:SentenceTransformer):
        """
        Initializes the ChromaDb with the given database path.

        Args:
            db_path (str): The path to the Chroma database.
        """
        self.db_path = db_path
        self.model=model

        # Here you would typically initialize the connection to the database

    def connect(self):
        """
        Connects to the Chroma database.
        This method should contain the logic to establish a connection.
        """
        self.client = chromadb.PersistentClient(self.db_path)
        self.collection = self.client.get_or_create_collection(name="documents")

    def add_document(self, texts):
        """
        :param document:
        :param metadata:
        :return:
        """
        embeddings = self.model.encode(texts).tolist()
        self.collection.add(documents= texts, embeddings=embeddings, ids=[f"doc{i}" for i in range(len(texts))])

    def query(self, query_text: str, top_n: int = 5):
        """
        Queries the Chroma database for similar documents.

        Args:
            query_text (str): The text to query against the database.
            top_n (int): The number of results to return.

        Returns:
            List of documents that match the query.
        """
        query_embedding = self.model.encode(query_text).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_n)
        return results


def shutdown_posthog():
    """
    Shuts down the PostHog client gracefully.
    This function should be called when the application is terminating.
    """
    posthog.shutdown()

def cleanup_threads():
    for thread in threading.enumerate():
        try:
            if thread is threading.main_thread() or isinstance(thread, threading._DummyThread):
                continue
            if not thread.is_alive():
                continue
            thread.join(timeout=1)  # Give threads time to finish
        except(RuntimeError,AttributeError, TypeError) as e:
            print(f"Error cleaning up thread {thread.name}: {e}")


if __name__== "__main__":
    # Example usage
    atexit.register(shutdown_posthog)
    atexit.register(cleanup_threads)
    db_path = "./chroma_db"
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_db = ChromaDb(db_path, model)
    chroma_db.connect()
    print("Chroma database connected successfully.")
    texts = [
        "The Eiffel Tower is in Paris.",
        "Cats are great pets.",
        "The Moon orbits the Earth.",
    ]
    chroma_db.add_document(texts)
    results=chroma_db.query("Tell me about space", 2)
    for doc, dist in zip(results['documents'][0], results['distances'][0]):
        print(f"- Match: {doc}, (Distance: {dist:.4f})")

    print("Done.")