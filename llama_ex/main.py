from fastapi import FastAPI

from llama_ex.llama_to_rag import Llama2RAG


def add_document():
    global doc
    documents = [
        "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from and make predictions on data without being explicitly programmed.",
        "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation to provide more accurate and contextual responses.",
        "Vector databases store high-dimensional vectors for similarity search operations, commonly used in AI applications for semantic search.",
        "Llama-2 is a large language model developed by Meta AI that comes in various sizes and is optimized for chat applications."
    ]
    for doc in documents:
        rag.add_document(doc)


def chat_query():
    global doc
    queries = [
        "Explain machine learning",
        "What is Python programming?",
        "How does RAG work?",
        "What are vector databases used for?"
    ]
    for query in queries:
        print(f"\n{'=' * 50}")
        print(f"Query: {query}")
        print('=' * 50)

        result = rag.chat(query)
        print(f"Response: {result['response']}")
        print(f"\nRetrieved {result['num_documents_retrieved']} documents:")

        for i, doc in enumerate(result['retrieved_documents']):
            print(f"{i + 1}. {doc['content']} (sim: {doc['similarity_score']:.3f}, rerank: {doc['rerank_score']:.3f})")

def init_service():
    app = FastAPI(
        title="Llama Ex API",
        description="Llama Ex API",
        version="0.1.0",
    )

if __name__ == "__main__":
    db_path = "/home/hd/py_ex/llama_ex/db/rag_database.db"
    model_gguf = "/home/hd/py_ex/models/llama-2-7b-chat.Q4_K_M.gguf"
    rag = Llama2RAG(
        llama2_model_path=model_gguf,  # Your Llama-2 model path
        db_path=db_path
    )

    # Add some example documents
    add_document()

    # Example queries
    chat_query()
