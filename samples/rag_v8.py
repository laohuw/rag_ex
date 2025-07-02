import cohere
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple


class CohereRerankedRAG:
    def __init__(self, cohere_api_key: str):
        # Initialize models
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cohere_client = cohere.Client(cohere_api_key)

        # Sample documents (in practice, these would be from your knowledge base)
        self.documents = [
            "Python is a high-level programming language known for its simplicity and readability.",
            "Machine learning algorithms can be implemented efficiently using Python libraries like scikit-learn.",
            "FastAPI is a modern web framework for building APIs with Python, featuring automatic documentation.",
            "Django is a comprehensive web framework that follows the model-view-template architectural pattern.",
            "NumPy provides support for large multi-dimensional arrays and mathematical functions in Python.",
            "Pandas is essential for data manipulation and analysis, offering data structures like DataFrames.",
            "TensorFlow and PyTorch are popular deep learning frameworks used for neural network development.",
            "REST APIs can be easily created using Flask, a lightweight Python web framework.",
            "Jupyter notebooks provide an interactive environment for data science and prototyping.",
            "Docker containers help in deploying Python applications with consistent environments."
        ]

        # Create embeddings and FAISS index
        self.setup_vector_index()

    def setup_vector_index(self):
        """Create embeddings and build FAISS index"""
        print("Creating embeddings...")
        self.embeddings = self.embedding_model.encode(self.documents)

        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))
        print(f"Index built with {len(self.documents)} documents")

    def initial_retrieval(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Step 1: Initial retrieval using embeddings"""
        print(f"\n1. Initial retrieval for: '{query}'")

        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # Return documents with scores
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            doc = self.documents[idx]
            results.append((doc, float(score)))
            print(f"   {i + 1}. Score: {score:.3f} - {doc[:80]}...")

        return results

    def rerank_with_cohere(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
        """Step 2: Rerank using Cohere's reranking model"""
        print(f"\n2. Reranking top {len(documents)} documents with Cohere...")

        try:
            # Use Cohere's rerank endpoint
            rerank_response = self.cohere_client.rerank(
                model='rerank-english-v3.0',  # or 'rerank-multilingual-v3.0'
                query=query,
                documents=documents,
                top_n=top_k,
                return_documents=True
            )

            # Format results
            reranked_results = []
            for result in rerank_response.results:
                reranked_results.append({
                    'document': result.document.text,
                    'relevance_score': result.relevance_score,
                    'index': result.index
                })
                print(f"   Score: {result.relevance_score:.3f} - {result.document.text[:80]}...")

            return reranked_results

        except Exception as e:
            print(f"Cohere reranking failed: {e}")
            # Fallback to original embedding results
            return [{'document': doc, 'relevance_score': score, 'index': i}
                    for i, (doc, score) in enumerate(zip(documents, [0.5] * len(documents)))]

    def retrieve_and_rerank(self, query: str, initial_k: int = 10, final_k: int = 3) -> List[Dict]:
        """Complete retrieval + reranking pipeline"""
        print("=" * 80)
        print(f"RAG with Cohere Reranking Pipeline")
        print("=" * 80)

        # Step 1: Initial retrieval with embeddings
        initial_results = self.initial_retrieval(query, initial_k)
        initial_docs = [doc for doc, score in initial_results]

        # Step 2: Rerank with Cohere
        final_results = self.rerank_with_cohere(query, initial_docs, final_k)

        print(f"\n3. Final top {len(final_results)} results after reranking:")
        for i, result in enumerate(final_results):
            print(f"   {i + 1}. Relevance: {result['relevance_score']:.3f}")
            print(f"      {result['document']}")
            print()

        return final_results

    def compare_methods(self, query: str):
        """Compare embedding-only vs embedding + reranking"""
        print("\n" + "=" * 80)
        print("COMPARISON: Embedding-only vs Embedding + Reranking")
        print("=" * 80)

        # Method 1: Embedding only (top 3)
        embedding_results = self.initial_retrieval(query, 3)
        print(f"\nEmbedding-only top 3:")
        for i, (doc, score) in enumerate(embedding_results):
            print(f"   {i + 1}. {doc}")

        # Method 2: Embedding + Reranking
        reranked_results = self.retrieve_and_rerank(query, initial_k=8, final_k=3)
        print(f"\nEmbedding + Reranking top 3:")
        for i, result in enumerate(reranked_results):
            print(f"   {i + 1}. {result['document']}")


# Example usage
def main():
    # You'll need a Cohere API key
    COHERE_API_KEY = "your-cohere-api-key-here"

    # Initialize the system
    rag_system = CohereRerankedRAG(COHERE_API_KEY)

    # Example queries
    queries = [
        "How to build web APIs with Python?",
        "What libraries are good for data analysis?",
        "Machine learning frameworks for deep learning"
    ]

    for query in queries:
        # Demonstrate the complete pipeline
        results = rag_system.retrieve_and_rerank(query)

        # Show comparison
        rag_system.compare_methods(query)

        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()

# Key Benefits of This Approach:
#
# 1. **Broad Initial Retrieval**: Embeddings cast a wide net to find potentially relevant docs
# 2. **Precise Reranking**: Cohere's reranker is specifically trained to understand query-document relevance
# 3. **Better Context**: More accurate top results mean better context for your LLM
# 4. **Handles Edge Cases**: Reranker can fix cases where embedding similarity doesn't match semantic relevance
#
# Example Output Pattern:
# - Embedding might rank "Python is a programming language" highly for "web API" query
# - Reranker would boost "FastAPI is a web framework" and "REST APIs with Flask" higher
# - Final results are more precisely relevant to the user's intent