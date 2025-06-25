import sqlite3
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import faiss
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    id: int
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict = None
    created_at: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class RetrievalResult:
    document: Document
    similarity_score: float
    rerank_score: float = 0.0


class VectorDatabase:
    """Persistent vector database using SQLite and FAISS"""

    def __init__(self, db_path: str, embedding_dim: int = 384):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        self._init_database()
        self._load_existing_data()

    def _init_database(self):
        """Initialize SQLite database tables"""
        cursor = self.conn.cursor()

        # Documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT,
                embedding_id INTEGER
            )
        ''')

        # FAISS index metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faiss_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER,
                embedding BLOB,
                FOREIGN KEY(doc_id) REFERENCES documents(id)
            )
        ''')

        self.conn.commit()

    def _load_existing_data(self):
        """Load existing embeddings into FAISS index"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT embedding FROM faiss_metadata ORDER BY id')

        embeddings = []
        for row in cursor.fetchall():
            embedding = pickle.loads(row[0])
            embeddings.append(embedding)

        if embeddings:
            embeddings_array = np.vstack(embeddings).astype(np.float32)
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings_array)
            self.faiss_index.add(embeddings_array)
            logger.info(f"Loaded {len(embeddings)} embeddings into FAISS index")

    def insert_document(self, document: Document):
        """Insert document with embedding into database"""
        cursor = self.conn.cursor()

        # Insert document
        cursor.execute('''
            INSERT INTO documents (content, metadata, created_at)
            VALUES (?, ?, ?)
        ''', (
            document.content,
            json.dumps(document.metadata),
            document.created_at
        ))

        doc_id = cursor.lastrowid
        document.id = doc_id

        # Store embedding
        if document.embedding is not None:
            # Normalize embedding for cosine similarity
            normalized_embedding = document.embedding.copy().astype(np.float32)
            faiss.normalize_L2(normalized_embedding.reshape(1, -1))

            # Add to FAISS index
            self.faiss_index.add(normalized_embedding.reshape(1, -1))

            # Store in database
            cursor.execute('''
                INSERT INTO faiss_metadata (doc_id, embedding)
                VALUES (?, ?)
            ''', (doc_id, pickle.dumps(document.embedding)))

        self.conn.commit()
        logger.info(f"Inserted document {doc_id}")

    def search_similar(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[Document, float]]:
        """Search for similar documents using FAISS"""
        if self.faiss_index.ntotal == 0:
            return []

        # Normalize query embedding
        query_normalized = query_embedding.copy().astype(np.float32)
        faiss.normalize_L2(query_normalized.reshape(1, -1))

        # Search FAISS index
        scores, indices = self.faiss_index.search(query_normalized.reshape(1, -1), k)

        results = []
        cursor = self.conn.cursor()

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue

            # Get document by FAISS index position
            cursor.execute('''
                SELECT d.id, d.content, d.metadata, d.created_at
                FROM documents d
                JOIN faiss_metadata f ON d.id = f.doc_id
                WHERE f.id = ?
            ''', (idx + 1,))  # FAISS indices are 0-based, SQLite IDs are 1-based

            row = cursor.fetchone()
            if row:
                doc = Document(
                    id=row[0],
                    content=row[1],
                    metadata=json.loads(row[2]) if row[2] else {},
                    created_at=row[3]
                )
                results.append((doc, float(score)))

        return results

    def get_all_documents(self) -> List[Document]:
        """Retrieve all documents from database"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, content, metadata, created_at FROM documents')

        documents = []
        for row in cursor.fetchall():
            doc = Document(
                id=row[0],
                content=row[1],
                metadata=json.loads(row[2]) if row[2] else {},
                created_at=row[3]
            )
            documents.append(doc)

        return documents


class Llama2Reranker:
    """Reranking using Llama-2-7B-Chat for better relevance scoring"""

    def __init__(self, model_path: str):
        self.llama = Llama(
            model_path=model_path,
            n_ctx=4096,  # Llama-2 supports longer context
            n_batch=512,
            n_gpu_layers=32,  # Offload layers to GPU if available
            n_threads=8,  # Optimize for multi-core CPU
            verbose=False
        )

    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 5) -> List[RetrievalResult]:
        """Rerank documents using Llama-2-7B-Chat scoring"""
        results = []

        for doc in documents:
            # Use Llama-2-Chat format with system prompt
            system_prompt = "You are a helpful assistant that evaluates document relevance."
            user_prompt = f"""Rate how relevant this document is to the query on a scale of 0-10. Only respond with a single number.

Query: {query}

Document: {doc.content[:800]}

Rating:"""

            # Format for Llama-2-Chat
            prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"

            # Get response from Llama-2
            response = self.llama(
                prompt,
                max_tokens=5,
                temperature=0.1,
                stop=["</s>", "[INST]", "\n"]
            )

            # Extract numerical score
            score_text = response['choices'][0]['text'].strip()
            try:
                rerank_score = float(score_text)
                # Normalize to 0-1 range
                rerank_score = min(max(rerank_score / 10.0, 0.0), 1.0)
            except ValueError:
                rerank_score = 0.0

            result = RetrievalResult(
                document=doc,
                similarity_score=0.0,  # Will be set by the RAG system
                rerank_score=rerank_score
            )
            results.append(result)

        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results[:top_k]


class Llama2RAG:
    """Complete RAG system with Llama-2-7B-Chat, embedding, reranking, and persistence"""

    def __init__(self,
                 llama2_model_path: str,
                 db_path: str ,
                 embedding_model: str = "all-MiniLM-L6-v2"):

        # Initialize components
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_db = VectorDatabase(db_path, self.embedding_model.get_sentence_embedding_dimension())
        self.reranker = Llama2Reranker(llama2_model_path)

        # Initialize Llama-2 for generation with optimized settings
        self.llama = Llama(
            model_path=llama2_model_path,
            n_ctx=4096,  # Longer context for better responses
            n_batch=512,
            n_gpu_layers=32,  # Use GPU if available
            n_threads=8,  # Multi-threading
            rope_scaling_type=1,  # Enable RoPE scaling for longer contexts
            verbose=False
        )

        logger.info("Llama-2-7B RAG system initialized")

    def add_document(self, content: str, metadata: Dict = None) -> Document:
        """Add a document to the knowledge base"""
        # Generate embedding
        embedding = self.embedding_model.encode(content)

        # Create document
        document = Document(
            id=0,  # Will be set by database
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )

        # Store in database
        self.vector_db.insert_document(document)
        return document

    def add_documents_from_file(self, file_path: str, chunk_size: int = 500) -> List[Document]:
        """Add documents from a text file, splitting into chunks"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Simple chunking by characters
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        documents = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Skip empty chunks
                metadata = {
                    "source_file": file_path,
                    "chunk_index": i,
                    "chunk_size": len(chunk)
                }
                doc = self.add_document(chunk, metadata)
                documents.append(doc)

        logger.info(f"Added {len(documents)} document chunks from {file_path}")
        return documents

    def retrieve_documents(self, query: str, top_k: int = 10, use_reranking: bool = True) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)

        # Vector similarity search
        similar_docs = self.vector_db.search_similar(query_embedding, top_k * 2)  # Get more for reranking

        if not similar_docs:
            return []

        # Create initial results
        results = []
        for doc, similarity in similar_docs:
            result = RetrievalResult(
                document=doc,
                similarity_score=similarity,
                rerank_score=0.0
            )
            results.append(result)

        # Apply reranking if requested
        if use_reranking and len(results) > 0:
            documents_to_rerank = [r.document for r in results]
            reranked_results = self.reranker.rerank_documents(query, documents_to_rerank, top_k)

            # Update similarity scores
            for reranked in reranked_results:
                for original in results:
                    if reranked.document.id == original.document.id:
                        reranked.similarity_score = original.similarity_score
                        break

            return reranked_results

        return results[:top_k]

    def generate_response(self, query: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate response using retrieved context with Llama-2-Chat format"""
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, top_k=5, use_reranking=True)

        # System prompt for Llama-2-Chat
        system_prompt = """You are a helpful, respectful and honest assistant. Answer questions based on the provided context. If you cannot answer based on the context, say so clearly."""

        if not retrieved_docs:
            # No relevant documents found, generate without context
            user_prompt = f"Question: {query}"
        else:
            # Build context from retrieved documents
            context_parts = []
            for i, result in enumerate(retrieved_docs[:3]):  # Use top 3 documents
                context_parts.append(f"Context {i + 1}: {result.document.content[:500]}")

            context = "\n\n".join(context_parts)

            user_prompt = f"""Based on the following context, answer the question:

{context}

Question: {query}"""

        # Format for Llama-2-Chat
        prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"

        # Generate response
        response = self.llama(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "[INST]", "<s>"],
            repeat_penalty=1.1
        )

        return response['choices'][0]['text'].strip()

    def chat(self, query: str) -> Dict:
        """Interactive chat interface with retrieval information"""
        retrieved_docs = self.retrieve_documents(query, top_k=5, use_reranking=True)
        response = self.generate_response(query)

        return {
            "query": query,
            "response": response,
            "retrieved_documents": [
                {
                    "content": result.document.content[:200] + "...",
                    "similarity_score": result.similarity_score,
                    "rerank_score": result.rerank_score,
                    "metadata": result.document.metadata
                }
                for result in retrieved_docs
            ],
            "num_documents_retrieved": len(retrieved_docs)
        }


# Example usage
def main():
    # Initialize RAG system with Llama-2-7B-Chat
    db_path= "/home/hd/py_ex/llama_ex/db/rag_database.db"
    rag = Llama2RAG(
        llama2_model_path="/home/hd/py_ex/models/llama-2-7b-chat.Q4_K_M.gguf",  # Your Llama-2 model path
        db_path=db_path
    )

    # Add some example documents
    documents = [
        "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from and make predictions on data without being explicitly programmed.",
        "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation to provide more accurate and contextual responses.",
        "Vector databases store high-dimensional vectors for similarity search operations, commonly used in AI applications for semantic search.",
        "Llama-2 is a large language model developed by Meta AI that comes in various sizes and is optimized for chat applications."
    ]

    for doc in documents:
        rag.add_document(doc)

    # Example queries
    queries = [
        "What is Python programming?",
        "Explain machine learning",
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


if __name__ == "__main__":
    main()