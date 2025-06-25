import logging
from typing import Dict, List

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from llama_ex.document import Document
from llama_ex.llama_to_reranker import Llama2Reranker
from llama_ex.retrieve_result import RetrievalResult
from llama_ex.vector_database import VectorDatabase


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
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Llama-2-7B RAG system initialized")

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

        self.logger.info(f"Added {len(documents)} document chunks from {file_path}")
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

        if not retrieved_docs and len(query.strip()) >0:
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
