from typing import List

from llama_cpp import Llama

from llama_ex.document import Document
from llama_ex.retrieve_result import RetrievalResult


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

