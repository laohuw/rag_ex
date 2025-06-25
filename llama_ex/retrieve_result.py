from dataclasses import dataclass

from llama_ex.document import Document


@dataclass
class RetrievalResult:
    document: Document
    similarity_score: float
    rerank_score: float = 0.0
