import cohere
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple
import re
from dataclasses import dataclass


@dataclass
class DocSection:
    """Represents a section of software documentation"""
    title: str
    content: str
    doc_type: str  # 'api', 'tutorial', 'reference', 'example', 'troubleshooting'
    code_examples: List[str]
    tags: List[str]
    url: str = ""


class SoftwareDocsReranker:
    def __init__(self, cohere_api_key: str):
        # Use code-aware embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cohere_client = cohere.Client(cohere_api_key)

        # Sample software documentation sections
        self.doc_sections = [
            DocSection(
                title="FastAPI Quick Start",
                content="FastAPI is a modern, fast web framework for building APIs with Python. Create your first API in minutes with automatic documentation generation.",
                doc_type="tutorial",
                code_examples=[
                    "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/')\ndef read_root():\n    return {'Hello': 'World'}"],
                tags=["fastapi", "getting-started", "web", "api"]
            ),
            DocSection(
                title="FastAPI Path Parameters",
                content="Path parameters are variable parts of URL paths. They are declared using Python type hints and automatically validated.",
                doc_type="reference",
                code_examples=[
                    "@app.get('/items/{item_id}')\ndef read_item(item_id: int):\n    return {'item_id': item_id}"],
                tags=["fastapi", "parameters", "routing", "validation"]
            ),
            DocSection(
                title="FastAPI Request Body",
                content="To send data from client to API, use request body. FastAPI uses Pydantic models for request/response validation.",
                doc_type="reference",
                code_examples=[
                    "from pydantic import BaseModel\nclass Item(BaseModel):\n    name: str\n    price: float\n@app.post('/items/')\ndef create_item(item: Item):\n    return item"],
                tags=["fastapi", "pydantic", "request-body", "validation"]
            ),
            DocSection(
                title="Django Models Tutorial",
                content="Django models define the structure of your database. Each model maps to a single database table.",
                doc_type="tutorial",
                code_examples=[
                    "from django.db import models\nclass Article(models.Model):\n    title = models.CharField(max_length=200)\n    pub_date = models.DateTimeField()"],
                tags=["django", "models", "database", "orm"]
            ),
            DocSection(
                title="Django URL Routing",
                content="URLconf maps URL patterns to views. Use path() function to define URL patterns with parameters.",
                doc_type="reference",
                code_examples=[
                    "from django.urls import path\nurlpatterns = [\n    path('articles/<int:year>/', views.year_archive),\n]"],
                tags=["django", "urls", "routing", "views"]
            ),
            DocSection(
                title="Python Requests Library",
                content="Requests is an elegant HTTP library for Python. It makes HTTP requests simple and human-friendly.",
                doc_type="reference",
                code_examples=[
                    "import requests\nr = requests.get('https://api.github.com/user', auth=('user', 'pass'))\nprint(r.json())"],
                tags=["requests", "http", "api", "client"]
            ),
            DocSection(
                title="Debugging FastAPI Applications",
                content="Common issues with FastAPI apps include import errors, validation failures, and async/await problems. Enable debug mode for detailed error messages.",
                doc_type="troubleshooting",
                code_examples=["app = FastAPI(debug=True)\n# or\nuvicorn main:app --reload --log-level debug"],
                tags=["fastapi", "debugging", "troubleshooting", "errors"]
            ),
            DocSection(
                title="FastAPI Authentication",
                content="Implement authentication using OAuth2, JWT tokens, or API keys. FastAPI provides security utilities for common authentication patterns.",
                doc_type="tutorial",
                code_examples=[
                    "from fastapi.security import HTTPBearer\nsecurity = HTTPBearer()\n@app.get('/protected')\ndef protected_route(credentials: HTTPAuthorizationCredentials = Depends(security)):\n    return {'token': credentials.credentials}"],
                tags=["fastapi", "authentication", "security", "oauth2", "jwt"]
            ),
            DocSection(
                title="Django Forms Validation",
                content="Django forms handle HTML form rendering, validation, and data cleaning. Use ModelForm for database-backed forms.",
                doc_type="reference",
                code_examples=[
                    "from django import forms\nclass ContactForm(forms.Form):\n    name = forms.CharField(max_length=100)\n    email = forms.EmailField()"],
                tags=["django", "forms", "validation", "html"]
            ),
            DocSection(
                title="API Rate Limiting Best Practices",
                content="Implement rate limiting to protect your API from abuse. Use sliding window or token bucket algorithms with Redis for distributed systems.",
                doc_type="tutorial",
                code_examples=[
                    "from slowapi import Limiter\nlimiter = Limiter(key_func=get_remote_address)\n@app.get('/limited')\n@limiter.limit('5/minute')\ndef limited_endpoint(request: Request):\n    return {'message': 'This endpoint is rate limited'}"],
                tags=["api", "rate-limiting", "security", "redis", "performance"]
            )
        ]

        self.setup_enhanced_index()

    def create_enhanced_document_text(self, doc: DocSection) -> str:
        """Create rich document representation for embedding"""
        # Combine title, content, code examples, and metadata
        doc_text = f"Title: {doc.title}\n"
        doc_text += f"Type: {doc.doc_type}\n"
        doc_text += f"Content: {doc.content}\n"

        if doc.code_examples:
            doc_text += "Code Examples:\n"
            for example in doc.code_examples:
                doc_text += f"{example}\n"

        if doc.tags:
            doc_text += f"Tags: {', '.join(doc.tags)}\n"

        return doc_text

    def setup_enhanced_index(self):
        """Create embeddings with enhanced document representation"""
        print("Creating enhanced embeddings for software docs...")

        # Create rich document representations
        self.document_texts = [
            self.create_enhanced_document_text(doc) for doc in self.doc_sections
        ]

        # Create embeddings
        self.embeddings = self.embedding_model.encode(self.document_texts)

        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))

        print(f"Enhanced index built with {len(self.doc_sections)} documentation sections")

    def classify_query_intent(self, query: str) -> Dict[str, float]:
        """Analyze query to understand user intent"""
        intent_keywords = {
            'how_to': ['how to', 'how do i', 'tutorial', 'guide', 'step by step'],
            'api_reference': ['api', 'function', 'method', 'parameter', 'reference'],
            'troubleshooting': ['error', 'bug', 'issue', 'problem', 'not working', 'fix'],
            'example': ['example', 'sample', 'demo', 'code'],
            'best_practices': ['best practice', 'recommendation', 'should', 'pattern']
        }

        query_lower = query.lower()
        intent_scores = {}

        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            intent_scores[intent] = score / len(keywords)  # Normalize

        return intent_scores

    def enhance_query_for_code_search(self, query: str) -> str:
        """Enhance query for better code-related retrieval"""
        # Add programming context keywords
        code_indicators = ['function', 'method', 'class', 'import', 'def', 'async', 'await']

        if any(indicator in query.lower() for indicator in code_indicators):
            query += " code example implementation"

        # Add framework context if detected
        frameworks = ['fastapi', 'django', 'flask', 'requests']
        detected_frameworks = [fw for fw in frameworks if fw in query.lower()]

        if detected_frameworks:
            query += f" {' '.join(detected_frameworks)} documentation"

        return query

    def initial_retrieval(self, query: str, top_k: int = 8) -> List[Tuple[DocSection, float]]:
        """Enhanced initial retrieval for software docs"""
        enhanced_query = self.enhance_query_for_code_search(query)
        print(f"\n1. Initial retrieval for: '{query}'")
        print(f"   Enhanced query: '{enhanced_query}'")

        # Encode enhanced query
        query_embedding = self.embedding_model.encode([enhanced_query])
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # Return doc sections with scores
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            doc = self.doc_sections[idx]
            results.append((doc, float(score)))
            print(f"   {i + 1}. Score: {score:.3f} - [{doc.doc_type}] {doc.title}")

        return results

    def create_rerank_documents(self, doc_sections: List[DocSection]) -> List[str]:
        """Create optimized documents for Cohere reranking"""
        rerank_docs = []

        for doc in doc_sections:
            # Create focused document for reranking
            rerank_text = f"{doc.title}\n"
            rerank_text += f"Documentation Type: {doc.doc_type}\n"
            rerank_text += f"{doc.content}\n"

            # Include relevant code snippet (first one, truncated)
            if doc.code_examples:
                code_snippet = doc.code_examples[0][:300]  # Limit length
                rerank_text += f"Code Example:\n{code_snippet}\n"

            # Add tags for context
            if doc.tags:
                rerank_text += f"Related: {', '.join(doc.tags[:5])}"  # Top 5 tags

            rerank_docs.append(rerank_text)

        return rerank_docs

    def rerank_with_intent_boost(self, query: str, doc_sections: List[DocSection],
                                 top_k: int = 5) -> List[Dict]:
        """Rerank with software documentation specific optimizations"""
        print(f"\n2. Reranking {len(doc_sections)} docs with Cohere + intent analysis...")

        # Analyze query intent
        intent_scores = self.classify_query_intent(query)
        dominant_intent = max(intent_scores.items(), key=lambda x: x[1])
        print(f"   Detected intent: {dominant_intent[0]} (confidence: {dominant_intent[1]:.2f})")

        # Create documents optimized for reranking
        rerank_documents = self.create_rerank_documents(doc_sections)

        try:
            # Cohere reranking
            rerank_response = self.cohere_client.rerank(
                model='rerank-english-v3.0',
                query=query,
                documents=rerank_documents,
                top_n=top_k,
                return_documents=True
            )

            # Process results with intent boosting
            reranked_results = []
            for result in rerank_response.results:
                doc = doc_sections[result.index]
                relevance_score = result.relevance_score

                # Apply intent-based boosting
                if dominant_intent[0] == 'troubleshooting' and doc.doc_type == 'troubleshooting':
                    relevance_score += 0.1
                elif dominant_intent[0] == 'how_to' and doc.doc_type == 'tutorial':
                    relevance_score += 0.1
                elif dominant_intent[0] == 'api_reference' and doc.doc_type == 'reference':
                    relevance_score += 0.1
                elif dominant_intent[0] == 'example' and doc.code_examples:
                    relevance_score += 0.05

                reranked_results.append({
                    'doc_section': doc,
                    'relevance_score': min(relevance_score, 1.0),  # Cap at 1.0
                    'original_index': result.index,
                    'intent_boosted': relevance_score != result.relevance_score
                })

                boost_indicator = " [BOOSTED]" if relevance_score != result.relevance_score else ""
                print(f"   Score: {relevance_score:.3f} - [{doc.doc_type}] {doc.title}{boost_indicator}")

            return reranked_results

        except Exception as e:
            print(f"Cohere reranking failed: {e}")
            # Fallback
            return [{'doc_section': doc, 'relevance_score': 0.5, 'original_index': i, 'intent_boosted': False}
                    for i, doc in enumerate(doc_sections)]

    def search_software_docs(self, query: str, initial_k: int = 8, final_k: int = 3) -> List[Dict]:
        """Complete software documentation search pipeline"""
        print("=" * 80)
        print(f"Software Documentation Search: '{query}'")
        print("=" * 80)

        # Step 1: Initial retrieval
        initial_results = self.initial_retrieval(query, initial_k)
        initial_docs = [doc for doc, score in initial_results]

        # Step 2: Rerank with intent analysis
        final_results = self.rerank_with_intent_boost(query, initial_docs, final_k)

        print(f"\n3. Final top {len(final_results)} results:")
        for i, result in enumerate(final_results):
            doc = result['doc_section']
            print(f"\n   {i + 1}. [{doc.doc_type.upper()}] {doc.title}")
            print(f"       Relevance: {result['relevance_score']:.3f}")
            print(f"       {doc.content[:100]}...")
            if doc.code_examples:
                print(f"       Code: {doc.code_examples[0][:80]}...")

        return final_results

    def demonstrate_software_queries(self):
        """Show different types of software documentation queries"""
        test_queries = [
            "How to create REST API with FastAPI?",  # Tutorial intent
            "FastAPI path parameters validation",  # Reference intent
            "FastAPI authentication not working error",  # Troubleshooting intent
            "Django model example with database",  # Example intent
            "API rate limiting best practices",  # Best practices intent
        ]

        for query in test_queries:
            results = self.search_software_docs(query)
            print("\n" + "-" * 50 + "\n")


# Example usage
def main():
    COHERE_API_KEY = "your-cohere-api-key-here"

    # Initialize the software docs reranker
    docs_search = SoftwareDocsReranker(COHERE_API_KEY)

    # Demonstrate with various software documentation queries
    docs_search.demonstrate_software_queries()


if __name__ == "__main__":
    main()

# Key Improvements for Software Documentation:
#
# 1. **Intent Analysis**: Detects if user wants tutorial, reference, troubleshooting, etc.
# 2. **Code-Aware Enhancement**: Boosts queries with programming context
# 3. **Rich Document Representation**: Includes code examples, tags, doc types in embeddings
# 4. **Intent-Based Boosting**: Boosts relevant doc types after reranking
# 5. **Structured Results**: Returns doc sections with metadata for better LLM context
#
# Benefits:
# - "How to" queries get tutorials, not just API references
# - Error-related queries surface troubleshooting docs first
# - Code examples are properly weighted and included
# - Framework-specific context is preserved and enhanced