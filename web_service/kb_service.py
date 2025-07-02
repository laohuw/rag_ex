import logging

import unicorn
import uvicorn
from fastapi import FastAPI

from llama_ex.llama_to_rag import Llama2RAG

db_path = "/home/hd/py_ex/llama_ex/db/rag_database.db"
model_gguf = "/home/hd/py_ex/models/llama-2-7b-chat.Q4_K_M.gguf"

logger= logging.getLogger("llama_ex.kb_service")
webApp = FastAPI(
        title="Llama Ex API",
        description="Llama Ex API",
        version="0.1.0",
        root_path="/api/v1",
    )

@webApp.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    rag = Llama2RAG(
        llama2_model_path=model_gguf,  # Your Llama-2 model path
        db_path=db_path
    )
    logger.info("Starting up Llama Ex API")

@webApp.get("/")
async def read_root():
    return {"message": "Welcome to the Llama Ex API"}

@webApp.post("/init")
async def init_doc():

    return {"message": "init successfully"}

@webApp.post("/add_document")
async def add_doc():
    documents = [
        "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from and make predictions on data without being explicitly programmed.",
        "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation to provide more accurate and contextual responses.",
        "Vector databases store high-dimensional vectors for similarity search operations, commonly used in AI applications for semantic search.",
        "Llama-2 is a large language model developed by Meta AI that comes in various sizes and is optimized for chat applications."
    ]
    for doc in documents:
        rag.add_document(doc)
    return {"message": "documents added successfully"}

@webApp.post("/query")
async def query_doc():
    return {"message": "Welcome to the Llama Ex API"}

if __name__ == "__main__":

    uvicorn.run("kb_service:webApp", prefix="api", port=8000, reload=True)


