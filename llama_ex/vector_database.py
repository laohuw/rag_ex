import json
import logging
import pickle
import sqlite3
from typing import List, Tuple

import faiss
import numpy as np

from llama_ex.document import Document


class VectorDatabase:
    """Persistent vector database using SQLite and FAISS"""

    def __init__(self, db_path: str, embedding_dim: int = 384):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.logger = logging.getLogger(self.__class__.__name__)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        # self._init_database()
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
            self.logger.info(f"Loaded {len(embeddings)} embeddings into FAISS index")

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
        self.logger.info(f"Inserted document {doc_id}")

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
            index=int(idx+1)
            cursor.execute('''
                SELECT d.id, d.content, d.metadata, d.created_at
                FROM documents d
                JOIN faiss_metadata f ON d.id = f.doc_id
                WHERE f.id = ?
            ''', (index,))  # FAISS indices are 0-based, SQLite IDs are 1-based

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

