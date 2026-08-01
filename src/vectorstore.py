import os
import faiss
import numpy as np
import pickle
from typing import List, Any, Dict, Optional
from sentence_transformers import SentenceTransformer
from .embeddings import EmbeddingPipeline


class FaissVectorStore:
    """Manages document embeddings using a FAISS vector store."""

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")
        self.load()

    def build_from_documents(self, documents: List[Any]):
        """Chunk documents, generate embeddings, and store in FAISS index."""
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = emb_pipe.chunk_documents(documents)
        texts = [doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in chunks]
        embeddings = emb_pipe.generate_embeddings(texts)
        metadatas = [getattr(doc, "metadata", {}) for doc in chunks]
        self.add_embeddings(texts, embeddings, metadatas)
        self.save()

    def add_embeddings(self, texts: List[str], embeddings: np.ndarray, metadatas: Optional[List[Dict[str, Any]]] = None):
        """Add pre-computed embeddings to the FAISS index."""
        embeddings = np.array(embeddings, dtype=np.float32)
        dimension = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        if metadatas is None:
            metadatas = [{} for _ in texts]
        for text, meta in zip(texts, metadatas):
            meta_entry = dict(meta)
            meta_entry["text"] = text
            self.metadata.append(meta_entry)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for top_k most similar documents for a query string."""
        if self.index is None or self.index.ntotal == 0:
            return []
        query_vector = np.array(self.model.encode([query]), dtype=np.float32)
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                item = dict(self.metadata[idx])
                item["score"] = float(dist)
                results.append(item)
        return results

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Alias for search()."""
        return self.search(query, top_k=top_k)

    def save(self):
        """Persist FAISS index and metadata to disk."""
        if self.index is not None:
            index_path = os.path.join(self.persist_dir, "index.faiss")
            meta_path = os.path.join(self.persist_dir, "metadata.pkl")
            faiss.write_index(self.index, index_path)
            with open(meta_path, "wb") as f:
                pickle.dump(self.metadata, f)
            print(f"[INFO] Saved vector store to {self.persist_dir}")

    def load(self):
        """Load FAISS index and metadata from disk if they exist."""
        index_path = os.path.join(self.persist_dir, "index.faiss")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"[INFO] Loaded existing vector store from {self.persist_dir}")
