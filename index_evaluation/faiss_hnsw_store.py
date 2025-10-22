"""FAISS HNSW Vector Store Implementation."""
import numpy as np
from typing import List, Dict, Any
import faiss
from .vector_store_interface import VectorStoreInterface


class FAISSHNSWVectorStore(VectorStoreInterface):
    """Vector store implementation using FAISS with HNSW indexing."""
    
    def __init__(self, embedding_dim: int, M: int = 32, ef_construction: int = 200, ef_search: int = 100):
        """
        Initialize FAISS HNSW vector store.
        
        Args:
            embedding_dim: Dimension of the embeddings
            M: Number of connections per layer (higher = better recall, more memory)
            ef_construction: Size of dynamic candidate list during construction (higher = better quality, slower build)
            ef_search: Size of dynamic candidate list during search (higher = better recall, slower search)
        """
        self.embedding_dim = embedding_dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        # Create HNSW index
        self.index = faiss.IndexHNSWFlat(embedding_dim, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
        
        self.documents = []
        self.is_built = False
        
    def build(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """Build the FAISS HNSW index from embeddings and documents."""
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        self.documents = documents
        
        # Ensure embeddings are contiguous and in float32
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        
        # Add all embeddings to the index
        self.index.add(embeddings)
        self.is_built = True
        
    def query(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Query the index for the top_k most similar documents."""
        if not self.is_built:
            raise RuntimeError("Index must be built before querying")
        
        # Ensure query embedding is in the right format
        query_embedding = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        
        # Search for nearest neighbors
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Return documents with their distances
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:  # FAISS returns -1 for invalid indices
                result = self.documents[idx].copy()
                result['distance'] = float(dist)
                # Convert L2 distance to similarity score
                result['score'] = 1.0 / (1.0 + dist)
                results.append(result)
        
        return results
    
    @property
    def name(self) -> str:
        """Returns the name of the vector store implementation."""
        return f"FAISS-HNSW (M={self.M}, ef_construction={self.ef_construction}, ef_search={self.ef_search})"
    
    def save(self, filepath: str):
        """Save the index to disk."""
        if self.is_built:
            faiss.write_index(self.index, filepath)
    
    def load(self, filepath: str):
        """Load the index from disk."""
        self.index = faiss.read_index(filepath)
        self.is_built = True
