"""Annoy (Spotify's Approximate Nearest Neighbors Oh Yeah) Vector Store Implementation."""
import numpy as np
from typing import List, Dict, Any
from annoy import AnnoyIndex
from .vector_store_interface import VectorStoreInterface


class AnnoyVectorStore(VectorStoreInterface):
    """Vector store implementation using Spotify's Annoy library."""
    
    def __init__(self, embedding_dim: int, n_trees: int = 10, metric: str = 'angular'):
        """
        Initialize Annoy vector store.
        
        Args:
            embedding_dim: Dimension of the embeddings
            n_trees: Number of trees to build (more trees = higher precision, slower build)
            metric: Distance metric ('angular', 'euclidean', 'manhattan', 'hamming', 'dot')
        """
        self.embedding_dim = embedding_dim
        self.n_trees = n_trees
        self.metric = metric
        self.index = AnnoyIndex(embedding_dim, metric)
        self.documents = []
        self.is_built = False
        
    def build(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """Build the Annoy index from embeddings and documents."""
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        self.documents = documents
        
        # Add all embeddings to the index
        for i, embedding in enumerate(embeddings):
            self.index.add_item(i, embedding)
        
        # Build the index
        self.index.build(self.n_trees)
        self.is_built = True
        
    def query(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Query the index for the top_k most similar documents."""
        if not self.is_built:
            raise RuntimeError("Index must be built before querying")
        
        # Get nearest neighbors
        indices, distances = self.index.get_nns_by_vector(
            query_embedding, 
            top_k, 
            include_distances=True
        )
        
        # Return documents with their distances
        results = []
        for idx, dist in zip(indices, distances):
            result = self.documents[idx].copy()
            result['distance'] = float(dist)
            result['score'] = 1.0 / (1.0 + dist)  # Convert distance to similarity score
            results.append(result)
        
        return results
    
    @property
    def name(self) -> str:
        """Returns the name of the vector store implementation."""
        return f"Annoy (n_trees={self.n_trees}, metric={self.metric})"
    
    def save(self, filepath: str):
        """Save the index to disk."""
        if self.is_built:
            self.index.save(filepath)
    
    def load(self, filepath: str):
        """Load the index from disk."""
        self.index.load(filepath)
        self.is_built = True
