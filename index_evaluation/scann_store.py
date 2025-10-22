"""ScaNN (Google's Scalable Nearest Neighbors) Vector Store Implementation."""
import numpy as np
from typing import List, Dict, Any
import scann
from .vector_store_interface import VectorStoreInterface


class ScaNNVectorStore(VectorStoreInterface):
    """Vector store implementation using Google's ScaNN library."""
    
    def __init__(self, embedding_dim: int, num_leaves: int = 100, num_leaves_to_search: int = 10):
        """
        Initialize ScaNN vector store.
        
        Args:
            embedding_dim: Dimension of the embeddings
            num_leaves: Number of partitions in the search tree (higher = better precision, more memory)
            num_leaves_to_search: Number of leaves to search (higher = better recall, slower search)
        """
        self.embedding_dim = embedding_dim
        self.num_leaves = num_leaves
        self.num_leaves_to_search = num_leaves_to_search
        self.searcher = None
        self.documents = []
        self.embeddings_array = None
        self.is_built = False
        
    def build(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """Build the ScaNN index from embeddings and documents."""
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        self.documents = documents
        
        # Ensure embeddings are contiguous and in float32
        self.embeddings_array = np.ascontiguousarray(embeddings, dtype=np.float32)
        
        # Build ScaNN searcher with tree and scoring quantization
        self.searcher = (
            scann.scann_ops_pybind.builder(self.embeddings_array, 10, "dot_product")
            .tree(
                num_leaves=self.num_leaves,
                num_leaves_to_search=self.num_leaves_to_search,
                training_sample_size=min(len(embeddings), 100000)
            )
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(100)
            .build()
        )
        self.is_built = True
        
    def query(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Query the index for the top_k most similar documents."""
        if not self.is_built:
            raise RuntimeError("Index must be built before querying")
        
        # Ensure query embedding is in the right format
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
        
        # Search for nearest neighbors
        indices, distances = self.searcher.search(query_embedding, final_num_neighbors=top_k)
        
        # Return documents with their distances
        results = []
        for idx, dist in zip(indices, distances):
            result = self.documents[idx].copy()
            result['distance'] = float(dist)
            # ScaNN returns dot product similarity, convert to score
            result['score'] = float(dist)
            results.append(result)
        
        return results
    
    @property
    def name(self) -> str:
        """Returns the name of the vector store implementation."""
        return f"ScaNN (num_leaves={self.num_leaves}, num_leaves_to_search={self.num_leaves_to_search})"
    
    def save(self, filepath: str):
        """Save the index to disk."""
        if self.is_built and self.searcher is not None:
            self.searcher.serialize(filepath)
    
    def load(self, filepath: str):
        """Load the index from disk - Note: ScaNN serialization requires embeddings."""
        # ScaNN doesn't support direct loading, need to rebuild
        raise NotImplementedError("ScaNN load functionality requires embeddings array")
