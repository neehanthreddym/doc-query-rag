from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class VectorStoreInterface(ABC):
    """A universal interface for vector store operations."""

    @abstractmethod
    def build(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """Builds the index from embeddings and documents."""
        pass

    @abstractmethod
    def query(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Queries the index for the top_k most similar documents."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the vector store implementation."""
        pass