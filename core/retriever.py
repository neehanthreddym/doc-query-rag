from typing import List, Dict, Any
from .vector_store import VectorStore
from .embedding_manager import EmbeddingManager

class RAGRetriever:
    """Handles retrieval of relevant documents from the vector store."""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Retrieves the top_k most relevant documents for a given query."""
        print(f"🔎 Retrieving top {top_k} documents for query: '{query}'")
        
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        retrieved_docs = []
        if results.get('documents'):
            for doc_id, content, metadata, distance in zip(
                results['ids'][0], results['documents'][0], results['metadatas'][0], results['distances'][0]
            ):
                similarity_score = 1 - distance
                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        'id': doc_id,
                        'content': content,
                        'metadata': metadata,
                        'score': similarity_score
                    })
        
        print(f"✅ Retrieved {len(retrieved_docs)} documents meeting the threshold.")
        return retrieved_docs