import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self._load_model()
        
    def _load_model(self) -> SentenceTransformer:
        """Loads the SentenceTransformer model."""
        try:
            print(f"🧠 Loading embedding model: {self.model_name}")
            model = SentenceTransformer(self.model_name)
            print(f"✅ Model loaded successfully. Embedding dimension: {model.get_sentence_embedding_dimension()}")
            return model
        except Exception as e:
            print(f"❌ Error loading the model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings for a list of texts."""
        if not self.model:
            raise ValueError("Embedding model is not loaded.")
        
        print(f"🧠 Generating embeddings for {len(texts)} text chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"✅ Generated embeddings of shape: {embeddings.shape}")
        return embeddings