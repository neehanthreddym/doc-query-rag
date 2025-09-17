import os
import uuid
import chromadb
import numpy as np
from typing import List
from langchain_core.documents import Document

class VectorStore:
    """Handles vector store operations using ChromaDB."""
    
    def __init__(self, collection_name: str, persist_directory: str):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = self._initialize_client()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Collection of document embeddings"}
        )
        print(f"✅ ChromaDB client initialized. Collection '{self.collection_name}' loaded.")
        print(f"   Existing documents in collection: {self.collection.count()}")

    def _initialize_client(self) -> chromadb.PersistentClient:
        """Initializes the persistent ChromaDB client."""
        os.makedirs(self.persist_directory, exist_ok=True)
        return chromadb.PersistentClient(path=self.persist_directory)
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Adds documents and their embeddings to the vector store."""
        if not documents:
            print("No documents to add.")
            return

        print(f"Adding {len(documents)} documents to the vector store...")
        
        ids = [f"doc_{uuid.uuid4().hex}" for _ in documents]
        metadatas = [doc.metadata for doc in documents]
        doc_contents = [doc.page_content for doc in documents]
        
        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=doc_contents,
                embeddings=embeddings.tolist()
            )
            print(f"✅ Successfully added {self.collection.count()} total documents to the store.")
        except Exception as e:
            print(f"❌ Error adding documents to vector store: {e}")
            raise