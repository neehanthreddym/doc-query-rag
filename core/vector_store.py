import os
import uuid
import chromadb
import numpy as np
from typing import List
from langchain_core.documents import Document

class VectorStore:
    """Handles vector store operations using ChromaDB."""

    MAX_BATCH_SIZE = 5000
    
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

        # Process in batches to avoid exceeding ChromaDB's max batch size
        total_added = 0
        num_batches = (len(documents) + self.MAX_BATCH_SIZE - 1) // self.MAX_BATCH_SIZE

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.MAX_BATCH_SIZE
            end_idx = min((batch_idx + 1) * self.MAX_BATCH_SIZE, len(documents))
            
            batch_docs = documents[start_idx:end_idx]
            batch_embeddings = embeddings[start_idx:end_idx]
        
            ids = [f"doc_{uuid.uuid4().hex}" for _ in batch_docs]
            metadatas = [doc.metadata for doc in batch_docs]
            doc_contents = [doc.page_content for doc in batch_docs]
        
            try:
                self.collection.add(
                    ids=ids,
                    metadatas=metadatas,
                    documents=doc_contents,
                    embeddings=batch_embeddings.tolist()
                )
                total_added += len(batch_docs)
                print(f"✅ Batch {batch_idx + 1}/{num_batches} added ({len(batch_docs)} documents). Total: {total_added}")
            except Exception as e:
                print(f"❌ Error adding batch {batch_idx + 1}: {e}")
                raise
        
        print(f"✅ Successfully added {self.collection.count()} total documents to the store.")