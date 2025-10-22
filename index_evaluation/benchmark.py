"""Benchmark script for evaluating different ANN algorithms on the RAG system."""
import os
import sys
import time
import numpy as np
import pandas as pd
import psutil
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import load_pdf_documents, split_documents
from core.embedding_manager import EmbeddingManager
from index_evaluation.annoy_store import AnnoyVectorStore
from index_evaluation.faiss_hnsw_store import FAISSHNSWVectorStore
from index_evaluation.scann_store import ScaNNVectorStore
from index_evaluation.vector_store_interface import VectorStoreInterface
import config


class ANNBenchmark:
    """Benchmark different ANN algorithms for vector search."""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.documents = []
        self.embeddings = None
        self.doc_metadata = []
        
    def load_and_prepare_data(self):
        """Load documents and generate embeddings."""
        print("=" * 80)
        print("LOADING AND PREPARING DATA")
        print("=" * 80)
        
        # Load documents
        print(f"\n📚 Loading documents from {config.PDF_DIR}...")
        documents = load_pdf_documents(config.PDF_DIR)
        if not documents:
            raise ValueError("No documents found")
        
        print(f"✅ Loaded {len(documents)} PDF documents")
        
        # Split documents into chunks
        print(f"\n✂️  Splitting documents (chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})...")
        chunked_docs = split_documents(
            documents,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        print(f"✅ Created {len(chunked_docs)} chunks")
        
        # Generate embeddings
        print(f"\n🧠 Generating embeddings...")
        texts = [doc.page_content for doc in chunked_docs]
        self.embeddings = self.embedding_manager.generate_embeddings(texts)
        
        # Store document metadata
        self.doc_metadata = [
            {
                'text': doc.page_content,
                'source': doc.metadata.get('source', 'unknown'),
                'page': doc.metadata.get('page', 0),
                'index': i
            }
            for i, doc in enumerate(chunked_docs)
        ]
        
        print(f"✅ Generated embeddings with shape: {self.embeddings.shape}")
        print(f"   Total documents: {len(self.doc_metadata)}")
        
    def generate_test_queries(self, num_queries: int = 20) -> Tuple[List[str], np.ndarray]:
        """Generate test queries from random document chunks."""
        print(f"\n🔍 Generating {num_queries} test queries from document corpus...")
        
        # Sample random documents as queries
        np.random.seed(42)
        query_indices = np.random.choice(len(self.doc_metadata), size=num_queries, replace=False)
        
        query_texts = [self.doc_metadata[idx]['text'] for idx in query_indices]
        query_embeddings = self.embeddings[query_indices]
        
        print(f"✅ Generated {len(query_texts)} test queries")
        
        return query_texts, query_embeddings, query_indices
    
    def measure_build_time(self, vector_store: VectorStoreInterface) -> float:
        """Measure time to build the index."""
        start_time = time.time()
        vector_store.build(self.embeddings, self.doc_metadata)
        build_time = time.time() - start_time
        return build_time
    
    def measure_query_performance(
        self, 
        vector_store: VectorStoreInterface,
        query_embeddings: np.ndarray,
        top_k: int = 10,
        num_warmup: int = 5
    ) -> Tuple[float, float]:
        """Measure query speed (QPS) and latency."""
        # Warmup queries
        for i in range(min(num_warmup, len(query_embeddings))):
            vector_store.query(query_embeddings[i], top_k)
        
        # Measure actual queries
        start_time = time.time()
        for query_emb in query_embeddings:
            vector_store.query(query_emb, top_k)
        total_time = time.time() - start_time
        
        qps = len(query_embeddings) / total_time
        avg_latency_ms = (total_time / len(query_embeddings)) * 1000
        
        return qps, avg_latency_ms
    
    def calculate_recall_at_k(
        self,
        vector_store: VectorStoreInterface,
        query_embeddings: np.ndarray,
        ground_truth_indices: np.ndarray,
        k: int = 10
    ) -> float:
        """Calculate Recall@k by comparing with ground truth (query's own index should be in top-k)."""
        recalls = []
        
        for i, query_emb in enumerate(query_embeddings):
            results = vector_store.query(query_emb, k)
            retrieved_indices = [r['index'] for r in results]
            
            # Check if the ground truth index is in the retrieved results
            if ground_truth_indices[i] in retrieved_indices:
                recalls.append(1.0)
            else:
                recalls.append(0.0)
        
        return np.mean(recalls)
    
    def measure_memory_usage(self, vector_store: VectorStoreInterface) -> float:
        """Measure memory usage of the index in MB."""
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Build the index
        vector_store.build(self.embeddings, self.doc_metadata)
        
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        mem_usage = mem_after - mem_before
        
        return max(mem_usage, 0)  # Ensure non-negative
    
    def benchmark_algorithm(
        self,
        vector_store: VectorStoreInterface,
        query_embeddings: np.ndarray,
        ground_truth_indices: np.ndarray,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Run complete benchmark for a single algorithm."""
        print(f"\n{'='*80}")
        print(f"BENCHMARKING: {vector_store.name}")
        print(f"{'='*80}")
        
        results = {'algorithm': vector_store.name}
        
        # 1. Measure build time
        print("\n⏱️  Measuring build time...")
        build_time = self.measure_build_time(vector_store)
        results['build_time_s'] = round(build_time, 4)
        print(f"   Build Time: {build_time:.4f}s")
        
        # 2. Measure query performance
        print("\n⚡ Measuring query performance...")
        qps, latency = self.measure_query_performance(vector_store, query_embeddings, top_k)
        results['qps'] = round(qps, 2)
        results['avg_latency_ms'] = round(latency, 4)
        print(f"   QPS: {qps:.2f} queries/second")
        print(f"   Avg Latency: {latency:.4f}ms")
        
        # 3. Calculate recall@k
        print(f"\n🎯 Calculating Recall@{top_k}...")
        recall = self.calculate_recall_at_k(vector_store, query_embeddings, ground_truth_indices, top_k)
        results[f'recall_at_{top_k}'] = round(recall, 4)
        print(f"   Recall@{top_k}: {recall:.4f}")
        
        # 4. Estimate memory usage (rough estimate based on index size)
        print("\n💾 Estimating memory usage...")
        # Create a fresh instance to measure memory
        if isinstance(vector_store, AnnoyVectorStore):
            fresh_store = AnnoyVectorStore(
                embedding_dim=vector_store.embedding_dim,
                n_trees=vector_store.n_trees,
                metric=vector_store.metric
            )
        elif isinstance(vector_store, FAISSHNSWVectorStore):
            fresh_store = FAISSHNSWVectorStore(
                embedding_dim=vector_store.embedding_dim,
                M=vector_store.M,
                ef_construction=vector_store.ef_construction,
                ef_search=vector_store.ef_search
            )
        else:  # ScaNNVectorStore
            fresh_store = ScaNNVectorStore(
                embedding_dim=vector_store.embedding_dim,
                num_leaves=vector_store.num_leaves,
                num_leaves_to_search=vector_store.num_leaves_to_search
            )
        
        memory_mb = self.measure_memory_usage(fresh_store)
        results['memory_mb'] = round(memory_mb, 2)
        print(f"   Memory Usage: {memory_mb:.2f} MB")
        
        print(f"\n✅ Benchmark complete for {vector_store.name}")
        
        return results
    
    def run_all_benchmarks(self, top_k: int = 10) -> pd.DataFrame:
        """Run benchmarks for all ANN algorithms."""
        print("\n" + "=" * 80)
        print("STARTING ANN ALGORITHM BENCHMARK")
        print("=" * 80)
        
        # Generate test queries
        query_texts, query_embeddings, ground_truth_indices = self.generate_test_queries(num_queries=20)
        
        embedding_dim = self.embeddings.shape[1]
        
        # Initialize vector stores
        vector_stores = [
            AnnoyVectorStore(embedding_dim=embedding_dim, n_trees=10, metric='angular'),
            FAISSHNSWVectorStore(embedding_dim=embedding_dim, M=32, ef_construction=200, ef_search=100),
            ScaNNVectorStore(embedding_dim=embedding_dim, num_leaves=100, num_leaves_to_search=10)
        ]
        
        # Run benchmarks
        all_results = []
        for vector_store in vector_stores:
            try:
                results = self.benchmark_algorithm(
                    vector_store,
                    query_embeddings,
                    ground_truth_indices,
                    top_k
                )
                all_results.append(results)
            except Exception as e:
                print(f"\n❌ Error benchmarking {vector_store.name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 80)
        print(df.to_string(index=False))
        
        return df


def main():
    """Main entry point for benchmarking."""
    # Initialize embedding manager
    print("🚀 Initializing Benchmark System...")
    embedding_manager = EmbeddingManager(config.EMBEDDING_MODEL_NAME)
    
    # Create benchmark instance
    benchmark = ANNBenchmark(embedding_manager)
    
    # Load and prepare data
    benchmark.load_and_prepare_data()
    
    # Run benchmarks
    results_df = benchmark.run_all_benchmarks(top_k=10)
    
    # Save results
    output_dir = Path("./index_evaluation")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "benchmark_results.csv"
    
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
