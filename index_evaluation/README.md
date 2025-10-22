# ANN Algorithm Benchmarking for RAG System

This directory contains implementations and benchmarking tools for comparing different Approximate Nearest Neighbor (ANN) algorithms for vector similarity search in the RAG-based Q&A system.

## 📁 Directory Structure

```
index_evaluation/
├── vector_store_interface.py    # Abstract interface for vector stores
├── annoy_store.py               # Spotify's Annoy implementation
├── faiss_hnsw_store.py          # FAISS HNSW implementation
├── scann_store.py               # Google's ScaNN implementation
├── benchmark.py                 # Benchmark execution script
├── analyze_results.py           # Analysis and visualization script
├── generate_sample_results.py  # Sample data generator (for demo)
├── benchmark_results.csv        # Benchmark results (CSV format)
├── benchmark_comparison.png     # Comparison bar charts
├── benchmark_radar.png          # Radar chart visualization
└── BENCHMARK_REPORT.md          # Detailed analysis report
```

## 🎯 Benchmarked Algorithms

### 1. Spotify's Annoy
- **Library**: `annoy`
- **Method**: Tree-based approximate nearest neighbors
- **Strengths**: Fast build time, low memory footprint
- **Use Case**: Resource-constrained environments, frequent index updates

### 2. FAISS HNSW
- **Library**: `faiss-cpu`
- **Method**: Hierarchical Navigable Small World graphs
- **Strengths**: High recall, balanced performance
- **Use Case**: High-accuracy requirements

### 3. Google's ScaNN
- **Library**: `scann`
- **Method**: Vector quantization with tree-based search
- **Strengths**: Highest query speed, low latency
- **Use Case**: High-traffic, latency-critical applications

## 📊 Evaluation Metrics

The benchmark evaluates each algorithm across five key metrics:

1. **Query Speed (QPS)** - Queries processed per second (higher is better)
2. **Search Accuracy (Recall@k)** - Percentage of correct results in top-k (higher is better)
3. **Memory Usage (MB)** - RAM consumed by the index (lower is better)
4. **Build Time (s)** - Time required to build the index (lower is better)
5. **Average Latency (ms)** - Average time per query (lower is better)

## 🚀 Usage

### Running the Full Benchmark

To run the complete benchmark on your document corpus:

```bash
# Make sure all dependencies are installed
pip install annoy scann pandas matplotlib psutil

# Run the benchmark (requires network access to download embedding model)
python index_evaluation/benchmark.py
```

**Note**: The benchmark script will:
1. Load all PDFs from `data/pdf_files`
2. Split them into chunks
3. Generate embeddings using sentence-transformers
4. Build indices for all three ANN algorithms
5. Run performance tests
6. Save results to `benchmark_results.csv`

### Generating Sample Results (Demo Mode)

If you want to see the analysis without running the full benchmark:

```bash
# Generate sample benchmark data
python index_evaluation/generate_sample_results.py

# This creates sample benchmark_results.csv for demonstration
```

### Analyzing Results

After running the benchmark (or generating sample data):

```bash
# Generate visualizations and report
python index_evaluation/analyze_results.py
```

This will create:
- `benchmark_comparison.png` - Bar charts comparing all metrics
- `benchmark_radar.png` - Radar chart showing normalized performance
- `BENCHMARK_REPORT.md` - Detailed analysis and recommendations

## 🔧 Customizing the Benchmark

### Adjusting Algorithm Parameters

Edit the parameters in `benchmark.py`:

```python
# Annoy parameters
AnnoyVectorStore(embedding_dim=384, n_trees=10, metric='angular')

# FAISS HNSW parameters
FAISSHNSWVectorStore(embedding_dim=384, M=32, ef_construction=200, ef_search=100)

# ScaNN parameters
ScaNNVectorStore(embedding_dim=384, num_leaves=100, num_leaves_to_search=10)
```

### Changing Test Configuration

Modify these settings in `benchmark.py`:

```python
# Number of test queries
benchmark.generate_test_queries(num_queries=20)

# Top-K retrieval
benchmark.run_all_benchmarks(top_k=10)
```

## 📈 Interpreting Results

### Quick Decision Guide

Choose based on your priorities:

- **Speed is critical** → ScaNN (highest QPS, lowest latency)
- **Accuracy is critical** → FAISS HNSW (highest recall)
- **Memory is limited** → Annoy (lowest memory usage)
- **Frequent updates** → Annoy (fastest build time)

### Understanding the Visualizations

**Comparison Plot** (`benchmark_comparison.png`):
- Shows side-by-side comparison of all metrics
- Each subplot focuses on one metric
- Includes actual values on top of bars

**Radar Chart** (`benchmark_radar.png`):
- Shows normalized performance across all dimensions
- Larger area = better overall performance
- Easy to spot algorithm strengths at a glance

## 🔬 Implementation Details

### VectorStoreInterface

All implementations follow this common interface:

```python
class VectorStoreInterface(ABC):
    @abstractmethod
    def build(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """Build the index from embeddings and documents"""
        pass
    
    @abstractmethod
    def query(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Query the index for top_k most similar documents"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the implementation"""
        pass
```

### Adding New Algorithms

To add a new ANN algorithm:

1. Create a new file (e.g., `new_algorithm_store.py`)
2. Implement the `VectorStoreInterface`
3. Add it to `__init__.py`
4. Include it in `benchmark.py`'s `vector_stores` list

## 📝 Benchmark Results

See `BENCHMARK_REPORT.md` for detailed analysis of the benchmark results, including:
- Performance comparison table
- Algorithm-specific strengths
- Recommendations for different use cases
- Methodology notes

## 🛠️ Dependencies

```
annoy>=1.17.3
faiss-cpu>=1.12.0
scann>=1.3.2
pandas>=2.0.0
matplotlib>=3.7.0
psutil>=5.9.0
sentence-transformers>=5.1.0
```

## 📚 References

- [Annoy Documentation](https://github.com/spotify/annoy)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [ScaNN Documentation](https://github.com/google-research/google-research/tree/master/scann)
- [Sentence Transformers](https://www.sbert.net/)

## 🤝 Contributing

To improve the benchmarking framework:

1. Add new evaluation metrics in `benchmark.py`
2. Implement additional visualizations in `analyze_results.py`
3. Test with different document collections
4. Share your benchmark results

## ⚠️ Known Limitations

- Network access required for downloading embedding model (first run)
- Memory usage measurements are approximate
- Results may vary based on hardware configuration
- ScaNN may require specific CPU features for optimal performance

## 📄 License

This benchmarking framework is part of the doc-query-rag project and follows the same license.
