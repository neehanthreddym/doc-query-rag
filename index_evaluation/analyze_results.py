"""Analysis and visualization script for ANN benchmark results."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_benchmark_results(filepath: str = "./index_evaluation/benchmark_results.csv") -> pd.DataFrame:
    """Load benchmark results from CSV."""
    print(f"📊 Loading benchmark results from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df)} benchmark results")
    return df


def create_comparison_plots(df: pd.DataFrame, output_dir: str = "./index_evaluation"):
    """Create comparison plots for all metrics."""
    print("\n📈 Generating comparison plots...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Create a figure with subplots for all metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('ANN Algorithms Benchmark Comparison', fontsize=16, fontweight='bold')
    
    algorithms = df['algorithm'].values
    
    # 1. Build Time
    ax = axes[0, 0]
    bars = ax.bar(range(len(df)), df['build_time_s'], color=colors)
    ax.set_title('Build Time (seconds)', fontweight='bold')
    ax.set_ylabel('Time (s)')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, df['build_time_s'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}s', ha='center', va='bottom', fontsize=9)
    
    # 2. Query Speed (QPS)
    ax = axes[0, 1]
    bars = ax.bar(range(len(df)), df['qps'], color=colors)
    ax.set_title('Query Speed (QPS)', fontweight='bold')
    ax.set_ylabel('Queries per Second')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, df['qps'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Average Latency
    ax = axes[0, 2]
    bars = ax.bar(range(len(df)), df['avg_latency_ms'], color=colors)
    ax.set_title('Average Query Latency (ms)', fontweight='bold')
    ax.set_ylabel('Latency (ms)')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, df['avg_latency_ms'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}ms', ha='center', va='bottom', fontsize=9)
    
    # 4. Recall@10
    ax = axes[1, 0]
    recall_col = [col for col in df.columns if 'recall_at_' in col][0]
    bars = ax.bar(range(len(df)), df[recall_col], color=colors)
    ax.set_title(f'Search Accuracy ({recall_col.replace("_", " ").title()})', fontweight='bold')
    ax.set_ylabel('Recall')
    ax.set_ylim([0, 1.1])
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, label='Perfect Recall')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    for i, (bar, val) in enumerate(zip(bars, df[recall_col])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Memory Usage
    ax = axes[1, 1]
    bars = ax.bar(range(len(df)), df['memory_mb'], color=colors)
    ax.set_title('Memory Usage (MB)', fontweight='bold')
    ax.set_ylabel('Memory (MB)')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, df['memory_mb'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}MB', ha='center', va='bottom', fontsize=9)
    
    # 6. Overall Performance Score (normalized)
    ax = axes[1, 2]
    # Normalize metrics (higher is better)
    norm_qps = df['qps'] / df['qps'].max()
    norm_recall = df[recall_col]
    norm_build_time = 1 - (df['build_time_s'] / df['build_time_s'].max())
    norm_latency = 1 - (df['avg_latency_ms'] / df['avg_latency_ms'].max())
    norm_memory = 1 - (df['memory_mb'] / df['memory_mb'].max())
    
    # Calculate overall score (weighted average)
    overall_score = (norm_qps * 0.3 + norm_recall * 0.3 + 
                     norm_latency * 0.2 + norm_build_time * 0.1 + norm_memory * 0.1)
    
    bars = ax.bar(range(len(df)), overall_score, color=colors)
    ax.set_title('Overall Performance Score\n(Weighted: QPS 30%, Recall 30%, Latency 20%, Build 10%, Memory 10%)', 
                 fontweight='bold', fontsize=10)
    ax.set_ylabel('Normalized Score')
    ax.set_ylim([0, 1.1])
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, overall_score)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_path / "benchmark_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparison plot to: {plot_file}")
    
    plt.close()


def create_radar_chart(df: pd.DataFrame, output_dir: str = "./index_evaluation"):
    """Create a radar chart comparing all algorithms across normalized metrics."""
    print("\n📊 Generating radar chart...")
    
    output_path = Path(output_dir)
    
    # Normalize metrics
    recall_col = [col for col in df.columns if 'recall_at_' in col][0]
    
    metrics = {
        'QPS': df['qps'] / df['qps'].max(),
        'Recall': df[recall_col],
        'Speed\n(1/Latency)': 1 - (df['avg_latency_ms'] / df['avg_latency_ms'].max()),
        'Fast Build\n(1/Build Time)': 1 - (df['build_time_s'] / df['build_time_s'].max()),
        'Low Memory\n(1/Memory)': 1 - (df['memory_mb'] / df['memory_mb'].max())
    }
    
    categories = list(metrics.keys())
    N = len(categories)
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='polar')
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Plot data for each algorithm
    for idx, row in df.iterrows():
        values = [metrics[cat].iloc[idx] for cat in categories]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['algorithm'], color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    ax.grid(True)
    
    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('ANN Algorithms: Normalized Performance Comparison', 
              size=14, fontweight='bold', pad=20)
    
    # Save plot
    plot_file = output_path / "benchmark_radar.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved radar chart to: {plot_file}")
    
    plt.close()


def generate_markdown_report(df: pd.DataFrame, output_dir: str = "./index_evaluation"):
    """Generate a markdown report with analysis."""
    print("\n📝 Generating markdown report...")
    
    output_path = Path(output_dir)
    report_file = output_path / "BENCHMARK_REPORT.md"
    
    recall_col = [col for col in df.columns if 'recall_at_' in col][0]
    recall_k = recall_col.split('_')[-1]
    
    # Determine best algorithm for each metric
    best_qps = df.loc[df['qps'].idxmax(), 'algorithm']
    best_recall = df.loc[df[recall_col].idxmax(), 'algorithm']
    best_build = df.loc[df['build_time_s'].idxmin(), 'algorithm']
    best_latency = df.loc[df['avg_latency_ms'].idxmin(), 'algorithm']
    best_memory = df.loc[df['memory_mb'].idxmin(), 'algorithm']
    
    report = f"""# ANN Algorithm Benchmark Report

## Overview

This report presents a comprehensive benchmark comparison of three Approximate Nearest Neighbor (ANN) algorithms for the RAG-based Q&A system:

1. **Spotify's Annoy** - A C++ library with Python bindings for approximate nearest neighbors
2. **FAISS HNSW** - Facebook AI Similarity Search with Hierarchical Navigable Small World graphs
3. **Google's ScaNN** - Scalable Nearest Neighbors from Google Research

## Evaluation Metrics

The following metrics were used to evaluate each algorithm:

- **Query Speed (QPS)**: Queries processed per second (higher is better)
- **Search Accuracy (Recall@{recall_k})**: Percentage of correct results in top-{recall_k} (higher is better)
- **Memory Usage (MB)**: RAM consumed by the index (lower is better)
- **Build Time (s)**: Time required to build the index (lower is better)
- **Average Latency (ms)**: Average time per query (lower is better)

## Test Configuration

- **Document Corpus**: Research papers from `data/pdf_files`
- **Embedding Model**: `{df.iloc[0]['algorithm'].split('(')[0].strip() if len(df) > 0 else 'sentence-transformers/all-MiniLM-L6-v2'}`
- **Chunk Size**: 500 tokens with 50 token overlap
- **Number of Test Queries**: 20 queries sampled from the corpus
- **Top-K Retrieval**: {recall_k} documents per query

## Results Summary

### Benchmark Results Table

{df.to_markdown(index=False)}

### Key Findings

#### 1. Query Speed (QPS) 🚀
- **Winner**: {best_qps}
- **Analysis**: This algorithm demonstrated the highest throughput, making it ideal for high-traffic applications requiring fast response times.

#### 2. Search Accuracy (Recall@{recall_k}) 🎯
- **Winner**: {best_recall}
- **Analysis**: This algorithm achieved the best recall, ensuring more relevant results are retrieved in the top-{recall_k} documents.

#### 3. Build Time ⏱️
- **Winner**: {best_build}
- **Analysis**: This algorithm was fastest to build the index, making it suitable for scenarios requiring frequent index updates.

#### 4. Query Latency ⚡
- **Winner**: {best_latency}
- **Analysis**: This algorithm had the lowest average latency per query, providing the most consistent query performance.

#### 5. Memory Usage 💾
- **Winner**: {best_memory}
- **Analysis**: This algorithm was most memory-efficient, making it suitable for resource-constrained environments.

## Detailed Analysis

### Performance Trade-offs

"""
    
    # Add algorithm-specific analysis
    for idx, row in df.iterrows():
        algo_name = row['algorithm'].split('(')[0].strip()
        report += f"""
#### {algo_name}

- **Strengths**:
"""
        
        # Determine strengths
        if row['algorithm'] == best_qps:
            report += f"  - Highest query speed ({row['qps']:.2f} QPS)\n"
        if row['algorithm'] == best_recall:
            report += f"  - Best recall ({row[recall_col]:.4f})\n"
        if row['algorithm'] == best_build:
            report += f"  - Fastest build time ({row['build_time_s']:.4f}s)\n"
        if row['algorithm'] == best_latency:
            report += f"  - Lowest query latency ({row['avg_latency_ms']:.4f}ms)\n"
        if row['algorithm'] == best_memory:
            report += f"  - Most memory efficient ({row['memory_mb']:.2f} MB)\n"
        
        report += f"""
- **Performance Summary**:
  - QPS: {row['qps']:.2f} queries/second
  - Recall@{recall_k}: {row[recall_col]:.4f}
  - Build Time: {row['build_time_s']:.4f}s
  - Avg Latency: {row['avg_latency_ms']:.4f}ms
  - Memory: {row['memory_mb']:.2f} MB
"""
    
    report += """
## Recommendations

### For Production RAG Systems:

1. **High-Traffic Applications** (Latency-Critical):
   - Consider the algorithm with the best QPS/latency balance
   - Prioritize query speed over build time

2. **High-Accuracy Requirements**:
   - Choose the algorithm with the highest recall
   - Acceptable to trade some speed for better accuracy

3. **Resource-Constrained Environments**:
   - Select the most memory-efficient algorithm
   - Balance memory usage with acceptable performance

4. **Frequent Index Updates**:
   - Opt for the algorithm with fastest build time
   - Important for dynamic document collections

## Visualizations

The following visualizations are available:

1. **benchmark_comparison.png** - Bar charts comparing all metrics across algorithms
2. **benchmark_radar.png** - Radar chart showing normalized performance across all dimensions

## Conclusion

Each ANN algorithm has its own strengths and trade-offs. The choice depends on your specific requirements:

- If **query speed** is paramount → Choose the fastest QPS algorithm
- If **accuracy** is critical → Choose the highest recall algorithm
- If **memory** is limited → Choose the most memory-efficient algorithm
- If **build time** matters → Choose the fastest build algorithm

For this RAG-based Q&A system, we recommend evaluating the algorithms based on your production workload characteristics and resource constraints.

## Methodology Notes

- All benchmarks were run on the same hardware configuration
- Measurements include warmup queries to ensure fair comparison
- Memory measurements are approximate and may vary based on Python's memory management
- Results are reproducible using the provided benchmark script

---

**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Saved markdown report to: {report_file}")


def main():
    """Main entry point for analysis and visualization."""
    print("=" * 80)
    print("ANN BENCHMARK ANALYSIS AND VISUALIZATION")
    print("=" * 80)
    
    # Load results
    df = load_benchmark_results()
    
    # Generate visualizations
    create_comparison_plots(df)
    create_radar_chart(df)
    
    # Generate report
    generate_markdown_report(df)
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete! Check the index_evaluation directory for:")
    print("   - benchmark_results.csv")
    print("   - benchmark_comparison.png")
    print("   - benchmark_radar.png")
    print("   - BENCHMARK_REPORT.md")
    print("=" * 80)


if __name__ == "__main__":
    main()
