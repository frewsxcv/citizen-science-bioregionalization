# Automatic Cluster Selection Guide

This guide explains how to use the automatic cluster optimization tools to determine the optimal number of clusters for your bioregionalization analysis.

## Overview

Instead of manually choosing the number of clusters (`k`), the optimization module evaluates multiple `k` values using statistical metrics to suggest the best choice. This makes your analysis:

- **More reproducible** - Others can understand why you chose a specific `k`
- **More defensible** - Backed by multiple statistical measures
- **More efficient** - Quickly evaluate many options

## Quick Start

### Basic Usage

```python
from src.cluster_optimization import ClusterOptimizer, create_metrics_report
from src.plot.cluster_optimization import plot_optimization_metrics

# After building your geocode_dataframe, distance_matrix, and connectivity_matrix:

# 1. Create optimizer
optimizer = ClusterOptimizer(
    geocode_dataframe=geocode_dataframe,
    distance_matrix=distance_matrix,
    connectivity_matrix=connectivity_matrix,
)

# 2. Evaluate range of k values (e.g., 5 to 20 clusters)
metrics_list = optimizer.evaluate_k_range(k_min=5, k_max=20)

# 3. Get optimal k suggestion
result = optimizer.suggest_optimal_k(metrics_list, method="multi_criteria")

# 4. Print report
print(create_metrics_report(result))

# 5. Visualize results
fig = plot_optimization_metrics(result)
fig.savefig("cluster_optimization.png", dpi=150, bbox_inches="tight")

# 6. Use the optimal k for clustering
optimal_k = result.optimal_k
geocode_cluster_dataframe = GeocodeClusterSchema.build(
    geocode_dataframe=geocode_dataframe,
    distance_matrix=distance_matrix,
    connectivity_matrix=connectivity_matrix,
    num_clusters=optimal_k,
)
```

## Selection Methods

The `suggest_optimal_k()` method supports four different approaches:

### 1. Multi-Criteria (Default) - Recommended

Balances multiple metrics using weighted ranking:
- **Silhouette score** (40% weight) - Measures cluster quality
- **Davies-Bouldin index** (30% weight) - Measures cluster separation
- **Calinski-Harabasz score** (20% weight) - Variance ratio
- **Cluster balance** (10% weight) - Penalizes very unbalanced clusters

```python
result = optimizer.suggest_optimal_k(metrics_list, method="multi_criteria")
```

**Use when:** You want a well-rounded solution that considers all aspects of cluster quality.

### 2. Silhouette

Chooses `k` with the highest silhouette score:

```python
result = optimizer.suggest_optimal_k(metrics_list, method="silhouette")
```

**Use when:** You prioritize well-separated, cohesive clusters above all else.

**Interpretation:**
- `> 0.7` - Excellent clustering
- `0.5 - 0.7` - Good clustering
- `0.3 - 0.5` - Fair clustering
- `< 0.3` - Poor clustering

### 3. Elbow Method

Finds the "elbow" in the inertia curve (point of diminishing returns):

```python
result = optimizer.suggest_optimal_k(metrics_list, method="elbow")
```

**Use when:** You want to balance cluster quality with parsimony (fewer clusters preferred).

### 4. Compromise

Finds the lowest `k` that still achieves good quality (silhouette > 0.3):

```python
result = optimizer.suggest_optimal_k(metrics_list, method="compromise")
```

**Use when:** You prefer simpler solutions (fewer clusters) as long as quality is acceptable.

## Understanding the Metrics

### Silhouette Score
- **Range:** -1 to 1
- **Interpretation:** Higher is better
- **Meaning:** Measures how similar each hexagon is to its own cluster compared to other clusters
- **Formula:** `(b - a) / max(a, b)` where `a` = mean intra-cluster distance, `b` = mean nearest-cluster distance

### Davies-Bouldin Index
- **Range:** 0 to ∞
- **Interpretation:** Lower is better
- **Meaning:** Ratio of within-cluster to between-cluster distances
- **Good values:** < 1.0

### Calinski-Harabasz Score
- **Range:** 0 to ∞
- **Interpretation:** Higher is better
- **Meaning:** Ratio of between-cluster variance to within-cluster variance
- **Good values:** Depends on data, but higher indicates better-defined clusters

### Inertia
- **Range:** 0 to ∞
- **Interpretation:** Lower is better (but look for elbow, not minimum)
- **Meaning:** Sum of squared distances to nearest cluster center
- **Use:** Find the point where adding more clusters doesn't significantly reduce inertia

## Example Workflow in Notebook

```python
import marimo as mo

# UI for optimization range
k_min_ui = mo.ui.number(value=5, label="Minimum clusters to evaluate")
k_max_ui = mo.ui.number(value=20, label="Maximum clusters to evaluate")
mo.hstack([k_min_ui, k_max_ui])

# UI for selection method
method_ui = mo.ui.dropdown(
    options={
        "multi_criteria": "Multi-Criteria (Recommended)",
        "silhouette": "Best Silhouette Score",
        "elbow": "Elbow Method",
        "compromise": "Quality Compromise",
    },
    value="multi_criteria",
    label="Selection method"
)
method_ui

# Run optimization
optimizer = ClusterOptimizer(
    geocode_dataframe=geocode_no_edges_dataframe,
    distance_matrix=geocode_distance_matrix,
    connectivity_matrix=geocode_connectivity_matrix,
)

metrics_list = optimizer.evaluate_k_range(
    k_min=k_min_ui.value,
    k_max=k_max_ui.value,
)

result = optimizer.suggest_optimal_k(metrics_list, method=method_ui.value)

# Display report
mo.md(f"```\n{create_metrics_report(result)}\n```")

# Plot results
fig = plot_optimization_metrics(result)
mo.mpl.interactive(fig)

# Use optimal k
optimal_k = result.optimal_k
mo.md(f"**Using k={optimal_k} for clustering**")
```

## Visualization Guide

### Main Optimization Plot

The `plot_optimization_metrics()` function creates a 6-panel figure:

1. **Silhouette Score** - Line plot showing quality across k values
2. **Davies-Bouldin Index** - Lower values indicate better separation
3. **Calinski-Harabasz Score** - Higher values indicate better-defined clusters
4. **Inertia (Elbow Plot)** - Look for the "elbow" where improvement slows
5. **Cluster Size Distribution** - Shows mean, min, and max cluster sizes
6. **Combined Score** - Normalized average of all metrics

### Focused Plots

For deeper analysis:

```python
from src.plot.cluster_optimization import (
    plot_silhouette_comparison,
    plot_elbow_detail,
)

# Detailed silhouette analysis
fig1 = plot_silhouette_comparison(metrics_list, result.optimal_k)
fig1.savefig("silhouette_detail.png")

# Detailed elbow analysis with derivatives
fig2 = plot_elbow_detail(metrics_list, result.optimal_k)
fig2.savefig("elbow_detail.png")
```

## Converting Metrics to DataFrame

For custom analysis or export:

```python
from src.cluster_optimization import metrics_to_dataframe

df = metrics_to_dataframe(metrics_list)
print(df)

# Export to CSV
df.write_csv("cluster_metrics.csv")

# Find k with best silhouette
best_silhouette = df.filter(
    pl.col("silhouette") == pl.col("silhouette").max()
)
print(best_silhouette)
```

## Tips and Best Practices

### 1. Choose Appropriate k Range

- **Too small** (k < 5): May not capture ecological diversity
- **Too large** (k > 20): Clusters become too granular, may overfit
- **Rule of thumb**: Evaluate `k_min = sqrt(n_hexagons / 2)` to `k_max = sqrt(n_hexagons * 2)`

### 2. Consider Your Domain Knowledge

Statistical metrics help, but ecological expertise matters:
- Do you expect major biomes (use lower k)?
- Are you looking for fine-scale habitat distinctions (use higher k)?
- Compare results with known ecoregion systems

### 3. Check Cluster Sizes

Look at `min_cluster_size` in the metrics:
- Very small clusters (< 5 hexagons) may be artifacts
- Very unbalanced sizes may indicate outliers or data quality issues

### 4. Multiple Methods Agreement

If different methods suggest similar k values, that's a strong signal:

```python
results = {}
for method in ["multi_criteria", "silhouette", "elbow", "compromise"]:
    result = optimizer.suggest_optimal_k(metrics_list, method=method)
    results[method] = result.optimal_k

print("Optimal k by method:")
for method, k in results.items():
    print(f"  {method}: k={k}")
```

### 5. Sensitivity Analysis

Run the analysis multiple times with different parameters:
- Different H3 resolutions
- Different temporal subsets
- Different quality filters

If optimal k is consistent, you have a robust solution.

## Computational Considerations

### Time Complexity

Evaluating each k value requires:
- Clustering: O(n² log n)
- Metric calculation: O(n²)

For large datasets:
- Reduce k_max to evaluate fewer options
- Use sampling if you have > 1000 hexagons

### Memory Usage

The optimizer stores distance matrices in memory. For very large datasets:

```python
# Evaluate k values in batches
all_metrics = []
for k in range(5, 21):
    metrics = optimizer.evaluate_k_range(k_min=k, k_max=k)
    all_metrics.extend(metrics)
```

## Troubleshooting

### "All metrics are similar across k values"

This suggests:
- Your data may not have natural clustering structure
- Try different distance metrics or preprocessing
- Consider if bioregionalization is appropriate for your data

### "Optimal k is at the boundary (k_min or k_max)"

- Extend your evaluation range
- The true optimum may lie outside your tested range

### "Different methods give very different results"

This is common and indicates:
- No single "obvious" optimal k
- Consider multiple k values for comparison
- Use domain knowledge to make final choice

### "Silhouette scores are all negative"

- Your data may have significant overlap between ecological communities
- Consider:
  - Increasing spatial resolution (lower H3 precision)
  - Filtering out transitional zones
  - Using softer clustering methods

## References

- **Silhouette Analysis:** Rousseeuw, P. J. (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis"
- **Davies-Bouldin Index:** Davies, D. L., & Bouldin, D. W. (1979). "A Cluster Separation Measure"
- **Calinski-Harabasz:** Caliński, T., & Harabasz, J. (1974). "A dendrite method for cluster analysis"
- **Elbow Method:** Thorndike, R. L. (1953). "Who belongs in the family?"

## Example Output

```
======================================================================
CLUSTER OPTIMIZATION REPORT
======================================================================

Selection Method: multi_criteria
Optimal k: 12
Reason: Best average rank across metrics: silhouette=0.487, DB=0.823, CH=156.3

All evaluated cluster counts:
----------------------------------------------------------------------
k    Silhouette   Davies-Bouldin   Calinski-H   Min Size  
----------------------------------------------------------------------
  5   0.523        0.891            142.7        12        
  6   0.509        0.864            148.2        10        
  7   0.498        0.847            151.5        9         
  8   0.512        0.834            153.8        8         
  9   0.501        0.829            155.1        7         
  10  0.495        0.826            155.9        6         
  11  0.489        0.824            156.2        6         
→ 12  0.487        0.823            156.3        5         
  13  0.481        0.825            155.8        5         
  14  0.476        0.829            154.9        4         
  15  0.468        0.835            153.7        4         
  16  0.461        0.843            152.1        3         
  17  0.453        0.852            150.3        3         
  18  0.445        0.863            148.2        3         
  19  0.437        0.876            145.8        2         
  20  0.428        0.891            143.1        2         
----------------------------------------------------------------------

Interpretation:
  • Silhouette: Higher is better (range: -1 to 1)
  • Davies-Bouldin: Lower is better (0 = perfect)
  • Calinski-Harabasz: Higher is better
  • Min Size: Minimum number of hexagons in smallest cluster
```
