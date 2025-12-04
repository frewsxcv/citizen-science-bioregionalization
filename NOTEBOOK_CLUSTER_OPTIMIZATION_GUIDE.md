# Notebook Cluster Optimization Guide

## Overview

The notebook has been updated to include **automatic cluster optimization**. Instead of manually choosing the number of clusters, the system evaluates multiple options using statistical metrics and suggests the optimal choice.

## Notebook Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INPUT PARAMETERS                                         │
│    ☑ Use automatic cluster optimization                    │
│    - If checked: k_min, k_max, selection method            │
│    - If unchecked: manual k value                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. DATA LOADING                                             │
│    - Darwin Core data → GeocodeDataFrame                    │
│    - Distance matrix (Bray-Curtis)                          │
│    - Connectivity matrix (H3 neighbors)                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. CLUSTER OPTIMIZATION (if enabled)                        │
│    - Evaluate k from k_min to k_max                         │
│    - Calculate metrics: Silhouette, Davies-Bouldin, etc.    │
│    - Select optimal k using chosen method                   │
│    - Display report and visualizations                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. CLUSTERING                                               │
│    - Use optimal k (auto) or manual k                       │
│    - Generate cluster assignments                           │
│    - Display: "Using k=X clusters (auto/manual)"            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. DOWNSTREAM ANALYSIS                                      │
│    - Cluster boundaries, diagnostic species, maps, etc.     │
│    (All use the selected k)                                 │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start: Using Automatic Cluster Selection in the Notebook

The notebook has been updated to include automatic cluster optimization. This guide shows you how to use it.

## What Changed

### Before
- Manual `num_clusters` input with no guidance
- No way to know if your choice was optimal
- Trial-and-error approach

### After
- **Checkbox to enable automatic optimization**
- Statistical evaluation of multiple k values (5-15 by default)
- Visual metrics and reports
- Automatic selection of optimal k
- Still supports manual override if needed

## How to Use

### Step 1: Enable Automatic Optimization

Near the top of the notebook, you'll see:

```
### Cluster Count Selection

Choose between manual selection or automatic optimization.

☑ Use automatic cluster optimization
```

**Check this box** to enable automatic optimization (it's checked by default).

### Step 2: Configure Optimization Parameters

When automatic optimization is enabled, you'll see:

```
┌─────────────────────────────────┬─────────────────────────────────┐
│ Minimum clusters to evaluate: 5 │ Maximum clusters to evaluate: 15│
└─────────────────────────────────┴─────────────────────────────────┘

Selection method: Multi-Criteria (Recommended) ▼
```

**Parameters:**
- **Minimum clusters** (k_min): Start of range (default: 5)
- **Maximum clusters** (k_max): End of range (default: 15)
- **Selection method**: 
  - `Multi-Criteria (Recommended)` - Balanced approach using all metrics
  - `Best Silhouette Score` - Maximize cluster quality
  - `Elbow Method` - Find diminishing returns point
  - `Quality Compromise` - Fewest clusters with acceptable quality

**Recommendations:**
- For regional analysis: k_min=5, k_max=15
- For fine-scale analysis: k_min=10, k_max=25
- For large areas: k_min=8, k_max=20

### Step 3: Run the Notebook

Execute all cells as usual. When it reaches the **Cluster Optimization** section, you'll see:

#### A. Optimization Report

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
...
→ 12  0.487        0.823            156.3        5         
...
----------------------------------------------------------------------
```

This shows:
- **Optimal k selected** (12 in this example)
- **Why it was selected** (reasoning)
- **All evaluated k values** with their metrics
- **Arrow (→)** points to the selected k

#### B. Optimization Visualizations

You'll see a **6-panel figure** showing:

1. **Silhouette Score** - Quality across k values (higher is better)
2. **Davies-Bouldin Index** - Separation quality (lower is better)
3. **Calinski-Harabasz Score** - Variance ratios (higher is better)
4. **Inertia Elbow Plot** - Diminishing returns curve
5. **Cluster Sizes** - Distribution of hexagons per cluster
6. **Combined Score** - Normalized multi-metric view

**Red vertical line** shows the optimal k across all panels.

#### C. Cluster Assignment

The next section shows:

```
Using k=12 clusters (automatically selected)
```

And the clustering proceeds with the optimal k value.

## Manual Mode (If You Prefer)

### Step 1: Uncheck the Box

```
☐ Use automatic cluster optimization
```

### Step 2: Enter Manual Value

```
Number of clusters (manual): 10
```

### Step 3: Run Normally

The optimization section will be skipped, and it will use your manual value:

```
Using k=10 clusters (manually specified)
```

## Understanding the Metrics

### Silhouette Score
- **Range:** -1 to 1
- **Interpretation:** Higher is better
- **Quality bands:**
  - > 0.7: Excellent
  - 0.5-0.7: Good
  - 0.3-0.5: Fair
  - < 0.3: Poor

### Davies-Bouldin Index
- **Range:** 0 to ∞
- **Interpretation:** Lower is better
- **Good values:** < 1.0

### Calinski-Harabasz Score
- **Range:** 0 to ∞
- **Interpretation:** Higher is better
- **Meaning:** Better-defined clusters have higher scores

### Min Size
- Smallest cluster size (# of hexagons)
- Very small values (< 5) may indicate problematic clusters
- Helps identify unbalanced clustering

## Tips & Best Practices

### 1. Check the Visualizations

Look for:
- **Silhouette plateau** - If it's flat across many k values, any in that range is fine
- **Elbow in inertia** - Clear bend indicates good k
- **Stable cluster sizes** - Avoid k where min_size is very small

### 2. Compare Methods

Try running with different selection methods and see if they agree:
- If they all suggest k≈10-12, that's strong evidence
- If they disagree widely, there may not be a clear optimal k

### 3. Validate Ecologically

After getting optimal k:
- Check if cluster boundaries make ecological sense
- Review diagnostic species (are they meaningful?)
- Compare with known ecoregion systems
- Look at the map - do regions look reasonable?

### 4. Adjust Range If Needed

If optimal k is at the boundary (k_min or k_max):
- The true optimum may be outside your range
- Extend the range and re-run
- Example: If optimal k=15 (your max), try k_max=25

### 5. Consider Your Study Goals

Statistics help, but domain knowledge matters:
- **Broad bioregions?** Use lower k (5-10)
- **Fine habitat distinctions?** Use higher k (12-20)
- **Conservation planning?** Fewer regions may be more practical
- **Ecological research?** More regions may reveal patterns

## Performance Notes

### Computation Time

Optimization evaluates multiple k values, so it takes longer:
- **100 hexagons:** ~10-30 seconds total
- **500 hexagons:** ~2-5 minutes total
- **1000 hexagons:** ~10-20 minutes total

Each additional k value adds time. Narrow your range if needed:
- Instead of k_min=5, k_max=20 (16 evaluations)
- Try k_min=8, k_max=15 (8 evaluations)

### Memory

The optimizer uses the same matrices already in memory. No extra memory overhead.

## Troubleshooting

### "Taxonomic coloring requires at least 10 clusters"

**Error:** `AssertionError: UMAP requires at least 10 clusters, got 7`

**Cause:** The taxonomic coloring method uses UMAP for dimensionality reduction, which requires at least 10 clusters.

**Solutions:**
- Set `k_min: 10` in the optimization parameters
- Or the system will automatically fall back to geographic coloring (with a warning)
- Or manually switch to `color_method="geographic"` in the notebook

### "All silhouette scores are similar"

This means your data doesn't have strong natural clustering. Try:
- Adjusting H3 precision (coarser or finer)
- Filtering data quality more strictly
- Using a different selection method

### "Optimal k keeps changing"

If you re-run and get different k values:
- Scores are very close for several k values
- Any in that range is acceptable
- Use the compromise method for consistency

### "Metrics disagree"

When different metrics suggest different k:
- No single "best" answer
- Look at the visualizations
- Consider your domain knowledge
- Choose based on your priorities (quality vs. simplicity)

### "Optimization is too slow"

To speed up:
- Reduce k_max (fewer values to evaluate)
- Use coarser H3 precision (fewer hexagons)
- Or switch to manual mode with educated guess

## Example Workflows

### Conservative Approach (Fewest Clusters)

```
k_min: 10  # Note: Use 10 if using taxonomic coloring
k_max: 12
method: Compromise
```

Gets you the simplest solution with acceptable quality.

**Note:** If using taxonomic coloring, `k_min` must be at least 10.

### Quality-First Approach (Best Clustering)

```
k_min: 8
k_max: 20
method: Silhouette
```

Maximizes cluster quality regardless of cluster count.

### Balanced Approach (Recommended)

```
k_min: 5
k_max: 15
method: Multi-Criteria
```

Best overall solution considering all factors.

### Exploratory Analysis

```
k_min: 5
k_max: 25
method: Multi-Criteria
```

Wide range to discover data structure.

## Exporting Results

The optimization results are stored in variables you can export:

```python
# In a new cell:
from src.cluster_optimization import metrics_to_dataframe

# Export metrics to CSV
df = metrics_to_dataframe(cluster_metrics_list)
df.write_csv("cluster_optimization_metrics.csv")

# Save report
from src.cluster_optimization import create_metrics_report
report = create_metrics_report(cluster_optimization_result)
with open("optimization_report.txt", "w") as f:
    f.write(report)

# Save figure
from src.plot.cluster_optimization import plot_optimization_metrics
fig = plot_optimization_metrics(cluster_optimization_result)
fig.savefig("cluster_optimization.png", dpi=300, bbox_inches="tight")
```

## Integration with Output

The selected k is automatically used for all downstream analyses:
- Cluster boundaries
- Diagnostic species
- Silhouette scores
- Visualizations
- GeoJSON output

The final output metadata should include the optimization method and selected k for reproducibility.

## Summary

**Key Benefits:**
- ✅ Statistical justification for cluster count
- ✅ Reproducible methodology
- ✅ Multiple evaluation metrics
- ✅ Visual confirmation
- ✅ Automatic or manual modes

**When to Use Automatic:**
- You want defensible, reproducible results
- You're unsure what k to use
- You want to explore your data structure
- You need statistical backing for publications

**When to Use Manual:**
- You have strong domain knowledge
- You want a specific number of regions
- You're matching an existing classification
- You need quick iterations

---

For more details, see:
- `CLUSTER_OPTIMIZATION.md` - Complete technical documentation
- `AUTOMATIC_CLUSTER_SELECTION_SUMMARY.md` - Implementation overview