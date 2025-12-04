# Quick Reference: Automatic Cluster Optimization in Notebook

## 🚀 Quick Start (3 Steps)

1. **Open notebook**: `marimo edit notebook.py`
2. **Check the box**: ☑ Use automatic cluster optimization
3. **Run**: Execute all cells

The system will automatically select optimal k and display justification.

---

## 📋 UI Controls

### Automatic Mode (Default)
```
☑ Use automatic cluster optimization

Minimum clusters to evaluate: 5
Maximum clusters to evaluate: 15
Selection method: Multi-Criteria (Recommended) ▼
```

### Manual Mode
```
☐ Use automatic cluster optimization

Number of clusters (manual): 10
```

---

## 🎯 Selection Methods

| Method | When to Use | Priority |
|--------|-------------|----------|
| **Multi-Criteria** ⭐ | General use, balanced solution | All metrics equally |
| **Silhouette** | Maximum cluster quality | Quality > simplicity |
| **Elbow** | Prefer fewer clusters | Simplicity > quality |
| **Compromise** | Minimum acceptable quality | Lowest k with quality > 0.3 |

---

## 📊 Understanding Metrics

| Metric | Good Values | What It Measures |
|--------|-------------|------------------|
| **Silhouette** | > 0.5 | How well hexagons fit their cluster |
| **Davies-Bouldin** | < 1.0 | Cluster separation quality |
| **Calinski-Harabasz** | Higher | Between/within variance ratio |
| **Min Size** | > 5 | Smallest cluster (avoid very small) |

**Quality Bands (Silhouette):**
- 🟢 > 0.7: Excellent
- 🟡 0.5-0.7: Good
- 🟠 0.3-0.5: Fair
- 🔴 < 0.3: Poor

---

## 🔧 Recommended Settings

### Regional Bioregions
```
k_min: 5
k_max: 12
method: Multi-Criteria
```

**Note:** If using taxonomic coloring, set `k_min: 10` (UMAP requirement).

### Fine-Scale Habitats
```
k_min: 10
k_max: 25
method: Silhouette
```

### Large Continental Areas
```
k_min: 8
k_max: 20
method: Multi-Criteria
```

### Conservative (Fewest Clusters)
```
k_min: 5
k_max: 12
method: Compromise
```

---

## 📈 What You'll See

### 1. Optimization Report
```
CLUSTER OPTIMIZATION REPORT
===========================
Optimal k: 12
Reason: Best average rank across metrics

k    Silhouette   Davies-Bouldin   
----------------------------------
5    0.523        0.891
...
→12  0.487        0.823  ← SELECTED
...
```

### 2. Visualization (6 Panels)
- Silhouette scores across k
- Davies-Bouldin indices
- Calinski-Harabasz scores
- Inertia elbow plot
- Cluster size distribution
- Combined normalized score

### 3. Clustering Proceeds
```
Using k=12 clusters (automatically selected)
```

---

## ⏱️ Performance Guide

| Hexagons | Time per k | k_min=5, k_max=15 (11 values) |
|----------|------------|-------------------------------|
| 100 | ~2 sec | ~20 seconds |
| 500 | ~15 sec | ~3 minutes |
| 1000 | ~90 sec | ~15 minutes |

**To speed up:**
- Reduce k_max (fewer evaluations)
- Use coarser H3 precision (fewer hexagons)
- Switch to manual mode

---

## ⚠️ Troubleshooting

### "Optimization is slow"
→ Reduce k_max or use manual mode

### "Methods suggest different k"
→ Normal! Look at visualizations, use domain knowledge

### "All metrics similar"
→ Data may lack natural clustering, try different H3 precision

### "Optimal k at boundary"
→ Extend range (if k=15 is optimal, try k_max=25)

### "Very small clusters"
→ If min_size < 5, increase k_min or use different method

### "Taxonomic coloring failed"
→ Requires k ≥ 10. Set k_min=10 or it will fallback to geographic coloring

---

## 📤 Exporting Results

```python
# In a new cell:
from src.cluster_optimization import metrics_to_dataframe, create_metrics_report

# Export metrics CSV
df = metrics_to_dataframe(cluster_metrics_list)
df.write_csv("cluster_metrics.csv")

# Save report
report = create_metrics_report(cluster_optimization_result)
with open("optimization_report.txt", "w") as f:
    f.write(report)

# Save figure
from src.plot.cluster_optimization import plot_optimization_metrics
fig = plot_optimization_metrics(cluster_optimization_result)
fig.savefig("optimization.png", dpi=300, bbox_inches="tight")
```

---

## ✅ Validation Checklist

After optimization runs:

- [ ] Review the selected k value
- [ ] Check silhouette score (> 0.3 is acceptable)
- [ ] Verify min cluster size (> 5 is good)
- [ ] Look at visualizations for quality trends
- [ ] Compare different selection methods
- [ ] Validate that clusters make ecological sense
- [ ] Review diagnostic species for meaningfulness
- [ ] Check cluster boundaries on map

---

## 📚 More Information

- **Full guide**: `NOTEBOOK_CLUSTER_OPTIMIZATION_GUIDE.md`
- **Technical docs**: `CLUSTER_OPTIMIZATION.md`
- **Implementation**: `AUTOMATIC_CLUSTER_SELECTION_SUMMARY.md`

---

## 🎓 Key Takeaway

**Before:** Manual guess → arbitrary k → no justification

**After:** Statistical evaluation → optimal k → full justification + visualization

This makes your bioregionalization **reproducible, defensible, and rigorous**.