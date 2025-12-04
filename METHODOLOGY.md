# Citizen Science Bioregionalization: Methodology

## Overview

This program identifies ecological regions (bioregions) from citizen science biodiversity data by clustering geographic areas based on species composition similarity. It combines spatial discretization, ecological distance metrics, and constrained clustering to produce statistically validated ecoregion boundaries.

**Input:** Species occurrence records (Darwin Core format) with coordinates and taxonomy  
**Output:** GeoJSON map of ecoregions with characteristic species and statistical validation

---

## Pipeline

### 1. Geographic Discretization
- Converts continuous coordinates to **H3 hexagonal grid** (user-specified precision, typically 4-6)
- Flags hexagons on bounding box edges (optionally excluded to avoid boundary effects)
- Ensures single connected component by adding edges between isolated hexagon groups

### 2. Species-Location Matrix
- Aggregates species observations within each hexagon
- Creates geocode × taxa abundance matrix

### 3. Ecological Distance
- Computes **Bray-Curtis dissimilarity** between all hexagon pairs: `BC = 1 - (2×C_ij)/(S_i + S_j)`
- Standard in community ecology; handles abundance data; robust to sampling differences
- Range [0,1]: 0 = identical, 1 = no shared species

### 4. Spatially-Constrained Clustering
- **Agglomerative hierarchical clustering** with Ward's linkage
- Connectivity constraint: only adjacent hexagons (H3 neighbors) can merge
- Produces k clusters (user-specified) that are ecologically similar AND geographically contiguous

### 5. Statistical Validation
- **PERMANOVA:** Tests if clusters explain ecological variation (permutation test, reports pseudo-F and p-value)
- **Silhouette scores:** Measures cluster assignment quality for each hexagon [-1 to 1 scale]

### 6. Ecoregion Characterization
- Identifies neighboring ecoregion pairs
- Statistical tests identify **diagnostic species** with significantly different abundances between regions
- Computes taxonomic distance between ecoregions

### 7. Visualization
- Generates boundary polygons for each ecoregion
- Color assignment via dimensionality reduction (taxonomic or geographic)
- UMAP/MDS projection for 2D visualization
- Heatmaps showing species composition patterns

---

## Key Mathematical Components

**Distance Metric:** Bray-Curtis dissimilarity (abundance-sensitive, zero-invariant)

**Clustering Constraint:** Spatial connectivity matrix C where C[i,j]=1 if hexagons are adjacent

**Validation:** PERMANOVA (H₀: clusters don't explain composition), Silhouette (assignment quality)

---

## Critical Parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| H3 Precision | 4-6 | Higher = finer resolution, more hexagons |
| Number of Clusters | 5-20 | More = finer ecological distinctions |
| Bounding Box | Geographic region | Study area limits |
| Significance Level | 0.01-0.05 | Threshold for diagnostic species |

---

## Assumptions & Limitations

**Assumptions:**
- Species composition reflects ecological conditions
- Sampling effort approximately uniform (or acceptable variation)
- Reliable taxonomic identifications

**Limitations:**
- Citizen science sampling bias toward accessible areas
- Temporal aggregation across observation periods
- Cluster number requires domain knowledge or optimization
- Resolution tradeoff: coarser hexagons = better statistics, less detail

---

## Validation

- Compare with environmental data (climate, soil, vegetation)
- Expert biological review
- Temporal stability analysis
- Sensitivity to parameter choices
- Comparison with established ecoregion systems

---

**Computational Complexity:** O(n²m) for distance calculation, O(n² log n) for clustering, where n=hexagons, m=species. Scales to thousands of hexagons.