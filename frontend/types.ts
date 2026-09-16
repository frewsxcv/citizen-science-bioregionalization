import type { Polygon } from "geojson";

export interface SignificantTaxa {
  gbif_taxon_id: number;
  taxon_id: number;
  scientific_name: string;
  image_url: string | null;
  log2_fold_change: number;
  cluster_count: number;
  neighbor_count: number;
  high_log2_high_count_score: number;
  low_log2_high_count_score: number;
}

export interface ClusterData {
  cluster: number;
  boundary: Polygon;
  significant_taxa: SignificantTaxa[];
  color: string;
  darkened_color: string;
}

export interface SelectedCluster {
  clusterId: number;
  significantTaxa: SignificantTaxa[];
}
