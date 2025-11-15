import type { Polygon } from "geojson";

export interface SignificantTaxa {
  gbif_taxon_id: number;
  scientific_name: string;
  image_url: string | null;
  p_value: number;
  log2_fold_change: number;
  cluster_count: number;
  neighbor_count: number;
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
