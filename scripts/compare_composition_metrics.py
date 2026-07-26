"""Compare the abundance and presence composition metrics on a single facet.

Phase 1 step 1 of CLADE_CONGRUENCE_PLAN.md asks whether switching cross-facet
work to presence/absence with Sorensen reproduces the existing pipeline's
result on one facet. If the two metrics disagree wildly here, then later
cross-facet numbers cannot be read as biology — they would be reporting the
metric change instead.

Everything upstream of the distance matrix is built once and shared, so the
only thing that differs between the two runs is the metric. UMAP is seeded in
both so the comparison is not measuring layout noise.

Usage:

    uv run python -m scripts.compare_composition_metrics \\
        --scope order:Coleoptera --geocode-precision 3 --limit-results 200000

Reads the same GBIF parquet source and bounding-box defaults as the notebook;
override with the flags below.
"""

import argparse
import logging
import sys

import numpy as np
import polars as pl
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from src import defaults
from src.dataframes.darwin_core import build_darwin_core_lf
from src.dataframes.geocode import build_geocode_lf, build_geocode_no_edges_lf
from src.dataframes.geocode_cluster import build_geocode_cluster_multi_k_df
from src.dataframes.geocode_neighbors import (
    build_geocode_neighbors_df,
    build_geocode_neighbors_no_edges_df,
)
from src.dataframes.geocode_silhouette_score import build_geocode_silhouette_score_df
from src.dataframes.geocode_taxa_counts import (
    build_geocode_taxa_counts_lf,
    filter_top_taxa_lf,
)
from src.dataframes.taxonomy import build_taxonomy_lf
from src.matrices.geocode_connectivity import GeocodeConnectivityMatrix
from src.matrices.geocode_distance import GeocodeDistanceMatrix
from src.taxon_scope import parse_scope
from src.types import COMPOSITION_METRICS, Bbox, CompositionMetric

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-source-path", default=defaults.PARQUET_SOURCE_PATH)
    parser.add_argument("--geocode-precision", type=int, default=3)
    parser.add_argument("--scope", default=defaults.TAXON_SCOPE)
    parser.add_argument("--limit-results", type=int, default=defaults.LIMIT_RESULTS)
    parser.add_argument("--min-lat", type=float, default=defaults.MIN_LAT)
    parser.add_argument("--max-lat", type=float, default=defaults.MAX_LAT)
    parser.add_argument("--min-lon", type=float, default=defaults.MIN_LON)
    parser.add_argument("--max-lon", type=float, default=defaults.MAX_LON)
    parser.add_argument("--min-clusters", type=int, default=defaults.MIN_CLUSTERS)
    parser.add_argument("--max-clusters", type=int, default=defaults.MAX_CLUSTERS)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="UMAP random_state. Held equal across both metrics.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to write the per-k comparison table.",
    )
    return parser.parse_args(argv)


def build_shared_inputs(args: argparse.Namespace):
    """Build everything up to (but not including) the distance matrix.

    Returns the taxa-counts frame, the geocode frame, and the connectivity
    matrix — the inputs both metrics share.
    """
    bbox = Bbox.from_coordinates(
        args.min_lat, args.max_lat, args.min_lon, args.max_lon
    )
    scope = parse_scope(args.scope)
    logger.info("Scope: %s", scope or "(all taxa)")

    darwin_core_lf = build_darwin_core_lf(
        source_path=args.parquet_source_path,
        bounding_box=bbox,
        limit=args.limit_results,
        scope=scope,
    )

    geocode_lf_with_edges = build_geocode_lf(
        darwin_core_lf, args.geocode_precision, bounding_box=bbox
    )
    geocode_unfiltered_lf = build_geocode_no_edges_lf(geocode_lf_with_edges)

    taxonomy_lf = build_taxonomy_lf(
        darwin_core_lf,
        args.geocode_precision,
        geocode_unfiltered_lf,
        bounding_box=bbox,
    )

    geocode_taxa_counts_lf = filter_top_taxa_lf(
        build_geocode_taxa_counts_lf(
            darwin_core_lf,
            args.geocode_precision,
            taxonomy_lf,
            geocode_unfiltered_lf,
            bounding_box=bbox,
        ),
        max_taxa=defaults.MAX_TAXA if defaults.MAX_TAXA_ENABLED else None,
        min_geocode_presence=(
            defaults.MIN_GEOCODE_PRESENCE
            if defaults.MIN_GEOCODE_PRESENCE_ENABLED
            else None
        ),
    )

    geocode_lf = geocode_unfiltered_lf.join(
        geocode_taxa_counts_lf.select(pl.col("geocode").unique()),
        on="geocode",
        how="semi",
    )

    neighbors_df = build_geocode_neighbors_no_edges_df(
        build_geocode_neighbors_df(geocode_lf_with_edges.collect()),
        geocode_lf.collect(),
    )
    connectivity = GeocodeConnectivityMatrix.build(neighbors_df)

    return geocode_taxa_counts_lf, geocode_lf, connectivity


def cluster_for_metric(
    metric: CompositionMetric,
    geocode_taxa_counts_lf: pl.LazyFrame,
    geocode_lf: pl.LazyFrame,
    connectivity: GeocodeConnectivityMatrix,
    args: argparse.Namespace,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cluster one metric across the whole k range.

    Returns (multi-k cluster assignments, silhouette scores).
    """
    logger.info("Building distance matrix for metric=%s", metric)
    distance_matrix = GeocodeDistanceMatrix.build(
        geocode_taxa_counts_lf,
        geocode_lf,
        metric=metric,
        random_state=args.seed,
    )
    multi_k_df = build_geocode_cluster_multi_k_df(
        geocode_lf,
        distance_matrix,
        connectivity,
        args.min_clusters,
        args.max_clusters,
    )
    silhouette_df = build_geocode_silhouette_score_df(distance_matrix, multi_k_df)
    return multi_k_df, silhouette_df


def overall_silhouette(silhouette_df: pl.DataFrame) -> dict[int, float]:
    """Mean silhouette score per k, keyed by k."""
    agg = silhouette_df.group_by("num_clusters").agg(
        pl.col("silhouette_score").mean().alias("mean_score")
    )
    return dict(zip(agg["num_clusters"].to_list(), agg["mean_score"].to_list()))


def compare(
    abundance_df: pl.DataFrame,
    presence_df: pl.DataFrame,
    abundance_sil: pl.DataFrame,
    presence_sil: pl.DataFrame,
) -> pl.DataFrame:
    """Per-k agreement between the two metrics' partitions.

    Both partitions cover the same geocodes in the same order, so labels can be
    compared directly once each is sorted by geocode. ARI and AMI are both
    chance-corrected, so 0 means "no better than a random partition of the same
    shape" — not "completely different".
    """
    abundance_sil_by_k = overall_silhouette(abundance_sil)
    presence_sil_by_k = overall_silhouette(presence_sil)

    rows = []
    for k in sorted(abundance_df["num_clusters"].unique().to_list()):
        a = abundance_df.filter(pl.col("num_clusters") == k).sort("geocode")
        p = presence_df.filter(pl.col("num_clusters") == k).sort("geocode")

        assert a["geocode"].equals(p["geocode"]), (
            f"Geocode sets differ between metrics at k={k}; the two runs must "
            f"share their support for ARI to mean anything."
        )

        a_labels = a["cluster"].to_numpy()
        p_labels = p["cluster"].to_numpy()

        rows.append(
            {
                "k": k,
                "ari": float(adjusted_rand_score(a_labels, p_labels)),
                "ami": float(adjusted_mutual_info_score(a_labels, p_labels)),
                "abundance_clusters": int(len(np.unique(a_labels))),
                "presence_clusters": int(len(np.unique(p_labels))),
                "abundance_silhouette": abundance_sil_by_k.get(k, float("nan")),
                "presence_silhouette": presence_sil_by_k.get(k, float("nan")),
            }
        )

    return pl.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)

    geocode_taxa_counts_lf, geocode_lf, connectivity = build_shared_inputs(args)
    num_geocodes = geocode_lf.select(pl.len()).collect().item()
    logger.info("Comparing over %d geocodes", num_geocodes)

    results = {}
    for metric in COMPOSITION_METRICS:
        results[metric] = cluster_for_metric(
            metric, geocode_taxa_counts_lf, geocode_lf, connectivity, args
        )

    table = compare(
        results["abundance"][0],
        results["presence"][0],
        results["abundance"][1],
        results["presence"][1],
    )

    print()
    print(f"Composition metric comparison over {num_geocodes} geocodes")
    print(f"scope={args.scope or '(all taxa)'} h3_res={args.geocode_precision} seed={args.seed}")
    print()
    with pl.Config(tbl_rows=-1, float_precision=3):
        print(table)
    print()
    print(
        "ARI/AMI are chance-corrected agreement between the two metrics' "
        "partitions at the same k.\nHigh agreement means the presence path "
        "reproduces the existing result; low agreement means\nany cross-facet "
        "number would be reporting the metric switch, not biology."
    )

    if args.output_csv:
        table.write_csv(args.output_csv)
        print(f"\nWrote {args.output_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
