"""Assemble what a run found, for `src/findings_page.py` to render.

Kept apart from the rendering so the numbers can be tested without parsing
HTML, and so a caller that wants them as JSON does not have to go through a
page. Every function here returns something empty rather than raising when its
inputs are missing: a run against a source with no rank columns, or outside the
reference framework's extent, should still emit a page saying so.
"""

import logging
from typing import Optional

import polars as pl
import polars_h3

from src.clade_congruence import (
    CladePartition,
    clade_taxon_ids,
    cluster_clade,
    congruence_by_k,
    latitude_spans,
)
from src.epa_reference import reference_region_lf, score_against_reference
from src.findings_page import CladeShare, FindingsData, RunContext
from src.types import CompositionSettings

logger = logging.getLogger(__name__)

#: The two clades the page compares. Birds and plants because they share a
#: landscape and little else, and because citizen-science data has enough of
#: both; the rank differs because "birds" is a class and "plants" a kingdom.
DEFAULT_CLADES: tuple[tuple[str, str, str], ...] = (
    ("Aves", "class", "Aves"),
    ("Plantae", "kingdom", "Plantae"),
)


def clade_shares(
    geocode_taxa_counts_lf: pl.LazyFrame,
    taxon_clade_lf: Optional[pl.LazyFrame],
    clades: tuple[tuple[str, str, str], ...] = DEFAULT_CLADES,
) -> list[CladeShare]:
    """Each clade's share of the records and of the distinct taxa."""
    if taxon_clade_lf is None:
        return []

    totals = (
        geocode_taxa_counts_lf.select(
            records=pl.col("count").sum(), taxa=pl.col("taxonId").n_unique()
        )
        .collect(engine="streaming")
        .row(0)
    )
    total_records, total_taxa = totals
    if not total_records or not total_taxa:
        return []

    shares = []
    for name, rank, value in clades:
        ids = clade_taxon_ids(taxon_clade_lf, rank, value)
        row = (
            geocode_taxa_counts_lf.join(ids, on="taxonId", how="semi")
            .select(records=pl.col("count").sum(), taxa=pl.col("taxonId").n_unique())
            .collect(engine="streaming")
            .row(0)
        )
        records, taxa = int(row[0] or 0), int(row[1] or 0)
        if not records:
            logger.info(f"clade_shares: no records for {name}; omitting it")
            continue
        shares.append(
            CladeShare(
                name=name,
                key=name.lower(),
                record_share=records / total_records,
                taxon_share=taxa / total_taxa,
                records=records,
                taxa=taxa,
            )
        )
    return shares


def reference_agreement_by_k(
    multi_k_df: pl.DataFrame,
    reference_lf: pl.LazyFrame,
    ks: list[int],
) -> list[tuple[int, object]]:
    """Score the combined partition against the reference at each cut."""
    out = []
    for k in ks:
        at_k = multi_k_df.filter(pl.col("num_clusters") == k).select(
            "geocode", "cluster"
        )
        if at_k.height == 0:
            continue
        agreement = score_against_reference(at_k, reference_lf)
        if agreement is not None:
            out.append((int(k), agreement))
    return out


def choose_span_cut(published_k: int, available_ks: set[int]) -> int:
    """Which of a clade's cuts to draw the latitude spans at.

    The published one when the clade was fit at it. A clade occupying few
    hexagons may not have been -- `cluster_clade` caps its range at the number
    of hexagons it holds -- so the nearest available cut is used instead, which
    keeps the chart describing roughly the grain the rest of the page does.

    Drawing this at the selector's k instead put a chart describing one
    partition beside figures describing another: on the published run it showed
    the degenerate Aves 1182/4 split from k=2 while everything else described
    k=4.
    """
    if published_k in available_ks:
        return published_k
    return min(available_ks, key=lambda k: (abs(k - published_k), k))


def build_findings_data(
    context: RunContext,
    geocode_taxa_counts_lf: pl.LazyFrame,
    geocode_lf: pl.LazyFrame,
    all_clusters_df: pl.DataFrame,
    taxon_clade_lf: Optional[pl.LazyFrame],
    settings: CompositionSettings,
    skipped: Optional[list[str]] = None,
) -> FindingsData:
    """Compute every number the findings page draws.

    Args:
        context: Carries `levels`, which is where the cuts to score come from.
            The page does not choose its own k values; scoring a set the run
            did not emit is how a figure ends up describing a partition no
            output contains.
        all_clusters_df: The run's multi-k clustering, which supplies the
            combined partition at each cut.
        taxon_clade_lf: `None` when the source carried no rank columns; the
            clade sections are then omitted rather than guessed at.
        settings: Passed through to `cluster_clade` so each clade is clustered
            the way the run's own map was.
    """
    data = FindingsData(context=context, skipped=list(skipped or []))

    data.clade_shares = clade_shares(geocode_taxa_counts_lf, taxon_clade_lf)

    # Cuts to score: the ones the run emitted, plus the ends of the range so
    # the page can show that agreement depends on grain rather than asserting
    # it from a single number. `emitted` already contains the published cut and
    # the selector's, so both are always scored and can be compared.
    #
    # Taken from the run rather than invented here. This used to be a hardcoded
    # {2, 4, 8, 12, 16}, which is a second opinion about which cuts matter and
    # could drift from the ladder the run actually emits.
    levels = context.levels
    candidate_ks = sorted(set(levels.emitted) | {levels.min_k, levels.max_k})
    available_ks = set(all_clusters_df["num_clusters"].unique().to_list())
    ks = [k for k in candidate_ks if k in available_ks]

    reference_lf = reference_region_lf(geocode_lf).collect(engine="streaming").lazy()
    data.reference_by_k = reference_agreement_by_k(  # type: ignore[assignment]
        all_clusters_df, reference_lf, ks
    )
    if not data.reference_by_k:
        data.skipped.append(
            "Agreement with EPA Level II ecoregions — no hexagon centre fell "
            "inside the checked-in reference extent, which covers the US East "
            "Coast only."
        )

    if taxon_clade_lf is None:
        data.skipped.append(
            "Clade congruence and the record/taxon split — the source carried "
            "no kingdom or class column."
        )
        return data

    partitions: dict[str, CladePartition] = {}
    for name, rank, value in DEFAULT_CLADES:
        partition = cluster_clade(
            name,
            geocode_taxa_counts_lf,
            clade_taxon_ids(taxon_clade_lf, rank, value),
            geocode_lf,
            min_k=levels.min_k,
            max_k=levels.max_k,
            settings=settings,
        )
        if partition is not None:
            partitions[name] = partition

    if len(partitions) < 2:
        data.skipped.append(
            "Clade congruence — fewer than two clades held enough hexagons to "
            "cluster separately."
        )
        return data

    (left, a), (right, b) = list(partitions.items())[:2]
    data.congruence_pair = (left, right)
    data.congruence = congruence_by_k(a, b)

    centres = (
        geocode_lf.select("geocode")
        .with_columns(lat=polars_h3.cell_to_lat("geocode"))
        .collect(engine="streaming")
    )
    for name, partition in partitions.items():
        clade_ks = set(partition.multi_k_df["num_clusters"].unique().to_list())
        data.clade_vs_reference[name] = [
            (k, agreement.adjusted_rand)  # type: ignore[union-attr]
            for k, agreement in reference_agreement_by_k(
                partition.multi_k_df, reference_lf, [k for k in ks if k in clade_ks]
            )
        ]
        spans = latitude_spans(
            partition, choose_span_cut(levels.published, clade_ks), centres
        )
        data.latitude_spans[name] = [
            (int(r["cluster"]), float(r["min_lat"]), float(r["max_lat"]), int(r["hexagons"]))
            for r in spans.iter_rows(named=True)
        ]

    return data
