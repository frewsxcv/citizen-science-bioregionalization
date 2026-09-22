"""Charts that state what a run found, rather than what it computed.

The rest of `src/plot/` visualises intermediates -- silhouettes, embeddings,
taxa heatmaps -- for someone reading the notebook while it runs. These answer
the questions a reader of the output asks instead: is this map biology or is it
a record of where people looked, and is the number of regions it settled on the
right one?

Everything here is derived from a single run's own outputs, so it stays true
when the parameters change. Two findings this project has established cannot be
computed this way and are deliberately absent:

- clade congruence, which needs one run per clade to compare
- the Sorensen/betasim comparison, which needs one run per index

Both are cross-run comparisons; see `scripts/` for those.
"""

from typing import Optional, cast

import altair as alt
import numpy as np
import polars as pl

# Okabe-Ito, published as colour-vision-deficiency safe. Assigned by role and
# never recycled, so adding a series does not repaint the others.
#
# Only the marks are coloured here. Titles, axes and labels are left to the
# theme, because the notebook exports against a dark surface and a fixed
# light-mode ink is near-illegible on it.
ACCENT = "#0072B2"
WARN = "#D55E00"
GOOD = "#009E73"


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, without pulling scipy into the plotting path."""
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def effort_vs_richness(
    geocode_taxa_counts_lf: pl.LazyFrame,
) -> tuple[alt.Chart, float]:
    """How much of "what lives here" is really "how hard anyone looked".

    Each point is a hexagon: records collected against distinct taxa seen. On
    citizen-science data these are nearly the same measurement, which is the
    single biggest threat to reading the map as biology. The rank correlation
    is returned alongside so the caller can state it.
    """
    per_hex = (
        geocode_taxa_counts_lf.group_by("geocode")
        .agg(
            pl.col("count").sum().alias("records"),
            pl.len().alias("taxa"),
        )
        # An H3 cell id is a u64 and routinely exceeds 2**53, which is where
        # JavaScript's number type stops being exact -- Vega raises "BigInt
        # exceeds integer number representation" and abandons the chart. Carry
        # it as text, which it is for display purposes anyway.
        .with_columns(pl.col("geocode").cast(pl.Utf8))
        .collect(engine="streaming")
    )
    rho = _spearman(per_hex["records"].to_numpy(), per_hex["taxa"].to_numpy())

    chart = (
        alt.Chart(per_hex.to_pandas())
        .mark_circle(size=18, opacity=0.35, color=ACCENT)
        .encode(
            x=alt.X(
                "records:Q", scale=alt.Scale(type="log"), title="Records in hexagon"
            ),
            y=alt.Y(
                "taxa:Q", scale=alt.Scale(type="log"), title="Distinct taxa observed"
            ),
            tooltip=[
                alt.Tooltip("geocode:N", title="Hexagon"),
                alt.Tooltip("records:Q", title="Records", format=","),
                alt.Tooltip("taxa:Q", title="Taxa", format=","),
            ],
        )
        .properties(
            height=280,
            title=alt.Title(
                "Observed richness is largely sampling effort",
                subtitle=f"Spearman {rho:.3f} across {per_hex.height:,} hexagons"
                " — a hexagon looks species-rich when it has been visited often",
            ),
        )
        .interactive()
    )
    return chart, rho


def dissimilarity_vs_effort(
    condensed: np.ndarray,
    geocode_taxa_counts_lf: pl.LazyFrame,
    max_pairs: int = 200_000,
    seed: int = 0,
) -> tuple[Optional[alt.Chart], float]:
    """Whether two hexagons look different because they *are* different.

    Bins every pair of hexagons by how far apart their record counts are, in
    orders of magnitude, and reports the median dissimilarity in each bin. A
    flat line is what an index that ignores sampling depth would give. A rising
    one means the map partly encodes effort -- pairs far apart in effort being
    called far apart in composition.

    Pairs are subsampled with a fixed seed above `max_pairs`; a thousand
    hexagons is already half a million pairs.

    Returns `(None, rho)` when no effort bin holds enough pairs to take a median
    of. A handful of hexagons -- the test fixture has ten, which is 45 pairs --
    cannot answer this question, and an empty chart claims otherwise.
    """
    per_hex = (
        geocode_taxa_counts_lf.group_by("geocode")
        .agg(pl.col("count").sum().alias("records"))
        .sort("geocode")
        .collect(engine="streaming")
    )
    log_records = np.log10(np.maximum(per_hex["records"].to_numpy(), 1))
    n = len(log_records)
    rows, cols = np.triu_indices(n, k=1)
    if len(rows) != len(condensed):
        raise ValueError(
            f"{len(condensed)} pairwise distances against {len(rows)} hexagon "
            f"pairs; the distance matrix and the counts disagree on the geocode set"
        )

    gap = np.abs(log_records[rows] - log_records[cols])
    if len(condensed) > max_pairs:
        idx = np.random.default_rng(seed).choice(
            len(condensed), max_pairs, replace=False
        )
        gap, dist = gap[idx], condensed[idx]
    else:
        dist = condensed
    rho = _spearman(gap, dist)

    edges = np.array([0, 0.25, 0.5, 1.0, 1.5, 2.0, 9.0])
    labels = ["<0.25", "0.25–0.5", "0.5–1", "1–1.5", "1.5–2", ">2"]
    # Scaled to the data rather than fixed: 30 is a sane floor on a real run and
    # silently empties the chart on a small one.
    min_pairs = max(5, len(dist) // 200)
    binned = []
    for i, label in enumerate(labels):
        m = (gap >= edges[i]) & (gap < edges[i + 1])
        if m.sum() >= min_pairs:
            binned.append(
                {
                    "bin": label,
                    "median": float(np.median(dist[m])),
                    "pairs": int(m.sum()),
                }
            )
    if not binned:
        return None, rho
    df = pl.DataFrame(binned).to_pandas()

    chart = (
        alt.Chart(df)
        .mark_bar(color=WARN, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=38)
        .encode(
            x=alt.X(
                "bin:N",
                sort=labels,
                title="Difference in sampling effort (orders of magnitude)",
            ),
            y=alt.Y(
                "median:Q", title="Median dissimilarity", scale=alt.Scale(domain=[0, 1])
            ),
            tooltip=[
                alt.Tooltip("bin:N", title="Effort gap"),
                alt.Tooltip("median:Q", title="Median dissimilarity", format=".4f"),
                alt.Tooltip("pairs:Q", title="Pairs", format=","),
            ],
        )
        .properties(
            height=260,
            title=alt.Title(
                "Does the map measure biota, or how often people visited?",
                subtitle=f"Spearman {rho:.3f} between dissimilarity and effort gap"
                " — flat bars would mean the index ignores sampling depth",
            ),
        )
    )
    return chart, rho


def metrics_by_k(
    all_cluster_metrics_df: pl.DataFrame,
    published_k: int,
    selector_k: Optional[int] = None,
) -> alt.LayerChart:
    """Every cut the tree was scored at, which one is published, which one the
    selector's score peaked at, and why those are not the same.

    The selector maximises a weighted score in which silhouette dominates, and
    silhouette falls with k almost by construction on data like this -- so its
    argmax is the bottom of the tested range whatever the data says. Showing
    the measures separately makes that visible rather than leaving the choice
    to look inevitable, and marking both cuts keeps the published one from
    being mistaken for the selector's.
    """
    # These are the column names build_cluster_metrics_df emits -- three of the
    # four carry a `_score` suffix. Getting them wrong is silent: the measure
    # just never appears, and the colour scale below still lists it in the
    # legend, so the chart claims to show four lines and draws one.
    keep = [
        c
        for c in (
            "silhouette_score",
            "calinski_harabasz_score",
            "davies_bouldin_score",
            "combined_score",
        )
        if c in all_cluster_metrics_df.columns
    ]
    long = (
        all_cluster_metrics_df.select(["num_clusters", *keep])
        # Each measure is on its own scale, and two scales on one axis is a lie.
        # Min-max per measure puts them on a shared 0-1 so their *shapes* can be
        # compared; absolute values stay in the tooltip.
        .unpivot(
            index="num_clusters", variable_name="metric", value_name="value"
        ).with_columns(
            normalized=(
                (pl.col("value") - pl.col("value").min().over("metric"))
                / (
                    pl.col("value").max().over("metric")
                    - pl.col("value").min().over("metric")
                )
            ).fill_nan(0.5)
        )
    )

    # Assigned by measure and never recycled, so a run missing one measure does
    # not repaint the others. The domain is drawn from `keep` rather than from a
    # fixed list, so the legend can only name a line the chart actually draws.
    palette = {
        "silhouette_score": ACCENT,
        "calinski_harabasz_score": GOOD,
        "davies_bouldin_score": WARN,
        "combined_score": "#CC79A7",
    }
    labels = {
        "silhouette_score": "silhouette",
        "calinski_harabasz_score": "Calinski-Harabasz",
        # Lower is better for this one, so its normalised curve runs opposite to
        # the others. Said in the legend rather than silently inverted, which
        # would put a number in the tooltip that is not the measure.
        "davies_bouldin_score": "Davies-Bouldin (lower is better)",
        "combined_score": "combined score",
    }
    long = long.with_columns(pl.col("metric").replace(labels))

    base = alt.Chart(long.to_pandas())
    lines = base.mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=45)).encode(
        x=alt.X(
            "num_clusters:Q", title="Number of regions", axis=alt.Axis(tickMinStep=1)
        ),
        y=alt.Y(
            "normalized:Q",
            title="Normalised score (each measure scaled to its own range)",
        ),
        color=alt.Color(
            "metric:N",
            title="Measure",
            scale=alt.Scale(
                domain=[labels[c] for c in keep],
                range=[palette[c] for c in keep],
            ),
        ),
        tooltip=[
            alt.Tooltip("num_clusters:Q", title="Regions"),
            alt.Tooltip("metric:N", title="Measure"),
            alt.Tooltip("value:Q", title="Value", format=".4f"),
        ],
    )
    # Both cuts, so a reader cannot take the selector's peak for the answer.
    # Solid for what is published, dashed for what the score happened to
    # maximise.
    #
    # Coloured literally rather than through an encoding: a second `color`
    # encoding on the same channel merges with the measures' scale, which puts
    # "published" in the measure legend and paints the rules from the measure
    # palette. Labelled in place instead, which is less to read than a legend.
    marks = [(published_k, "published", "#1a1a19", [])]
    if selector_k is not None and selector_k != published_k:
        marks.append((selector_k, "selector's peak", "#918f86", [5, 4]))

    marker_layers = []
    for value, label, color, dash in marks:
        frame = pl.DataFrame({"num_clusters": [value], "what": [label]}).to_pandas()
        marker_layers.append(
            alt.Chart(frame)
            .mark_rule(strokeWidth=2, opacity=0.9, color=color, strokeDash=dash)
            .encode(
                x="num_clusters:Q",
                tooltip=[alt.Tooltip("what:N", title="Cut"), "num_clusters:Q"],
            )
        )
        marker_layers.append(
            alt.Chart(frame)
            .mark_text(
                align="left", dx=5, dy=3, baseline="top", fontSize=11, color=color
            )
            .encode(x="num_clusters:Q", y=alt.value(0), text="what:N")
        )
    marker = alt.layer(*marker_layers)
    return cast(
        alt.LayerChart,
        (lines + marker)
        .properties(
            height=300,
            title=alt.Title(
                f"Published: {published_k} regions"
                + (
                    f" — the selector's score peaked at {selector_k}"
                    if selector_k is not None and selector_k != published_k
                    else ""
                ),
                subtitle="Silhouette falls with k on saturated ecological"
                " distances and carries the most weight in the combined score,"
                " so the selector's peak is the bottom of the tested range"
                " whatever the data says",
            ),
        )
        .interactive(),
    )
