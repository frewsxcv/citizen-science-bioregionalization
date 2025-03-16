import os

import numpy as np
import polars as pl
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import RobustScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
import umap
from src.data_container import DataContainer
from src.dataframes import geocode_taxa_counts
from src.dataframes.geocode import GeocodeDataFrame
from src.logging import log_action, logger
from contexttimer import Timer


def pivot_taxon_counts(taxon_counts: pl.DataFrame) -> pl.DataFrame:
    """
    Create a matrix where each row is a geocode and each column is a taxon ID

    Example input:

    ```txt
    ┌─────────┬─────────┬───────┐
    │ geocode ┆ taxonId ┆ count │
    │ ---     ┆ ---     ┆ ---   │
    │ str     ┆ u32     ┆ u32   │
    ╞═════════╪═════════╪═══════╡
    │ 9eu     ┆ 12345   ┆ 1     │
    │ 9ev     ┆ 23456   ┆ 1     │
    │ 9ev     ┆ 34567   ┆ 1     │
    │ 9ev     ┆ 45678   ┆ 2     │
    │ 9ev     ┆ 56789   ┆ 4     │
    └─────────┴─────────┴───────┘
    ```

    Example output:

    ```txt
    ┌─────────┬───────┬───────┬───────┬───┬───────┐
    │ geocode ┆ 12345 ┆ 23456 ┆ 34567 ┆ … ┆ 56789 │
    │ ---     ┆ ---   ┆ ---   ┆ ---   ┆   ┆ ---   │
    │ str     ┆ u32   ┆ u32   ┆ u32   ┆   ┆ u32   │
    ╞═════════╪═══════╪═══════╪═══════╪═══╪═══════╡
    │ 9eu     ┆ 1     ┆ null  ┆ null  ┆ … ┆ null  │
    │ 9ev     ┆ null  ┆ 1     ┆ 1     ┆ … ┆ 4     │
    └─────────┴───────┴───────┴───────┴───┴───────┘
    ```
    """
    return taxon_counts.pivot(
        on="taxonId",
        index="geocode",
        values="count",
    )


def build_X(
    geocode_taxa_counts_dataframe: geocode_taxa_counts.GeocodeTaxaCountsDataFrame,
    geocode_dataframe: GeocodeDataFrame,
) -> pl.DataFrame:
    X = log_action(
        "Building matrix",
        lambda: geocode_taxa_counts_dataframe.df.pipe(pivot_taxon_counts),
    )

    assert X.height > 1, "More than one geocode is required to cluster"

    # fill null values with 0
    X = log_action("Filling null values", lambda: X.fill_null(np.uint32(0)))

    assert X["geocode"].to_list() == geocode_dataframe.df["geocode"].to_list()

    X = log_action("Dropping geocode column", lambda: X.drop("geocode"))

    return log_action("Scaling values", lambda: X.pipe(scale_values))


def scale_values(X: pl.DataFrame) -> pl.DataFrame:
    scaler = RobustScaler()
    return pl.from_numpy(scaler.fit_transform(X.to_numpy()))


def reduce_dimensions_pca(X: pl.DataFrame) -> pl.DataFrame:
    pca = IncrementalPCA(n_components=3000, copy=True, batch_size=3000)
    return pl.from_numpy(pca.fit_transform(X.to_numpy()))


def reduce_dimensions_umap(X: pl.DataFrame) -> pl.DataFrame:
    # https://github.com/lmcinnes/umap/issues/201
    assert (
        X.height > 10
    ), "Temporary constraint until I have a better understanding of this operation"
    reducer = umap.UMAP(
        n_components=X.height - 2,
        metric="braycurtis",
        min_dist=0.5,
    )
    return pl.from_numpy(reducer.fit_transform(X.to_numpy()))


def reduce_dimensions_tsne(X: pl.DataFrame) -> pl.DataFrame:
    tsne = TSNE(
        n_components=3000,
        random_state=42,
        metric="braycurtis",
        method="exact",
        init="random",
        # perplexity=min(30, X.shape[0] - 1), # HACK FOR SMALLER DATASETS
    )
    return pl.from_numpy(tsne.fit_transform(X.to_numpy()))


class DistanceMatrix(DataContainer):
    """
    A distance matrix where each column and row is a geocode, and the cell at the intersection of a
    column and row is the similarity (or distance) between the two geocodes. Internally it is stored
    as a condensed distance matrix, which is a one-dimensional array containing the upper triangular
    part of the distance matrix.
    """

    _condensed: np.ndarray

    def __init__(self, condensed: np.ndarray):
        self._condensed = condensed

    @classmethod
    def build(
        cls,
        geocode_taxa_counts_dataframe: geocode_taxa_counts.GeocodeTaxaCountsDataFrame,
        geocode_dataframe: GeocodeDataFrame,
    ) -> "DistanceMatrix":
        X = build_X(geocode_taxa_counts_dataframe, geocode_dataframe)

        # filtered.group_by("geocode").agg(pl.col("len").filter(on == value).sum().alias(str(value)) for value in set(taxonKeys)).collect()

        logger.info(f"Reducing dimensions with PCA. Previously: {X.shape}")

        # Use PCA to reduce the number of dimensions
        X = log_action("Fitting PCA", lambda: X.pipe(reduce_dimensions_umap))

        logger.info(
            f"Reduced dimensions with PCA. Now: {X.shape[0]} geocodes, {X.shape[1]} taxon IDs"
        )

        Y = log_action(
            f"Running pdist on matrix: {X.shape[0]} geocodes, {X.shape[1]} taxon IDs",
            lambda: pdist(X, metric="braycurtis"),
        )

        return cls(Y)

    def condensed(self) -> np.ndarray:
        return self._condensed

    def squareform(self) -> np.ndarray:
        return squareform(self._condensed)
