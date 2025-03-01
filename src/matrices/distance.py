from typing import Self
import os

import numpy as np
import polars as pl
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import RobustScaler
from scipy.spatial.distance import pdist, squareform

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
    ┌─────────┬──────────┬────────────────────────┬───────┐
    │ geocode ┆ kingdom  ┆ scientificName         ┆ count │
    │ ---     ┆ ---      ┆ ---                    ┆ ---   │
    │ str     ┆ enum     ┆ str                    ┆ u32   │
    ╞═════════╪══════════╪════════════════════════╪═══════╡
    │ 9eu     ┆ Animalia ┆ Mydas xanthopterus     ┆ 1     │
    │ 9ev     ┆ Animalia ┆ Plecia plagiata        ┆ 1     │
    │ 9ev     ┆ Animalia ┆ Palpada vinetorum      ┆ 1     │
    │ 9ev     ┆ Animalia ┆ Dioprosopa clavatus    ┆ 2     │
    │ 9ev     ┆ Animalia ┆ Toxomerus politus      ┆ 4     │
    │ …       ┆ …        ┆ …                      ┆ …     │
    │ f8k     ┆ Animalia ┆ Polydontomyia curvipes ┆ 1     │
    │ f8k     ┆ Animalia ┆ Eristalis arbustorum   ┆ 3     │
    │ f8k     ┆ Animalia ┆ Lucilia sericata       ┆ 3     │
    │ f8k     ┆ Animalia ┆ Bittacomorpha clavipes ┆ 1     │
    │ f8k     ┆ Animalia ┆ Pollenia vagabunda     ┆ 2     │
    └─────────┴──────────┴────────────────────────┴───────┘
    ```

    Example output:

    ```txt
    ┌─────────┬───────────┬───────────┬───────────┬───┬───────────┬───────────┬───────────┬───────────┐
    │ geocode ┆ {"Animali ┆ {"Animali ┆ {"Animali ┆ … ┆ {"Animali ┆ {"Animali ┆ {"Animali ┆ {"Animali │
    │ ---     ┆ a","Mydas ┆ a","Pleci ┆ a","Palpa ┆   ┆ a","Pegom ┆ a","Phyto ┆ a","Dolic ┆ a","Cysti │
    │ str     ┆ xanthopte ┆ a plagiat ┆ da vineto ┆   ┆ ya solenn ┆ myza      ┆ hopus     ┆ phora     │
    │         ┆ ru…       ┆ a"}       ┆ rum…      ┆   ┆ is"…      ┆ lineata…  ┆ palaes…   ┆ taraxa…   │
    │         ┆ ---       ┆ ---       ┆ ---       ┆   ┆ ---       ┆ ---       ┆ ---       ┆ ---       │
    │         ┆ u32       ┆ u32       ┆ u32       ┆   ┆ u32       ┆ u32       ┆ u32       ┆ u32       │
    ╞═════════╪═══════════╪═══════════╪═══════════╪═══╪═══════════╪═══════════╪═══════════╪═══════════╡
    │ 9eu     ┆ 1         ┆ null      ┆ null      ┆ … ┆ null      ┆ null      ┆ null      ┆ null      │
    │ 9ev     ┆ null      ┆ 1         ┆ 1         ┆ … ┆ null      ┆ null      ┆ null      ┆ null      │
    │ 9ey     ┆ null      ┆ null      ┆ null      ┆ … ┆ null      ┆ null      ┆ null      ┆ null      │
    │ 9ez     ┆ null      ┆ null      ┆ 1         ┆ … ┆ null      ┆ null      ┆ null      ┆ null      │
    │ 9gb     ┆ 2         ┆ 1         ┆ null      ┆ … ┆ null      ┆ null      ┆ null      ┆ null      │
    │ …       ┆ …         ┆ …         ┆ …         ┆ … ┆ …         ┆ …         ┆ …         ┆ …         │
    │ f8c     ┆ null      ┆ null      ┆ null      ┆ … ┆ null      ┆ null      ┆ null      ┆ null      │
    │ f8d     ┆ null      ┆ null      ┆ null      ┆ … ┆ null      ┆ null      ┆ null      ┆ null      │
    │ f8e     ┆ null      ┆ null      ┆ null      ┆ … ┆ null      ┆ null      ┆ null      ┆ null      │
    │ f8h     ┆ null      ┆ null      ┆ null      ┆ … ┆ null      ┆ null      ┆ null      ┆ null      │
    │ f8k     ┆ null      ┆ null      ┆ null      ┆ … ┆ null      ┆ null      ┆ null      ┆ null      │
    └─────────┴───────────┴───────────┴───────────┴───┴───────────┴───────────┴───────────┴───────────┘
    ```
    """
    return taxon_counts.pivot(
        # There are some scientific names that are repeated with different taxon ranks, so we need to include the taxon rank in the pivot
        on=["kingdom", "scientificName", "taxonRank"],
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


def reduce_dimensions(X: pl.DataFrame) -> pl.DataFrame:
    pca = IncrementalPCA(n_components=3000, copy=True, batch_size=3000)
    return pl.from_numpy(pca.fit_transform(X.to_numpy()))


class DistanceMatrix(DataContainer):
    _condensed: np.ndarray

    def __init__(self, condensed: np.ndarray):
        self._condensed = condensed

    @classmethod
    def build(
        cls,
        geocode_taxa_counts_dataframe: geocode_taxa_counts.GeocodeTaxaCountsDataFrame,
        geocode_dataframe: GeocodeDataFrame,
    ) -> Self:
        X = build_X(geocode_taxa_counts_dataframe, geocode_dataframe)

        # filtered.group_by("geocode").agg(pl.col("len").filter(on == value).sum().alias(str(value)) for value in set(taxonKeys)).collect()

        logger.info(f"Reducing dimensions with PCA. Previously: {X.shape}")

        # Use PCA to reduce the number of dimensions
        X = log_action("Fitting PCA", lambda: X.pipe(reduce_dimensions))

        logger.info(
            f"Reduced dimensions with PCA. Now: {X.shape[0]} geocodees, {X.shape[1]} taxon IDs"
        )

        Y = log_action(
            f"Running pdist on matrix: {X.shape[0]} geocodees, {X.shape[1]} taxon IDs",
            lambda: pdist(X, metric="braycurtis"),
        )

        return cls(Y)

    def condensed(self) -> np.ndarray:
        return self._condensed

    def squareform(self) -> np.ndarray:
        return squareform(self._condensed)
