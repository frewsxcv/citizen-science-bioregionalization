# Citizen Science Bioregionalization

## Overview

This project analyzes citizen science data to identify and visualize biological regions (bioregions) based on species distribution. It processes Darwin Core formatted CSV data, clusters geographic locations based on species similarity, and outputs the results as GeoJSON for visualization.

![sample output map](https://github.com/user-attachments/assets/8e72d089-dbb8-4b78-b645-4dd88d420384)


## Installation

Ensure Python is installed (version specified in `.python-version`).

Install dependencies using:
```bash
pip install -r pyproject.toml
```

## Usage

### Interactive Mode

Open the notebook in your browser for interactive exploration:

```bash
uv run marimo edit notebook.py
```

### Command Line Mode

Run the notebook with CLI arguments (using `--key=value` format after `--`):

```bash
uv run marimo run notebook.py -- [OPTIONS]
```

### Options:

- `--geocode-precision=N`: Precision level for H3 geocoding (default: 4).
- `--min-clusters=N`: Minimum number of clusters to test (default: 2).
- `--max-clusters=N`: Maximum number of clusters to test (default: 20).
- `--log-file=PATH`: Path to save the log file (default: run.log).
- `--parquet-source-path=PATH`: Path to the parquet data source.
- `--scope=RANK:TAXON`: Optional taxonomic scope, e.g. `--scope=order:Coleoptera`.
  Matches the taxon name, case-sensitively. Omit for all taxa. See
  [Taxonomic scoping](#taxonomic-scoping).
- `--min-lat=N`, `--max-lat=N`, `--min-lon=N`, `--max-lon=N`: Bounding box coordinates.
- `--limit-results=N`: Limit number of results for testing (defaults to on, at 1000).
  Takes the first N records in scan order, which is fine for a quick run and
  wrong for a representative one: a GBIF snapshot's file order is grouped by
  source dataset, so the head is whichever datasets sort earliest.
- `--sample-records=N`: Keep approximately N records, chosen uniformly rather
  than by scan order. Costs one counting pass over the source. Use this, not
  `--limit-results`, when a run has to cap its input for memory but the result
  is meant to describe the region. Supersedes `--limit-results` when both are
  given.
- `--no-limit`: Process every record. Required for a full run — `--limit-results`
  can only change the cap, not remove it.
- `--max-taxa=N`: Keep only top N taxa by occurrence count.
- `--min-geocode-presence=N`: Keep only taxa present in at least this fraction of geocodes.
- `--min-hex-records=N`: Drop hexagons holding fewer than N occurrence records.
  Omit it and a floor is derived from the data — a tenth of what the median
  hexagon holds, bounded to between 20 and 100. Pass `--no-hex-floor` to keep
  every hexagon.
  A hexagon observed once yields a composition vector of a single taxon, which
  says more about survey effort than about what lives there. See
  [Why there is a sampling floor](#why-there-is-a-sampling-floor).
- `--no-hex-floor`: Keep every hexagon, however sparsely surveyed.
- `--num-clusters=N`: Draw exactly N regions instead of choosing a number. See
  [Asking for more regions](#asking-for-more-regions).
- `--metric-weights=S,C,D`: Weights for the combined score — silhouette,
  Calinski-Harabasz, Davies-Bouldin. Defaults to `0.7,0.15,0.15`.
- `--composition-metric=presence|abundance`: How a hexagon's composition is
  represented. Defaults to `presence`. See
  [Presence or abundance](#presence-or-abundance).
- `--terrestrial-only`: Drop hexagons whose centre falls in the sea. Country-code
  filtering includes the maritime zone, so coastal clusters can otherwise be driven
  by fish and seabirds. Uses the checked-in Natural Earth 1:50m coastline, which is
  generalised to ~50 km — small offshore islands are dropped along with the ocean.
- `--seed=N`: Seed for UMAP and the PERMANOVA permutation test (default: 0).
  Runs are reproducible: the same seed and input produce byte-identical output.
- `--no-seed`: Opt out of seeding for a faster, multithreaded UMAP, at the cost
  of results that vary between runs.
- `--no-images`: Skip the Wikidata image lookup, the pipeline's only network call
  after data loading.
- `--no-stop`: Bypass the run button when running from command line.

### Example:

```bash
uv run marimo run notebook.py -- \
  --geocode-precision=5 \
  --min-clusters=2 \
  --max-clusters=15 \
  --log-file=run.log \
  --parquet-source-path="gs://public-datasets-gbif/occurrence/2025-11-01/occurrence.parquet/*" \
  --no-stop
```

### Export to HTML:

```bash
uv run marimo export html notebook.py -o output/index.html -- \
  --geocode-precision=5 \
  --min-clusters=2 \
  --max-clusters=15 \
  --parquet-source-path="gs://public-datasets-gbif/occurrence/2025-11-01/occurrence.parquet/*" \
  --no-stop
```

### Taxonomic scoping

Runs can be restricted to a clade with `--scope=RANK:TAXON`, where `RANK` is one
of `kingdom`, `phylum`, `class`, `order`, `family`, `genus`:

```bash
uv run marimo run notebook.py -- --scope=class:Aves --no-stop
```

Filtering matches the Darwin Core rank *name* column (`class`, `order`, ...).
Matching is case-sensitive; GBIF capitalises names at every rank above species,
so `class:Aves` matches and `class:aves` does not.

Scoping used to filter on GBIF's integer backbone keys (`classKey` and friends),
which was the better predicate — keys survive backbone releases that rename
taxa, and integer equality prunes parquet row groups better than string
comparison. Current snapshots no longer carry those columns at all: of the
backbone keys only `taxonkey` and `specieskey` remain. A key-based scope could
not run against this project's own default data source, so `--scope=order:1470`
is no longer accepted and fails with an explanatory error.

Names are not validated against a list of known taxa. The data is the authority,
and a name absent from it simply matches nothing, which surfaces as "More than
one geocode is required to cluster" rather than as a curated registry that has
to be kept in step with the backbone.

Two consequences worth knowing:

- **Homonyms.** Names are unique only within a kingdom, so `genus:Oenanthe`
  matches both the wheatears (birds) and the water dropworts (plants). A
  backbone key would not have. There is no way around this while the keys are
  absent from the data; pair the scope with a coarser rank if it matters.
- **The GBIF backbone does not always match textbook taxonomy.** `Squamata` and
  `Testudines` are backbone *classes* rather than orders, and `Reptilia` and
  `Actinopterygii` are absent entirely. Scope to what the backbone calls things,
  not to what a field guide does.




### Asking for more regions

Nothing in the selection metrics prefers more regions. Measured on Colombia at
presence/absence with the sampling floor, k from 2 to 12:

| k | silhouette (as the selector sees it) | Calinski-Harabasz | Davies-Bouldin |
|---|---|---|---|
| 2 | 0.4300 | 109.39 | 4.03 |
| 6 | 0.1865 | 41.86 | 5.35 |
| 12 | 0.1159 | 27.06 | 5.83 |

All three are monotone: silhouette and Calinski-Harabasz fall as k rises,
Davies-Bouldin rises. Every criterion prefers the smallest k in range, so
`--min-clusters=2` reliably yields two regions and no reweighting changes that.

If you want more regions, ask for them: `--num-clusters=6`. The run logs that it
was asked rather than derived —

```
k pinned to 6; the combined score preferred 2. This is a choice about how many
regions to draw, not a claim that 6 fits the data better.
```

— and still reports every metric for the k it drew, so the cost is visible. On
Colombia, k=6 scores 0.1865 against k=2's 0.4300.

This is defensible rather than merely indulgent, because separation is weak at
*every* k: the best composition-space silhouette on the published dataset is
0.1262, well under the 0.25 threshold. The selector is choosing between options
that are all poorly separated, so its preference for k=2 is not a strong signal
about the data. Six regions is not less true than two; it is a finer partition
of the same gradient.

`--metric-weights` is also available and now does what it says. Silhouette used
to be normalised onto its theoretical [-1, 1] range while the other two were
min-max normalised across the k values tested, so on real data silhouette's
whole range after normalisation was 0.0113 and a weight of 0.4 could move the
combined score by at most 0.0045 — against 0.3 for Calinski-Harabasz. The
weights were decorative. All three are now normalised the same way.

### Presence or abundance

By default a hexagon is described by *which* taxa were seen there, not how many
of each — every count becomes a 1 or a 0. Bray-Curtis over presence bits is
Sørensen dissimilarity, so the metric is unchanged; only what it is given
changes. Pass `--composition-metric=abundance` for the counts.

Presence is the default because the counts are not trustworthy. In a Colombian
extract, `individualCount` has a **median of 2** and a **maximum of
35,182,100**, and **19% of records carry no count at all** and are filled with
1. A single record claiming 35 million individuals can dominate a hexagon's
entire profile — which is how a moss once reached the top of a region's
indicator taxa with 140 million individuals.

Measured at k=4 against Bray-Curtis over raw counts, with the sampling floor
applied:

| representation | Colombia R² | SE Australia R² | best silhouette |
|---|---|---|---|
| abundance (counts) | 0.0351 | 0.0468 | 0.0170 |
| **presence/absence** | **0.0591** | **0.1114** | **0.0298** |

That is +68% and +138% explained variance, and the best separation of seven
representations tried — the others being `RobustScaler` (the previous default),
column-max scaling, `log1p`, Hellinger, and Wisconsin double standardisation.

Presence also makes per-taxon scaling moot, which matters because the scaling
that was there did very little: `RobustScaler` divides by the interquartile
range, and 99.4% of taxa appear in under 25% of hexagons, so their IQR is zero
and scikit-learn silently falls back to a scale of 1.0. It altered 0.27% of
cells. Note that it is *not* skipped by accident on the presence path but
deliberately: on a binary column whose median is 1, centring maps the column to
0 and −1, handing Bray-Curtis the negative values it is not defined for.

What this does not fix is separation. The best silhouette across every
representation tested is 0.0298, which is close to none. Explained variance
improves; the regions do not become distinct.

### Why the land mask counts records

`--terrestrial-only` keeps a hexagon when a majority of its own records fall on
land. Two earlier versions tested a single synthetic point instead, and both
failed the same way — a point derived from the data need not sit where the data
is:

- **The hexagon's geometric centre.** At H3 resolution 4 a cell spans roughly
  1,770 km², so its midpoint can be 18 km from the records. Manhattan's cell
  centres in Long Island Sound, 17.7 km from Central Park, so one of the most
  intensively recorded hexagons on the map was discarded.
- **The marginal median of the records.** Median latitude and median longitude
  are taken independently, so the resulting point need not be near any actual
  record. On Manhattan — a narrow island between two rivers — it lands in the
  East River, while **88.0% of that cell's ten million records are on land**.

A share has no such failure mode. Records are sampled rather than all tested,
and the sample is **shuffled with a fixed seed rather than taken from the head**,
which matters more than it sounds: the snapshot is ordered by source dataset, so
a cell's first records all come from whichever dataset appears earliest. On
Manhattan the first 2,000 records are 45.8% on land against a true 88.9% — the
difference between dropping the cell and keeping it.

### Why there is a sampling floor

A hexagon with three records has a three-taxon composition vector that is
maximally distant from everything else, so Ward peels it off as a cluster of its
own. On a country-scale run that is not hypothetical: hiding 1% of Colombia's
records turned a partition of 1538/1545/790/193 hexagons into 2460/1604/1/1 —
two of the four regions became single hexagons.

Measured as the mean adjusted Rand index between a partition and the same
partition after hiding 5% of observed records, three draws each, at k=4:

| region | median records/hexagon | no floor | fixed 50 | derived |
|---|---|---|---|---|
| Alps / Central Europe | 250 | 0.975 | 0.926 | **0.984** |
| California | 25 | −0.000 | 0.911 | 0.903 |
| SE Australia | 163 | −0.001 | 0.619 | **0.716** |
| Colombia | 340 | −0.000 | 0.509 | **0.517** |

Without a floor, three of the four regions collapse to chance agreement — the
partition carries no information that survives a 5% perturbation. The Alps do
not, because they are evenly surveyed and have no tail of sparse hexagons, which
is also why a fixed floor of 50 *degrades* them: it discards data that was doing
no harm.

Deriving the floor from each region's own median handles both cases, and keeps
more hexagons than the fixed floor in every region measured (Colombia 3000
against 2860; California 1367 against 891). The absolute term matters where the
whole extent is thin: California's median hexagon holds 25 records, so a purely
relative floor would come out at 2 and filter nothing.

The derived floor is bounded at both ends. The lower bound is for thin extents,
as above. The upper bound is for dense ones, and was found in CI rather than in
testing: the published run is at H3 resolution 4, where a hexagon covers seven
times the area of a resolution-5 one and its median holds around 73,000 records.
A tenth of that derived a floor of 7,274 — discarding hexagons with thousands of
observations, which is not what the floor is for. All four regions above are
resolution 5 and derive between 20 and 34, so none of them exposed it.

Two caveats on those numbers. Only Colombia is a full-density extract; the other
three are 8.4% subsamples of the GBIF snapshot, so their hexagons are roughly
twelve times sparser than a complete extract would be, and the absolute
thresholds do not transfer even though the pattern does. And the measurement is
noisy — the same Colombia configuration scored 0.934 on one draw and 0.319 on
another, which is why the table reports means over three.

### Working from a local country cache

Pointing a run straight at a GBIF snapshot means a full pass over ~266 GiB every
time, because each snapshot file is a single row group whose coordinate statistics
span the globe — a geographic predicate prunes nothing. When iterating on one
country, extract it once and run against the local copy instead:

```bash
# One full pass over the snapshot (~30 minutes), narrowed to a country code.
# Reads GBIF's AWS mirror, which needs no credentials.
uv run python scripts/extract_country_parquet.py \
  --country=CO --snapshot=2026-09-01 --output=data/colombia.parquet

# Optional: drop taxa seen in only a handful of hexagons.
uv run python scripts/filter_sparse_taxa.py \
  --input=data/colombia.parquet --output=data/colombia_res5.parquet \
  --precision=5 --min-taxon-hexes=10

uv run marimo run notebook.py -- \
  --parquet-source-path=data/colombia_res5.parquet \
  --geocode-precision=5 --no-limit --no-stop
```

`filter_sparse_taxa.py` overlaps with the notebook's `--max-taxa` /
`--min-geocode-presence` flags, which work fine — the notebook re-derives its
geocode set from the filtered counts, so hexagons left with no taxa simply drop
out of the run. The script is worth using when you want that filtering done once
and cached across many runs, or when you want the threshold expressed as "seen in
at least N hexagons at precision P" rather than as a fraction of the surviving
hexagons. Either way, note that filtering removes hexagons from the map, and that
a threshold aggressive enough to empty every hexagon fails the run outright.

### Outputs:

All outputs are saved to the `output/` directory:
- GeoJSON file: `output/output.geojson`
- HTML report: `output/output.html`
- Log file: depends on the path provided in `--log-file` option, but defaults to the output directory

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0.en.html). 
