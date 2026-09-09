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
  Accepts a name resolved against the checked-in GBIF backbone key registry, or a
  raw backbone key (`--scope=order:1470`). Omit for all taxa. See
  [Taxonomic scoping](#taxonomic-scoping).
- `--min-lat=N`, `--max-lat=N`, `--min-lon=N`, `--max-lon=N`: Bounding box coordinates.
- `--limit-results=N`: Limit number of results for testing (defaults to on, at 1000).
- `--no-limit`: Process every record. Required for a full run — `--limit-results`
  can only change the cap, not remove it.
- `--max-taxa=N`: Keep only top N taxa by occurrence count.
- `--min-geocode-presence=N`: Keep only taxa present in at least this fraction of geocodes.
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
uv run marimo run notebook.py -- --scope=order:1470 --no-stop   # raw backbone key
```

Filtering is done on GBIF's integer backbone keys rather than taxon names: keys
are stable across backbone releases that rename taxa, and an integer equality
predicate prunes parquet row groups far better than a string comparison.

Names are resolved offline against `src/data/taxon_keys.json` so that runs and
tests never need network access. To add a taxon, extend `CURATED` in
`scripts/fetch_taxon_keys.py` and re-run it:

```bash
uv run python scripts/fetch_taxon_keys.py
```

Note that the GBIF backbone does not always match textbook taxonomy — `Squamata`
and `Testudines` are backbone *classes* rather than orders, and `Reptilia` and
`Actinopterygii` are absent entirely. The registry follows the backbone, and the
generator script fails loudly on a rank mismatch rather than recording a wrong
key.

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
