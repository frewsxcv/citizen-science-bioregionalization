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
- `--limit-results=N`: Limit number of results for testing.
- `--max-taxa=N`: Keep only top N taxa by occurrence count.
- `--min-geocode-presence=N`: Keep only taxa present in at least this fraction of geocodes.
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

### Outputs:

All outputs are saved to the `output/` directory:
- GeoJSON file: `output/output.geojson`
- HTML report: `output/output.html`
- Log file: depends on the path provided in `--log-file` option, but defaults to the output directory

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0.en.html). 
