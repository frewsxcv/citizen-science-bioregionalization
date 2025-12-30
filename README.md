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

Run the script with the following command:

```bash
python run.py [OPTIONS] INPUT_FILE
```

### Options:

- `--geocode-precision`: Precision level for H3 geocoding (required).
- `--min-clusters`: Minimum number of clusters to test (required).
- `--max-clusters`: Maximum number of clusters to test (required).
- `--log-file`: Path to save the log file (required).
- `--taxon-filter`: Optional filter for specific taxon.
- `--min-lat`, `--max-lat`, `--min-lon`, `--max-lon`: Bounding box coordinates.
- `--limit-results`: Limit number of results for testing.

### Example:

```bash
python notebook.py --geocode-precision 5 --min-clusters 2 --max-clusters 15 --log-file run.log gs://public-datasets-gbif/occurrence/2025-11-01/occurrence.parquet/*
```

### Outputs:

All outputs are saved to the `output/` directory:
- GeoJSON file: `output/output.geojson`
- HTML report: `output/output.html`
- Log file: depends on the path provided in `--log-file` option, but defaults to the output directory

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0.en.html). 
