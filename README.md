# Citizen Science Ecoregions

## Overview

This project analyzes biodiversity data to identify and visualize ecological regions (ecoregions) based on species distribution. It processes Darwin Core formatted CSV data, clusters geographic locations based on species similarity, and outputs the results as GeoJSON for visualization.

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

- `--geohash-precision`: Precision level for geocoding (required).
- `--num-clusters`: Number of clusters to generate (required).
- `--log-file`: Path to save the log file (required).
- `--plot`: Flag to plot the clusters visually (optional).

### Example:

```bash
python run.py --geohash-precision 5 --num-clusters 10 data/input.csv --plot
```

### Outputs:

All outputs are saved to the `output/` directory:
- GeoJSON file: `output/output.geojson`
- HTML report: `output/output.html`
- Log file: depends on the path provided in `--log-file` option, but defaults to the output directory

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0.en.html). 
