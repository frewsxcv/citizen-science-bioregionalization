import argparse


def parse_cli_input() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process Darwin Core CSV data and generate clusters."
    )

    # Add required options
    parser.add_argument(
        "--geocode-precision", type=int, required=True, help="Precision of the geocode"
    )
    parser.add_argument(
        "--num-clusters", type=int, required=True, help="Number of clusters to generate"
    )
    parser.add_argument(
        "--log-file", type=str, required=True, help="Path to the log file"
    )

    # Add optional arguments
    parser.add_argument(
        "--taxon-filter",
        type=str,
        default=None,
        help="Filter to a specific taxon (e.g., 'Aves')",
    )

    # Positional arguments
    parser.add_argument("input_file", type=str, help="Path to the input file")

    return parser.parse_args()
