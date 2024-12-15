import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster geohash data.")
    parser.add_argument(
        "--geohash-precision",
        type=int,
        help="Precision of the geohash",
        required=True,
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to the log file",
        required=True,
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input file",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output file",
    )
    parser.add_argument(
        "--show-dendrogram",
        action="store_true",
        help="Show the dendrogram",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the clusters",
    )
    return parser.parse_args()
