import argparse
import typing


class Inputs(typing.NamedTuple):
    geohash_precision: int
    num_clusters: int
    log_file: str
    input_file: str
    output_file: str
    show_dendrogram: bool
    plot: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> typing.Self:
        return cls(
            geohash_precision=args.geohash_precision,
            num_clusters=args.num_clusters,
            log_file=args.log_file,
            input_file=args.input_file,
            output_file=args.output_file,
            show_dendrogram=args.show_dendrogram,
            plot=args.plot,
        )
