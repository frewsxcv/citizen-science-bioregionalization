from src.cluster_stats import Stats
import matplotlib as mpl

import random


class ClusterColorBuilder:
    @classmethod
    def blue(cls) -> str:
        """Return a random blue color."""
        return mpl.colors.to_hex(mpl.colormaps["Blues"](random.random()))

    @classmethod
    def red(cls) -> str:
        """Return a random red color."""
        return mpl.colors.to_hex(mpl.colormaps["Reds"](random.random()))

    @classmethod
    def green(cls) -> str:
        """Return a random green color."""
        return mpl.colors.to_hex(mpl.colormaps["Greens"](random.random()))

    @classmethod
    def determine_color_for_cluster(cls, stats: Stats) -> str:
        # If the number of Anseriformes (waterfowl) and Charadriiformes (wading
        # birds) is greater than Passeriformes (perching birds) for the given
        # cluster, use blue.
        if (
            stats.order_count("Anseriformes") + stats.order_count("Charadriiformes")
        ) > stats.order_count("Passeriformes"):
            return cls.blue()

        # If the number of Piciformes (perching birds) is greater than
        # Anseriformes (waterfowl) for the given cluster, use green.
        elif stats.order_count("Piciformes") > stats.order_count("Anseriformes"):
            return cls.green()

        return cls.red()
