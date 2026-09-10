"""Parsing of taxonomic scopes.

A scope is written ``rank:name`` (``order:Coleoptera``) and filters occurrences
on the Darwin Core rank name column.

Scopes used to be written ``rank:key`` as well, resolved against a checked-in
registry of GBIF backbone keys. Current snapshots no longer carry the per-rank
key columns, so there is nothing for a key to match; see ``TaxonScope``.
"""

from typing import Optional, cast

from src.types import TAXON_RANKS, TaxonRank, TaxonScope


def parse_scope(raw: Optional[str]) -> Optional[TaxonScope]:
    """Parse a ``rank:name`` scope string.

    An empty or missing value means "no taxonomic scoping" and yields ``None``,
    which callers treat as an unfiltered run.

    The name is not validated against a list of known taxa: the authority is the
    data itself, and a name absent from it simply matches nothing. A scope that
    matches nothing fails later with "More than one geocode is required to
    cluster", which is a clearer signal than a curated registry that has to be
    kept in step with the backbone.

    Raises:
        ValueError: if the string is malformed or names an unsupported rank.
    """
    if raw is None:
        return None

    raw = raw.strip()
    if not raw:
        return None

    if ":" not in raw:
        raise ValueError(
            f"Malformed scope {raw!r}: expected 'rank:name', "
            f"e.g. 'order:Coleoptera'. Valid ranks: {', '.join(TAXON_RANKS)}"
        )

    rank_str, _, name = raw.partition(":")
    rank_str = rank_str.strip().lower()
    name = name.strip()

    if rank_str not in TAXON_RANKS:
        raise ValueError(
            f"Unsupported rank {rank_str!r} in scope {raw!r}. "
            f"Valid ranks: {', '.join(TAXON_RANKS)}"
        )
    rank = cast(TaxonRank, rank_str)

    if not name:
        raise ValueError(f"Scope {raw!r} is missing a taxon name")

    # Reject the old rank:key form rather than filtering for a taxon literally
    # named "1470" and reporting an empty result.
    if name.isdigit():
        raise ValueError(
            f"Scope {raw!r} looks like a GBIF backbone key. Scoping now matches "
            f"the taxon name, because current GBIF snapshots no longer carry the "
            f"per-rank key columns. Pass a name instead, e.g. '{rank}:Aves'."
        )

    return TaxonScope(rank=rank, name=name)
