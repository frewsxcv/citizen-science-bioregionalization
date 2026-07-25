"""Parsing and resolution of taxonomic scopes.

A scope is written `rank:name` (``order:Coleoptera``) or `rank:key`
(``order:1470``). Names are resolved against a checked-in registry of GBIF
backbone keys so that pipeline runs and tests never depend on network access;
regenerate it with ``scripts/fetch_taxon_keys.py``.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, cast

from src.types import TAXON_RANKS, TaxonRank, TaxonScope

REGISTRY_PATH = Path(__file__).resolve().parent / "data/taxon_keys.json"


@lru_cache(maxsize=1)
def load_registry() -> dict[str, dict[str, int]]:
    """Load the checked-in name -> GBIF backbone key registry."""
    with REGISTRY_PATH.open() as f:
        return json.load(f)


def known_names(rank: TaxonRank) -> list[str]:
    """Names resolvable offline for the given rank."""
    return sorted(load_registry().get(rank, {}))


def parse_scope(raw: Optional[str]) -> Optional[TaxonScope]:
    """Parse a ``rank:name`` or ``rank:key`` scope string.

    An empty or missing value means "no taxonomic scoping" and yields ``None``,
    which callers treat as an unfiltered run.

    Raises:
        ValueError: if the string is malformed, names an unsupported rank, or
            names a taxon absent from the registry.
    """
    if raw is None:
        return None

    raw = raw.strip()
    if not raw:
        return None

    if ":" not in raw:
        raise ValueError(
            f"Malformed scope {raw!r}: expected 'rank:name' or 'rank:key', "
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
        raise ValueError(f"Scope {raw!r} is missing a taxon name or key")

    # A bare integer is taken as a backbone key directly, which lets callers
    # reference taxa that aren't in the curated registry without a network trip.
    if name.isdigit():
        return TaxonScope(rank=rank, key=int(name), label=name)

    registry = load_registry().get(rank, {})
    if name not in registry:
        raise ValueError(
            f"Unknown {rank} {name!r}. Either pass the GBIF backbone key "
            f"directly (e.g. '{rank}:1470'), or add the name to CURATED in "
            f"scripts/fetch_taxon_keys.py and re-run it. "
            f"Known {rank} names: {', '.join(known_names(rank)) or '(none)'}"
        )

    return TaxonScope(rank=rank, key=registry[name], label=name)
