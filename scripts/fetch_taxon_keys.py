#!/usr/bin/env python
"""Regenerate the checked-in GBIF backbone key registry.

The pipeline filters occurrences by integer GBIF backbone keys rather than name
strings (see `src/taxon_scope.py`). This script resolves a curated list of
commonly used taxonomic scopes to their keys via the GBIF species-match API and
writes `src/data/taxon_keys.json`.

Run it when adding a scope to the curated list, or after a GBIF backbone release
that may have changed keys:

    uv run python scripts/fetch_taxon_keys.py

The registry is checked in so that normal pipeline runs — and the test suite —
never touch the network.
"""

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "src/data/taxon_keys.json"

API_URL = "https://api.gbif.org/v1/species/match"

# Curated scopes worth having offline. Ordered roughly by expected use in
# cross-clade comparison work; extend as needed and re-run.
CURATED: dict[str, list[str]] = {
    "kingdom": [
        "Animalia",
        "Archaea",
        "Bacteria",
        "Chromista",
        "Fungi",
        "Plantae",
        "Protozoa",
        "Viruses",
    ],
    "phylum": [
        "Chordata",
        "Arthropoda",
        "Mollusca",
        "Annelida",
        "Cnidaria",
        "Tracheophyta",
        "Bryophyta",
        "Ascomycota",
        "Basidiomycota",
    ],
    # NOTE: rank assignments here must match the GBIF backbone, which does not
    # always agree with textbook taxonomy. Squamata and Testudines are backbone
    # *classes*, not orders. Reptilia (paraphyletic) and Actinopterygii are not
    # in the backbone at all, so they are not offerable as scopes.
    "class": [
        "Aves",
        "Mammalia",
        "Insecta",
        "Arachnida",
        "Amphibia",
        "Squamata",
        "Testudines",
        "Gastropoda",
        "Magnoliopsida",
        "Liliopsida",
        "Pinopsida",
        "Polypodiopsida",
        "Agaricomycetes",
        "Lecanoromycetes",
    ],
    "order": [
        "Coleoptera",
        "Lepidoptera",
        "Hymenoptera",
        "Diptera",
        "Hemiptera",
        "Odonata",
        "Orthoptera",
        "Araneae",
        "Passeriformes",
        "Anseriformes",
        "Charadriiformes",
        "Carnivora",
        "Rodentia",
        "Chiroptera",
        "Anura",
        "Asterales",
        "Fabales",
        "Poales",
        "Rosales",
        "Lamiales",
        "Caryophyllales",
        "Ericales",
        "Pinales",
        "Agaricales",
    ],
    "family": [
        "Fabaceae",
        "Asteraceae",
        "Poaceae",
        "Orchidaceae",
        "Formicidae",
        "Apidae",
    ],
}


def resolve(rank: str, name: str) -> int:
    """Resolve a taxon name to its GBIF backbone key for the given rank."""
    query = urllib.parse.urlencode({"rank": rank.upper(), "name": name})
    with urllib.request.urlopen(f"{API_URL}?{query}", timeout=30) as response:
        payload = json.load(response)

    if payload.get("matchType") != "EXACT":
        raise RuntimeError(
            f"{rank}:{name} did not match exactly "
            f"(matchType={payload.get('matchType')!r})"
        )
    if payload.get("rank", "").lower() != rank.lower():
        raise RuntimeError(
            f"{rank}:{name} resolved to rank {payload.get('rank')!r}, expected {rank!r}"
        )

    # The rank-specific key (e.g. orderKey) is the stable backbone identifier the
    # occurrence snapshot stores; usageKey coincides with it for an exact match
    # at that rank, but prefer the explicit field.
    key = payload.get(f"{rank}Key", payload.get("usageKey"))
    if not isinstance(key, int):
        raise RuntimeError(f"{rank}:{name} returned no usable key: {payload!r}")
    return key


def main() -> int:
    registry: dict[str, dict[str, int]] = {}
    failures: list[str] = []

    for rank, names in CURATED.items():
        resolved: dict[str, int] = {}
        for name in names:
            try:
                resolved[name] = resolve(rank, name)
            except Exception as exc:  # noqa: BLE001 - reported and surfaced below
                failures.append(f"{rank}:{name}: {exc}")
                continue
            print(f"  {rank}:{name} -> {resolved[name]}", file=sys.stderr)
        registry[rank] = dict(sorted(resolved.items()))

    if failures:
        print("\nFailed to resolve:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {REGISTRY_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
