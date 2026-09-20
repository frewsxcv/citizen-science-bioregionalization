"""Render what a run found into a self-contained page.

This replaces a hand-written page whose numbers were transcribed from runs and
then went stale. Everything here is computed by the run that emits it, so a
section either carries this run's numbers or says it could not compute them --
there is no third state where an old number survives a change in the pipeline.

No build step and no network: inline SVG built from the run's own values, one
`<details>` table behind every chart that carries numbers, and an `aria-label`
describing the shape of each.

Colours are Okabe-Ito, which is published as colour-vision-deficiency safe, and
are assigned by entity rather than by position, so a run missing one clade does
not repaint the others.
"""

import dataclasses
import html
import json
from dataclasses import dataclass, field
from typing import Optional, Sequence

from src.clade_congruence import Congruence
from src.epa_reference import ReferenceAgreement
from src.types import ClusterLevels

# Assigned by entity and never recycled.
COLORS = {
    "aves": ("#0072B2", "#4FA3D8"),
    "plantae": ("#009E73", "#35C79A"),
    "combined": ("#CC79A7", "#E3A0C4"),
    "reference": ("#6d6b63", "#918f86"),
    "records": ("#D55E00", "#F08A4B"),
    "taxa": ("#0072B2", "#4FA3D8"),
}

_CSS = """
:root{--surface:#fff;--sunken:#f7f7f5;--ink:#1a1a19;--ink2:#45443f;--muted:#6d6b63;
--rule:#e3e2dd;--grid:#ececE7}
@media(prefers-color-scheme:dark){:root{--surface:#191917;--sunken:#222220;
--ink:#f2f1ec;--ink2:#c9c7bf;--muted:#918f86;--rule:#34332f;--grid:#2b2a27}}
*{box-sizing:border-box}
body{margin:0;padding:2.5rem 1.25rem 5rem;background:var(--surface);color:var(--ink);
font:16px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:52rem;margin:0 auto}
h1{font-size:1.9rem;line-height:1.25;margin:0 0 .5rem}
h2{font-size:1.3rem;margin:3rem 0 .25rem;padding-top:1.5rem;border-top:1px solid var(--rule)}
h3{font-size:1.05rem;font-weight:600;margin:1.75rem 0 .5rem;color:var(--ink2)}
p{margin:.75rem 0;color:var(--ink2)}
.sub{color:var(--muted);margin-top:0}
figure{margin:1.25rem 0 0}
figcaption{color:var(--muted);font-size:.9rem;margin-top:.5rem}
svg{display:block;width:100%;height:auto;overflow:visible}
.gridline{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--rule);stroke-width:1}
.tick{fill:var(--muted);font-size:11px}
.lbl{fill:var(--ink2);font-size:12px;font-weight:600}
details{margin-top:.75rem}
summary{cursor:pointer;color:var(--muted);font-size:.9rem}
table{border-collapse:collapse;margin-top:.5rem;font-size:.9rem;width:100%}
th,td{text-align:right;padding:.3rem .6rem;border-bottom:1px solid var(--rule)}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600}
.note{background:var(--sunken);border-left:3px solid var(--rule);padding:.75rem 1rem;
margin:1rem 0;color:var(--ink2);font-size:.95rem}
.missing{color:var(--muted);font-style:italic}
"""


@dataclass
class RunContext:
    """What the run was, so a reader can tell two pages apart."""

    source: str
    bbox: str
    geocode_precision: int
    hexagons: int
    #: Taxa in the taxonomy, before the run's taxa filters.
    taxa: int
    #: Taxa that survived `filter_top_taxa_lf` and so were actually clustered.
    #: Reported separately because the clade shares below are fractions of
    #: *this* number, not of `taxa`. Showing only the larger one invites the
    #: reader to divide by it and get a different answer -- on the published
    #: run, 671 Aves taxa is 6.7% of the 10,000 analysed and 0.3% of the
    #: 208,296 in the taxonomy.
    taxa_analysed: int
    records: Optional[int]
    #: Which cuts the run emitted and which one it publishes. One object rather
    #: than two ints, because the distinction between them was got wrong twice
    #: when they travelled separately -- see `types.ClusterLevels`. Every figure
    #: on this page describes `levels.published`.
    levels: ClusterLevels
    composition_metric: str
    seed: Optional[int]


@dataclass
class CladeShare:
    """One clade's share of the records and of the taxa."""

    name: str
    key: str
    record_share: float
    taxon_share: float
    records: int
    taxa: int


@dataclass
class FindingsData:
    """Everything the page draws, as computed by one run."""

    context: RunContext
    clade_shares: list[CladeShare] = field(default_factory=list)
    congruence: list[Congruence] = field(default_factory=list)
    congruence_pair: tuple[str, str] = ("", "")
    reference_by_k: list[tuple[int, ReferenceAgreement]] = field(default_factory=list)
    clade_vs_reference: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    latitude_spans: dict[str, list[tuple[int, float, float, int]]] = field(
        default_factory=dict
    )
    skipped: list[str] = field(default_factory=list)


#: Two cuts whose Adjusted Rand scores differ by less than this are reported as
#: a tie rather than ranked. On the published run the top two are 0.0001 apart,
#: and presenting that as a winner would be a precision the measure does not
#: have.
ARI_TIE = 0.01


def _esc(value: object) -> str:
    return html.escape(str(value))


def _join(parts) -> str:
    """"a", "a and b", "a, b and c"."""
    items = list(parts)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _line_chart(
    series: Sequence[tuple[str, str, list[tuple[float, float]]]],
    x_label: str,
    y_label: str,
    aria: str,
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> str:
    """A multi-series line chart over integer x, drawn as inline SVG.

    `series` is (label, colour-key, [(x, y), ...]). Empty series are dropped
    rather than drawn flat at zero, which would read as a measured result.
    """
    series = [s for s in series if s[2]]
    if not series:
        return '<p class="missing">Not computed in this run.</p>'

    w, h = 720, 300
    pad_l, pad_r, pad_t, pad_b = 48, 132, 16, 40
    xs = sorted({x for _, _, pts in series for x, _ in pts})
    x_lo, x_hi = min(xs), max(xs)
    span = max(x_hi - x_lo, 1)

    def px(x: float) -> float:
        return pad_l + (x - x_lo) / span * (w - pad_l - pad_r)

    def py(y: float) -> float:
        frac = (y - y_min) / (y_max - y_min) if y_max > y_min else 0.0
        return h - pad_b - frac * (h - pad_t - pad_b)

    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{_esc(aria)}">',
    ]
    for i in range(5):
        y = y_min + (y_max - y_min) * i / 4
        parts.append(
            f'<line class="gridline" x1="{pad_l}" x2="{w - pad_r}" '
            f'y1="{py(y):.1f}" y2="{py(y):.1f}"/>'
            f'<text class="tick" x="{pad_l - 8}" y="{py(y) + 4:.1f}" '
            f'text-anchor="end">{y:.2f}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{pad_l}" x2="{w - pad_r}" '
        f'y1="{h - pad_b}" y2="{h - pad_b}"/>'
    )
    for x in xs:
        parts.append(
            f'<text class="tick" x="{px(x):.1f}" y="{h - pad_b + 16}" '
            f'text-anchor="middle">{x:g}</text>'
        )
    parts.append(
        f'<text class="lbl" x="{(pad_l + w - pad_r) / 2:.0f}" y="{h - 4}" '
        f'text-anchor="middle">{_esc(x_label)}</text>'
        f'<text class="lbl" x="14" y="{h / 2:.0f}" text-anchor="middle" '
        f'transform="rotate(-90 14 {h / 2:.0f})">{_esc(y_label)}</text>'
    )

    for idx, (label, key, pts) in enumerate(series):
        light, dark = COLORS.get(key, COLORS["combined"])
        pts = sorted(pts)
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}"
            for i, (x, y) in enumerate(pts)
        )
        parts.append(
            f'<path d="{d}" fill="none" stroke="{light}" stroke-width="2.5" '
            f'class="s{idx}"/>'
        )
        for x, y in pts:
            parts.append(
                f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" fill="{light}">'
                f"<title>{_esc(label)} at {x:g}: {y:.3f}</title></circle>"
            )
        ly = pad_t + 14 + idx * 20
        parts.append(
            f'<circle cx="{w - pad_r + 10}" cy="{ly - 4}" r="5" fill="{light}"/>'
            f'<text class="tick" x="{w - pad_r + 22}" y="{ly}">{_esc(label)}</text>'
        )
        parts.append(
            f"<style>@media(prefers-color-scheme:dark){{"
            f".s{idx}{{stroke:{dark}}}}}</style>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _share_chart(shares: Sequence[CladeShare]) -> str:
    """Paired bars: share of records against share of taxa, per clade."""
    if not shares:
        return '<p class="missing">Not computed in this run.</p>'

    row_h, bar_h, gap = 64, 22, 6
    w = 720
    pad_l, pad_r = 120, 60
    h = len(shares) * row_h + 30
    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="'
        + _esc(
            "; ".join(
                f"{s.name} is {s.record_share * 100:.1f} percent of records and "
                f"{s.taxon_share * 100:.1f} percent of taxa"
                for s in shares
            )
        )
        + '">'
    ]
    scale = w - pad_l - pad_r
    for i, s in enumerate(shares):
        top = i * row_h + 10
        parts.append(
            f'<text class="lbl" x="{pad_l - 10}" y="{top + bar_h}" '
            f'text-anchor="end">{_esc(s.name)}</text>'
        )
        for j, (label, key, value, count) in enumerate(
            (
                ("records", "records", s.record_share, s.records),
                ("taxa", "taxa", s.taxon_share, s.taxa),
            )
        ):
            light, dark = COLORS[key]
            y = top + j * (bar_h + gap)
            width = max(value * scale, 1.5)
            parts.append(
                f'<rect x="{pad_l}" y="{y}" width="{width:.1f}" height="{bar_h}" '
                f'rx="3" fill="{light}" class="b{i}{j}">'
                f"<title>{_esc(s.name)}: {value * 100:.1f}% of {label} "
                f"({count:,})</title></rect>"
                f'<text class="tick" x="{pad_l + width + 8:.1f}" y="{y + bar_h - 6}">'
                f"{value * 100:.1f}% of {label}</text>"
                f"<style>@media(prefers-color-scheme:dark)"
                f"{{.b{i}{j}{{fill:{dark}}}}}</style>"
            )
    parts.append("</svg>")
    return "".join(parts)


def _span_chart(spans: dict[str, list[tuple[int, float, float, int]]]) -> str:
    """Latitude range of each cluster, one row per clade."""
    rows = [(clade, s) for clade, entries in spans.items() for s in entries]
    if not rows:
        return '<p class="missing">Not computed in this run.</p>'

    lats = [v for _, s in rows for v in (s[1], s[2])]
    lo, hi = min(lats), max(lats)
    span = max(hi - lo, 0.5)
    w, pad_l, pad_r = 720, 120, 40
    row_h = 30
    h = len(rows) * row_h + 44
    scale = w - pad_l - pad_r

    def px(lat: float) -> float:
        return pad_l + (lat - lo) / span * scale

    aria = "; ".join(
        f"{clade} cluster {s[0]} spans {s[1]:.1f} to {s[2]:.1f} degrees north"
        for clade, s in rows
    )
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{_esc(aria)}">']
    for i, (clade, (cluster, min_lat, max_lat, hexes)) in enumerate(rows):
        key = clade.lower()
        light, dark = COLORS.get(key, COLORS["combined"])
        y = i * row_h + 12
        x0, x1 = px(min_lat), px(max_lat)
        parts.append(
            f'<text class="lbl" x="{pad_l - 10}" y="{y + 14}" text-anchor="end">'
            f"{_esc(clade)} · {cluster}</text>"
            f'<rect x="{x0:.1f}" y="{y}" width="{max(x1 - x0, 2):.1f}" height="18" '
            f'rx="3" fill="{light}" class="r{i}">'
            f"<title>{_esc(clade)} cluster {cluster}: {min_lat:.2f} to "
            f"{max_lat:.2f} N, {hexes:,} hexagons</title></rect>"
            f"<style>@media(prefers-color-scheme:dark)"
            f"{{.r{i}{{fill:{dark}}}}}</style>"
        )
    for frac in (0.0, 0.5, 1.0):
        lat = lo + span * frac
        parts.append(
            f'<text class="tick" x="{px(lat):.1f}" y="{h - 8}" '
            f'text-anchor="middle">{lat:.1f}°N</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    if not rows:
        return ""
    head = "".join(f"<th>{_esc(x)}</th>" for x in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>" for row in rows
    )
    return (
        "<details><summary>Show as table</summary>"
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        "</details>"
    )


def render_findings_page(data: FindingsData) -> str:
    """Build the page. Returns a complete HTML document."""
    c = data.context
    out: list[str] = []

    out.append(
        "<h1>What this run found</h1>"
        '<p class="sub">Every number below was computed by the run that produced '
        "this page. A section with nothing to show says so rather than carrying "
        "a number from an earlier run.</p>"
        '<div class="note">'
        f"<strong>{c.hexagons:,} hexagons</strong> at H3 resolution "
        f"{c.geocode_precision}, <strong>{c.taxa_analysed:,} taxa</strong> "
        f"clustered out of {c.taxa:,} in the taxonomy"
        + (f", from <strong>{c.records:,} records</strong>" if c.records else "")
        + f". Composition measured with <code>{_esc(c.composition_metric)}</code>; "
        f"published at <strong>{c.levels.published} regions</strong>"
        + (
            "."
            if c.levels.selector_agrees
            else f", where the selector's score peaked at {c.levels.selector}."
        )
        + f"<br>Extent {_esc(c.bbox)} · source <code>{_esc(c.source)}</code>"
        + (f" · seed {c.seed}" if c.seed is not None else " · unseeded")
        + "</div>"
    )

    # --- Congruence -------------------------------------------------------
    left, right = data.congruence_pair
    out.append("<h2>Congruence — do two biotas draw the same map?</h2>")
    if data.congruence:
        best = max(data.congruence, key=lambda x: x.adjusted_rand)
        out.append(
            f"<p>{_esc(left)} and {_esc(right)} were clustered separately, over "
            f"the hexagons each occupies, and the two partitions compared at "
            f"every cut. They agree best at <strong>{best.num_clusters} regions"
            f"</strong> (ARI {best.adjusted_rand:.3f}). Adjusted Rand corrects "
            f"for chance, so zero is what two unrelated maps would score.</p>"
        )
        series: list[tuple[str, str, list[tuple[float, float]]]] = [
            (
                f"{left} vs {right}",
                "combined",
                [(float(c.num_clusters), c.adjusted_rand) for c in data.congruence],
            )
        ]
        for clade, points in data.clade_vs_reference.items():
            series.append(
                (
                    f"{clade} vs EPA",
                    clade.lower(),
                    [(float(k), v) for k, v in points],
                )
            )
        out.append(
            "<figure>"
            + _line_chart(
                series,
                "Number of regions",
                "Adjusted Rand Index",
                aria=(
                    f"Adjusted Rand Index against number of regions. "
                    f"{left} and {right} agree at "
                    f"{data.congruence[0].adjusted_rand:.2f} at "
                    f"{data.congruence[0].num_clusters} regions, peaking at "
                    f"{best.adjusted_rand:.2f} at {best.num_clusters}."
                ),
                y_min=min(0.0, min(x.adjusted_rand for x in data.congruence)),
            )
            + "<figcaption>Clade agreement against the published framework, by "
            "grain. Compared only over hexagons both partitions kept.</figcaption>"
            + _table(
                ["Regions", "ARI", "Hexagons compared"],
                [
                    [x.num_clusters, f"{x.adjusted_rand:.4f}", f"{x.compared:,}"]
                    for x in data.congruence
                ],
            )
            + "</figure>"
        )
    else:
        out.append(
            '<p class="missing">Not computed in this run — no two clades held '
            "enough hexagons to cluster separately.</p>"
        )

    if data.latitude_spans:
        out.append("<h3>Where each clade puts the boundary</h3>")
        out.append(
            "<figure>"
            + _span_chart(data.latitude_spans)
            + f"<figcaption>Latitude range of every cluster at {c.levels.published} "
            "regions, the cut this run publishes. "
            "Disjoint ranges mean a north/south split; clusters that all span "
            "the extent mean the partition is not geographic.</figcaption>"
            + _table(
                ["Clade", "Cluster", "Min °N", "Max °N", "Hexagons"],
                [
                    [clade, s[0], f"{s[1]:.2f}", f"{s[2]:.2f}", f"{s[3]:,}"]
                    for clade, entries in data.latitude_spans.items()
                    for s in entries
                ],
            )
            + "</figure>"
        )

    # --- Record/taxon inversion -------------------------------------------
    out.append("<h2>Who the records are, and who the species are</h2>")
    if data.clade_shares:
        out.append(
            "<p>Citizen-science effort is not distributed like biodiversity. A "
            "clade can dominate the record count and contribute almost none of "
            "the species, which is what the composition metric has to see "
            "past.</p>"
            "<figure>"
            + _share_chart(data.clade_shares)
            + (
                "<figcaption>Share of records against share of distinct taxa, "
                f"per clade, as fractions of the {c.records:,} records and "
                f"{c.taxa_analysed:,} taxa that were actually clustered — not "
                f"of the {c.taxa:,} taxa in the taxonomy.</figcaption>"
                if c.records
                else "<figcaption>Share of records against share of distinct "
                f"taxa, per clade, as fractions of the {c.taxa_analysed:,} "
                f"taxa that were actually clustered.</figcaption>"
            )
            + _table(
                ["Clade", "Records", "% of records", "Taxa", "% of taxa"],
                [
                    [
                        s.name,
                        f"{s.records:,}",
                        f"{s.record_share * 100:.2f}%",
                        f"{s.taxa:,}",
                        f"{s.taxon_share * 100:.2f}%",
                    ]
                    for s in data.clade_shares
                ],
            )
            + "</figure>"
        )
    else:
        out.append(
            '<p class="missing">Not computed in this run — the source carried no '
            "rank columns to split the taxa by.</p>"
        )

    # --- Validation against the framework ---------------------------------
    out.append("<h2>Validation — does this agree with a framework somebody drew?</h2>")
    if data.reference_by_k:
        by_k = dict(data.reference_by_k)
        best_k, best_a = max(data.reference_by_k, key=lambda x: x[1].adjusted_rand)
        published = by_k.get(c.levels.published)
        selector = by_k.get(c.levels.selector)

        # Cuts too close to the best to be distinguished by this measure.
        # Reporting a bare argmax reads as though one cut won; on the published
        # run the top two are 0.0001 apart, which nothing here resolves.
        near = sorted(
            k
            for k, a in data.reference_by_k
            if k != best_k and best_a.adjusted_rand - a.adjusted_rand < ARI_TIE
        )
        tie_note = (
            ""
            if not near
            else (
                " That is a tie: "
                + _join(f"{k}" for k in near)
                + f" scores within {ARI_TIE} of it, which is below what this "
                "measure distinguishes."
            )
        )
        out.append(
            "<p>The combined partition scored against EPA/CEC Level II "
            "ecoregions, which the pipeline is never shown. Agreement is "
            f"highest at <strong>{best_k} regions</strong> "
            f"(ARI {best_a.adjusted_rand:.3f})"
            + (
                ", which is the cut this run publishes."
                if best_k == c.levels.published
                else (
                    f"; this run publishes {c.levels.published}"
                    + (
                        f" (ARI {published.adjusted_rand:.3f})."
                        if published is not None
                        else "."
                    )
                )
            )
            + (
                f" The selector's {c.levels.selector} scores "
                f"{selector.adjusted_rand:.3f}."
                if selector is not None and not c.levels.selector_agrees
                else ""
            )
            + tie_note
            + "</p>"
            "<figure>"
            + _line_chart(
                [
                    ("ARI", "combined", [
                        (k, a.adjusted_rand) for k, a in data.reference_by_k
                    ]),
                    ("V-measure", "reference", [
                        (k, a.v_measure) for k, a in data.reference_by_k
                    ]),
                ],
                "Number of regions",
                "Agreement with EPA Level II",
                # Says the same as the prose above, ties included. A
                # screen-reader user should not get the confident version.
                aria=(
                    f"Adjusted Rand and V-measure against EPA Level II "
                    f"ecoregions by number of regions. ARI is highest at "
                    f"{best_a.adjusted_rand:.2f} at {best_k} regions"
                    + (
                        f", tied with " + _join(f"{k}" for k in near) + "."
                        if near
                        else "."
                    )
                    + " V-measure rises with the number of regions throughout."
                ),
            )
            + "<figcaption>ARI penalises splitting a reference region; "
            "V-measure rewards splitting it consistently. They move apart as "
            "the cut gets finer, which is what a nested hierarchy should "
            "do.</figcaption>"
            + _table(
                ["Regions", "ARI", "V-measure", "Hexagons compared", "Unmatched"],
                [
                    [
                        k,
                        f"{a.adjusted_rand:.4f}",
                        f"{a.v_measure:.4f}",
                        f"{a.compared:,}",
                        f"{a.unmatched:,}",
                    ]
                    for k, a in data.reference_by_k
                ],
            )
            + "</figure>"
        )
    else:
        out.append(
            '<p class="missing">Not computed in this run — no hexagon centre fell '
            "inside the checked-in reference extent, which covers the US East "
            "Coast only.</p>"
        )

    if data.skipped:
        out.append(
            "<h2>What this run did not compute</h2><ul>"
            + "".join(f"<li>{_esc(x)}</li>" for x in data.skipped)
            + "</ul>"
        )

    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>What this run found</title>"
        f"<style>{_CSS}</style></head><body><main>"
        + "".join(out)
        + "</main></body></html>"
    )


def write_findings_page(data: FindingsData, path: str) -> None:
    """Render and write the page."""
    import logging
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(render_findings_page(data))
    logging.getLogger(__name__).info(f"write_findings_page: wrote {path}")


def findings_summary_json(data: FindingsData) -> str:
    """The same numbers as JSON, for anything that would otherwise scrape them."""
    return json.dumps(
        {
            "context": dataclasses.asdict(data.context),
            "clade_shares": [s.__dict__ for s in data.clade_shares],
            "congruence": [x._asdict() for x in data.congruence],
            "congruence_pair": list(data.congruence_pair),
            "reference_by_k": [
                {"num_clusters": k, **a._asdict()} for k, a in data.reference_by_k
            ],
            "clade_vs_reference": data.clade_vs_reference,
            "latitude_spans": data.latitude_spans,
            "skipped": data.skipped,
        },
        indent=2,
        sort_keys=True,
    )
