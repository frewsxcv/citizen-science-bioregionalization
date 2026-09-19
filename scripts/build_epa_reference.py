"""Turn the EPA/CEC Level II ecoregion shapefile into a checked-in GeoJSON.

Run offline, when the reference framework or the published bounding box
changes. The output is committed; runs do not download anything.

    curl -o /tmp/l2.zip \\
      https://dmap-prod-oms-edc.s3.us-east-1.amazonaws.com/ORD/Ecoregions/cec_na/na_cec_eco_l2.zip
    unzip -d /tmp/epa /tmp/l2.zip
    uv run --with pyproj scripts/build_epa_reference.py \\
        /tmp/epa/NA_CEC_Eco_Level2.shp src/data/epa_l2_east_coast.geojson

`pyproj` is passed with `--with` rather than added to the project: the source is
in a spherical Lambert Azimuthal Equal Area projection and has to be unprojected
to WGS84, but that happens here, once, and nothing at runtime needs it. Runtime
reads the result with `json` and `shapely`, which is what `src/geocode.py`
already does for the Natural Earth coastline.

The polygons are clipped to the published bounding box and simplified. Level II
is the right grain: Level I puts the whole eastern United States in one region,
so it cannot agree or disagree with a north/south split.
"""

import argparse
import json
import sys
from pathlib import Path

import pyogrio
import shapely
import shapely.geometry
import shapely.ops

# The published run's bounding box, padded by a degree so that a hexagon near
# the edge still lands inside a polygon rather than just outside one.
DEFAULT_BBOX = (-88.0, 24.0, -65.0, 48.0)

# Chosen against an H3 resolution-4 hexagon, which spans roughly 1,770 km2 --
# about 0.4 degrees. Simplifying to a hundredth of a degree is far below the
# grain anything is compared at, and takes the file from tens of megabytes to
# something reasonable to check in.
DEFAULT_TOLERANCE = 0.01


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shapefile", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()

    # Imported here, and unresolved for the type checker on purpose: pyproj is
    # not a project dependency. It is supplied by `uv run --with pyproj` for
    # this script alone, because nothing at runtime reprojects anything.
    from pyproj import CRS, Transformer  # pyright: ignore[reportMissingImports]

    info = pyogrio.read_info(str(args.shapefile))
    source_crs = CRS.from_user_input(info["crs"])
    to_wgs84 = Transformer.from_crs(source_crs, CRS.from_epsg(4326), always_xy=True)

    _, table = pyogrio.raw.read_arrow(str(args.shapefile))  # pyright: ignore[reportAttributeAccessIssue]
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    geometries = columns.pop("wkb_geometry", None) or columns.pop("geometry")

    clip = shapely.geometry.box(*DEFAULT_BBOX)
    features = []
    for i, wkb in enumerate(geometries):
        if wkb is None:
            continue
        geom = shapely.ops.transform(to_wgs84.transform, shapely.from_wkb(wkb))
        geom = geom.intersection(clip)
        if geom.is_empty:
            continue
        geom = geom.simplify(args.tolerance, preserve_topology=True)
        if geom.is_empty:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "code": columns["NA_L2CODE"][i],
                    "name": columns["NA_L2NAME"][i],
                },
                "geometry": shapely.geometry.mapping(geom),
            }
        )

    # Sorted so the file is stable between prep runs: pyogrio's feature order is
    # the shapefile's, but the merge below groups by code and a dict's insertion
    # order would otherwise leak into the output.
    features.sort(key=lambda f: (f["properties"]["code"], json.dumps(f["geometry"])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)

    codes = sorted({f["properties"]["code"] for f in features})
    print(
        f"{len(features)} polygons across {len(codes)} Level II regions "
        f"-> {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)"
    )
    print(f"codes: {', '.join(codes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
