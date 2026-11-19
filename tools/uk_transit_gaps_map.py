#!/usr/bin/env python3
"""
uk_transit_gaps_map.py
======================
Draws a poster-style map of British council areas that have no station,
in the manner of the "Counties without public transit" map of the US.

Every council area in Great Britain has at least one bus stop, so the UK
version of the question is about stations. Each area is put in one of
three groups:

    dark    no station of the chosen kind
    yellow  no station, though a railway line passes through
    grey    at least one station

Open data used (all free, no key needed)
----------------------------------------
Stops       NaPTAN, Department for Transport      (Great Britain only)
Boundaries  ONS Local Authority Districts, generalised
Rail lines  OpenStreetMap, through the Overpass API

Install
-------
    pip install geopandas matplotlib requests

Run
---
    python uk_transit_gaps_map.py                     # downloads everything
    python uk_transit_gaps_map.py --modes rail,tram   # count tram and metro stops too
    python uk_transit_gaps_map.py --modes all         # any stop, bus included
    python uk_transit_gaps_map.py --boundaries lads.geojson --stops Stops.csv \
        --rail rail.geojson                           # use files you already have

Downloads are cached in ./data so later runs are quick. If one of the web
addresses below has moved, download the file by hand and pass it in.
"""
import argparse
import io
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import LineString

# ---------------------------------------------------------------- sources
NAPTAN_CSV = "https://naptan.api.dft.gov.uk/v1/access-nodes?dataFormat=csv"
LAD_GEOJSON = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Local_Authority_Districts_December_2023_Boundaries_UK_BUC/FeatureServer/0/"
    "query?where=1%3D1&outFields=*&outSR=4326&f=geojson"
)
OVERPASS = "https://overpass-api.de/api/interpreter"
OVERPASS_QUERY = """
[out:json][timeout:600];
area["ISO3166-1"="GB"][admin_level=2]->.gb;
way["railway"="rail"]["usage"~"^(main|branch)$"]["service"!~"."](area.gb);
out geom;
"""

# NaPTAN StopType codes, grouped by mode
STOP_TYPES = {
    "rail": {"RLY", "RSE", "RPL"},
    "tram": {"MET", "TMU", "PLT"},
    "bus": {"BCT", "BCS", "BCE", "BST", "BCQ"},
    "ferry": {"FER", "FBT", "FTD"},
}

# ------------------------------------------------------------------ style
STYLE = {
    "background": "#eeede8",
    "has": "#d9d7d1",
    "none": "#2b2d31",
    "passes": "#f2c12e",
    "rail": "#d84a35",
    "ink": "#1d1d1f",
    "muted": "#77756f",
    "title_font": ["Roboto Slab", "Rockwell", "Zilla Slab", "DejaVu Serif"],
    "body_font": ["Barlow", "Public Sans", "Arial", "DejaVu Sans"],
}
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # quiet missing-font notes
BNG = 27700  # British National Grid, so areas and lengths are in metres


# ------------------------------------------------------------------ loading
def fetch(url, cache, **kwargs):
    """Download once and keep a copy in the cache folder."""
    import requests

    if cache.exists():
        return cache.read_bytes()
    print(f"Downloading {url[:70]}...")
    send = requests.post if "data" in kwargs else requests.get
    r = send(url, timeout=900, **kwargs)
    r.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(r.content)
    return r.content


def load_boundaries(src, cache_dir, name_col):
    if src:
        gdf = gpd.read_file(src)
    else:
        gdf = gpd.read_file(io.BytesIO(fetch(LAD_GEOJSON, cache_dir / "lads.geojson")))
    if name_col is None:  # pick the first column that looks like a name
        name_col = next((c for c in gdf.columns if c.upper().endswith("NM")), gdf.columns[0])
    code_col = next((c for c in gdf.columns if c.upper().endswith("CD")), None)
    if code_col is not None:  # NaPTAN does not cover Northern Ireland
        gdf = gdf[~gdf[code_col].astype(str).str.startswith("N")]
    gdf = gdf.rename(columns={name_col: "name"})[["name", "geometry"]]
    return gdf.to_crs(BNG).reset_index(drop=True)


def load_stops(src, cache_dir, modes):
    raw = Path(src).read_bytes() if src else fetch(NAPTAN_CSV, cache_dir / "naptan.csv")
    df = pd.read_csv(io.BytesIO(raw), low_memory=False, encoding="utf-8", encoding_errors="replace")
    wanted = set().union(*(STOP_TYPES[m] for m in modes))
    df = df[df["StopType"].isin(wanted)]
    if "Status" in df.columns:
        df = df[df["Status"].astype(str).str.lower().isin({"active", "act"})]
    df = df.dropna(subset=["Longitude", "Latitude"])
    pts = gpd.points_from_xy(df["Longitude"], df["Latitude"], crs=4326)
    return gpd.GeoDataFrame(df[["CommonName", "StopType"]], geometry=pts).to_crs(BNG)


def load_rail(src, cache_dir):
    if src:
        return gpd.read_file(src).to_crs(BNG)[["geometry"]]
    raw = fetch(OVERPASS, cache_dir / "rail_osm.json", data={"data": OVERPASS_QUERY})
    ways = json.loads(raw)["elements"]
    lines = [
        LineString([(p["lon"], p["lat"]) for p in w["geometry"]])
        for w in ways
        if w.get("type") == "way" and len(w.get("geometry", [])) > 1
    ]
    return gpd.GeoDataFrame(geometry=lines, crs=4326).to_crs(BNG)


# ---------------------------------------------------------------- analysis
def classify(areas, stops, rail, min_track_m=1000):
    """Add 'stations', 'track_km' and 'group' to each area."""
    joined = gpd.sjoin(stops, areas, predicate="within")
    areas["stations"] = joined.groupby("index_right").size().reindex(areas.index, fill_value=0)

    track = []
    sindex = rail.sindex
    for geom in areas.geometry:
        nearby = rail.iloc[sindex.query(geom, predicate="intersects")]
        track.append(nearby.intersection(geom).length.sum() if len(nearby) else 0.0)
    areas["track_km"] = [t / 1000 for t in track]

    areas["group"] = "has"
    empty = areas["stations"] == 0
    areas.loc[empty, "group"] = "none"
    areas.loc[empty & (areas["track_km"] * 1000 >= min_track_m), "group"] = "passes"
    return areas


# ---------------------------------------------------------------- drawing
def draw(areas, rail, modes, out, dpi):
    s = STYLE
    counts = areas["group"].value_counts()
    n_none, n_pass = int(counts.get("none", 0)), int(counts.get("passes", 0))
    thing = {"rail": "RAILWAY STATION", "all": "PUBLIC TRANSPORT STOP"}.get(
        "all" if "bus" in modes else "rail", "STATION")
    label = "railway station" if modes == ["rail"] else "station or stop"

    fig = plt.figure(figsize=(10.8, 13.5), facecolor=s["background"])
    ax = fig.add_axes([0.04, 0.2, 0.92, 0.6])
    ax.set_axis_off()
    ax.set_aspect("equal")

    for group in ("has", "none", "passes"):
        part = areas[areas["group"] == group]
        if len(part):
            part.plot(ax=ax, color=s[group], edgecolor=s["background"], linewidth=0.35)
    rail.plot(ax=ax, color=s["rail"], linewidth=0.7, alpha=0.95)

    fig.text(0.06, 0.955, f"COUNCIL AREAS WITHOUT\nA {thing}", fontsize=41, fontweight="bold",
             family=s["title_font"], color=s["ink"], va="top", linespacing=0.98)
    fig.text(0.06, 0.845, f"{n_none + n_pass} of {len(areas)} council areas in Great Britain "
             f"have no {label}.", fontsize=15, family=s["body_font"], color=s["ink"], va="top")
    fig.text(0.06, 0.822, f"In {n_pass} of them, a railway line runs straight through without stopping.",
             fontsize=15, family=s["body_font"], color=s["muted"], va="top")

    handles = [
        Patch(facecolor=s["none"], label=f"No {label}"),
        Patch(facecolor=s["passes"], label="None, though a railway line passes through"),
        Patch(facecolor=s["has"], label="At least one"),
        Line2D([0], [0], color=s["rail"], linewidth=2, label="Railway line"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.05, 0.085), frameon=False,
               prop={"family": s["body_font"], "size": 13}, labelspacing=1.1, handlelength=2.2)

    fig.text(0.06, 0.055, "Map by Jasmine Brittan", fontsize=12, fontweight="bold",
             family=s["body_font"], color=s["ink"])
    fig.text(0.06, 0.036, "Stops from NaPTAN, Department for Transport. Boundaries from the Office for "
             "National Statistics.\nRailway lines © OpenStreetMap contributors. Great Britain only.",
             fontsize=9.5, family=s["body_font"], color=s["muted"], va="top", linespacing=1.5)

    fig.savefig(out, dpi=dpi, facecolor=s["background"])
    print(f"Saved {out}")


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boundaries", help="boundary file (GeoJSON, shapefile, GeoPackage); downloaded if omitted")
    ap.add_argument("--name-col", help="column holding each area's name; guessed if omitted")
    ap.add_argument("--stops", help="NaPTAN Stops.csv; downloaded if omitted")
    ap.add_argument("--rail", help="railway lines file; fetched from OpenStreetMap if omitted")
    ap.add_argument("--modes", default="rail", help="rail, tram, bus, ferry, comma separated, or 'all'")
    ap.add_argument("--min-track-m", type=float, default=1000,
                    help="metres of track inside an area before a line counts as passing through")
    ap.add_argument("--out", default="uk_transit_gaps.png")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--cache-dir", default="data")
    a = ap.parse_args()

    modes = list(STOP_TYPES) if a.modes == "all" else [m.strip() for m in a.modes.split(",")]
    unknown = [m for m in modes if m not in STOP_TYPES]
    if unknown:
        sys.exit(f"Unknown mode: {', '.join(unknown)}. Choose from {', '.join(STOP_TYPES)} or all.")
    cache = Path(a.cache_dir)

    areas = load_boundaries(a.boundaries, cache, a.name_col)
    stops = load_stops(a.stops, cache, modes)
    rail = load_rail(a.rail, cache)
    print(f"{len(areas)} areas, {len(stops)} stops, {len(rail)} railway segments")

    areas = classify(areas, stops, rail, a.min_track_m)
    table = Path(a.out).with_suffix(".csv")
    areas.drop(columns="geometry").sort_values(["group", "name"]).to_csv(table, index=False)
    print(areas["group"].value_counts().to_string())
    print(f"Saved {table}")

    draw(areas, rail, modes, a.out, a.dpi)


if __name__ == "__main__":
    main()
