"""
Interactive map: Esri World Imagery basemap with school buffers colored by NDVI mean.

Reads the GEE greenness CSV (e.g. outputs/school_ndvi_nlcd_500m.csv) and writes HTML
under visualizations/.

Example:
  python visualizations/make_school_ndvi_buffer_map.py \\
    --greenness-csv outputs/school_ndvi_nlcd_500m.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import folium
import pandas as pd
from branca.colormap import LinearColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.visualization_config import VIS, ndvi_to_color

DEFAULT_GREENNESS_CSV = ROOT / "outputs" / "school_ndvi_nlcd_500m.csv"
DEFAULT_OUT = ROOT / "visualizations" / "school_ndvi_buffer_map.html"

ESRI_WORLD_IMAGERY = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer"
    "/tile/{z}/{y}/{x}"
)


def load_greenness_for_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"lat", "lon", "ndvi_mean", "school"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Greenness CSV missing columns: {sorted(missing)}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(
        description="Folium map: satellite basemap and NDVI-colored school buffers."
    )
    p.add_argument("--greenness-csv", type=Path, default=DEFAULT_GREENNESS_CSV)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--buffer-m",
        type=float,
        default=500.0,
        help="Circle radius in meters (match GEE --buffer-m)",
    )
    p.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Colormap minimum NDVI (default: config or data min)",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Colormap maximum NDVI (default: config or data max)",
    )
    args = p.parse_args()

    df = load_greenness_for_map(args.greenness_csv)
    df = df[pd.to_numeric(df["lat"], errors="coerce").notna()].copy()
    df = df[pd.to_numeric(df["lon"], errors="coerce").notna()].copy()
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    df["ndvi_mean"] = pd.to_numeric(df["ndvi_mean"], errors="coerce")
    df = df[df["ndvi_mean"].notna()].copy()
    if df.empty:
        raise SystemExit("No rows with valid lat, lon, and ndvi_mean.")

    vmin = args.vmin if args.vmin is not None else VIS.ndvi_map_vmin
    vmax = args.vmax if args.vmax is not None else VIS.ndvi_map_vmax
    if vmin is None:
        vmin = float(df["ndvi_mean"].min())
    if vmax is None:
        vmax = float(df["ndvi_mean"].max())
    if vmax < vmin:
        vmin, vmax = vmax, vmin

    center_lat = float(df["lat"].mean())
    center_lon = float(df["lon"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=None)
    folium.TileLayer(
        tiles=ESRI_WORLD_IMAGERY,
        attr="Esri",
        name="Satellite (Esri)",
        max_zoom=19,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", max_zoom=19).add_to(m)

    fg = folium.FeatureGroup(name="School buffers (NDVI)").add_to(m)

    for _, row in df.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        ndvi = float(row["ndvi_mean"])
        school = str(row["school"])
        city = str(row["city"]) if "city" in row and pd.notna(row["city"]) else ""
        sid = str(row["ID"]) if "ID" in row and pd.notna(row["ID"]) else ""
        fill = ndvi_to_color(ndvi, vmin, vmax)
        stroke = ndvi_to_color(ndvi, vmin, vmax)
        lines = [f"<b>{school}</b>"]
        if sid:
            lines.append(f"ID: {sid}")
        if city:
            lines.append(f"{city}")
        lines.append(f"NDVI mean: {ndvi:.4f}")
        popup_html = "<br>".join(lines)
        folium.Circle(
            location=[lat, lon],
            radius=float(args.buffer_m),
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=school[:80],
            color=stroke,
            weight=VIS.ndvi_map_stroke_weight,
            opacity=VIS.ndvi_map_stroke_opacity,
            fill=True,
            fill_color=fill,
            fill_opacity=VIS.ndvi_map_fill_opacity,
        ).add_to(fg)

    sw = [float(df["lat"].min()), float(df["lon"].min())]
    ne = [float(df["lat"].max()), float(df["lon"].max())]
    m.fit_bounds([sw, ne], padding=(24, 24))

    LinearColormap(
        colors=[VIS.ndvi_map_low_color, VIS.ndvi_map_high_color],
        vmin=vmin,
        vmax=vmax,
        caption="NDVI mean (within buffer)",
    ).add_to(m)

    folium.LayerControl(position="topright").add_to(m)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
