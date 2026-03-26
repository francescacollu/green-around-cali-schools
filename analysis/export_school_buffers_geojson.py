"""
Export circular school buffers with NDVI mean properties as GeoJSON via Google Earth Engine.

Uses the same Sentinel-2 median NDVI and reduction as analysis/gee_school_greenness.py.
Requires Earth Engine authentication (earthengine authenticate).

Example:
  python analysis/export_school_buffers_geojson.py \\
    --input data/cleaned/public_schools_frpm_santaclara_analysis.csv \\
    --output outputs/school_buffers_ndvi.geojson
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import ee

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.gee_school_greenness import (
    DEFAULT_GEE_PROJECT,
    build_ndvi_median,
    load_schools_csv,
    reduce_ndvi,
    schools_to_feature_collection,
)


def run(
    input_path: Path,
    output_path: Path,
    buffer_m: float,
    start_date: str,
    end_date: str,
    gee_project: str | None,
) -> None:
    schools, dropped = load_schools_csv(input_path)
    if len(dropped):
        print(f"Dropped {len(dropped)} row(s) with invalid coordinates.", file=sys.stderr)
    if schools.empty:
        raise SystemExit("No valid school rows after coordinate checks.")

    if gee_project and str(gee_project).strip():
        project = str(gee_project).strip()
    else:
        project = os.environ.get("GEE_PROJECT") or DEFAULT_GEE_PROJECT
    ee.Initialize(project=project)

    fc = schools_to_feature_collection(schools)
    buffered = fc.map(lambda f: f.buffer(buffer_m))

    region = buffered.geometry()
    ndvi_img = build_ndvi_median(region, start_date, end_date)
    ndvi_fc = reduce_ndvi(buffered, ndvi_img, scale_m=10)

    info = ndvi_fc.getInfo()
    features = info.get("features") or []
    geojson = {"type": "FeatureCollection", "features": features}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)
    print(f"Wrote {output_path} ({len(features)} features)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export GEE school buffers + NDVI mean as GeoJSON."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "cleaned" / "public_schools_frpm_santaclara_analysis.csv",
        help="Schools CSV (same schema as gee_school_greenness.py)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "school_buffers_ndvi.geojson",
        help="Output .geojson path",
    )
    p.add_argument("--buffer-m", type=float, default=500.0)
    p.add_argument("--start-date", type=str, default="2023-06-01")
    p.add_argument("--end-date", type=str, default="2025-09-30")
    p.add_argument(
        "--project",
        type=str,
        default=None,
        help="GCP project id for ee.Initialize; else GEE_PROJECT env or default",
    )
    args = p.parse_args()
    run(
        args.input,
        args.output,
        args.buffer_m,
        args.start_date,
        args.end_date,
        args.project,
    )


if __name__ == "__main__":
    main()
