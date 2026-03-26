"""
Summarize Sentinel-2 NDVI (median composite) and NLCD tree canopy cover within
buffers around school points, then write a CSV.

Setup (one-time):
  1. Register for Earth Engine and link a Google Cloud project:
     https://earthengine.google.com/signup/
  2. Install deps: pip install -r requirements.txt
  3. Authenticate: earthengine authenticate
  4. Project id for ee.Initialize (first match wins):
       --project on the command line, else environment variable GEE_PROJECT,
       else default green-around-cali-schools.

Example:
  python analysis/gee_school_greenness.py \\
    --input data/cleaned/public_schools_frpm_santaclara_analysis.csv \\
    --output outputs/school_ndvi_nlcd_500m.csv

Input CSV may use: (1) legacy columns ID, school, lat, lon, city; (2) merged
columns cds_code, school_name, latitude, longitude, city; or (3) analysis slim
columns ID (cds_code values), school_name, latitude, longitude, city.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import ee
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GEE_PROJECT = "green-around-cali-schools"

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
NLCD_TCC_COLLECTION = "USGS/NLCD_RELEASES/2023_REL/TCC/v2023-5"
NLCD_CANOPY_BAND = "NLCD_Percent_Tree_Canopy_Cover"
NLCD_YEAR = 2023


def _parse_float(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _normalize_school_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map merged or analysis-slim columns to ID, school, lat, lon, city."""
    if "cds_code" in df.columns:
        mapping = {
            "cds_code": "ID",
            "school_name": "school",
            "latitude": "lat",
            "longitude": "lon",
        }
        use = {k: v for k, v in mapping.items() if k in df.columns}
        return df.rename(columns=use)
    if "ID" in df.columns and "school_name" in df.columns:
        mapping = {
            "school_name": "school",
            "latitude": "lat",
            "longitude": "lon",
        }
        use = {k: v for k, v in mapping.items() if k in df.columns}
        return df.rename(columns=use)
    return df


def load_schools_csv(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (valid rows for GEE, dropped invalid rows)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = _normalize_school_columns(df)
    required = {"ID", "school", "lat", "lon", "city"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    lat = df["lat"].map(_parse_float)
    lon = df["lon"].map(_parse_float)
    ok = lat.notna() & lon.notna() & lat.between(-90, 90) & lon.between(-180, 180)
    dropped = df.loc[~ok].copy()
    valid = df.loc[ok].copy()
    valid["_lat"] = lat[ok].astype(float)
    valid["_lon"] = lon[ok].astype(float)
    return valid, dropped


def schools_to_feature_collection(df: pd.DataFrame) -> ee.FeatureCollection:
    feats = []
    for _, row in df.iterrows():
        lon = float(row["_lon"])
        lat = float(row["_lat"])
        geom = ee.Geometry.Point([lon, lat])
        feats.append(
            ee.Feature(
                geom,
                {
                    "ID": str(row["ID"]),
                    "school": str(row["school"]),
                    "lat": lat,
                    "lon": lon,
                    "city": str(row["city"]),
                },
            )
        )
    return ee.FeatureCollection(feats)


def mask_s2_sr_clouds(image: ee.Image) -> ee.Image:
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    scl = image.select("SCL")
    scl_ok = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    return image.updateMask(mask).updateMask(scl_ok)


def add_ndvi(image: ee.Image) -> ee.Image:
    nir = image.select("B8").multiply(0.0001)
    red = image.select("B4").multiply(0.0001)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("ndvi")
    return image.addBands(ndvi)


def build_ndvi_median(
    region: ee.Geometry,
    start_date: str,
    end_date: str,
) -> ee.Image:
    col = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .map(mask_s2_sr_clouds)
        .map(add_ndvi)
    )
    return col.select("ndvi").median().rename("ndvi")


def nlcd_canopy_image() -> ee.Image:
    ic = ee.ImageCollection(NLCD_TCC_COLLECTION)
    tcc = (
        ic.filter(ee.Filter.eq("year", NLCD_YEAR))
        .filter(ee.Filter.eq("study_area", "CONUS"))
        .first()
    )
    canopy = tcc.select(NLCD_CANOPY_BAND).rename("nlcd_canopy")
    return canopy


def reduce_ndvi(
    buffered: ee.FeatureCollection,
    ndvi: ee.Image,
    scale_m: float,
) -> ee.FeatureCollection:
    return ndvi.reduceRegions(
        collection=buffered,
        reducer=ee.Reducer.mean(),
        scale=scale_m,
        tileScale=4,
    )


def reduce_nlcd(
    buffered: ee.FeatureCollection,
    canopy: ee.Image,
    threshold_pct: float,
    scale_m: float,
) -> ee.FeatureCollection:
    high = canopy.gte(threshold_pct).rename("nlcd_high_canopy_mask").float()
    stacked = canopy.addBands(high)
    return stacked.reduceRegions(
        collection=buffered,
        reducer=ee.Reducer.mean(),
        scale=scale_m,
        tileScale=4,
    )


def fc_to_dataframe(fc: ee.FeatureCollection) -> pd.DataFrame:
    features = fc.getInfo()["features"]
    rows = []
    for f in features:
        props = dict(f["properties"])
        rows.append(props)
    return pd.DataFrame(rows)


def _normalize_ndvi_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ndvi_mean" in out.columns:
        return out
    if "ndvi" in out.columns:
        return out.rename(columns={"ndvi": "ndvi_mean"})
    meta = {"ID", "school", "lat", "lon", "city", "system:index"}
    stat_cols = [c for c in out.columns if c not in meta]
    if len(stat_cols) == 1:
        return out.rename(columns={stat_cols[0]: "ndvi_mean"})
    raise ValueError(f"Could not infer NDVI column from columns: {list(out.columns)}")


def _normalize_nlcd_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mean_key = "nlcd_canopy_mean"
    frac_key = "nlcd_high_canopy_frac"
    if mean_key not in out.columns:
        for alt in ("nlcd_canopy", "nlcd_canopy_mean_mean"):
            if alt in out.columns:
                out = out.rename(columns={alt: mean_key})
                break
    if frac_key not in out.columns:
        for alt in (
            "nlcd_high_canopy_mask",
            "nlcd_high_canopy_mask_mean",
            "nlcd_high_canopy_frac_mean",
        ):
            if alt in out.columns:
                out = out.rename(columns={alt: frac_key})
                break
    missing = [k for k in (mean_key, frac_key) if k not in out.columns]
    if missing:
        raise ValueError(
            f"Could not infer NLCD columns {missing} from columns: {list(out.columns)}"
        )
    return out


def run(
    input_path: Path,
    output_path: Path,
    buffer_m: float,
    start_date: str,
    end_date: str,
    canopy_threshold: float,
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
    canopy_img = nlcd_canopy_image()

    ndvi_fc = reduce_ndvi(buffered, ndvi_img, scale_m=10)
    nlcd_fc = reduce_nlcd(buffered, canopy_img, canopy_threshold, scale_m=30)

    ndvi_df = _normalize_ndvi_columns(fc_to_dataframe(ndvi_fc))
    nlcd_df = _normalize_nlcd_columns(fc_to_dataframe(nlcd_fc))

    merge_cols = ["ID"]
    for c in ("nlcd_canopy_mean", "nlcd_high_canopy_frac"):
        if c in nlcd_df.columns:
            merge_cols.append(c)
    out = ndvi_df.merge(nlcd_df[merge_cols], on="ID", how="left")

    for col in ("ndvi_mean", "nlcd_canopy_mean", "nlcd_high_canopy_frac"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    preferred = [
        "ID",
        "school",
        "lat",
        "lon",
        "city",
        "ndvi_mean",
        "nlcd_canopy_mean",
        "nlcd_high_canopy_frac",
    ]
    rest = [c for c in out.columns if c not in preferred]
    out = out[[c for c in preferred if c in out.columns] + rest]

    if "ndvi_mean" in out.columns:
        bad_ndvi = out["ndvi_mean"].notna() & (
            (out["ndvi_mean"] < -1.05) | (out["ndvi_mean"] > 1.05)
        )
        if bad_ndvi.any():
            n = int(bad_ndvi.sum())
            print(
                f"Warning: {n} row(s) have ndvi_mean outside [-1, 1] (check masks).",
                file=sys.stderr,
            )
    if "nlcd_canopy_mean" in out.columns:
        bad_tc = out["nlcd_canopy_mean"].notna() & (
            (out["nlcd_canopy_mean"] < 0) | (out["nlcd_canopy_mean"] > 100)
        )
        if bad_tc.any():
            n = int(bad_tc.sum())
            print(
                f"Warning: {n} row(s) have nlcd_canopy_mean outside [0, 100].",
                file=sys.stderr,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(out)} rows)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="NDVI + NLCD tree canopy within buffers around school points (Google Earth Engine)."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "cleaned" / "public_schools_frpm_santaclara_analysis.csv",
        help="Schools CSV: ID, school, lat, lon, city (merged or analysis slim schema; see module docstring)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "school_ndvi_nlcd_500m.csv",
        help="Output CSV path",
    )
    p.add_argument("--buffer-m", type=float, default=500.0, help="Buffer radius in meters")
    p.add_argument(
        "--start-date",
        type=str,
        default="2023-06-01",
        help="Inclusive start date for Sentinel-2 composite (YYYY-MM-DD)",
    )
    p.add_argument(
        "--end-date",
        type=str,
        default="2025-09-30",
        help="Exclusive end date for Sentinel-2 composite (YYYY-MM-DD)",
    )
    p.add_argument(
        "--canopy-threshold",
        type=float,
        default=10.0,
        help="NLCD canopy %% threshold for nlcd_high_canopy_frac (mean of mask)",
    )
    p.add_argument(
        "--project",
        type=str,
        default=None,
        help=(
            "Google Cloud project id for ee.Initialize; if omitted, uses GEE_PROJECT "
            f"env or default {DEFAULT_GEE_PROJECT}"
        ),
    )
    args = p.parse_args()
    run(
        args.input,
        args.output,
        args.buffer_m,
        args.start_date,
        args.end_date,
        args.canopy_threshold,
        args.project,
    )


if __name__ == "__main__":
    main()
