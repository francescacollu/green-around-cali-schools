from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_GRENESS_CSV = ROOT / "outputs" / "school_ndvi_nlcd_500m.csv"
DEFAULT_SCHOOLS_CSV = (
    ROOT / "data" / "cleaned" / "public_schools_frpm_santaclara_analysis.csv"
)
DEFAULT_OUTPUT_CSV = ROOT / "outputs" / "school_ndvi_nlcd_frpm_500m.csv"


GRENESS_REQUIRED_COLS = {
    "ID",
    "school",
    "lat",
    "lon",
    "city",
    "ndvi_mean",
    "nlcd_canopy_mean",
    "nlcd_high_canopy_frac",
}

SCHOOLS_REQUIRED_COLS = {
    "ID",
    "school_name",
    "address",
    "latitude",
    "longitude",
    "percent_eligible_frpm_k12",
}


def _ensure_columns(df: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} CSV missing columns: {sorted(missing)}")


def load_greenness(greenness_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(greenness_csv)
    _ensure_columns(df, GRENESS_REQUIRED_COLS, name="Greenness")
    # Ensure expected numeric types where possible.
    for c in ("lat", "lon", "ndvi_mean", "nlcd_canopy_mean", "nlcd_high_canopy_frac"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_schools(schools_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(schools_csv)
    _ensure_columns(df, SCHOOLS_REQUIRED_COLS, name="Schools")
    for c in ("latitude", "longitude", "percent_eligible_frpm_k12"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def merge_greenness_and_frpm(
    greenness: pd.DataFrame,
    schools: pd.DataFrame,
    *,
    coord_tolerance_deg: float,
) -> pd.DataFrame:
    # Include latitude/longitude temporarily for a coordinate-consistency check,
    # but we will drop them before returning the final output.
    schools_subset = schools[
        ["ID", "school_name", "address", "percent_eligible_frpm_k12", "latitude", "longitude"]
    ].copy()

    merged = greenness.merge(schools_subset, on="ID", how="left")

    # Validate that the coordinate pair matches across the two upstream pipelines.
    lat_diff = (merged["lat"] - merged["latitude"]).abs()
    lon_diff = (merged["lon"] - merged["longitude"]).abs()
    both_present = merged["lat"].notna() & merged["lon"].notna() & merged["latitude"].notna() & merged[
        "longitude"
    ].notna()

    if both_present.any():
        bad = both_present & ((lat_diff > coord_tolerance_deg) | (lon_diff > coord_tolerance_deg))
        if bad.any():
            n_bad = int(bad.sum())
            max_lat = float(lat_diff[bad].max()) if n_bad else 0.0
            max_lon = float(lon_diff[bad].max()) if n_bad else 0.0
            print(
                f"Warning: {n_bad} row(s) have lat/lon disagreement > {coord_tolerance_deg} deg "
                f"(max_lat_diff={max_lat}, max_lon_diff={max_lon}). Proceeding with greenness lat/lon.",
                file=sys.stderr,
            )

    missing_frpm = int(merged["percent_eligible_frpm_k12"].isna().sum())
    if missing_frpm:
        raise ValueError(
            f"After merge, {missing_frpm} row(s) are missing percent_eligible_frpm_k12 for known `ID`s."
        )

    # Drop coordinate duplicates. Output keeps only `lat`/`lon` from the greenness CSV.
    merged = merged.drop(columns=["latitude", "longitude"])

    out_cols = [
        "ID",
        "school",
        "school_name",
        "city",
        "address",
        "lat",
        "lon",
        "ndvi_mean",
        "nlcd_canopy_mean",
        "nlcd_high_canopy_frac",
        "percent_eligible_frpm_k12",
    ]

    return merged[out_cols]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge NDVI/NLCD greenness (500m) with FRPM analysis fields by school `ID`."
    )
    p.add_argument("--greenness", type=Path, default=DEFAULT_GRENESS_CSV)
    p.add_argument("--schools", type=Path, default=DEFAULT_SCHOOLS_CSV)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument(
        "--coord-tolerance-deg",
        type=float,
        default=1e-5,
        help="Max allowed lat/lon disagreement (degrees) between the two input CSVs; warns if exceeded.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    greenness = load_greenness(args.greenness)
    schools = load_schools(args.schools)

    merged = merge_greenness_and_frpm(
        greenness,
        schools,
        coord_tolerance_deg=args.coord_tolerance_deg,
    )

    if len(merged) != len(greenness):
        raise ValueError(f"Row count mismatch after merge: got {len(merged)} expected {len(greenness)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    print(f"Wrote merged CSV: {args.output}")


if __name__ == "__main__":
    main()

