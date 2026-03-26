"""
Export analysis-ready slim CSV: ID from cds_code, core location fields, one FRPM column.

Reads:  data/cleaned/public_schools_frpm_sv_merged.csv (default)
Writes: data/cleaned/public_schools_frpm_sv_analysis.csv (default)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = ROOT / "data" / "cleaned" / "public_schools_frpm_sv_merged.csv"
DEFAULT_OUTPUT = ROOT / "data" / "cleaned" / "public_schools_frpm_sv_analysis.csv"

NO_DATA = "No Data"

OUTPUT_COLUMNS = [
    "ID",
    "school_name",
    "city",
    "address",
    "latitude",
    "longitude",
    "percent_eligible_frpm_k12",
]


def _is_missing_cds(val: object) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return True
    s = str(val).strip()
    return not s or s == NO_DATA


def _to_fraction_maybe(x: object) -> float | None:
    """Convert legacy percent strings (e.g. '51.9%') to a fraction (0-1)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s or s == NO_DATA:
        return None
    s = s.replace("%", "").strip()
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        return None

    # If the data is already in 0-1 form, keep it; otherwise assume it's 0-100.
    return f if f <= 1 else (f / 100.0)


def run(input_path: Path, output_path: Path) -> int:
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    if "cds_code" not in df.columns:
        print("Input CSV must contain column cds_code.", file=sys.stderr)
        return 1
    missing = [c for c in ("school_name", "city", "street", "latitude", "longitude", "percent_eligible_frpm_k12") if c not in df.columns]
    if missing:
        print(f"Input CSV missing columns: {missing}", file=sys.stderr)
        return 1

    bad = df["cds_code"].map(_is_missing_cds)
    n_bad = int(bad.sum())
    if n_bad:
        print(f"Dropping {n_bad} row(s) with missing cds_code.", file=sys.stderr)
        df = df.loc[~bad].copy()

    out = pd.DataFrame(
        {
            "ID": df["cds_code"].astype(str).str.strip(),
            "school_name": df["school_name"],
            "city": df["city"],
            "address": df["street"],
            "latitude": df["latitude"],
            "longitude": df["longitude"],
            "percent_eligible_frpm_k12": df["percent_eligible_frpm_k12"].map(_to_fraction_maybe),
        }
    )
    out = out[OUTPUT_COLUMNS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(out)} rows)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Merged public schools + FRPM CSV")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Slim analysis CSV path")
    args = p.parse_args()
    return run(args.input, args.output)


if __name__ == "__main__":
    sys.exit(main())
