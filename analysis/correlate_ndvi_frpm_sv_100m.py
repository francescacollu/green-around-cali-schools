from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_INPUT_CSV = ROOT / "outputs" / "school_ndvi_nlcd_frpm_sv_100m.csv"
DEFAULT_OUT_CSV = ROOT / "outputs" / "ndvi_frpm_correlation_sv_100m.csv"

DEFAULT_METRIC_COL = "ndvi_mean"
FRPM_COL = "percent_eligible_frpm_k12"


def load_and_clean(path: Path, *, metric_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in (metric_col, FRPM_COL):
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")
    df = df.copy()
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df[FRPM_COL] = pd.to_numeric(df[FRPM_COL], errors="coerce")
    df = df[df[metric_col].notna() & df[FRPM_COL].notna()].copy()
    df = df[df[FRPM_COL].between(0.0, 1.0)].copy()
    return df


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pearson and Spearman correlation between a metric column and FRPM (SV, 100m buffer)."
    )
    p.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    p.add_argument(
        "--metric-col",
        type=str,
        default=DEFAULT_METRIC_COL,
        help="Numeric metric column to correlate against FRPM.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help="Write one-row summary (n, r, p-values, metadata).",
    )
    p.add_argument(
        "--no-out-csv",
        action="store_true",
        help="Do not write the summary CSV.",
    )
    p.add_argument(
        "--buffer-m",
        type=float,
        default=100.0,
        help="Buffer radius in meters (metadata only; must match how NDVI was computed).",
    )
    args = p.parse_args()

    df = load_and_clean(args.input_csv, metric_col=args.metric_col)
    n = len(df)
    if n < 3:
        raise SystemExit(f"Need at least 3 schools for correlation; got n={n}.")

    x = df[args.metric_col].to_numpy()
    y = df[FRPM_COL].to_numpy()

    r_pearson, p_pearson = pearsonr(x, y)
    r_spearman, p_spearman = spearmanr(x, y)

    print(f"Dataset: {args.input_csv.name}")
    print(f"Metric column: {args.metric_col}")
    print(f"n (schools after cleaning): {n}")
    print(f"Pearson r:  {r_pearson:.6f}  (two-sided p = {p_pearson:.6g})")
    print(f"Spearman r: {r_spearman:.6f}  (two-sided p = {p_spearman:.6g})")

    if not args.no_out_csv:
        row = {
            "dataset": args.input_csv.name,
            "metric_col": args.metric_col,
            "buffer_m": args.buffer_m,
            "n": n,
            "pearson_r": r_pearson,
            "pearson_p_two_sided": p_pearson,
            "spearman_r": r_spearman,
            "spearman_p_two_sided": p_spearman,
        }
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
