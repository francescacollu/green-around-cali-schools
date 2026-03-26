from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.visualization_config import VIS, apply_common_layout

DEFAULT_GREENNESS_CSV = ROOT / "outputs" / "school_ndvi_nlcd_sv_100m.csv"
DEFAULT_FRPM_MERGED_CSV = ROOT / "data" / "cleaned" / "public_schools_frpm_sv_merged.csv"
DEFAULT_OUTPATH = ROOT / "visualizations" / "frpm_vs_noneligible_conditional_ndvi_bin_probability_sv_100m.html"


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def load_greenness(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ID": "string"})
    _require_columns(df, {"ID", "ndvi_mean"}, "Greenness CSV")
    df = df[["ID", "ndvi_mean"]].copy()
    df["ndvi_mean"] = pd.to_numeric(df["ndvi_mean"], errors="coerce")
    df = df.dropna(subset=["ID", "ndvi_mean"]).copy()
    return df


def load_frpm_merged(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"cds_code": "string"})
    _require_columns(df, {"cds_code", "enrollment_k12", "frpm_count_k12"}, "FRPM merged CSV")

    df = df[["cds_code", "enrollment_k12", "frpm_count_k12"]].copy()
    df["enrollment_k12"] = pd.to_numeric(df["enrollment_k12"], errors="coerce")
    df["frpm_count_k12"] = pd.to_numeric(df["frpm_count_k12"], errors="coerce")
    df = df.dropna(subset=["cds_code", "enrollment_k12", "frpm_count_k12"]).copy()

    df["eligible_count"] = df["frpm_count_k12"].astype(float)
    df["non_eligible_count"] = (df["enrollment_k12"] - df["frpm_count_k12"]).astype(float)
    df.loc[df["eligible_count"] < 0, "eligible_count"] = 0.0
    df.loc[df["non_eligible_count"] < 0, "non_eligible_count"] = 0.0

    return df[["cds_code", "eligible_count", "non_eligible_count"]]


def assign_ndvi_bins(df: pd.DataFrame, *, col: str = "ndvi_mean") -> pd.DataFrame:
    eps = 1e-9
    ndvi = pd.to_numeric(df[col], errors="coerce")
    ok = ndvi.notna() & (ndvi >= 0.0) & (ndvi <= 1.0)
    out = df.loc[ok].copy()

    out["_ndvi_adj"] = ndvi.loc[ok].clip(lower=0.0, upper=1.0 - eps)
    edges = [i / 10 for i in range(11)]
    ordered_labels = [f"{i*10}\u2013{(i+1)*10}%" for i in range(10)]

    bins = pd.cut(
        out["_ndvi_adj"],
        bins=edges,
        right=False,
        include_lowest=True,
        labels=ordered_labels,
    )
    out["ndvi_bin_label"] = bins.astype("string")
    return out


def compute_group_bin_probabilities(
    df: pd.DataFrame,
    *,
    bin_label_col: str,
    weight_col: str,
    group_name: str,
    ordered_bin_labels: list[str],
) -> pd.DataFrame:
    work = df[[bin_label_col, weight_col]].copy()
    work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce")
    work = work.dropna(subset=[bin_label_col, weight_col]).copy()
    work = work[work[weight_col] > 0].copy()

    total_w = float(work[weight_col].sum())
    by_bin = work.groupby(bin_label_col, as_index=False)[weight_col].sum()
    by_bin["probability"] = by_bin[weight_col] / total_w if total_w > 0 else 0.0

    base = pd.DataFrame({bin_label_col: ordered_bin_labels})
    out = base.merge(by_bin[[bin_label_col, "probability"]], on=bin_label_col, how="left")
    out["probability"] = out["probability"].fillna(0.0)
    out["group"] = group_name
    return out


def make_grouped_bar_chart(df_long: pd.DataFrame, *, title: str, ordered_bin_labels: list[str]):
    fig = px.bar(
        df_long,
        x="ndvi_bin_label",
        y="probability",
        color="group",
        barmode="group",
        category_orders={"ndvi_bin_label": ordered_bin_labels},
        color_discrete_sequence=list(VIS.color_palette),
    )
    fig.update_xaxes(title_text="NDVI bin")
    fig.update_yaxes(title_text="Probability", tickformat=".1%")
    fig.update_traces(
        hovertemplate="group=%{fullData.name}<br>bin=%{x}<br>prob=%{y:.2%}<extra></extra>"
    )
    return apply_common_layout(fig, title=title)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot P(a<ndvi_mean<b | FRPM-eligible) and P(a<ndvi_mean<b | non-eligible) using 10% NDVI bins."
    )
    p.add_argument("--greenness-csv", type=Path, default=DEFAULT_GREENNESS_CSV)
    p.add_argument("--frpm-merged-csv", type=Path, default=DEFAULT_FRPM_MERGED_CSV)
    p.add_argument("--outpath", type=Path, default=DEFAULT_OUTPATH)
    args = p.parse_args()

    greenness = load_greenness(args.greenness_csv)
    frpm = load_frpm_merged(args.frpm_merged_csv)

    merged = greenness.merge(frpm, left_on="ID", right_on="cds_code", how="inner")
    merged = assign_ndvi_bins(merged, col="ndvi_mean")

    ordered_labels = [f"{i*10}\u2013{(i+1)*10}%" for i in range(10)]

    df_eligible = compute_group_bin_probabilities(
        merged,
        bin_label_col="ndvi_bin_label",
        weight_col="eligible_count",
        group_name="FRPM-eligible",
        ordered_bin_labels=ordered_labels,
    )
    df_non = compute_group_bin_probabilities(
        merged,
        bin_label_col="ndvi_bin_label",
        weight_col="non_eligible_count",
        group_name="Non-eligible",
        ordered_bin_labels=ordered_labels,
    )
    df_long = pd.concat([df_eligible, df_non], ignore_index=True)

    fig = make_grouped_bar_chart(
        df_long,
        title="Conditional NDVI bin probability by eligibility (SV, 100m, 10% bins)",
        ordered_bin_labels=ordered_labels,
    )

    args.outpath.parent.mkdir(parents=True, exist_ok=True)
    args.outpath.write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")
    print(f"Wrote Plotly HTML figure to: {args.outpath}")


if __name__ == "__main__":
    main()

