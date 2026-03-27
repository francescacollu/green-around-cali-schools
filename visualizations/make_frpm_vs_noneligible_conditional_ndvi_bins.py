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


def _ordered_equal_width_bin_labels(bin_width_pct: float) -> list[str]:
    n_bins = int(round(100 / bin_width_pct))
    if abs((n_bins * bin_width_pct) - 100.0) > 1e-6:
        raise ValueError("bin_width_pct must divide 100 evenly (e.g. 10, 20, 25, 50).")
    w = int(round(bin_width_pct))
    return [f"{i * w}\u2013{(i + 1) * w}%" for i in range(n_bins)]


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def load_greenness(path: Path, *, metric_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ID": "string"})
    _require_columns(df, {"ID", metric_col}, "Greenness CSV")
    df = df[["ID", metric_col]].copy()
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=["ID", metric_col]).copy()
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


def assign_bins(
    df: pd.DataFrame,
    *,
    col: str,
    bin_width_pct: float = 10.0,
    bin_strategy: str = "equal_width_0_1",
) -> tuple[pd.DataFrame, list[str]]:
    values = pd.to_numeric(df[col], errors="coerce")
    out = df.loc[values.notna()].copy()

    if bin_strategy == "equal_width_0_1":
        eps = 1e-9
        ok = values.notna() & (values >= 0.0) & (values <= 1.0)
        out = df.loc[ok].copy()
        ordered_labels = _ordered_equal_width_bin_labels(bin_width_pct)
        n_bins = len(ordered_labels)
        step = bin_width_pct / 100.0
        edges = [i * step for i in range(n_bins + 1)]
        out["_metric_adj"] = values.loc[ok].clip(lower=0.0, upper=1.0 - eps)
        bins = pd.cut(
            out["_metric_adj"],
            bins=edges,
            right=False,
            include_lowest=True,
            labels=ordered_labels,
        )
    elif bin_strategy == "quantile":
        n_bins = int(round(100 / bin_width_pct))
        if abs((n_bins * bin_width_pct) - 100.0) > 1e-6:
            raise ValueError("bin_width_pct must divide 100 evenly (e.g. 10, 20, 25, 50).")
        ordered_labels = [f"Q{i+1}" for i in range(n_bins)]
        bins = pd.qcut(values.loc[out.index], q=n_bins, labels=ordered_labels, duplicates="drop")
        observed_labels = pd.Index(bins.dropna().astype(str).unique().tolist())
        ordered_labels = [lbl for lbl in ordered_labels if lbl in observed_labels.tolist()]
    else:
        raise ValueError(f"Unknown bin_strategy: {bin_strategy}")

    out["ndvi_bin_label"] = bins.astype("string")
    return out, ordered_labels


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
        description=(
            "Plot conditional probabilities for FRPM-eligible vs non-eligible groups across metric bins."
        )
    )
    p.add_argument("--greenness-csv", type=Path, default=DEFAULT_GREENNESS_CSV)
    p.add_argument("--frpm-merged-csv", type=Path, default=DEFAULT_FRPM_MERGED_CSV)
    p.add_argument(
        "--metric-col",
        type=str,
        default="ndvi_mean",
        help="Metric column to bin (e.g., ndvi_mean or greenery_index_ndvi_nlcd).",
    )
    p.add_argument(
        "--bin-strategy",
        type=str,
        default="equal_width_0_1",
        choices=["equal_width_0_1", "quantile"],
        help="Bin construction strategy. Use quantile for non-[0,1] metrics.",
    )
    p.add_argument(
        "--outpath",
        type=Path,
        default=None,
        help=f"Default: {DEFAULT_OUTPATH.name} for 10%% bins; else ..._<pct>pct_bins.html",
    )
    p.add_argument(
        "--bin-width-pct",
        type=float,
        default=10.0,
        help="Width of each NDVI bin as percent of the 0-1 scale (must divide 100).",
    )
    args = p.parse_args()

    bin_w = float(args.bin_width_pct)
    if args.outpath is None:
        if bin_w == 10.0:
            args.outpath = (
                ROOT / "visualizations" / f"frpm_vs_noneligible_conditional_{args.metric_col}_bin_probability_sv_100m.html"
            )
        else:
            args.outpath = (
                ROOT
                / "visualizations"
                / f"frpm_vs_noneligible_conditional_{args.metric_col}_bin_probability_sv_100m_{int(round(bin_w))}pct_bins.html"
            )

    greenness = load_greenness(args.greenness_csv, metric_col=args.metric_col)
    frpm = load_frpm_merged(args.frpm_merged_csv)

    merged = greenness.merge(frpm, left_on="ID", right_on="cds_code", how="inner")
    merged, ordered_labels = assign_bins(
        merged,
        col=args.metric_col,
        bin_width_pct=bin_w,
        bin_strategy=args.bin_strategy,
    )

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

    pct = int(round(bin_w))
    fig = make_grouped_bar_chart(
        df_long,
        title=f"Conditional {args.metric_col} bin probability by eligibility (SV, 100m, {pct}% bins)",
        ordered_bin_labels=ordered_labels,
    )

    args.outpath.parent.mkdir(parents=True, exist_ok=True)
    args.outpath.write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")
    print(f"Wrote Plotly HTML figure to: {args.outpath}")


if __name__ == "__main__":
    main()

