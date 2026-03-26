from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.visualization_config import VIS, apply_common_layout

DEFAULT_GRENESS_CSV = ROOT / "outputs" / "school_ndvi_nlcd_500m.csv"
DEFAULT_SCHOOLS_CSV = (
    ROOT / "data" / "cleaned" / "public_schools_frpm_santaclara_analysis.csv"
)
DEFAULT_OUTDIR = ROOT / "visualizations"

CITIES_DEFAULT = ["San Jose", "Palo Alto", "Mountain View"]


def load_greenness(greenness_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(greenness_csv)
    expected = {"ID", "city", "ndvi_mean", "nlcd_canopy_mean", "nlcd_high_canopy_frac"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"GEE greenness CSV missing columns: {sorted(missing)}")
    return df


def load_schools(schools_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(schools_csv)
    expected = {"ID", "city", "percent_eligible_frpm_k12"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Schools CSV missing columns: {sorted(missing)}")
    return df[["ID", "city", "percent_eligible_frpm_k12"]]


def merge_greenness_and_frpm(greenness: pd.DataFrame, schools: pd.DataFrame) -> pd.DataFrame:
    merged = greenness.merge(
        schools[["ID", "percent_eligible_frpm_k12"]],
        on="ID",
        how="left",
    )
    return merged


def make_boxplot(df: pd.DataFrame, x: str, y: str, color: str, title: str):
    fig = px.box(
        df,
        x=x,
        y=y,
        color=color,
        color_discrete_sequence=list(VIS.color_palette),
        category_orders={x: CITIES_DEFAULT},
        points=False,
    )
    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)
    return apply_common_layout(fig, title=title)


def make_scatter(df: pd.DataFrame, x: str, y: str, color: str, title: str):
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        color_discrete_sequence=list(VIS.color_palette),
        category_orders={"city": CITIES_DEFAULT},
        opacity=0.85,
    )
    # x is a numeric fraction (0-1); format it as percent for readable axis + hover.
    fig.update_xaxes(title_text=x, tickformat=".1%")
    fig.update_yaxes(title_text=y)
    hovertemplate = (
        f"city=%{{fullData.name}}<br>"
        f"{x}=%{{x:.1%}}<br>"
        f"{y}=%{{y}}"
        "<extra></extra>"
    )
    fig.update_traces(hovertemplate=hovertemplate)
    return apply_common_layout(fig, title=title)


def main() -> None:
    p = argparse.ArgumentParser(description="Create city-level plots for NDVI and tree canopy vs FRPM.")
    p.add_argument("--greenness-csv", type=Path, default=DEFAULT_GRENESS_CSV)
    p.add_argument("--schools-csv", type=Path, default=DEFAULT_SCHOOLS_CSV)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument(
        "--cities",
        type=str,
        default=",".join(CITIES_DEFAULT),
        help="Comma-separated list of cities to include.",
    )
    args = p.parse_args()

    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    if not cities:
        raise SystemExit("No cities provided.")

    greenness = load_greenness(args.greenness_csv)
    schools = load_schools(args.schools_csv)

    df = merge_greenness_and_frpm(greenness, schools)
    df = df[df["city"].isin(cities)].copy()

    # NDVI
    fig_ndvi = make_boxplot(
        df,
        x="city",
        y="ndvi_mean",
        color="city",
        title="School NDVI mean (Sentinel-2) within 500m buffers",
    )
    # Tree canopy
    fig_canopy = make_boxplot(
        df,
        x="city",
        y="nlcd_canopy_mean",
        color="city",
        title="School NLCD tree canopy mean within 500m buffers",
    )
    # High-canopy fraction
    fig_frac = make_boxplot(
        df,
        x="city",
        y="nlcd_high_canopy_frac",
        color="city",
        title="School fraction of area with NLCD canopy >= 10% within 500m buffers",
    )
    # FRPM relationships
    fig_ndvi_vs_frpm = make_scatter(
        df,
        x="percent_eligible_frpm_k12",
        y="ndvi_mean",
        color="city",
        title="NDVI mean vs FRPM percent by city",
    )
    fig_canopy_vs_frpm = make_scatter(
        df,
        x="percent_eligible_frpm_k12",
        y="nlcd_canopy_mean",
        color="city",
        title="NLCD canopy mean vs FRPM percent by city",
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "school_ndvi_boxplot.html").write_text(
        fig_ndvi.to_html(full_html=True, include_plotlyjs="cdn"),
        encoding="utf-8",
    )
    (args.outdir / "school_canopy_boxplot.html").write_text(
        fig_canopy.to_html(full_html=True, include_plotlyjs="cdn"),
        encoding="utf-8",
    )
    (args.outdir / "school_high_canopy_frac_boxplot.html").write_text(
        fig_frac.to_html(full_html=True, include_plotlyjs="cdn"),
        encoding="utf-8",
    )
    (args.outdir / "school_ndvi_vs_frpm_scatter.html").write_text(
        fig_ndvi_vs_frpm.to_html(full_html=True, include_plotlyjs="cdn"),
        encoding="utf-8",
    )
    (args.outdir / "school_canopy_vs_frpm_scatter.html").write_text(
        fig_canopy_vs_frpm.to_html(full_html=True, include_plotlyjs="cdn"),
        encoding="utf-8",
    )

    print(f"Wrote Plotly HTML figures to: {args.outdir}")


if __name__ == "__main__":
    main()

