from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.visualization_config import VIS, apply_common_layout

DEFAULT_GRENESS_CSV = ROOT / "outputs" / "school_ndvi_nlcd_frpm_sv_100m.csv"
DEFAULT_SCHOOLS_CSV = (
    ROOT / "data" / "cleaned" / "public_schools_frpm_sv_analysis.csv"
)
DEFAULT_OUTDIR = ROOT / "visualizations"

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
    keep_cols = ["ID", "city", "percent_eligible_frpm_k12"]
    if "school_name" in df.columns:
        keep_cols.append("school_name")
    elif "school" in df.columns:
        keep_cols.append("school")
    return df[keep_cols]


def merge_greenness_and_frpm(greenness: pd.DataFrame, schools: pd.DataFrame) -> pd.DataFrame:
    school_col: str | None = None
    if "school_name" in schools.columns:
        school_col = "school_name"
    elif "school" in schools.columns:
        school_col = "school"

    if "percent_eligible_frpm_k12" in greenness.columns:
        merged = greenness.copy()
        fill_cols = ["ID", "percent_eligible_frpm_k12"]
        if school_col is not None:
            fill_cols.append(school_col)
        fill_df = schools[fill_cols].rename(columns={"percent_eligible_frpm_k12": "_frpm_from_schools"})
        if school_col is not None:
            fill_df = fill_df.rename(columns={school_col: "_school_from_schools"})
        merged = merged.merge(fill_df, on="ID", how="left")
        merged["percent_eligible_frpm_k12"] = merged["percent_eligible_frpm_k12"].fillna(
            merged["_frpm_from_schools"]
        )
        if school_col is not None:
            if school_col in merged.columns:
                merged[school_col] = merged[school_col].fillna(merged["_school_from_schools"])
            else:
                merged[school_col] = merged["_school_from_schools"]
            merged = merged.drop(columns=["_school_from_schools"])
        merged = merged.drop(columns=["_frpm_from_schools"])
    else:
        merge_cols = ["ID", "percent_eligible_frpm_k12"]
        if school_col is not None:
            merge_cols.append(school_col)
        merged = greenness.merge(schools[merge_cols], on="ID", how="left")
    return merged


def make_boxplot(
    df: pd.DataFrame, x: str, y: str, color: str, title: str, city_order: list[str]
):
    fig = px.box(
        df,
        x=x,
        y=y,
        color=color,
        color_discrete_sequence=list(VIS.color_palette),
        category_orders={x: city_order},
        points=False,
    )
    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)
    return apply_common_layout(fig, title=title)


def make_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    title: str,
    city_order: list[str],
    include_school_name_in_hover: bool = False,
):
    custom_data: list[str] | None = None
    school_name_col: str | None = None
    if include_school_name_in_hover:
        if "school_name" in df.columns:
            school_name_col = "school_name"
        elif "school" in df.columns:
            school_name_col = "school"
        if school_name_col is not None:
            custom_data = [school_name_col]

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        color_discrete_sequence=list(VIS.color_palette),
        category_orders={"city": city_order},
        opacity=0.85,
        custom_data=custom_data,
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
    if school_name_col is not None:
        hovertemplate = (
            f"school_name=%{{customdata[0]}}<br>"
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
        default="",
        help="Optional comma-separated list of cities to include (default: all cities).",
    )
    args = p.parse_args()

    greenness = load_greenness(args.greenness_csv)
    schools = load_schools(args.schools_csv)

    df = merge_greenness_and_frpm(greenness, schools)
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    if cities:
        df = df[df["city"].isin(cities)].copy()
    if df.empty:
        raise SystemExit("No rows available after city filter.")
    city_order = sorted(df["city"].dropna().astype(str).unique().tolist())

    # NDVI
    fig_ndvi = make_boxplot(
        df,
        x="city",
        y="ndvi_mean",
        color="city",
        title="School NDVI mean (Sentinel-2) within 100m buffers",
        city_order=city_order,
    )
    # Tree canopy
    fig_canopy = make_boxplot(
        df,
        x="city",
        y="nlcd_canopy_mean",
        color="city",
        title="School NLCD tree canopy mean within 100m buffers",
        city_order=city_order,
    )
    # High-canopy fraction
    fig_frac = make_boxplot(
        df,
        x="city",
        y="nlcd_high_canopy_frac",
        color="city",
        title="School fraction of area with NLCD canopy >= 10% within 100m buffers",
        city_order=city_order,
    )
    # FRPM relationships
    fig_ndvi_vs_frpm = make_scatter(
        df,
        x="percent_eligible_frpm_k12",
        y="ndvi_mean",
        color="city",
        title="NDVI mean vs FRPM percent by city",
        city_order=city_order,
    )
    fig_canopy_vs_frpm = make_scatter(
        df,
        x="percent_eligible_frpm_k12",
        y="nlcd_canopy_mean",
        color="city",
        title="NLCD canopy mean vs FRPM percent by city",
        city_order=city_order,
    )
    fig_gi_vs_frpm = None
    if "greenery_index_ndvi_nlcd" in df.columns:
        fig_gi_vs_frpm = make_scatter(
            df,
            x="percent_eligible_frpm_k12",
            y="greenery_index_ndvi_nlcd",
            color="city",
            title="Greenery index (NDVI + NLCD canopy) vs FRPM percent by city",
            city_order=city_order,
            include_school_name_in_hover=True,
        )
    else:
        print(
            "Skipping greenery index vs FRPM scatter: missing 'greenery_index_ndvi_nlcd' column."
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "school_ndvi_boxplot_sv_100m.html").write_text(
        fig_ndvi.to_html(full_html=True, include_plotlyjs="cdn"),
        encoding="utf-8",
    )
    (args.outdir / "school_canopy_boxplot_sv_100m.html").write_text(
        fig_canopy.to_html(full_html=True, include_plotlyjs="cdn"),
        encoding="utf-8",
    )
    (args.outdir / "school_high_canopy_frac_boxplot_sv_100m.html").write_text(
        fig_frac.to_html(full_html=True, include_plotlyjs="cdn"),
        encoding="utf-8",
    )
    (args.outdir / "school_ndvi_vs_frpm_scatter_sv_100m.html").write_text(
        fig_ndvi_vs_frpm.to_html(full_html=True, include_plotlyjs="cdn"),
        encoding="utf-8",
    )
    (args.outdir / "school_canopy_vs_frpm_scatter_sv_100m.html").write_text(
        fig_canopy_vs_frpm.to_html(full_html=True, include_plotlyjs="cdn"),
        encoding="utf-8",
    )
    if fig_gi_vs_frpm is not None:
        (args.outdir / "school_greenery_index_vs_frpm_scatter_sv_100m.html").write_text(
            fig_gi_vs_frpm.to_html(full_html=True, include_plotlyjs="cdn"),
            encoding="utf-8",
        )

    print(f"Wrote Plotly HTML figures to: {args.outdir}")


if __name__ == "__main__":
    main()

