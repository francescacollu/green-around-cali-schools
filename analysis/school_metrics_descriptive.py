from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from analysis.visualization_config import VIS, apply_common_layout

DEFAULT_GREENNESS_CSV = ROOT / "outputs" / "school_ndvi_nlcd_500m.csv"
DEFAULT_SCHOOLS_CSV = (
    ROOT / "data" / "cleaned" / "public_schools_frpm_santaclara_analysis.csv"
)
DEFAULT_OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_VIZ_DIR = ROOT / "visualizations"

METRICS = [
    "ndvi_mean",
    "nlcd_canopy_mean",
    "nlcd_high_canopy_frac",
    "percent_eligible_frpm_k12",
]

SUMMARY_CSV = "school_metrics_summary.csv"
OUTLIERS_CSV = "school_metrics_outliers_iqr.csv"
CLUSTERS_CSV = "school_metrics_clusters.csv"
CLUSTERING_PARAMS_CSV = "school_metrics_clustering_params.csv"


def load_greenness(greenness_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(greenness_csv)
    expected = {"ID", "city", "school", "ndvi_mean", "nlcd_canopy_mean", "nlcd_high_canopy_frac"}
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
    return df[["ID", "percent_eligible_frpm_k12"]]


def merge_greenness_and_frpm(greenness: pd.DataFrame, schools: pd.DataFrame) -> pd.DataFrame:
    return greenness.merge(schools, on="ID", how="left")


def build_summary(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for m in metrics:
        s = df[m]
        rows.append(
            {
                "metric": m,
                "n": int(s.notna().sum()),
                "min": s.min(),
                "max": s.max(),
                "mean": s.mean(),
            }
        )
    return pd.DataFrame(rows)


def build_iqr_outliers_long(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    records: list[dict] = []
    id_col, school_col, city_col = "ID", "school", "city"
    for m in metrics:
        q1 = df[m].quantile(0.25)
        q3 = df[m].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = df[m].notna() & ((df[m] < lower) | (df[m] > upper))
        sub = df.loc[mask]
        for _, row in sub.iterrows():
            records.append(
                {
                    id_col: row[id_col],
                    school_col: row[school_col],
                    city_col: row[city_col],
                    "metric": m,
                    "value": row[m],
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_fence": lower,
                    "upper_fence": upper,
                }
            )
    return pd.DataFrame(records)


def make_histogram(df: pd.DataFrame, column: str, title: str, x_title: str) -> px.Figure:
    fig = px.histogram(df, x=column, nbins=30, color_discrete_sequence=[VIS.color_palette[0]])
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text="Count")
    fig = apply_common_layout(fig, title=title)
    if column == "percent_eligible_frpm_k12":
        fig.update_xaxes(tickformat=".0%")
    return fig


def pick_k_and_cluster(
    X: np.ndarray, k_min: int = 2, k_max: int = 10
) -> tuple[int, float, np.ndarray] | None:
    n = X.shape[0]
    k_hi = min(k_max, n - 1)
    if k_hi < k_min:
        return None
    best_k: int | None = None
    best_s = -1.0
    best_labels: np.ndarray | None = None
    for k in range(k_min, k_hi + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        try:
            s = float(silhouette_score(X, labels))
        except ValueError:
            continue
        if s > best_s:
            best_s, best_k, best_labels = s, k, labels
    if best_k is None or best_labels is None:
        return None
    return best_k, best_s, best_labels


def main() -> None:
    p = argparse.ArgumentParser(
        description="Summary stats, histograms, IQR outliers, and K-means clusters for school metrics."
    )
    p.add_argument("--greenness-csv", type=Path, default=DEFAULT_GREENNESS_CSV)
    p.add_argument("--schools-csv", type=Path, default=DEFAULT_SCHOOLS_CSV)
    p.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    p.add_argument("--viz-dir", type=Path, default=DEFAULT_VIZ_DIR)
    p.add_argument(
        "--cities",
        type=str,
        default="",
        help="Optional comma-separated cities to filter (default: all cities).",
    )
    args = p.parse_args()

    greenness = load_greenness(args.greenness_csv)
    schools = load_schools(args.schools_csv)
    df = merge_greenness_and_frpm(greenness, schools)

    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    if cities:
        df = df[df["city"].isin(cities)].copy()

    for m in METRICS:
        if m not in df.columns:
            raise ValueError(f"Merged frame missing metric column: {m}")

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    args.viz_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(df, METRICS)
    summary_path = args.outputs_dir / SUMMARY_CSV
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    outliers = build_iqr_outliers_long(df, METRICS)
    outliers_path = args.outputs_dir / OUTLIERS_CSV
    outliers.to_csv(outliers_path, index=False)
    print(f"Wrote {outliers_path} ({len(outliers)} outlier rows)")

    hist_specs = [
        ("ndvi_mean", "Histogram: NDVI mean (500m buffer)", "NDVI mean"),
        ("nlcd_canopy_mean", "Histogram: NLCD canopy mean (%)", "NLCD canopy mean (%)"),
        (
            "nlcd_high_canopy_frac",
            "Histogram: fraction of area with NLCD canopy >= 10%",
            "High-canopy fraction",
        ),
        (
            "percent_eligible_frpm_k12",
            "Histogram: percent eligible FRPM (K-12)",
            "Percent eligible FRPM",
        ),
    ]
    for col, title, x_title in hist_specs:
        fig_h = make_histogram(df.dropna(subset=[col]), col, title, x_title)
        out_html = args.viz_dir / f"hist_{col}.html"
        out_html.write_text(fig_h.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")
        print(f"Wrote {out_html}")

    complete = df.dropna(subset=METRICS)
    dropped = len(df) - len(complete)
    if dropped:
        print(f"Clustering: dropped {dropped} row(s) with missing metric value(s); n={len(complete)}")

    cluster_result = pick_k_and_cluster(
        StandardScaler().fit_transform(complete[METRICS].to_numpy())
    )
    if cluster_result is None:
        print("Clustering skipped: not enough samples or no valid k.")
        (args.outputs_dir / CLUSTERS_CSV).write_text(
            "ID,school,city,cluster,ndvi_mean,nlcd_canopy_mean,nlcd_high_canopy_frac,"
            "percent_eligible_frpm_k12\n",
            encoding="utf-8",
        )
    else:
        best_k, best_s, labels = cluster_result
        print(f"Clustering: chosen k={best_k}, silhouette={best_s:.4f}")
        out_cluster = complete[["ID", "school", "city"] + METRICS].copy()
        out_cluster["cluster"] = labels.astype(int)
        cols = ["ID", "school", "city", "cluster"] + METRICS
        cluster_path = args.outputs_dir / CLUSTERS_CSV
        out_cluster[cols].to_csv(cluster_path, index=False)
        print(f"Wrote {cluster_path}")

        params_path = args.outputs_dir / CLUSTERING_PARAMS_CSV
        pd.DataFrame([{"k": best_k, "silhouette": best_s, "n_samples": len(complete)}]).to_csv(
            params_path, index=False
        )
        print(f"Wrote {params_path}")

        fig_sc1 = px.scatter(
            out_cluster,
            x="ndvi_mean",
            y="nlcd_canopy_mean",
            color="cluster",
            hover_data=["school", "city"],
            color_discrete_sequence=list(VIS.color_palette),
        )
        fig_sc1.update_xaxes(title_text="NDVI mean")
        fig_sc1.update_yaxes(title_text="NLCD canopy mean (%)")
        # Cluster labels are from standardized 4D feature space, not geography.
        fig_sc1 = apply_common_layout(fig_sc1, title="K-means clusters: NDVI vs NLCD canopy mean")
        p1 = args.viz_dir / "scatter_clusters_ndvi_vs_canopy.html"
        p1.write_text(fig_sc1.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")
        print(f"Wrote {p1}")

        fig_sc2 = px.scatter(
            out_cluster,
            x="percent_eligible_frpm_k12",
            y="nlcd_high_canopy_frac",
            color="cluster",
            hover_data=["school", "city"],
            color_discrete_sequence=list(VIS.color_palette),
        )
        fig_sc2.update_xaxes(title_text="Percent eligible FRPM", tickformat=".0%")
        fig_sc2.update_yaxes(title_text="High-canopy fraction")
        fig_sc2 = apply_common_layout(
            fig_sc2, title="K-means clusters: FRPM vs high-canopy fraction"
        )
        p2 = args.viz_dir / "scatter_clusters_frpm_vs_high_canopy_frac.html"
        p2.write_text(fig_sc2.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")
        print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
