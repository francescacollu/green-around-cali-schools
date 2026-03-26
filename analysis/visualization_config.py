from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go


@dataclass(frozen=True)
class VizConfig:
    color_palette: tuple[str, ...] = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    )
    font_family: str = "Arial"
    title_size: int = 16
    axis_size: int = 12
    legend_size: int = 11
    template: str = "plotly_white"
    fig_width: int = 1000
    fig_height: int = 650
    # NDVI buffer map (Folium): low NDVI grey, high NDVI dark green.
    ndvi_map_low_color: str = "#c8c8c8"
    ndvi_map_high_color: str = "#0d3d0d"
    ndvi_map_fill_opacity: float = 0.55
    ndvi_map_stroke_opacity: float = 0.85
    ndvi_map_stroke_weight: int = 1
    #: If set, colormap uses these bounds; if None, uses data min/max per run.
    ndvi_map_vmin: float | None = None
    ndvi_map_vmax: float | None = None


VIS = VizConfig()


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.strip().lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected #RRGGBB, got {hex_color!r}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def ndvi_to_color(
    ndvi: float,
    vmin: float,
    vmax: float,
    low_hex: str = VIS.ndvi_map_low_color,
    high_hex: str = VIS.ndvi_map_high_color,
) -> str:
    """Linear grey-to-green (or custom endpoints) for NDVI in [vmin, vmax]."""
    if vmax <= vmin:
        t = 0.5
    else:
        t = (float(ndvi) - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = _hex_to_rgb(low_hex)
    r2, g2, b2 = _hex_to_rgb(high_hex)
    r = int(round(r1 + (r2 - r1) * t))
    g = int(round(g1 + (g2 - g1) * t))
    b = int(round(b1 + (b2 - b1) * t))
    return _rgb_to_hex(r, g, b)


def apply_common_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        template=VIS.template,
        width=VIS.fig_width,
        height=VIS.fig_height,
        title={"text": title, "x": 0.01, "font": {"size": VIS.title_size}},
        font={"family": VIS.font_family, "size": VIS.axis_size},
        legend={"font": {"size": VIS.legend_size}},
        margin={"l": 60, "r": 30, "t": 60, "b": 55},
    )
    return fig
