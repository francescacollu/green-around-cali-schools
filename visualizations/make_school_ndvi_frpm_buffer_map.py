from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

import folium
import pandas as pd
from branca.colormap import LinearColormap
from branca.element import Element

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.visualization_config import VIS, ndvi_to_color

DEFAULT_GRENESS_CSV = ROOT / "outputs" / "school_ndvi_nlcd_frpm_sv_100m.csv"
DEFAULT_OUT = ROOT / "visualizations" / "school_ndvi_frpm_buffer_map_sv_100m.html"
DEFAULT_SCHOOLS_MERGED_CSV = ROOT / "data" / "cleaned" / "public_schools_frpm_sv_merged.csv"

# NDVI ramp endpoints for the interactive map.
# Low -> High: #7A8480 -> #2ECC71
NDVI_LOW_HEX = "#dad7cd"
NDVI_HIGH_HEX = "#344e41"

# Constant dot size (pixels) for all schools.
DOT_RADIUS_PX = 8.0

CARTO_DARK_TILES = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
CARTO_ATTR = "&copy; OpenStreetMap contributors &copy; CARTO"


def load_greenness_for_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"ID", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Greenness CSV missing columns: {sorted(missing)}")
    return df


def _pick_school_name(row: pd.Series) -> str:
    if "school_name" in row and pd.notna(row["school_name"]):
        return str(row["school_name"])
    if "school" in row and pd.notna(row["school"]):
        return str(row["school"])
    return ""


def frpm_to_quartile_label(frpm: float) -> str | None:
    # Assumes `percent_eligible_frpm_k12` is stored as a fraction in [0, 1].
    if frpm < 0.0 or frpm > 1.0:
        return None
    if frpm < 0.25:
        return "0-25%"
    if frpm < 0.50:
        return "25-50%"
    if frpm < 0.75:
        return "50-75%"
    return "75-100%"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Interactive dark basemap map: red-to-green NDVI dots with enrollment-sized markers and FRPM slider filtering."
    )
    p.add_argument("--greenness-csv", type=Path, default=DEFAULT_GRENESS_CSV)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--metric-col",
        type=str,
        default="ndvi_mean",
        help="Numeric metric column used for marker color.",
    )
    p.add_argument(
        "--buffer-m",
        type=float,
        default=100.0,
        help="GEE buffer radius used to compute metrics (for labeling/reference; markers are dots).",
    )
    p.add_argument(
        "--schools-merged-csv",
        type=Path,
        default=DEFAULT_SCHOOLS_MERGED_CSV,
        help="Schools CSV with `cds_code` and `enrollment_k12` used for dot sizing.",
    )
    p.add_argument(
        "--frpm-col",
        type=str,
        default="percent_eligible_frpm_k12",
        help="Column name for FRPM percentage/fraction values.",
    )
    p.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Colormap minimum for metric (default: config or data min).",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Colormap maximum for metric (default: config or data max).",
    )
    args = p.parse_args()

    df = load_greenness_for_map(args.greenness_csv)
    if args.metric_col not in df.columns:
        raise ValueError(f"Greenness CSV missing columns: {sorted({args.metric_col} - set(df.columns))}")
    if args.frpm_col not in df.columns:
        raise ValueError(f"Greenness CSV missing columns: {sorted({args.frpm_col} - set(df.columns))}")
    df = df[pd.to_numeric(df["lat"], errors="coerce").notna()].copy()
    df = df[pd.to_numeric(df["lon"], errors="coerce").notna()].copy()
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    df[args.metric_col] = pd.to_numeric(df[args.metric_col], errors="coerce")
    df[args.frpm_col] = pd.to_numeric(df[args.frpm_col], errors="coerce")
    df = df[df[args.metric_col].notna()].copy()
    df = df[df[args.frpm_col].notna()].copy()
    df = df[df[args.frpm_col].between(0.0, 1.0)].copy()
    if df.empty:
        raise SystemExit(f"No rows with valid lat/lon, {args.metric_col}, and FRPM in [0, 1].")

    vmin = args.vmin if args.vmin is not None else VIS.ndvi_map_vmin
    vmax = args.vmax if args.vmax is not None else VIS.ndvi_map_vmax
    if vmin is None:
        vmin = float(df[args.metric_col].min())
    if vmax is None:
        vmax = float(df[args.metric_col].max())
    if vmax < vmin:
        vmin, vmax = vmax, vmin

    # Merge enrollment for dot sizing.
    df["ID"] = df["ID"].astype(str)
    schools_df = pd.read_csv(args.schools_merged_csv)
    if "cds_code" not in schools_df.columns:
        raise ValueError("Schools merged CSV missing `cds_code` column for join key.")
    if "enrollment_k12" not in schools_df.columns:
        raise ValueError("Schools merged CSV missing `enrollment_k12` column for dot sizing.")

    schools_df = schools_df.rename(columns={"cds_code": "ID"})
    schools_df["ID"] = schools_df["ID"].astype(str)
    schools_df["enrollment_k12"] = pd.to_numeric(schools_df["enrollment_k12"], errors="coerce")
    df = df.merge(schools_df[["ID", "enrollment_k12"]], on="ID", how="left")

    # Constant dot size for all schools (requested).
    df["radius_px"] = float(DOT_RADIUS_PX)

    center_lat = float(df["lat"].mean())
    center_lon = float(df["lon"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles=CARTO_DARK_TILES,
        name="CartoDB Dark Matter",
        attr=CARTO_ATTR,
        max_zoom=19,
    ).add_to(m)

    # Create JS-ready school marker data for interactive FRPM filtering.
    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        metric_val = float(row[args.metric_col])
        frpm = float(row[args.frpm_col])
        radius_px = float(row["radius_px"])

        color = ndvi_to_color(
            metric_val,
            vmin,
            vmax,
            low_hex=NDVI_LOW_HEX,
            high_hex=NDVI_HIGH_HEX,
        )

        school_name = _pick_school_name(row)
        city = str(row["city"]) if "city" in row and pd.notna(row["city"]) else ""
        frpm_pct = frpm * 100.0

        tooltip_html = (
            f"<b>{html.escape(school_name)}</b>"
            f"<br>{html.escape(city)}"
            f"<br>FRPM: {frpm_pct:.1f}%"
            f"<br>{html.escape(args.metric_col)}: {metric_val:.4f}"
        )

        records.append(
            {
                "lat": lat,
                "lon": lon,
                "frpm": frpm,
                "radius": radius_px,
                "color": color,
                "tooltip": tooltip_html,
            }
        )

    records_json = json.dumps(records, ensure_ascii=True)
    map_js_name = m.get_name()

    # Fit bounds before injecting JS controls/markers (so map is centered correctly).
    sw = [float(df["lat"].min()), float(df["lon"].min())]
    ne = [float(df["lat"].max()), float(df["lon"].max())]
    m.fit_bounds([sw, ne], padding=(24, 24))

    js = """
    <script>
    (function() {
      var initialized = false;

      function init() {
        // Use window[...] so we don't get a ReferenceError before Folium declares the map variable.
        var mapObj = window[__MAP_VAR__];
        if (!mapObj || typeof mapObj.getCenter !== 'function') {
          setTimeout(init, 100);
          return;
        }
        if (initialized) {
          return;
        }
        initialized = true;

        var markerLayer = L.layerGroup().addTo(mapObj);

        var schools = __RECORDS__;
        var markers = [];
        for (var i = 0; i < schools.length; i++) {
          var s = schools[i];
          var marker = L.circleMarker([s.lat, s.lon], {
            radius: s.radius,
            color: s.color,
            weight: __STROKE_WEIGHT__,
            opacity: __STROKE_OPACITY__,
            fillColor: s.color,
            fillOpacity: __FILL_OPACITY__
          });
          marker.bindTooltip(s.tooltip, {sticky: true});
          markers.push(marker);
        }

        var frpmControl = L.control({position: 'topleft'});
        frpmControl.onAdd = function() {
          var div = L.DomUtil.create('div', 'frpm-filter');
          div.style.background = 'rgba(0, 0, 0, 0.55)';
          div.style.color = 'white';
          div.style.padding = '10px';
          div.style.borderRadius = '8px';
          div.style.fontFamily = '__FONT__';
          div.style.fontSize = '13px';
          div.innerHTML = `
            <div style="font-weight:bold; margin-bottom:6px;">FRPM Filter</div>
            <div style="margin-bottom:8px; opacity:0.9; max-width:240px; line-height:1.2;">
              Shows only schools whose FRPM-eligible share falls within the selected range.
            </div>
            <div style="margin-bottom:6px;">
              Min: <span id="frpmMinVal">0%</span>
              <span style="opacity:0.8;">&nbsp; | &nbsp;</span>
              Max: <span id="frpmMaxVal">100%</span>
            </div>

            <div style="position:relative; width:210px; height:26px; margin:8px 0 10px;">
              <div id="frpmTrackBar" style="position:absolute; left:0; right:0; top:50%; transform:translateY(-50%); height:6px; border-radius:9999px; background: rgba(255,255,255,0.25);"></div>
              <input id="frpmMin" class="frpm-range" type="range" min="0" max="100" value="0" step="1" />
              <input id="frpmMax" class="frpm-range" type="range" min="0" max="100" value="100" step="1" />
            </div>

            <div style="display:flex; gap:6px; margin-top:2px; flex-wrap:wrap;">
              <button type="button" onclick="window.setFrpmPreset('all')" style="cursor:pointer;">All schools</button>
              <button type="button" onclick="window.setFrpmPreset('low_frpm')" style="cursor:pointer;">Low % of FRPM eligible students</button>
              <button type="button" onclick="window.setFrpmPreset('high_frpm')" style="cursor:pointer;">High % of FRPM eligible students</button>
            </div>
          `;

          // Prevent map interactions when using the UI control.
          // Without this, dragging the slider can pan the map in Chrome/Edge.
          if (L && L.DomEvent) {
            L.DomEvent.disableClickPropagation(div);
            L.DomEvent.disableScrollPropagation(div);
          }
          var stopEvents = ['mousedown', 'mousemove', 'mouseup', 'dblclick', 'click', 'contextmenu', 'wheel', 'touchstart', 'touchmove', 'touchend', 'pointerdown', 'pointermove', 'pointerup'];
          for (var ei = 0; ei < stopEvents.length; ei++) {
            div.addEventListener(stopEvents[ei], function(e) { e.stopPropagation(); }, {passive: true});
          }

          var dragDisabledByControl = false;
          function disableMapDrag() {
            if (dragDisabledByControl) { return; }
            if (mapObj && mapObj.dragging && typeof mapObj.dragging.disable === 'function') {
              mapObj.dragging.disable();
              dragDisabledByControl = true;
            }
          }
          function enableMapDrag() {
            if (!dragDisabledByControl) { return; }
            if (mapObj && mapObj.dragging && typeof mapObj.dragging.enable === 'function') {
              mapObj.dragging.enable();
              dragDisabledByControl = false;
            }
          }

          var minEl = div.querySelector('#frpmMin');
          var maxEl = div.querySelector('#frpmMax');
          var downEvents = ['mousedown', 'touchstart', 'pointerdown'];
          for (var di = 0; di < downEvents.length; di++) {
            if (minEl) { minEl.addEventListener(downEvents[di], disableMapDrag, {passive: true}); }
            if (maxEl) { maxEl.addEventListener(downEvents[di], disableMapDrag, {passive: true}); }
          }
          if (!window.__frpmMapDragReleaseHooked) {
            window.__frpmMapDragReleaseHooked = true;
            document.addEventListener('mouseup', enableMapDrag, {passive: true});
            document.addEventListener('touchend', enableMapDrag, {passive: true});
            document.addEventListener('pointerup', enableMapDrag, {passive: true});
            document.addEventListener('pointercancel', enableMapDrag, {passive: true});
          }

          // Inject CSS once for the dual-thumb slider controls.
          if (!document.getElementById('frpm-range-style')) {
            var style = document.createElement('style');
            style.id = 'frpm-range-style';
            style.innerHTML = `
              .frpm-range {
                position:absolute;
                left:0;
                right:0;
                top:0;
                bottom:0;
                width:100%;
                margin:0;
                background: transparent;
                -webkit-appearance: none;
                appearance: none;
                /* Prevent Chromium's native blue track/progress fill. */
                accent-color: rgba(255,255,255,0);
                /* Two range inputs overlap; let only thumbs receive pointer events. */
                pointer-events: none;
              }
              .frpm-range::-webkit-slider-runnable-track {
                background: transparent;
                border: 0;
              }
              .frpm-range::-moz-range-track {
                background: transparent;
                border: 0;
              }

              .frpm-range::-webkit-slider-thumb {
                -webkit-appearance: none;
                appearance: none;
                width: 14px;
                height: 14px;
                border-radius: 50%;
                background: white;
                border: 2px solid rgba(0,0,0,0.3);
                pointer-events: auto;
                cursor: ew-resize;
              }
              .frpm-range::-moz-range-thumb {
                width: 14px;
                height: 14px;
                border-radius: 50%;
                background: white;
                border: 2px solid rgba(0,0,0,0.3);
                pointer-events: auto;
                cursor: ew-resize;
              }
            `;
            document.head.appendChild(style);
          }
          return div;
        };
        frpmControl.addTo(mapObj);

        function updateText() {
          var minPct = parseFloat(document.getElementById('frpmMin').value);
          var maxPct = parseFloat(document.getElementById('frpmMax').value);
          document.getElementById('frpmMinVal').innerText = minPct.toFixed(0) + '%';
          document.getElementById('frpmMaxVal').innerText = maxPct.toFixed(0) + '%';
        }

        function updateTrack(minPct, maxPct) {
          var bar = document.getElementById('frpmTrackBar');
          if (!bar) { return; }
          // Only highlight the selected range between minPct and maxPct.
          bar.style.background = 'linear-gradient(to right,' +
            'rgba(255,255,255,0.25) 0%,' +
            'rgba(255,255,255,0.25) ' + minPct + '%,' +
            '__ACTIVE__ ' + minPct + '%,' +
            '__ACTIVE__ ' + maxPct + '%,' +
            'rgba(255,255,255,0.25) ' + maxPct + '%,' +
            'rgba(255,255,255,0.25) 100%)';
        }

        function setSliderValues(minPct, maxPct) {
          document.getElementById('frpmMin').value = minPct;
          document.getElementById('frpmMax').value = maxPct;
          updateText();
        }

        function updateFilter() {
          var minPct = parseFloat(document.getElementById('frpmMin').value);
          var maxPct = parseFloat(document.getElementById('frpmMax').value);
          if (minPct > maxPct) {
            maxPct = minPct;
            document.getElementById('frpmMax').value = maxPct;
          }
          updateText();
          updateTrack(minPct, maxPct);

          // Ensure the appropriate thumb is on top when close together.
          // (Otherwise the max slider can block grabbing the min thumb.)
          var minEl2 = document.getElementById('frpmMin');
          var maxEl2 = document.getElementById('frpmMax');
          if (minEl2 && maxEl2) {
            if (minPct >= (maxPct - 2)) {
              minEl2.style.zIndex = '6';
              maxEl2.style.zIndex = '5';
            } else {
              minEl2.style.zIndex = '5';
              maxEl2.style.zIndex = '6';
            }
          }

          var minFrac = minPct / 100.0;
          var maxFrac = maxPct / 100.0;

          markerLayer.clearLayers();
          for (var i = 0; i < schools.length; i++) {
            var s = schools[i];
            if (s.frpm >= minFrac && s.frpm <= maxFrac) {
              markerLayer.addLayer(markers[i]);
            }
          }
        }

        function setPreset(preset) {
          if (preset === 'all') {
            setSliderValues(0, 100);
          } else if (preset === 'low_frpm') {
            setSliderValues(0, 25);
          } else if (preset === 'high_frpm') {
            setSliderValues(50, 75);
          }
          updateFilter();
        }

        window.setFrpmPreset = setPreset;

        document.getElementById('frpmMin').addEventListener('input', updateFilter);
        document.getElementById('frpmMax').addEventListener('input', updateFilter);

        // Initialize.
        updateFilter();
      }

      init();
    })();
    </script>
    """.replace(
        "__MAP_VAR__",
        json.dumps(map_js_name),
    )
    js = js.replace("__RECORDS__", records_json)
    js = js.replace("__STROKE_WEIGHT__", str(VIS.ndvi_map_stroke_weight))
    js = js.replace("__STROKE_OPACITY__", str(VIS.ndvi_map_stroke_opacity))
    js = js.replace("__FILL_OPACITY__", str(VIS.ndvi_map_fill_opacity))
    js = js.replace("__FONT__", VIS.font_family)
    js = js.replace("__ACTIVE__", "rgba(255,255,255,0.9)")
    m.get_root().html.add_child(Element(js))

    LinearColormap(
        colors=[NDVI_LOW_HEX, NDVI_HIGH_HEX],
        vmin=vmin,
        vmax=vmax,
        caption=f"{args.metric_col} (within buffer)",
    ).add_to(m)

    # Make the legend readable on dark basemap and add low/high labels.
    legend_js = r"""
    <style>
      /* Branca colormap legend container */
      .legend {
        color: rgba(255,255,255,0.9) !important;
      }
      .legend .caption {
        color: rgba(255,255,255,0.9) !important;
      }
      .ndvi-end-labels {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        margin-top: 4px;
        font-size: 11px;
        color: rgba(255,255,255,0.85);
        font-family: __FONT__;
        line-height: 1.2;
      }
    </style>
    <script>
      (function() {
        function addEndLabels() {
          var legends = document.getElementsByClassName('legend');
          if (!legends || legends.length === 0) {
            setTimeout(addEndLabels, 100);
            return;
          }
          for (var i = 0; i < legends.length; i++) {
            var el = legends[i];
            if (el.querySelector('.ndvi-end-labels')) { continue; }
            var labels = document.createElement('div');
            labels.className = 'ndvi-end-labels';
            labels.innerHTML = '<div>Lower values</div><div>Higher values</div>';
            el.appendChild(labels);
          }
        }
        addEndLabels();
      })();
    </script>
    """.replace("__FONT__", VIS.font_family)
    m.get_root().html.add_child(Element(legend_js))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

