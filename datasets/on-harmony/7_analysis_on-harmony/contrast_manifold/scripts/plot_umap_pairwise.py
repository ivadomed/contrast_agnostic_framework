#!/usr/bin/env python
"""
Generate per-contrast UMAP pair plots from a pre-computed umap_joint_coords.csv.

For each target contrast (T2w, FLAIR, bold, dwi, epi, GRE, …) produces one
interactive 3-D HTML showing:
  - T1w points       → red, full opacity
  - target contrast  → its modality colour, full opacity
  - all synthetic    → semi-transparent grey background

This lets you inspect how well the synthetic distribution bridges between T1w
and each real contrast, without re-fitting UMAP.

Usage:
  set_slot 0 .venv/bin/python analysis/contrast_manifold/scripts/plot_umap_pairwise.py \
      --coords analysis/contrast_manifold/outputs/plots/v19/v19_c_r1/synthseg_mask_31/umap/umap_joint_coords.csv \
      --output_dir analysis/contrast_manifold/outputs/plots/v19/v19_c_r1/synthseg_mask_31/umap
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODALITY_PALETTE = {
    "T1w":   "#E53935",
    "T2w":   "#1E88E5",
    "FLAIR": "#43A047",
    "dwi":   "#FB8C00",
    "bold":  "#8E24AA",
    "epi":   "#00ACC1",
    "GRE":   "#6D4C41",
}

PLOTLY_SYMS = ["circle", "square", "diamond", "cross", "x",
               "circle-open", "square-open", "diamond-open"]

SYNTH_COLOR = "#888888"
SYNTH_ALPHA = 0.15

_JS = """
(function() {
    var gd = document.querySelector('.plotly-graph-div');
    gd.on('plotly_click', function(data) {
        var pt = data.points[0];
        var cd = pt.customdata;
        if (!cd) return;
        var text = Array.isArray(cd) ? cd[0] : cd;
        var toast = document.createElement('div');
        toast.style.cssText = [
            'position:fixed','bottom:24px','left:50%','transform:translateX(-50%)',
            'background:#222','color:#fff','padding:10px 18px','border-radius:6px',
            'font-size:13px','font-family:monospace','z-index:9999',
            'max-width:80%','word-break:break-all','box-shadow:0 2px 8px rgba(0,0,0,.4)'
        ].join(';');
        if (text && text !== 'none') {
            // execCommand must run synchronously inside the click handler (Safari requirement).
            var el = document.createElement('textarea');
            el.value = text;
            el.style.cssText = 'position:fixed;opacity:0;pointer-events:none;';
            document.body.appendChild(el);
            el.select();
            var syncOk = document.execCommand('copy');
            document.body.removeChild(el);
            if (syncOk) {
                toast.textContent = '✓ Copied: ' + text;
            } else if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(text).then(function() {
                    toast.textContent = '✓ Copied: ' + text;
                }).catch(function() { toast.textContent = text; });
            } else {
                toast.textContent = text;
            }
        } else {
            toast.textContent = pt.text ? pt.text.replace(/<[^>]+>/g,' ') : 'No path';
        }
        document.body.appendChild(toast);
        setTimeout(function() {
            toast.style.transition = 'opacity 0.4s';
            toast.style.opacity = '0';
            setTimeout(function() { document.body.removeChild(toast); }, 400);
        }, 3500);
    });
})();
"""


def base_mod(m: str) -> str:
    return m.split("_")[0] if m else "unknown"


def hover(row: pd.Series) -> tuple[str, str]:
    path = str(row.get("image_path", "") or "")
    fname = Path(path).name if path else "—"
    txt = (
        f"<b>{row.get('subject','?')}</b> {row.get('session','?')}<br>"
        f"modality: {row.get('modality','?')}<br>"
        f"scanner:  {row.get('scanner','?')}<br>"
        f"file:     {fname}<br>"
        f"<i>{'click to copy path' if path else 'no path'}</i>"
    )
    return txt, (path or "none")


def make_pair_html(
    df_t1w: pd.DataFrame,
    df_target: pd.DataFrame,
    df_synth: pd.DataFrame,
    target_label: str,
    synth_label: str,
    output_path: Path,
) -> None:
    traces: list[go.Scatter3d] = []

    # ── Synthetic background ──────────────────────────────────────────────
    hover_synth = [hover(r)[0] for _, r in df_synth.iterrows()]
    paths_synth = [hover(r)[1] for _, r in df_synth.iterrows()]
    traces.append(go.Scatter3d(
        x=df_synth["umap1"].values,
        y=df_synth["umap2"].values,
        z=df_synth["umap3"].values,
        mode="markers",
        marker=dict(size=2.5, color=SYNTH_COLOR, opacity=SYNTH_ALPHA),
        name=synth_label,
        text=hover_synth,
        customdata=[[p] for p in paths_synth],
        hoverinfo="text",
    ))

    # ── T1w ──────────────────────────────────────────────────────────────
    scanners = sorted(pd.concat([df_t1w, df_target])["scanner"].dropna().unique())
    symbol_map = {sc: PLOTLY_SYMS[i % len(PLOTLY_SYMS)] for i, sc in enumerate(scanners)}

    for sc, grp in df_t1w.groupby("scanner"):
        hover_t = [hover(r)[0] for _, r in grp.iterrows()]
        paths_t = [hover(r)[1] for _, r in grp.iterrows()]
        traces.append(go.Scatter3d(
            x=grp["umap1"].values, y=grp["umap2"].values, z=grp["umap3"].values,
            mode="markers",
            marker=dict(size=4, color=MODALITY_PALETTE["T1w"],
                        symbol=symbol_map.get(sc, "circle"), opacity=0.9),
            name=f"T1w / {sc}",
            legendgroup="T1w",
            text=hover_t,
            customdata=[[p] for p in paths_t],
            hoverinfo="text",
        ))

    # ── Target contrast ───────────────────────────────────────────────────
    color = MODALITY_PALETTE.get(base_mod(target_label), "#888888")
    for sc, grp in df_target.groupby("scanner"):
        hover_t = [hover(r)[0] for _, r in grp.iterrows()]
        paths_t = [hover(r)[1] for _, r in grp.iterrows()]
        traces.append(go.Scatter3d(
            x=grp["umap1"].values, y=grp["umap2"].values, z=grp["umap3"].values,
            mode="markers",
            marker=dict(size=4, color=color,
                        symbol=symbol_map.get(sc, "circle"), opacity=0.9),
            name=f"{target_label} / {sc}",
            legendgroup=target_label,
            text=hover_t,
            customdata=[[p] for p in paths_t],
            hoverinfo="text",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"UMAP — T1w + {target_label} + {synth_label}",
        scene=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3"),
        legend=dict(font=dict(size=9)),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn", post_script=_JS)
    log.info("Saved → %s", output_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--coords", type=Path, required=True,
        help="Path to umap_joint_coords.csv produced by plot_umap_joint.py",
    )
    p.add_argument(
        "--output_dir", type=Path, required=True,
        help="Directory to write output HTML files into",
    )
    p.add_argument(
        "--targets", nargs="*", default=None,
        help="Base modality names to pair with T1w (default: all except T1w). "
             "E.g. --targets T2w FLAIR bold",
    )
    p.add_argument(
        "--group_by_base", action="store_true", default=True,
        help="Group GRE_echo-N variants into a single 'GRE' panel (default: True)",
    )
    p.add_argument(
        "--no_group_by_base", dest="group_by_base", action="store_false",
        help="One HTML per exact modality string instead of grouped by base",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.coords.exists():
        raise FileNotFoundError(args.coords)

    df = pd.read_csv(args.coords)
    if "scanner" not in df.columns and "scanner_model" in df.columns:
        df = df.rename(columns={"scanner_model": "scanner"})
    if "scanner" not in df.columns:
        df["scanner"] = "unknown"

    # Distinguish synthetic rows by modality prefix
    df["is_synth"] = df["modality"].str.startswith("syn-")
    synth_label = (
        df.loc[df["is_synth"], "image_path"]
        .dropna()
        .str.extract(r"(synthetic_[^/]+)")[0]
        .dropna()
        .iloc[0]
        if df["is_synth"].any()
        else "synthetic"
    )

    df_synth = df[df["is_synth"]].copy()
    df_orig  = df[~df["is_synth"]].copy()

    # Optionally collapse multi-echo GRE etc. to base modality
    if args.group_by_base:
        df_orig["modality_group"] = df_orig["modality"].apply(base_mod)
    else:
        df_orig["modality_group"] = df_orig["modality"]

    groups = sorted(df_orig["modality_group"].unique())
    log.info("Original modality groups: %s", groups)

    df_t1w = df_orig[df_orig["modality_group"] == "T1w"].copy()
    if df_t1w.empty:
        raise ValueError("No T1w rows found in coords CSV")
    log.info("T1w rows: %d", len(df_t1w))

    targets = args.targets if args.targets else [g for g in groups if g != "T1w"]
    log.info("Target contrasts: %s", targets)

    for tgt in targets:
        df_tgt = df_orig[df_orig["modality_group"] == tgt].copy()
        if df_tgt.empty:
            log.warning("No rows for target '%s', skipping", tgt)
            continue
        log.info("  %s: %d rows", tgt, len(df_tgt))
        out = args.output_dir / f"umap_joint_3d_T1w_vs_{tgt}_synth.html"
        make_pair_html(df_t1w, df_tgt, df_synth, tgt, synth_label, out)

    log.info("Done. %d pair plots written to %s", len(targets), args.output_dir)


if __name__ == "__main__":
    main()
