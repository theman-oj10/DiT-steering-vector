"""
Visualization utilities for plotting and grid viewing
"""

from logging import error
import os
import math
import re
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def load_csv_any(path: Path) -> pd.DataFrame:
    """Load CSV with header awareness"""
    # Always try to read with header first
    df = pd.read_csv(path)
    
    # Check if we got valid column names
    if not all(str(c).startswith("Unnamed") for c in df.columns):
        # Valid header found, use it
        return df
    
    # No valid header, use fallback column names
    # (This should rarely happen if CSV is properly formatted)
    cols = [
        "concept", "extracted_layer", "start_layer", "end_layer",
        "weight", "steer_target", "seed", "base_prompt", "neg_prompt",
        "delta_clip_score", "lpips", "image_path", "notes"
    ]
    return pd.read_csv(path, names=cols, header=None, on_bad_lines="skip")


def parse_weight(w):
    """Parse weight values from various formats"""
    if pd.isna(w):
        return np.nan
    s = str(w).strip()
    try:
        return float(s)
    except Exception:
        pass
    neg = s.startswith("m")
    s2 = s[1:] if neg else s
    s2 = s2.replace("p", ".")
    try:
        val = float(s2)
        return -val if neg else val
    except Exception:
        return np.nan


def infer_extracted_from_filename(path_str):
    """Extract layer info from filename pattern"""
    m = re.search(r"_ext(\d+)_", str(path_str))
    return int(m.group(1)) if m else np.nan


def load_image(path_str: str, images_dir: Path = None):
    """Load image from path with fallback to images directory"""
    p = Path(path_str) if path_str else None
    if p and p.exists():
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            pass
    if p and images_dir:
        cand = images_dir / p.name
        if cand.exists():
            try:
                return Image.open(cand).convert("RGB")
            except Exception:
                pass
    return None


def fmt_metrics(r):
    """Format metrics for display"""
    clip_v = r.get("delta_clip_score", np.nan)
    lp_v = r.get("lpips", np.nan)
    dino_v = r.get("dino_sim", np.nan)

    clip_txt = "—" if pd.isna(clip_v) else f"{clip_v:.3f}"
    lpips_txt = "—" if pd.isna(lp_v) else f"{lp_v:.3f}"
    dino_txt = "-" if pd.isna(dino_v) else f"{dino_v:.3f}"
    return clip_txt, lpips_txt, dino_txt


def create_concept_grid(
    results_csv_path: Path,
    images_dir: Path,
    cfg_concepts=None,
    cfg_extracted=None,
    cfg_start_layers=None,
    cfg_end_layers=None,
    cfg_targets=["image", "text", "both"],
    cfg_weights=None,
    cfg_seeds=None,
    max_images_per_concept=None
):
    """Create concept grids for visualization"""
    
    # Load and process data
    df = load_csv_any(results_csv_path)
    print(f"Loaded {len(df)} rows from {results_csv_path}")

    # Ensure required columns exist
    for c in ["concept", "extracted_layer", "start_layer", "end_layer", "weight", "steer_target", "seed",
              "image_path", "delta_clip_score", "lpips", "base_prompt", "neg_prompt", "notes"]:
        if c not in df.columns:
            df[c] = np.nan

    # Clean data
    df["concept"] = df["concept"].fillna("unknown").astype(str)
    df["steer_target"] = df["steer_target"].fillna("").astype(str)
    df["image_path"] = df["image_path"].fillna("").astype(str)
    df["weight"] = df["weight"].apply(parse_weight)

    for col in ["start_layer", "end_layer", "extracted_layer", "seed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["delta_clip_score", "lpips"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # If extracted_layer missing, infer from filename
    if df["extracted_layer"].isna().all():
        df["extracted_layer"] = df["image_path"].apply(infer_extracted_from_filename)
        df.loc[df["extracted_layer"].isna(), "extracted_layer"] = df["start_layer"]

    # Filter data
    q = df.copy()
    print(f"\nFiltering {len(q)} rows...")
    print(f"  cfg_targets: {cfg_targets}")
    
    if cfg_concepts:
        q = q[q["concept"].isin(cfg_concepts)]
        print(f"  After cfg_concepts filter: {len(q)} rows")
    if cfg_extracted:
        q = q[q["extracted_layer"].isin(cfg_extracted)]
        print(f"  After cfg_extracted filter: {len(q)} rows")
    if cfg_start_layers:
        q = q[q["start_layer"].isin(cfg_start_layers)]
        print(f"  After cfg_start_layers filter: {len(q)} rows")
    if cfg_end_layers:
        q = q[q["end_layer"].isin(cfg_end_layers)]
        print(f"  After cfg_end_layers filter: {len(q)} rows")
    if cfg_targets:
        print(f"  Applying cfg_targets filter: {cfg_targets}")
        print(f"  Unique steer_target values before filter: {q['steer_target'].unique().tolist()}")
        q = q[q["steer_target"].isin(cfg_targets)]
        print(f"  After cfg_targets filter: {len(q)} rows")
    if cfg_weights:
        q = q[q["weight"].isin(cfg_weights)]
        print(f"  After cfg_weights filter: {len(q)} rows")
    if cfg_seeds:
        q = q[q["seed"].isin(cfg_seeds)]
        print(f"  After cfg_seeds filter: {len(q)} rows")

    if len(q) == 0:
        print("\n=== Debugging Info ===")
        print(f"Original dataframe has {len(df)} rows")
        print(f"Columns in CSV: {df.columns.tolist()}")
        try:
            print(f"Unique concepts in CSV: {df['concept'].unique().tolist()}")
        except:
            print(f"Error reading concept column")
        try:
            print(f"Unique steer_targets in CSV: {df['steer_target'].unique().tolist()}")
        except:
            print(f"Error reading steer_target column")
        try:
            print(f"Unique extracted_layers in CSV: {df['extracted_layer'].dropna().unique().tolist()}")
        except:
            print(f"Error reading extracted_layer column")
        print(f"cfg_targets filter: {cfg_targets}")
        print(f"cfg_concepts filter: {cfg_concepts}")
        print(f"cfg_extracted filter: {cfg_extracted}")
        print(f"Rows after filtering: {len(q)}")
        raise RuntimeError("No rows after filtering. Relax filters or check CSV columns.")

    # Create grids for each concept
    concepts = sorted(q["concept"].unique(), key=lambda x: x.lower())
    thumb_max = 224

    for concept in concepts:
        sub = q[q["concept"] == concept].copy()
        layers = sorted(sub["extracted_layer"].dropna().unique().tolist())
        if cfg_extracted:
            layers = [l for l in layers if l in cfg_extracted]
        if not layers:
            print(f"[skip] {concept}: no extracted_layer after filtering.")
            continue

        # Decide weight order
        if cfg_weights:
            weight_order = list(cfg_weights)
        else:
            weight_order = sorted(sub["weight"].dropna().unique().tolist())

        if not weight_order:
            print(f"[skip] {concept}: no weights after filtering.")
            continue

        # Get steer targets from data if not specified
        if cfg_targets:
            steer_targets = cfg_targets
        else:
            steer_targets = sorted(sub["steer_target"].dropna().unique().tolist())
        
        if not steer_targets:
            print(f"[skip] {concept}: no steer_targets after filtering.")
            continue

        # Grid dimensions
        rows = len(layers) * len(steer_targets)
        cols = len(weight_order)

        # Create figure
        fig_w = cols * 3.0
        fig_h = rows * 2.8
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
        if rows == 1:
            axes = np.array([axes])
        axes = np.array(axes).reshape(rows, cols)

        # Suptitle
        fig.suptitle(f"Concept: {concept}", fontsize=14, y=0.995)

        # Column headers (weights)
        for j, w in enumerate(weight_order):
            axes[0, j].set_title(f"w={w}", fontsize=9, pad=6)

        # Fill grid
        for li, layer in enumerate(layers):
            base_row = li * len(steer_targets)

            # Layer annotation
            axes[base_row, 0].text(
                -0.35, 0.5, f"ext={int(layer)}",
                ha="right", va="center", rotation=90, fontsize=10,
                transform=axes[base_row, 0].transAxes
            )

            for ti, tgt in enumerate(steer_targets):
                r_idx = base_row + ti

                # Target annotation
                axes[r_idx, -1].text(
                    1.04, 0.5, tgt,
                    ha="left", va="center", fontsize=9,
                    transform=axes[r_idx, -1].transAxes
                )

                for cj, w in enumerate(weight_order):
                    ax = axes[r_idx, cj]
                    ax.axis("off")

                    # Find matching row
                    cand = sub[
                        (sub["extracted_layer"] == layer) &
                        (sub["steer_target"] == tgt) &
                        (sub["weight"] == w)
                    ].sort_values("image_path", ascending=False).head(1)

                    if len(cand) == 0:
                        ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=12)
                        continue

                    rec = cand.iloc[0]
                    img = load_image(str(rec.get("image_path", "")), images_dir)
                    if img is not None:
                        img = ImageOps.contain(img, (thumb_max, thumb_max))
                        ax.imshow(img)
                    else:
                        ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=10)

                    # Metrics text
                    clip_txt, lpips_txt, dino_txt = fmt_metrics(rec)
                    ax.text(
                        0.02, -0.04, f"ΔC={clip_txt} · L={lpips_txt} · D={dino_txt}",
                        transform=ax.transAxes, ha="left", va="top", fontsize=7
                    )

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()


def create_line_charts(results_csv_path: Path, output_dir: Path = None):
    """Create line charts for metrics vs weight"""
    
    if output_dir is None:
        output_dir = results_csv_path.parent / "plots"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    df = pd.read_csv(results_csv_path)

    # Clean data
    for c in ["extracted_layer", "start_layer", "end_layer"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["delta_clip_score"] = pd.to_numeric(df["delta_clip_score"], errors="coerce")
    df["lpips"] = pd.to_numeric(df["lpips"], errors="coerce")
    df["dino_sim"] = pd.to_numeric(df.get("dino_sim", np.nan), errors="coerce")

    df = df.dropna(subset=["concept", "extracted_layer", "weight", "steer_target"])

    # Create combo key
    df["combo"] = df["concept"].astype(str) + "_L" + df["extracted_layer"].astype(int).astype(str)

    # Aggregate duplicates
    agg = (
        df.groupby(["combo", "weight", "steer_target"], as_index=False)[["delta_clip_score", "lpips", "dino_sim"]]
          .mean()
          .sort_values(["combo", "steer_target", "weight"])
    )

    # Combined metric
    agg["clip_per_lpips"] = np.where(
        (agg["lpips"] > 0) & np.isfinite(agg["lpips"]),
        agg["delta_clip_score"] / agg["lpips"],
        np.nan
    )

    def _plot_metric_for_combo(sub, metric_col, y_label, file_suffix):
        """Plot metric vs weight for a single combo"""
        wide = sub.pivot_table(index="weight", columns="steer_target", values=metric_col, aggfunc="mean")
        wide = wide.sort_index()
        wide = wide.dropna(axis=1, how="all")

        if wide.empty:
            return None

        plt.figure()
        for col in wide.columns:
            series = wide[col].replace([np.inf, -np.inf], np.nan).dropna()
            if not series.empty:
                plt.plot(series.index.values, series.values, marker="o", label=str(col))

        plt.xlabel("weight")
        plt.ylabel(y_label)
        plt.title(f"{sub['combo'].iloc[0]} — {y_label} vs weight")
        plt.legend(title="steer_target")
        plt.grid(True, which="both", axis="both", alpha=0.3)

        fname = f"{sub['combo'].iloc[0]}_{file_suffix}.png".replace("/", "_")
        path = output_dir / fname
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.show()
        plt.close()
        return path

    # Generate plots
    saved = []
    for combo, sub in agg.groupby("combo"):
        # LPIPS
        p1 = _plot_metric_for_combo(sub, "lpips", "LPIPS (lower is better)", "lpips_by_weight")
        if p1:
            saved.append(p1)
        # CLIP delta
        p2 = _plot_metric_for_combo(sub, "delta_clip_score", "delta CLIPScore (higher is better)", "clip_by_weight")
        if p2:
            saved.append(p2)
        # Combined
        p3 = _plot_metric_for_combo(sub, "clip_per_lpips", "CLIP / LPIPS (higher is better)", "clip_per_lpips_by_weight")
        if p3:
            saved.append(p3)
        p4 = _plot_metric_for_combo(sub, "dino_sim", "DinoV2 Similarity (Higher is Better)", "dino_by_weight" )
        if p4:
            saved.append(p4)
    print(f"Saved {len(saved)} plots to: {output_dir}")
    return saved


def create_scatter_plots(results_csv_path: Path, output_dir: Path = None):
    """Create scatter plots for Pareto analysis"""
    
    if output_dir is None:
        output_dir = results_csv_path.parent / "plots"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    df = pd.read_csv(results_csv_path)

    # Clean data
    for c in ["extracted_layer", "weight", "delta_clip_score", "lpips"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["concept", "extracted_layer", "weight", "steer_target", "delta_clip_score", "lpips"])

    df["combo"] = df["concept"].astype(str) + "_L" + df["extracted_layer"].astype(int).astype(str)

    # Aggregate
    agg = (
        df.groupby(["combo", "weight", "steer_target"], as_index=False)[["delta_clip_score", "lpips"]]
          .mean()
    )

    def size_from_weight(w, wmin=None, wmax=None):
        """Map weights to point sizes"""
        w = np.asarray(w, dtype=float)
        if wmin is None:
            wmin = np.nanmin(w)
        if wmax is None:
            wmax = np.nanmax(w)
        span = max(wmax - wmin, 1e-9)
        norm = (w - wmin) / span
        return 30.0 + 120.0 * norm

    def pareto_frontier_mask(x_lpips, y_clip):
        """Calculate Pareto frontier mask"""
        order = np.argsort(x_lpips)
        x_sorted = np.asarray(x_lpips)[order]
        y_sorted = np.asarray(y_clip)[order]
        mask_sorted = np.zeros_like(y_sorted, dtype=bool)
        best = -np.inf
        for i, y in enumerate(y_sorted):
            if y >= best:
                mask_sorted[i] = True
                best = y
        mask = np.zeros_like(mask_sorted, dtype=bool)
        mask[order] = mask_sorted
        return mask

    saved = []
    for combo, sub in agg.groupby("combo"):
        plt.figure()
        
        # Scatter plot
        for tgt, g in sub.groupby("steer_target"):
            plt.scatter(
                g["lpips"].values, g["delta_clip_score"].values,
                s=size_from_weight(g["weight"].values, sub["weight"].min(), sub["weight"].max()),
                label=str(tgt), alpha=0.85
            )

        # Pareto frontier
        mask = pareto_frontier_mask(sub["lpips"].values, sub["delta_clip_score"].values)
        if mask.any():
            xf = sub["lpips"].values[mask]
            yf = sub["delta_clip_score"].values[mask]
            order = np.argsort(xf)
            plt.plot(xf[order], yf[order], marker="x", linewidth=2, linestyle="-", label="Pareto frontier")

        plt.xlabel("LPIPS (lower is better)")
        plt.ylabel("delta CLIPScore (higher is better)")
        plt.title(f"{combo} — Trade-off: CLIP vs LPIPS (size = weight)")
        plt.legend(title="steer_target")
        plt.grid(True, which="both", axis="both", alpha=0.3)
        plt.tight_layout()

        fname = f"{combo}_scatter_clip_vs_lpips.png".replace("/", "_")
        path = output_dir / fname
        plt.savefig(path, dpi=160)
        plt.show()
        plt.close()
        saved.append(path)

    print(f"Saved {len(saved)} scatter plots to: {output_dir}")
    return saved
