#!/usr/bin/env python3
"""Qualitative grid and four-way audit figures from already generated outputs.

  figs/qual.png            baseline (top) vs erased (bottom)    (paper Figs. 6-7, notebook cell 18)
  figs/audit.png           four-way LLaVA audit buckets         (paper Figs. 2, 8, 9, notebook cell 20)
  figs/audit_examples.png  one example image per audit bucket   (paper Fig. 10)

Needs scripts/generate.py and scripts/evaluate.py to have run. No GPU needed.
Example:
    python scripts/figures.py --model sd3.5-medium
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from _common import C, parser, setup  # noqa: E402

SHOW = [("Dogs", "object", "Van_Gogh", "Dogs"), ("Cats", "object", "Cartoon", "Cats"),
        ("Towers", "object", "Watercolor", "Towers"),
        ("Van_Gogh", "style", "Van_Gogh", "Dogs"), ("Watercolor", "style", "Watercolor", "Cats")]
BUCKETS = ["clean_removal", "replaced", "target_remains", "destroyed"]
COLORS = ["#2a9d8f", "#8ab17d", "#e9c46a", "#e76f51"]


def qualitative(spec, paths):
    sd0 = C.EVAL_SEEDS[0]
    fig, ax = plt.subplots(2, len(SHOW), figsize=(3.4 * len(SHOW), 7.2))
    for c, (concept, _kind, st, ob) in enumerate(SHOW):
        for r, (method, name) in enumerate((("none", "none"), ("ours", concept))):
            p = paths.image(method, name, st, ob, sd0)
            if os.path.exists(p):
                ax[r, c].imshow(Image.open(p))
            ax[r, c].axis("off")
        ax[0, c].set_title(f"erase {concept}\n({st}/{ob})", fontsize=9)
    plt.suptitle(f"Joint entry-point projection erasure — {spec.model_id}")
    plt.tight_layout()
    out = os.path.join(paths.figs, "qual.png")
    plt.savefig(out, dpi=150); plt.close()
    print("wrote", out)


def audit(paths):
    df = pd.read_csv(paths.classifications)
    df["bucket"] = df["bucket"].fillna("")
    aud = df[(df.method == "ours") & (df.bucket != "")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, kind in enumerate(("object", "style")):
        g = aud[aud.kind == kind]
        frac = [100 * (g.bucket == b).mean() if len(g) else 0 for b in BUCKETS]
        axs[i].bar([b.replace("_", "\n") for b in BUCKETS], frac, color=COLORS)
        axs[i].set_title(f"{kind} targets (n={len(g)})"); axs[i].set_ylabel("%")
        for x, v in enumerate(frac):
            axs[i].text(x, v + 1, f"{v:.0f}", ha="center", fontsize=9)
    plt.suptitle("What did erasure produce? (LLaVA audit of erased target images)")
    plt.tight_layout()
    out = os.path.join(paths.figs, "audit.png")
    plt.savefig(out, dpi=150); plt.close()
    print("wrote", out)

    strip = aud.groupby("bucket").head(1)
    if len(strip):
        fig, axs2 = plt.subplots(1, len(strip), figsize=(3.2 * len(strip), 3.4))
        axs2 = np.atleast_1d(axs2)
        for a, (_, r) in zip(axs2, strip.iterrows()):
            a.imshow(Image.open(r.path)); a.axis("off"); a.set_title(r.bucket, fontsize=9)
        plt.tight_layout()
        out = os.path.join(paths.figs, "audit_examples.png")
        plt.savefig(out, dpi=150); plt.close()
        print("wrote", out)


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    spec, paths, _ = setup(args)
    qualitative(spec, paths)
    if os.path.exists(paths.classifications):
        audit(paths)
    else:
        print(f"{paths.classifications} not found: skipping audit figures (run scripts/evaluate.py)")


if __name__ == "__main__":
    main()
