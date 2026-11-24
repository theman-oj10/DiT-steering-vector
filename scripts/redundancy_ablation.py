#!/usr/bin/env python3
"""Carrier redundancy ablation (paper Sec. 5.1, Figs. 1 and 5).

Erases one concept from each carrier alone, from the sequence carriers together,
and from all carriers jointly, on two evaluation prompts. FLUX modes:
none / pooled / seq / all; SD3.5 modes: none / pooled / clip_seq / t5_seq /
seq / all. Mirrors notebook cell 7. Examples:
    python scripts/redundancy_ablation.py --model sd3.5-medium --concept Dogs
    python scripts/redundancy_ablation.py --model flux-schnell --concept Towers --strength 1.5
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import parser, setup  # noqa: E402
from carrier_erasure import benchmark as B  # noqa: E402
from carrier_erasure.erasure import make_eraser  # noqa: E402
from carrier_erasure.prompts import make_eval_prompts  # noqa: E402


def main():
    p = parser(__doc__.splitlines()[0])
    p.add_argument("--concept", default="Dogs", help="concept to erase, e.g. Dogs, Towers, Van_Gogh")
    p.add_argument("--kind", default="object", choices=["object", "style"])
    p.add_argument("--anchor", default="empty_scene",
                   choices=["empty_scene", "hypernym", "something", "no_animal"], help="object anchor (ignored for styles)")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=188)
    args = p.parse_args()
    spec, paths, device = setup(args)
    er = make_eraser(spec, device, args.cpu_offload)

    anchor = args.anchor if args.kind == "object" else "hypernym"
    sub = B.get_sub(er, paths, args.concept, args.kind, anchor)
    prompts = make_eval_prompts(args.concept, args.kind)
    modes = (["none", "pooled", "seq", "all"] if spec.family == "flux"
             else ["none", "pooled", "clip_seq", "t5_seq", "seq", "all"])

    fig, ax = plt.subplots(len(prompts), len(modes), figsize=(3.8 * len(modes), 4.2 * len(prompts)))
    if len(prompts) == 1:
        ax = np.array([ax])
    for r, prompt in enumerate(prompts):
        for c, mode in enumerate(modes):
            img = er.generate(prompt, args.seed, sub=sub, mode=mode, strength=args.strength,
                              token_frac=1.0, neutralize_tokens=False, negative_concept=None)
            ax[r, c].imshow(img); ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(mode, fontsize=11)
            img.save(os.path.join(paths.figs, f"redund_{args.concept}_{anchor}_{mode}_p{r}.jpg"))
    plt.suptitle(f"Clean redundancy ablation — target={args.concept}, anchor={anchor}\n"
                 f"strength={args.strength}, hard=False, neg=False")
    plt.tight_layout()
    out = os.path.join(paths.figs, f"redundancy_{args.concept}_{anchor}.png")
    plt.savefig(out, dpi=140); plt.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
