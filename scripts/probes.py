#!/usr/bin/env python3
"""Recoverability and hard-concept probes (paper App. F and G).

  figs/recoverability.png  Dogs and Van Gogh paraphrases under the selected configs
  figs/hard_concept.png    Impressionism at strengths 1.0 / 1.5 / 2.0 vs baseline

Mirrors notebook cell 22. Example:
    python scripts/probes.py --model sd3.5-medium
"""
from __future__ import annotations

import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import parser, setup  # noqa: E402
from carrier_erasure import benchmark as B  # noqa: E402
from carrier_erasure.erasure import make_eraser  # noqa: E402
from carrier_erasure.prompts import bench_prompt  # noqa: E402

PARAS = {
    "Dogs": ["A puppy playing on the grass.",
             "A canine companion outdoors.",
             "A photo of a golden retriever in a park."],
    "Van_Gogh": ["A mountain painted with swirling starry brushstrokes.",
                 "A post-impressionist oil painting of a mountain.",
                 "A mountain like Starry Night."],
}
HARD, HARD_OBJECT = "Impressionism", "Dogs"
SEED = 188


def wrap(text, width):
    return "\n".join(textwrap.wrap(text, width=width))


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    spec, paths, device = setup(args)
    selection = B.load_selection(paths)
    er = make_eraser(spec, device, args.cpu_offload)

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    for r, (concept, kind) in enumerate((("Dogs", "object"), ("Van_Gogh", "style"))):
        cfg = selection[kind]["cfg"]
        sub_u = B.get_sub(er, paths, concept, kind, B.anchor_for(kind, cfg))
        for c, prompt in enumerate(PARAS[concept]):
            img = B.ours_generate(er, prompt, SEED, sub_u, cfg, concept, kind)
            axs[r, c].imshow(img); axs[r, c].axis("off")
            axs[r, c].set_title(f"Erase: {concept.replace('_', ' ')}\n{wrap(prompt, 38)}", fontsize=8)
            img.save(os.path.join(paths.figs, f"recoverability_{concept}_{c}.jpg"))
    plt.suptitle("Recoverability probes", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(paths.figs, "recoverability.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print("wrote", out)

    prompt = bench_prompt(HARD, HARD_OBJECT)
    sub_h = B.get_sub(er, paths, HARD, "style", "hypernym")
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.2))
    for c, (label, s_use, strength) in enumerate((("Baseline", None, 0.0), ("s=1.0", sub_h, 1.0),
                                                  ("s=1.5", sub_h, 1.5), ("s=2.0", sub_h, 2.0))):
        img = er.generate(prompt, SEED, sub=s_use, mode=("none" if s_use is None else "all"), strength=strength)
        axs[c].imshow(img); axs[c].axis("off")
        axs[c].set_title(f"{label}\n{wrap(prompt, 34)}", fontsize=8)
        img.save(os.path.join(paths.figs, f"hard_concept_{HARD}_s{strength}.jpg"))
    plt.suptitle(f"Hard concept probe: {HARD}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    out = os.path.join(paths.figs, "hard_concept.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
