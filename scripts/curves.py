#!/usr/bin/env python3
"""Operating curves: UA, OTR and coherence vs erasure strength (paper Fig. 3, Figs. 11-12).

Re-runs the selected configuration at strengths 0.5-3.0 on the proxy grid for
Dogs (object) and Van_Gogh (style), scores with LLaVA and plots the curves.
Per-carrier strengths are scaled proportionally. Mirrors notebook cell 19.
Writes figs/curve.png and tables/curve.csv. Example:
    python scripts/curves.py --model flux-schnell
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from _common import C, free_gpu, parser, setup  # noqa: E402
from carrier_erasure import benchmark as B  # noqa: E402
from carrier_erasure.erasure import make_eraser  # noqa: E402
from carrier_erasure.llava_eval import COHERENCE_QUESTION, LLaVA  # noqa: E402
from carrier_erasure.metrics import add_norm_columns, norm_label, pct  # noqa: E402
from carrier_erasure.prompts import bench_prompt  # noqa: E402

CURVE_TARGETS = (("object", "Dogs"), ("style", "Van_Gogh"))


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    spec, paths, device = setup(args)
    selection = B.load_selection(paths)
    er = make_eraser(spec, device, args.cpu_offload)

    curve_rows = []
    for kind, concept in CURVE_TARGETS:
        base_cfg = dict(selection[kind]["cfg"])
        sub_u = B.get_sub(er, paths, concept, kind, B.anchor_for(kind, base_cfg))
        for s in C.CURVE_STRENGTHS:
            cfg = dict(base_cfg); cfg["strength"] = s
            if cfg.get("strengths"):
                base_s = base_cfg.get("strength", 1.0) or 1.0
                cfg["strengths"] = {k: v * s / base_s for k, v in base_cfg["strengths"].items()}
            for st, ob, sd in C.PROXY:
                fp = paths.image(f"curve_s{s}", concept, st, ob, sd)
                if not os.path.exists(fp):
                    B.ours_generate(er, bench_prompt(st, ob), sd, sub_u, cfg, concept, kind).save(fp)
                curve_rows.append(dict(kind=kind, concept=concept, s=s, style=st, obj=ob, path=fp))

    er.pipe.to("cpu"); free_gpu()
    llava = LLaVA()
    crows = [{**r,
              "pred_style": llava.classify(r["path"], C.FULL_STYLES, "style"),
              "pred_obj": llava.classify(r["path"], C.FULL_OBJECTS, "object"),
              "coherent": bool(llava.yes(r["path"], COHERENCE_QUESTION))}
             for r in tqdm(curve_rows, desc="curve LLaVA")]
    llava.unload()

    cdf = add_norm_columns(pd.DataFrame(crows), cols=("pred_style", "pred_obj", "style", "obj", "kind", "concept"))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for i, (kind, concept) in enumerate(CURVE_TARGETS):
        cn = norm_label(concept)
        g = cdf[cdf.kind_n == norm_label(kind)]
        ua, otr, coh = [], [], []
        for s in C.CURVE_STRENGTHS:
            gs = g[g.s == s]
            if kind == "object":
                t = gs[gs.obj_n == cn]
                ua.append(pct(t.pred_obj_n != cn)); otr.append(pct(t.pred_style_n == t.style_n))
            else:
                t = gs[gs.style_n == cn]
                ua.append(pct(t.pred_style_n != cn)); otr.append(pct(t.pred_obj_n == t.obj_n))
            coh.append(pct(gs.coherent))
        axs[i].plot(C.CURVE_STRENGTHS, ua, "o-", label="UA (removal)")
        axs[i].plot(C.CURVE_STRENGTHS, otr, "s-", label="OTR (scene kept)")
        axs[i].plot(C.CURVE_STRENGTHS, coh, "^--", label="coherent")
        axs[i].set_title(f"{kind}: {concept}")
        axs[i].set_xlabel("strength s"); axs[i].set_ylim(0, 105)
        axs[i].legend(fontsize=8); axs[i].grid(alpha=.3)
    plt.suptitle("Operating curve — removal vs retention vs coherence")
    plt.tight_layout()
    out = os.path.join(paths.figs, "curve.png")
    plt.savefig(out, dpi=150); plt.close()
    cdf.to_csv(os.path.join(paths.tables, "curve.csv"), index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
