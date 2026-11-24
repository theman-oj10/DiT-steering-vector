#!/usr/bin/env python3
"""Classify every benchmark image with LLaVA-1.6 (paper Sec. 4.2). Resume-safe.

For each generated image: style and object predictions over the full label
sets, plus the four-way audit bucket on erased target images ("ours" only).
Writes tables/classifications.csv, checkpointing every 50 images. Does not load
the diffusion model. Mirrors notebook cell 16. Example:
    python scripts/evaluate.py --model flux-dev
"""
from __future__ import annotations

import os

import pandas as pd
from tqdm.auto import tqdm

from _common import C, parser, setup
from carrier_erasure import benchmark as B
from carrier_erasure.llava_eval import LLaVA
from generate import targets


def jobs(spec, paths, selection):
    for concept, kind, _ in targets(selection):
        methods = (["ours"] + (["pswap"] if concept in C.BASELINE_TARGETS else [])
                   + (["negp"] if (spec.family == "sd3" and concept in C.BASELINE_TARGETS) else []))
        for m in methods:
            for st, ob, sd in C.benchmark_grid():
                fp = paths.image(m, concept, st, ob, sd)
                if os.path.exists(fp):
                    yield dict(method=m, concept=concept, kind=kind, style=st, obj=ob, seed=sd, path=fp)


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    spec, paths, _ = setup(args)
    selection = B.load_selection(paths)
    out = paths.classifications
    rows, done = [], set()
    if os.path.exists(out):
        prev = pd.read_csv(out)
        rows, done = prev.to_dict("records"), set(prev.path)

    llava = LLaVA()
    todo = [j for j in jobs(spec, paths, selection) if j["path"] not in done]
    for i, r in enumerate(tqdm(todo, desc="LLaVA")):
        r["pred_style"] = llava.classify(r["path"], C.FULL_STYLES, "style")
        r["pred_obj"] = llava.classify(r["path"], C.FULL_OBJECTS, "object")
        is_target = (r["obj"] == r["concept"]) if r["kind"] == "object" else (r["style"] == r["concept"])
        r["bucket"] = llava.bucket(r["path"], r["concept"], r["kind"]) if (is_target and r["method"] == "ours") else ""
        rows.append(r)
        if (i + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    llava.unload()
    print("wrote", out)


if __name__ == "__main__":
    main()
