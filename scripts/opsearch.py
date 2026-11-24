#!/usr/bin/env python3
"""Operating-point search (paper Sec. 4.4 / App. A).

For each target type, generates the proxy grid under every candidate
configuration, scores it with LLaVA by 0.4 UA + 0.2 (IRA + CRA + OTR), and
selects the best configuration whose proxy images are coherent >= 80% of the
time. Writes tables/opsearch_<kind>.csv and tables/selected_configs.json, which
the later scripts read.

Mirrors notebook cell 14. Example:
    python scripts/opsearch.py --model flux-schnell
"""
from __future__ import annotations

import os

import pandas as pd
from tqdm.auto import tqdm

from _common import C, free_gpu, parser, setup
from carrier_erasure import benchmark as B
from carrier_erasure.erasure import make_eraser
from carrier_erasure.llava_eval import COHERENCE_QUESTION, LLaVA
from carrier_erasure.metrics import add_norm_columns, composite, metr, pct
from carrier_erasure.prompts import bench_prompt


def run_opsearch(er, llava, paths, device, kind):
    concept = C.PROXY_TARGET[kind]
    cfgs = B.candidate_configs(kind, er.family)

    # Phase A: generate proxy images
    recs = []
    for name, cfg in cfgs:
        sub_u = B.get_sub(er, paths, concept, kind, B.anchor_for(kind, cfg))
        for st, ob, sd in C.PROXY:
            fp = paths.image(f"ops_{name}", concept, st, ob, sd)
            if not os.path.exists(fp):
                B.ours_generate(er, bench_prompt(st, ob), sd, sub_u, cfg, concept, kind).save(fp)
            recs.append(dict(config=name, style=st, obj=ob, seed=sd, path=fp))

    # Phase B: classify (diffusion pipeline parked on CPU while LLaVA runs)
    er.pipe.to("cpu"); free_gpu()
    rows = []
    for r in tqdm(recs, desc=f"opsearch {kind}"):
        rows.append({**r,
                     "pred_style": llava.classify(r["path"], C.FULL_STYLES, "style"),
                     "pred_obj": llava.classify(r["path"], C.FULL_OBJECTS, "object"),
                     "coherent": bool(llava.yes(r["path"], COHERENCE_QUESTION))})
    llava.unload(); er.pipe.to(device)

    df = add_norm_columns(pd.DataFrame(rows), cols=("pred_style", "pred_obj", "style", "obj"))
    out = []
    for name, _ in cfgs:
        g = df[df["config"] == name]
        if not len(g):
            print(f"no rows for config {name}; skipping")
            continue
        m = metr(g, concept, kind)
        out.append(dict(config=name, **{k: round(v, 1) for k, v in m.items()},
                        coherent=round(pct(g["coherent"]), 1), composite=composite(m)))
    if not out:
        raise SystemExit(f"opsearch produced no scored configs for {kind}")

    T = pd.DataFrame(out).sort_values("composite", ascending=False, na_position="last")
    print(T.to_string(index=False))
    T.to_csv(os.path.join(paths.tables, f"opsearch_{kind}.csv"), index=False)

    ok = T[T["coherent"] >= C.COHERENCE_MIN]
    best = (ok.iloc[0] if len(ok) else T.iloc[0])["config"]
    print(f"SELECTED {kind}: {best}  (inspect its proxy images before the full benchmark)")
    return {"name": best, "cfg": dict(cfgs)[best]}


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    spec, paths, device = setup(args)
    er = make_eraser(spec, device, args.cpu_offload)
    llava = LLaVA()
    selection = {kind: run_opsearch(er, llava, paths, device, kind) for kind in ("object", "style")}
    B.save_selection(paths, selection)
    print("wrote", paths.selected_configs)


if __name__ == "__main__":
    main()
