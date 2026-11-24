#!/usr/bin/env python3
"""Generate every benchmark image (paper Sec. 4.1, 4.3). Resume-safe.

Produces, on the 5-style x 8-object x 2-seed grid:
  - the unedited baseline grid (method "none"),
  - joint carrier erasure for every object and style target ("ours"),
  - the prompt-swap reference for the controls subset ("pswap"),
  - SD3.5 only: the native negative-prompt reference ("negp").

Uses the operating points chosen by scripts/opsearch.py. Mirrors notebook cell 15.
Example:
    python scripts/generate.py --model sd3.5-medium
"""
from __future__ import annotations

import os

from tqdm.auto import tqdm

from _common import C, free_gpu, parser, setup
from carrier_erasure import benchmark as B
from carrier_erasure.erasure import make_eraser
from carrier_erasure.prompts import bench_prompt, negative_concept_prompt, swap_prompt


def targets(selection):
    return ([(c, "object", selection["object"]["cfg"]) for c in C.OBJECT_TARGETS]
            + [(c, "style", selection["style"]["cfg"]) for c in C.STYLE_TARGETS])


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    spec, paths, device = setup(args)
    selection = B.load_selection(paths)
    er = make_eraser(spec, device, args.cpu_offload)
    grid = C.benchmark_grid()

    for st, ob, sd in tqdm(grid, desc="shared baseline"):
        fp = paths.image("none", "none", st, ob, sd)
        if not os.path.exists(fp):
            er.generate(bench_prompt(st, ob), sd, sub=None, mode="none").save(fp)

    for concept, kind, cfg in targets(selection):
        sub_c = B.get_sub(er, paths, concept, kind, B.anchor_for(kind, cfg))
        for st, ob, sd in tqdm(grid, desc=f"ours {concept}"):
            fp = paths.image("ours", concept, st, ob, sd)
            if not os.path.exists(fp):
                B.ours_generate(er, bench_prompt(st, ob), sd, sub_c, cfg, concept, kind).save(fp)
        free_gpu()

    for concept, kind, _ in targets(selection):
        if concept not in C.BASELINE_TARGETS:
            continue
        for st, ob, sd in tqdm(grid, desc=f"promptswap {concept}"):
            fp = paths.image("pswap", concept, st, ob, sd)
            if not os.path.exists(fp):
                er.generate(swap_prompt(st, ob, concept, kind), sd, sub=None, mode="none").save(fp)
        if spec.family == "sd3":
            for st, ob, sd in tqdm(grid, desc=f"negprompt {concept}"):
                fp = paths.image("negp", concept, st, ob, sd)
                if not os.path.exists(fp):
                    er.generate(bench_prompt(st, ob), sd, sub=None, mode="none",
                                negative_concept=negative_concept_prompt(concept, kind)).save(fp)
    print("images in", paths.img)


if __name__ == "__main__":
    main()
