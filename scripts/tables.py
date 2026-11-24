#!/usr/bin/env python3
"""Build the results tables from tables/classifications.csv (paper Tables 1-4).

Writes, per target type:
  tables/results_<kind>.csv   joint carrier erasure on every target (Tables 1, 2)
  tables/controls_<kind>.csv  ours vs prompt-swap (vs negative-prompt on SD3.5)
                              on the controls subset (Tables 3, 4)
Values are mean ± std over seeds. Mirrors notebook cell 17. Example:
    python scripts/tables.py --model flux-schnell
"""
from __future__ import annotations

import os

import pandas as pd

from _common import C, parser, setup
from carrier_erasure.metrics import results_table


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    _, paths, _ = setup(args)
    if not os.path.exists(paths.classifications):
        raise SystemExit(f"{paths.classifications} not found: run scripts/evaluate.py first")
    df = pd.read_csv(paths.classifications)

    for kind, all_targets in (("object", C.OBJECT_TARGETS), ("style", C.STYLE_TARGETS)):
        ours = results_table(df[df.method == "ours"], all_targets, kind, C.EVAL_SEEDS)
        controls = results_table(df, [c for c in all_targets if c in C.BASELINE_TARGETS], kind, C.EVAL_SEEDS)
        for name, T in (("results", ours), ("controls", controls)):
            if not len(T):
                print(f"no rows for {name} / {kind}")
                continue
            if name == "controls":
                T = T[T.Concept != "AVERAGE"]
            out = os.path.join(paths.tables, f"{name}_{kind}.csv")
            T.to_csv(out, index=False)
            print(f"\n== {name} ({kind}) — {paths.model_key} ==\n{T.to_string(index=False)}\nwrote {out}")


if __name__ == "__main__":
    main()
